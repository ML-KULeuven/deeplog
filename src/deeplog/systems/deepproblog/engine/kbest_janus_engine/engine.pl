%  Copyright (c) 2024-2026. KU Leuven
% K-best heap-based prover. Adapted from DeepProbLog's engine_heap.pl, but
% targets deeplog's ID:fact/2 and ID:rule/2 program representation and
% defers formula construction to the Python-side factory via py_call.

:- use_module(library(heaps)).
:- use_module("../janus_engine/janus_translation.pl").
:- use_module("../janus_engine/builtins.pl").
:- include("heuristics.pl").

:- dynamic rule/2.
:- dynamic fact/2.
:- dynamic extern_builtin/2.
:- dynamic engine_id/2.

% --- Entry point -----------------------------------------------------------

% Enumerates up to K top-ranked proof formulas for Query; each solution
% returns the corresponding (GroundQuery, Formula) pair. Python collects the
% set and folds them with factory.disjoin to build the per-goal result.
kbest_prove_query(ID, QuerySymbol, Factory, K, HeuristicName, GroundQuery, Formula) :-
    from_symbol(QuerySymbol, Query),
    initial_heuristic(HeuristicName, InitialHeur),
    py_call(Factory:get_true(), InitFormula),
    singleton_heap(Heap, InitialHeur, n([ID:Query], Query, InitFormula, [])),
    step(Heap, K, Factory, [], Proofs),
    member(GroundTerm-Formula, Proofs),
    to_symbol(GroundTerm, GroundQuery).

initial_heuristic(pp, H) :- partial_probability(H).
initial_heuristic(gm, H) :- geometric_mean(H).

% --- Heap step loop --------------------------------------------------------

step(Heap, _, _, Proofs, FinalProofs) :-
    empty_heap(Heap), !,
    reverse(Proofs, FinalProofs).

step(_, K, _, Proofs, FinalProofs) :-
    length(Proofs, L), L >= K, !,
    reverse(Proofs, FinalProofs).

step(Heap, K, Factory, Acc, Proofs) :-
    get_from_heap(Heap, Heur, n(Goals, GroundTerm, Formula, Assigned), RestHeap),
    (   Goals == []
    ->  step(RestHeap, K, Factory, [GroundTerm-Formula|Acc], Proofs)
    ;   findall(NewHeur-NewNode,
                expand(Heur, n(Goals, GroundTerm, Formula, Assigned), Factory, NewHeur, NewNode),
                Children),
        foldl(add_pair, Children, RestHeap, NewHeap),
        step(NewHeap, K, Factory, Acc, Proofs)
    ).

add_pair(Key-Value, H0, H1) :- add_to_heap(H0, Key, Value, H1).

% --- Node expansion --------------------------------------------------------
%
% A node is n(Goals, GroundTerm, Formula, Assigned) paired with a heuristic
% key in the heap. Goals is a list of ID:Goal items still to prove;
% GroundTerm is the top-level query term (becomes ground as the proof
% unifies); Formula is the partial proof formula built by conjoining fact
% leaves; Assigned is the per-branch list of Goal-Bool commitments for
% probabilistic facts (Goal ground; true if committed positively, false
% if committed negatively via NAF hoisting).

% Top-level conjunction: push both goals in head order.
expand(Heur, n([ID:','(G1,G2)|Rest], GT, F, A), _, Heur,
       n([ID:G1, ID:G2|Rest], GT, F, A)) :- !.

% Disjunction: spawn a child for each branch.
expand(Heur, n([ID:';'(G1,_G2)|Rest], GT, F, A), _, Heur,
       n([ID:G1|Rest], GT, F, A)).
expand(Heur, n([ID:';'(_G1,G2)|Rest], GT, F, A), _, Heur,
       n([ID:G2|Rest], GT, F, A)) :- !.

% `true` — consume with no effect.
expand(Heur, n([_:true|Rest], GT, F, A), _, Heur, n(Rest, GT, F, A)) :- !.

% Negation as failure, restricted to the deterministic part of the proof.
% Resolves Goal against the per-branch assignment A; on the first
% uncommitted probabilistic fact encountered, hoists that choice into the
% outer heap and re-attempts NAF on the next pop. This preserves
% ProbLog's distribution semantics: NAF's truth in a world is determined
% by RV values that must be committed beforehand.
%
% Two kinds of hoist:
%   - Boolean: an ordinary probabilistic fact splits into two children
%     (Goal-true / Goal-false).
%   - Categorical: an AD splits into N+1 children — one per branch
%     value (cat(CatId)-Idx) plus one "none" child (cat(CatId)-none)
%     whose probability is the residual 1 - Σ P_i.
expand(Heur, n([ID:not(Goal)|Rest], GT, F, A), Factory, NewHeur, NewNode) :- !,
    catch(
        catch(
            (   naf_resolve(ID:Goal, A)
            ->  Outcome = success
            ;   Outcome = failure
            ),
            naf_needs_choice(ChoiceGoal, ChoiceLabel),
            Outcome = needs_choice(ChoiceGoal, ChoiceLabel)
        ),
        naf_needs_categorical_choice(CID, CatId),
        Outcome = needs_cat_choice(CID, CatId)
    ),
    naf_dispatch(Outcome, Heur, n([ID:not(Goal)|Rest], GT, F, A),
                 Factory, NewHeur, NewNode).

% Pure Prolog builtin: execute and consume.
expand(Heur, n([_:Goal|Rest], GT, F, A), _, Heur, n(Rest, GT, F, A)) :-
    nonvar(Goal),
    allowed_builtin(Goal), !,
    call(Goal).

% Python-registered extern builtin: enumerate solutions, unify, consume.
expand(Heur, n([ID:Goal|Rest], GT, F, A), _, Heur, n(Rest, GT, F, A)) :-
    nonvar(Goal),
    functor(Goal, Name, Arity),
    ID:engine_id(Engine, EngineID),
    EngineID:extern_builtin(Name, Arity), !,
    to_symbol(Goal, GoalSym),
    py_call(Engine:'_call_builtin'(GoalSym), ResultSyms),
    member(ResultSym, ResultSyms),
    from_symbol(ResultSym, Unified),
    Goal = Unified.

% Categorical (annotated-disjunction) fact: enforces per-proof mutual
% exclusivity. AD branches arrive in the program with their probability
% wrapped as '@cat'(InnerLabel, CatId, ValueIdx). The clause head
% matches this shape directly, so Prolog backtracks over each AD branch
% as a separate proof; the non-categorical fact clause below is guarded
% with `Label \= '@cat'(_,_,_)` to keep the two clauses mutually
% exclusive.
%
% Three cases against the per-branch assignment A:
%   (a) cat(CatId)-ValueIdx already in A → this exact branch was
%       committed earlier in the proof; the leaf is already in F and
%       the heuristic is already updated, so consume without redoing
%       work.
%   (b) cat(CatId)-Other already in A with Other \= ValueIdx → another
%       outcome of the same AD has been committed (different value, or
%       the "none" outcome from a NAF hoist); the proof is inconsistent,
%       so fail the branch.
%   (c) No cat(CatId) entry in A → fresh consumption; unwrap, conjoin
%       the leaf, update the heuristic with the inner probability, and
%       record cat(CatId)-ValueIdx in A.
expand(Heur, n([ID:Goal|Rest], GT, F, A), Factory, NewHeur,
       n(Rest, GT, NewF, NewA)) :-
    ID:fact(Goal, '@cat'(InnerLabel, CatId, ValueIdx)),
    (   member(cat(CatId)-Choice, A)
    ->  Choice == ValueIdx,
        NewHeur = Heur, NewF = F, NewA = A
    ;   to_symbol(Goal, GoalSym),
        to_symbol(InnerLabel, LabelSym),
        atom_string(CatId, CatIdStr),
        py_call(Factory:get_categorical_value(GoalSym, LabelSym,
                                              CatIdStr, ValueIdx),
                Leaf),
        py_call(Factory:conjoin(F, Leaf), NewF),
        py_call(Factory:get_scalar_probability(LabelSym), P),
        add_probability_to_heuristic(P, Heur, NewHeur),
        NewA = [cat(CatId)-ValueIdx|A]
    ).

% Fact: build the leaf formula, conjoin into the accumulator, update
% the heuristic with the fact's scalar probability, and record the
% positive commitment in the assignment so later NAF subproofs can
% see it.
%
% The `Label \= '@cat'(_,_,_)` guard keeps this clause mutually
% exclusive with the categorical-fact clause above.
%
% Three cases against the per-branch assignment A:
%   (a) Goal-false already in A → this branch was committed by a NAF
%       hoist to Goal=false; consuming the fact positively would be
%       inconsistent, so fail the branch.
%   (b) Goal-true already in A → already committed positively (either
%       by a prior fact consumption or by a NAF hoist's true branch);
%       the leaf is already in F and the heuristic is already updated,
%       so consume the goal without re-conjoining or re-multiplying.
%   (c) Goal not in A → fresh consumption; conjoin the leaf, update
%       the heuristic, and record Goal-true in A.
expand(Heur, n([ID:Goal|Rest], GT, F, A), Factory, NewHeur,
       n(Rest, GT, NewF, NewA)) :-
    ID:fact(Goal, Label),
    Label \= '@cat'(_,_,_),
    (   member(Goal-false, A)
    ->  fail
    ;   member(Goal-true, A)
    ->  NewHeur = Heur, NewF = F, NewA = A
    ;   to_symbol(Goal, GoalSym),
        to_symbol(Label, LabelSym),
        py_call(Factory:get_boolean(GoalSym, LabelSym), Leaf),
        py_call(Factory:conjoin(F, Leaf), NewF),
        py_call(Factory:get_scalar_probability(LabelSym), P),
        add_probability_to_heuristic(P, Heur, NewHeur),
        NewA = [Goal-true|A]
    ).

% Rule: replace the head goal with the rule body (a nested conjunction).
expand(Heur, n([ID:Goal|Rest], GT, F, A), _, Heur, n([ID:Body|Rest], GT, F, A)) :-
    ID:rule(Goal, Body).

% Unknown predicate: fires only when no other expand clause applies.
expand(_, n([ID:Goal|_], _, _, _), _, _, _) :-
    nonvar(Goal),
    Goal \= ','(_,_),
    Goal \= ';'(_,_),
    Goal \= true,
    Goal \= not(_),
    \+ allowed_builtin(Goal),
    functor(Goal, Name, Arity),
    functor(Probe, Name, Arity),
    \+ ID:fact(Probe, _),
    \+ ID:rule(Probe, _),
    (   ID:engine_id(_, EngineID)
    ->  \+ EngineID:extern_builtin(Name, Arity)
    ;   true
    ),
    throw(error(unknown_procedure(Name, Arity), ID)).

% --- NAF dispatch ----------------------------------------------------------

% NAF subproof failed → not(Goal) succeeds → consume goal, no formula change.
naf_dispatch(failure, Heur, n([_|Rest], GT, F, A), _, Heur, n(Rest, GT, F, A)).

% NAF subproof succeeded → not(Goal) fails → branch dies (no children).
naf_dispatch(success, _, _, _, _, _) :- fail.

% NAF subproof needs an undecided RV → spawn two outer-heap children with
% the choice committed positively / negatively. The not(Goal) goal stays
% on the goal stack; it'll be re-resolved on the next pop under the
% extended assignment.
naf_dispatch(needs_choice(ChoiceGoal, ChoiceLabel),
             Heur, n(Goals, GT, F, A), Factory, NewHeur,
             n(Goals, GT, NewF, NewA)) :-
    to_symbol(ChoiceGoal, GoalSym),
    to_symbol(ChoiceLabel, LabelSym),
    py_call(Factory:get_boolean(GoalSym, LabelSym), Leaf),
    py_call(Factory:get_scalar_probability(LabelSym), P),
    (   % True branch.
        NewA = [ChoiceGoal-true|A],
        py_call(Factory:conjoin(F, Leaf), NewF),
        add_probability_to_heuristic(P, Heur, NewHeur)
    ;   % False branch.
        NewA = [ChoiceGoal-false|A],
        py_call(Factory:negate(Leaf), NegLeaf),
        py_call(Factory:conjoin(F, NegLeaf), NewF),
        Q is 1.0 - P,
        add_probability_to_heuristic(Q, Heur, NewHeur)
    ).

% NAF subproof needs an undecided categorical RV → spawn N+1 outer-heap
% children: one per AD branch (cat(CatId)-Idx committed positively, the
% branch's MV literal conjoined, heuristic multiplied by P_i) plus a
% "none" child (cat(CatId)-none committed, a single
% "RV took the residual outcome" leaf conjoined, heuristic multiplied
% by the residual 1 - Σ P_i).
%
% Each child emits exactly one new leaf via the factory's MV-aware
% methods — `get_categorical_value` for branch values, `get_categorical_none`
% for the residual. Downstream MV-SDD compilation reads the resulting
% (cat_id, value_idx | "none") tagging off the engine result and turns
% them into single multi-valued literals at the cat_id RV's vtree leaf.
%
% After the split, not(Goal) stays on the goal stack and is re-resolved
% on the next pop. The single child whose committed value matches the
% goal's own value-idx will see naf_resolve succeed, naf_dispatch will
% fail it, and the branch dies. The remaining N children survive — that
% disjunctive set is exactly the worlds where the queried AD branch is
% false.
naf_dispatch(needs_cat_choice(CID, CatId),
             Heur, n(Goals, GT, F, A), Factory, NewHeur,
             n(Goals, GT, NewF, NewA)) :-
    findall(BGoal-BIdx-BInner,
            CID:fact(BGoal, '@cat'(BInner, CatId, BIdx)),
            Branches),
    atom_string(CatId, CatIdStr),
    (   % Per-branch positive child.
        member(BGoal-BIdx-BInner, Branches),
        to_symbol(BGoal, GoalSym),
        to_symbol(BInner, LabelSym),
        py_call(Factory:get_categorical_value(GoalSym, LabelSym,
                                              CatIdStr, BIdx),
                Leaf),
        py_call(Factory:conjoin(F, Leaf), NewF),
        py_call(Factory:get_scalar_probability(LabelSym), P),
        add_probability_to_heuristic(P, Heur, NewHeur),
        NewA = [cat(CatId)-BIdx|A]
    ;   % "None" child: a single MV literal for the residual outcome.
        cat_residual_probability(Branches, Factory, 0.0, SumP),
        Residual is 1.0 - SumP,
        py_call(Factory:get_categorical_none(CatIdStr), NoneLeaf),
        py_call(Factory:conjoin(F, NoneLeaf), NewF),
        add_probability_to_heuristic(Residual, Heur, NewHeur),
        NewA = [cat(CatId)-none|A]
    ).

% Sum the scalar probabilities of all branches under a CatId — used by
% the categorical-NAF "none" child to compute its residual probability.
cat_residual_probability([], _, AccP, AccP).
cat_residual_probability([_BGoal-_BIdx-BInner|Rest], Factory, AccP, OutP) :-
    to_symbol(BInner, LabelSym),
    py_call(Factory:get_scalar_probability(LabelSym), P),
    NextP is AccP + P,
    cat_residual_probability(Rest, Factory, NextP, OutP).

% --- NAF subproof resolver -------------------------------------------------
%
% Stripped-down SLD resolver that mirrors the main engine's control flow
% but does not introduce probabilistic choices: probabilistic facts are
% looked up in the per-branch assignment A. On an uncommitted RV it
% throws naf_needs_choice/2, which the outer engine catches and turns
% into a heap-level split (see naf_dispatch).

naf_resolve(_:true, _) :- !.
naf_resolve(ID:','(G1,G2), A) :- !,
    naf_resolve(ID:G1, A),
    naf_resolve(ID:G2, A).
naf_resolve(ID:';'(G1,G2), A) :- !,
    (   naf_resolve(ID:G1, A)
    ;   naf_resolve(ID:G2, A)
    ).
naf_resolve(ID:not(G), A) :- !,
    \+ naf_resolve(ID:G, A).
naf_resolve(_:Goal, _) :-
    nonvar(Goal),
    allowed_builtin(Goal), !,
    call(Goal).
naf_resolve(ID:Goal, _) :-
    nonvar(Goal),
    functor(Goal, Name, Arity),
    ID:engine_id(Engine, EngineID),
    EngineID:extern_builtin(Name, Arity), !,
    to_symbol(Goal, GoalSym),
    py_call(Engine:'_call_builtin'(GoalSym), ResultSyms),
    member(ResultSym, ResultSyms),
    from_symbol(ResultSym, Unified),
    Goal = Unified.
% NAF over a categorical fact: look up the per-branch assignment for
% this CatId. If the AD has been committed to some outcome, succeed
% iff the committed value matches this branch's index (any other value
% — or the "none" outcome — means this Goal is false in the world).
% Otherwise, throw to trigger the multi-way hoist in naf_dispatch.
naf_resolve(ID:Goal, A) :-
    ID:fact(Goal, '@cat'(_, CatId, ValueIdx)), !,
    (   member(cat(CatId)-Choice, A)
    ->  Choice == ValueIdx
    ;   throw(naf_needs_categorical_choice(ID, CatId))
    ).
naf_resolve(ID:Goal, A) :-
    ID:fact(Goal, Label),
    Label \= '@cat'(_,_,_),
    (   member(Goal-true, A)
    ->  true
    ;   member(Goal-false, A)
    ->  fail
    ;   throw(naf_needs_choice(Goal, Label))
    ).
naf_resolve(ID:Goal, A) :-
    ID:rule(Goal, Body),
    naf_resolve(ID:Body, A).

% Unknown predicate inside a NAF subproof: mirror the main engine's
% behavior and throw, rather than silently treating it as failure.
% Fires only when no fact/rule/extern_builtin is registered for the
% predicate name+arity (matching the bottom-of-stack expand clause).
naf_resolve(ID:Goal, _) :-
    nonvar(Goal),
    Goal \= ','(_,_),
    Goal \= ';'(_,_),
    Goal \= true,
    Goal \= not(_),
    \+ allowed_builtin(Goal),
    functor(Goal, Name, Arity),
    functor(Probe, Name, Arity),
    \+ ID:fact(Probe, _),
    \+ ID:rule(Probe, _),
    (   ID:engine_id(_, EngineID)
    ->  \+ EngineID:extern_builtin(Name, Arity)
    ;   true
    ),
    throw(error(unknown_procedure(Name, Arity), ID)).

% --- Performance optimization notes ----------------------------------------
%
% The current NAF implementation prioritizes simplicity over performance.
% Things to consider revisiting:
%
%   1. Assignment representation. `Assigned` is a plain list with O(n)
%      member/2 lookups. Switch to library(assoc) or a hashed term for
%      O(log n) lookups; this matters on long proofs with many facts.
%
%   2. Eager-throw policy in naf_resolve. The resolver throws on the
%      *first* uncommitted RV it encounters, even if a different proof
%      branch could have decided NAF without that RV. Collecting all
%      possible-success branches first (and only hoisting if no branch
%      succeeds without a fresh choice) would avoid unnecessary outer-
%      heap fan-out.
%
%   3. Dependency-driven hoisting. Rather than discovering RVs lazily,
%      a static pass over the program could pre-compute the RV-dependency
%      set of each predicate. NAF over a goal whose dependency set is
%      already a subset of the committed assignment can skip resolution
%      entirely; otherwise hoist the missing RVs as a batch.
%
%   4. NAF result caching. naf_resolve is re-run from scratch after each
%      hoist. Memoizing partial results (keyed by the relevant slice of
%      A) would avoid redoing deterministic work.
%
%   5. Multi-valued (annotated-disjunction) RVs. The current hoist
%      hardcodes a Boolean split (P / 1-P). For ADs, the false branch
%      should distribute residual probability across the remaining
%      values rather than collapsing them.
