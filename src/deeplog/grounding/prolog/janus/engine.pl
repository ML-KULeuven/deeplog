%  Copyright (c) 2024-2026. KU Leuven
:- module(deeplog_janus_grounder, [prove_query/6]).
:- use_module(deeplog(grounding)).

:- table prove_and_compile(_,_,_,lattice(disjoin_formulas/3)).

:- dynamic factory/1.

disjoin_formulas(Formula1,Formula2,NewFormula) :-
    factory(Factory),
    py_call(Factory:'disjoin'(Formula1,Formula2), NewFormula).

% `disjoin_formulas/3` is the table's lattice aggregation and so has no room
% for the factory in its arguments; it reads this fact instead. Clearing it here
% rather than only on the way out keeps a query that ended early -- an exception,
% an abandoned solution generator -- from leaving a builder behind for the next
% one to aggregate with.
prove_query(ID,Prover,QuerySymbol,Factory,GroundQuery,Formula) :-
    retractall(factory(_)),
    assertz(factory(Factory)),
    from_symbol(QuerySymbol,Query),
    prove_and_compile(ID:Query,Prover,Factory,Formula),
    to_symbol(Query,GroundQuery),
    retractall(factory(_)).

% Entry point: dispatch to prove_compile which handles all goal types.
prove_and_compile(Query,Prover,Factory,Formula) :-
    prove_compile(Query,Prover,Factory,Formula).

% Conjunction: compile each subgoal via tabled prove_and_compile,
% producing factored formulas through intermediate aggregation.
prove_compile(ID:','(G1,G2),Prover,Factory,Formula) :- !,
    prove_and_compile(ID:G1,Prover,Factory,F1),
    prove_and_compile(ID:G2,Prover,Factory,F2),
    py_call(Factory:'conjoin'(F1,F2),Formula).

prove_compile(ID:';'(G1,G2),Prover,Factory,Formula) :- !,
    (   prove_and_compile(ID:G1,Prover,Factory,Formula)
    ;   prove_and_compile(ID:G2,Prover,Factory,Formula)
    ).

prove_compile(ID:not(Goal),Prover,Factory,Formula) :- !,
    py_call(Factory:get_false(),Init),
    findall(F,prove_and_compile(ID:Goal,Prover,Factory,F),Formulas),
    foldl_disjoin(Factory,Formulas,Init,Agg),
    py_call(Factory:'negate'(Agg),Formula).

prove_compile(_:true,_,Factory,Formula) :- !,
    py_call(Factory:get_true(),Formula).

prove_compile(_:Goal,Prover,Factory,Formula) :-
    is_builtin(Prover,Goal),!,
    call_builtin(Prover,Goal),
    py_call(Factory:get_true(),Formula).

% Resolve against a rule. When the head predicate is open, the rule declares
% its domain: each derived ground head instance is itself a leaf, conjoined
% with the body's proof.
prove_compile(ID:Goal,Prover,Factory,Formula) :-
    ID:rule(Goal,Body),
    prove_and_compile(ID:Body,Prover,Factory,BodyFormula),
    functor(Goal,Name,Arity),
    (   ID:open_predicate(Name,Arity)
    ->  (   ground(Goal)
        ->  true
        ;   non_ground_open_predicate(Goal)
        ),
        to_symbol(Goal,GoalSymbol),
        py_call(Factory:leaf(GoalSymbol),Leaf),
        py_call(Factory:conjoin(BodyFormula,Leaf),Formula)
    ;   Formula = BodyFormula
    ).

% Open (leaf) predicate: emit a boolean leaf for the ground atom. Whatever
% meaning attaches to that leaf (a probability, an annotated-disjunction
% branch, ...) is the caller's concern, applied after grounding -- the
% grounder only records the atom.
prove_compile(ID:Goal,_,Factory,Formula) :-
    ID:leaf(Goal),
    to_symbol(Goal,GoalSymbol),
    py_call(Factory:leaf(GoalSymbol),Formula).

prove_compile(ID:Goal,_,_,_) :-
    functor(Goal,Name,Arity),
    functor(NewGoal,Name,Arity),
    \+(ID:rule(NewGoal,_) ; ID:leaf(NewGoal)),
    unknown_predicate(Goal).

% Helper: fold disjoin over a list of formulas
foldl_disjoin(_,[],Acc,Acc).
foldl_disjoin(Factory,[F|Fs],Acc,Result) :-
    py_call(Factory:'disjoin'(Acc,F),NewAcc),
    foldl_disjoin(Factory,Fs,NewAcc,Result).
