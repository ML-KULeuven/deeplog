%  Copyright (c) 2024-2026. KU Leuven
:- use_module("janus_translation.pl").
:- use_module("builtins.pl").

:- table prove_and_compile(_,_,lattice(disjoin_formulas/3)).

:- dynamic rule/2.
:- dynamic fact/2.
:- dynamic extern_builtin/2.
:- dynamic engine_id/2.
:- dynamic factory/1.

disjoin_formulas(Formula1,Formula2,NewFormula) :-
    factory(Factory),
    py_call(Factory:'disjoin'(Formula1,Formula2), NewFormula).

prove_query(ID,QuerySymbol,Factory,GroundQuery,Formula) :-
    assertz(factory(Factory)),
    from_symbol(QuerySymbol,Query),
    prove_and_compile(ID:Query,Factory,Formula),
    to_symbol(Query,GroundQuery),
    retractall(factory(_)).

% Entry point: dispatch to prove_compile which handles all goal types.
prove_and_compile(Query,Factory,Formula) :-
    prove_compile(Query,Factory,Formula).

% Conjunction: compile each subgoal via tabled prove_and_compile,
% producing factored formulas through intermediate aggregation.
prove_compile(ID:','(G1,G2),Factory,Formula) :- !,
    prove_and_compile(ID:G1,Factory,F1),
    prove_and_compile(ID:G2,Factory,F2),
    py_call(Factory:'conjoin'(F1,F2),Formula).

prove_compile(ID:';'(G1,G2),Factory,Formula) :- !,
    (prove_and_compile(ID:G1,Factory,Formula) ; prove_and_compile(ID:G2,Factory,Formula)).

prove_compile(ID:not(Goal),Factory,Formula) :- !,
    py_call(Factory:get_false(),Init),
    findall(F,prove_and_compile(ID:Goal,Factory,F),Formulas),
    foldl_disjoin(Factory,Formulas,Init,Agg),
    py_call(Factory:'negate'(Agg),Formula).

prove_compile(_:true,Factory,Formula) :- !,
    py_call(Factory:get_true(),Formula).

prove_compile(_:Goal,Factory,Formula) :-
    allowed_builtin(Goal),!,
    call(Goal),
    py_call(Factory:get_true(),Formula).

prove_compile(ID:Goal,Factory,Formula) :-
    functor(Goal,Name,Arity),
    ID:engine_id(Engine,EngineID),
    EngineID:extern_builtin(Name,Arity),!,
    to_symbol(Goal,Symbol),
    py_call(Engine:'_call_builtin'(Symbol),ResultSymbols),
    maplist(from_symbol,ResultSymbols,Results),
    member(Goal,Results),
    py_call(Factory:get_true(),Formula).

prove_compile(ID:Goal,Factory,Formula) :-
    ID:rule(Goal,Body),
    prove_and_compile(ID:Body,Factory,Formula).

% Categorical (annotated-disjunction) fact: emit a single MV literal
% (cat_id, value_idx). The JanusEngine doesn't enforce per-proof mutex
% itself — downstream MV-SDD compilation reads `categoricals` off the
% engine result and fans this leaf out as one value of the cat_id RV,
% which encodes mutex structurally. Inner label still goes to the
% factory's labels map for probability lookup.
prove_compile(ID:Goal,Factory,Formula) :-
    ID:fact(Goal, '@cat'(InnerLabel, CatId, ValueIdx)), !,
    to_symbol(Goal, GoalSymbol),
    to_symbol(InnerLabel, LabelSymbol),
    atom_string(CatId, CatIdStr),
    py_call(Factory:get_categorical_value(GoalSymbol, LabelSymbol,
                                          CatIdStr, ValueIdx),
            Formula).

prove_compile(ID:Goal,Factory,Formula) :-
    ID:fact(Goal,Label),
    Label \= '@cat'(_,_,_),
    to_symbol(Goal,GoalSymbol),
    to_symbol(Label,LabelSymbol),
    py_call(Factory:get_boolean(GoalSymbol,LabelSymbol),Formula).

prove_compile(ID:Goal,_,_) :-
    functor(Goal,Name,Arity),
    functor(NewGoal,Name,Arity),
    \+(ID:rule(NewGoal,_) ; ID:fact(NewGoal,_)),
    throw(error(unknown_procedure(Name,Arity),ID)).

% Helper: fold disjoin over a list of formulas
foldl_disjoin(_,[],Acc,Acc).
foldl_disjoin(Factory,[F|Fs],Acc,Result) :-
    py_call(Factory:'disjoin'(Acc,F),NewAcc),
    foldl_disjoin(Factory,Fs,NewAcc,Result).
