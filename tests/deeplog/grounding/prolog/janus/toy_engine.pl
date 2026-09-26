%  Copyright (c) 2024-2026. KU Leuven
% A prover over `fact/1` programs, written only against DeepLog's Prolog library.
% `leaf/1` stands for an open predicate, whose instances must be ground.

:- module(deeplog_test_toy_engine, [prove/4]).
:- use_module(deeplog(grounding)).

prove(Program, Prover, GoalSymbol, AnswerSymbol) :-
    from_symbol(GoalSymbol, Goal),
    solve(Program, Prover, Goal),
    to_symbol(Goal, AnswerSymbol).

solve(_, Prover, Goal) :-
    is_builtin(Prover, Goal), !,
    call_builtin(Prover, Goal).
solve(_, _, leaf(X)) :-
    \+ ground(X), !,
    non_ground_open_predicate(leaf(X)).
solve(Program, _, Goal) :-
    Program:fact(Goal).
solve(Program, _, Goal) :-
    functor(Goal, Name, Arity),
    functor(Probe, Name, Arity),
    \+ Program:fact(Probe),
    unknown_predicate(Goal).
