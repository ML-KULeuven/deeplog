%  Copyright (c) 2024-2026. KU Leuven
% A second prover defining the same predicate names as toy_engine.pl.

:- module(deeplog_test_other_engine, [prove/4]).
:- use_module(deeplog(grounding)).

prove(_, _, _, Answer) :-
    to_symbol(other, Answer).

solve(_, _, _).
