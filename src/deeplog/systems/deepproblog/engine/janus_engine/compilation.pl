%  Copyright (c) 2024-2026. KU Leuven
:- module(compilation,[compile_formula/3]).

is_true(true).
is_false(false).


compile_formula(true, Factory, Formula) :- py_call(Factory:get_true(), Formula).
compile_formula(false, Factory, Formula) :- py_call(Factory:get_false(), Formula).

compile_formula(and([]), Factory, Formula) :- py_call(Factory:get_true(), Formula).
compile_formula(or([]), Factory, Formula) :- py_call(Factory:get_false(), Formula).


compile_formula(and([Proof|Proofs]), Factory, Formula) :-
    compile_formula(Proof, Factory, Formula1),
    compile_formula(and(Proofs), Factory, Formula2),
    py_call(Factory:'conjoin'(Formula1,Formula2), Formula).
%    py_call(Formula1:'__and__'(Formula2), Formula).

compile_formula(or([Proof|Proofs]), Factory, Formula) :-
    compile_formula(Proof, Factory, Formula1),
    compile_formula(or(Proofs), Factory, Formula2),
    py_call(Factory:'disjoin'(Formula1,Formula2), Formula).
%    py_call(Formula1:'__or__'(Formula2), Formula).

compile_formula(not(Proof), Factory, NotFormula) :-
    compile_formula(Proof, Factory, Formula),
    py_call(Factory:'negate'(Formula), NotFormula).
%    py_call(Formula:'__neg__'(), NotFormula).

compile_formula(fact(Goal,Label), Factory, Formula) :-
    to_symbol(Goal,GoalSymbol),
    to_symbol(Label,LabelSymbol),
    py_call(Factory:get_boolean(GoalSymbol,LabelSymbol), Formula).
