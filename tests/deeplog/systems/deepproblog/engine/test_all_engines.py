#  Copyright (c) 2024-2026. KU Leuven

import math

import pytest

from deeplog.circuit import to_module
from deeplog.formula import DeepLogModuleFactory
from deeplog.formula import SymbolicFormulaFactory
from deeplog.symbol import is_variable
from deeplog.symbol import parse_symbol
from deeplog.symbol import symbol_to_pretty_string
from deeplog.systems.deepproblog.engine import Engine
from deeplog.systems.deepproblog.engine import JanusEngine
from deeplog.systems.deepproblog.engine import SimpleEngine
from deeplog.systems.deepproblog.engine import UnknownPredicateException
from deeplog.systems.deepproblog.program import str_to_rule
from deeplog.systems.deepproblog.program import str_to_rules


available_engines: list[type[Engine]] = [SimpleEngine]

if JanusEngine.is_available():
    available_engines.append(JanusEngine)


@pytest.mark.parametrize("engine_class", available_engines)
class TestEngines:
    def test_not(self, engine_class: type[Engine]):
        engine = engine_class()

        program = tuple(
            str_to_rule(r)
            for r in [
                "la::a.",
                "?- not(a).",
            ]
        )
        result = engine.get_query_result(program, SymbolicFormulaFactory())
        formula = result.formulas[("not", ("a",))]
        assert formula == "not a_boolean"
        assert result.labels[("a",)] == ("la",)

    def test_identical_conjunction(self, engine_class: type[Engine]):
        engine = engine_class()

        program = tuple({str_to_rule("?-a,a.")})
        facts = tuple({str_to_rule("la::a.")})

        result = engine.get_query_result(program + facts, SymbolicFormulaFactory())
        assert len(result.formulas) == 1 and (",", ("a",), ("a",)) in result.formulas

    # TODO. What is the correct answer. laxla or la?

    def test_program_overlap(self, engine_class: type[Engine]):
        engine = engine_class()

        program = tuple({str_to_rule("a:-b,c."), str_to_rule("?-a.")})
        facts = tuple({str_to_rule("lb::b."), str_to_rule("lc::c.")})

        result = engine.get_query_result(program + facts, SymbolicFormulaFactory())
        assert len(result.formulas) == 1 and ("a",) in result.formulas

        with pytest.raises(UnknownPredicateException):
            engine.get_query_result(program, SymbolicFormulaFactory())

    def test_variable_in_query(self, engine_class: type[Engine]):
        engine = engine_class()
        program = tuple(
            {
                str_to_rule("la0::a(0)."),
                str_to_rule("la1::a(1)."),
                str_to_rule("?-a(X)."),
            }
        )

        result = engine.get_query_result(program, SymbolicFormulaFactory())
        assert set(result.formulas) == {("a", ("0",)), ("a", ("1",))}

    def test_rule_with_label(self, engine_class: type[Engine]):
        engine = engine_class()
        program = tuple({str_to_rule("classifier(X) :: output(X) :- between(0,9,X).")})
        result = engine.get_result(
            program, parse_symbol("output(X)"), SymbolicFormulaFactory()
        )
        assert len(result.formulas) == 10
        for i in range(10):
            formula = result.formulas[("output", (str(i),))]
            # Labeled rule is rewritten to aux fact: classifier(X) :: aux0(X).
            assert formula == f"aux0({i})_boolean"
            # The label for the auxiliary atom maps back to the classifier annotation
            assert result.labels[("aux0", (str(i),))] == ("classifier", (str(i),))

    def test_reasoning(self, engine_class: type[Engine]):
        program = tuple(
            str_to_rules(
                """
        edge(0,1).
        edge(1,2).
        edge(1,3).

        connected(X,Y) :- edge(X,Y).
        connected(X,Y) :- edge(X,Z), connected(Z,Y).
        ?-connected(X,Y).
        """
            )
        )
        connections = set(
            engine_class().get_query_result(program, SymbolicFormulaFactory()).formulas
        )
        connected = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        expected_connections = {
            ("connected", (str(x),), (str(y),)) for x, y in connected
        }
        assert connections == expected_connections

    def test_add_builtin(self, engine_class: type[Engine]):
        engine = engine_class()

        def square(lhs, rhs):
            if is_variable(lhs):
                if not is_variable(rhs):
                    yield {lhs: (str(math.sqrt(int(rhs[0]))),)}
            else:
                if is_variable(rhs):
                    yield {rhs: (str(int(lhs[0]) ** 2),)}
                else:
                    if int(rhs[0]) == int(lhs[0]) ** 2:
                        yield {}

        engine.add_builtin("square", 2, square)
        result = engine.get_query_result(
            tuple(str_to_rules("?-square(2,X).")), SymbolicFormulaFactory()
        )
        assert (
            len(result.formulas) == 1 and parse_symbol("square(2,4)") in result.formulas
        )

    def test_multiple_proofs(self, engine_class: type[Engine]):
        engine = engine_class()

        code = """
        0.5::digit(I,0).
        0.5::digit(I,1).
        a(I1) :- digit(I1,N1).
        ?- a(input(0)).
        """
        program = tuple(str_to_rules(code))
        result = engine.get_query_result(program, SymbolicFormulaFactory())
        assert len(result.formulas) == 1
        sym_query, weighted_formula = list(result.formulas.items())[0]
        assert symbol_to_pretty_string(sym_query) == "a(input(0))"

        # TODO fix test
        # assert (str(weighted_formula.get_formula()) == "digit(input(0),0) ; digit(input(0),1)" or
        # 		str(weighted_formula.get_formula()) == "digit(input(0),1) ; digit(input(0),0)")

    def test_unused_variable(self, engine_class: type[Engine]):
        engine = engine_class()

        code = """
               0.5::b(0).
               0.5::c(input(0)).
               a(X) :- b(Y).
               a(X) :- c(X).
               ?- a(input(0)).
               """
        program = tuple(str_to_rules(code))
        result = engine.get_query_result(program, SymbolicFormulaFactory())
        sym_query, weighted_formula = list(result.formulas.items())[0]

        # TODO test
        # assert (str(weighted_formula.get_formula()) == "b(0) ; c(input(0))" or
        # 		str(weighted_formula.get_formula()) == "c(input(0)) ; b(0)")

    def test_substitution(self, engine_class: type[Engine]):
        engine = engine_class()

        code = """
        fact(t(1,2,X), t(2,1,X)).

        ?- fact(t(1,2,3), Z).
        """

        program = tuple(str_to_rules(code))
        result = engine.get_query_result(program, SymbolicFormulaFactory())
        sym_query, _ = list(result.formulas.items())[0]
        assert sym_query == (
            "fact",
            ("t", ("1",), ("2",), ("3",)),
            ("t", ("2",), ("1",), ("3",)),
        )

    def test_list_predicates_with_rules(self, engine_class: type[Engine]):
        engine = engine_class()
        code = """
        cons([H|T], H, T).
        head(L, H) :- cons(L, H, _).
        tail(L, T) :- cons(L, _, T).
        ?- head([a,b,c], H).
        ?- tail([a,b,c], T).
        """
        program = tuple(str_to_rules(code))
        results = engine.get_query_result(program, SymbolicFormulaFactory())

        list_term = ("cons", ("a",), ("cons", ("b",), ("cons", ("c",), ("nil",))))
        head_key = ("head", list_term, ("a",))
        tail_term = ("cons", ("b",), ("cons", ("c",), ("nil",)))
        tail_key = ("tail", list_term, tail_term)

        assert head_key in results.formulas
        assert tail_key in results.formulas

    def test_labeled_formula_no_builder(self, engine_class: type[Engine]):
        program_code = """
            addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
            classifier(I,N) :: digit(I,N).
            ?- addition(i1,i2,S).
            """
        engine = engine_class()
        program = tuple(str_to_rules(program_code))

        factory = DeepLogModuleFactory()
        result = engine.get_query_result(program, factory)
        answers, nodes = zip(*result.formulas.items(), strict=True)
        module = to_module(
            *(n.root for n in nodes),
            names=answers,
        )
        print(list(module.get_input_shape()))
        print(list(module.get_output_shape()))
