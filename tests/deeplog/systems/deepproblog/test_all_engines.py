#  Copyright (c) 2024-2026. KU Leuven

import pytest

from deeplog import to_module
from deeplog.formula import AstFactory
from deeplog.formula import SymbolicFormulaFactory
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.grounding import JanusGrounder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import UnknownPredicateException
from deeplog.grounding import str_to_rule
from deeplog.grounding import str_to_rules
from deeplog.symbol import parse_symbol
from deeplog.symbol import symbol_to_pretty_string
from deeplog.symbol import with_structure
from deeplog.systems.deepproblog import Solver

from ...testing_formulas import leaf
from ...testing_formulas import operands


available_engines = [lambda: Solver(SimpleGrounder())]

if JanusGrounder.is_available():
    available_engines.append(lambda: Solver(JanusGrounder()))


@pytest.mark.parametrize("engine_class", available_engines)
class TestEngines:
    def test_not(self, engine_class):
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

    def test_identical_conjunction(self, engine_class):
        engine = engine_class()

        program = tuple({str_to_rule("?-a,a.")})
        facts = tuple({str_to_rule("la::a.")})

        result = engine.get_query_result(program + facts, SymbolicFormulaFactory())
        assert len(result.formulas) == 1 and (",", ("a",), ("a",)) in result.formulas
        # The label is recorded once per atom, so a ∧ a carries la (not la×la);
        # the idempotent conjunction collapses to a under WMC.
        assert result.labels[("a",)] == ("la",)

    def test_program_overlap(self, engine_class):
        engine = engine_class()

        program = tuple({str_to_rule("a:-b,c."), str_to_rule("?-a.")})
        facts = tuple({str_to_rule("lb::b."), str_to_rule("lc::c.")})

        result = engine.get_query_result(program + facts, SymbolicFormulaFactory())
        assert len(result.formulas) == 1 and ("a",) in result.formulas

        with pytest.raises(UnknownPredicateException):
            engine.get_query_result(program, SymbolicFormulaFactory())

    def test_variable_in_query(self, engine_class):
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

    def test_rule_with_label(self, engine_class):
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

    def test_multiple_proofs(self, engine_class):
        engine = engine_class()

        code = """
        0.5::digit(I,0).
        0.5::digit(I,1).
        a(I1) :- digit(I1,N1).
        ?- a(input(0)).
        """
        program = tuple(str_to_rules(code))
        result = engine.get_query_result(program, AstFactory())
        assert len(result.formulas) == 1
        sym_query, formula = list(result.formulas.items())[0]
        assert symbol_to_pretty_string(sym_query) == "a(input(0))"

        # a(input(0)) proves through digit(input(0),0) and digit(input(0),1); the
        # two engines emit the disjuncts in different orders.
        assert set(operands(formula, "or")) == {
            leaf("digit(input(0),0)"),
            leaf("digit(input(0),1)"),
        }

    def test_unused_variable(self, engine_class):
        engine = engine_class()

        code = """
               0.5::b(0).
               0.5::c(input(0)).
               a(X) :- b(Y).
               a(X) :- c(X).
               ?- a(input(0)).
               """
        program = tuple(str_to_rules(code))
        result = engine.get_query_result(program, AstFactory())
        assert len(result.formulas) == 1
        sym_query, formula = list(result.formulas.items())[0]
        assert symbol_to_pretty_string(sym_query) == "a(input(0))"

        # a(input(0)) proves via a(X):-b(Y) (b grounds to b(0), Y unused) and
        # a(X):-c(X) (c(input(0))); operand order is engine-dependent.
        assert set(operands(formula, "or")) == {leaf("b(0)"), leaf("c(input(0))")}

    def test_labeled_formula_no_builder(self, engine_class):
        program_code = """
            addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).
            classifier(I,N) :: digit(I,N).
            ?- addition(i1,i2,S).
            """
        engine = engine_class()
        program = tuple(str_to_rules(program_code))

        factory = CircuitFactory()
        result = engine.get_query_result(program, factory)
        answers, nodes = zip(*result.formulas.items(), strict=True)
        module = to_module(
            *nodes,
            names=answers,
        )

        # addition(i1,i2,S) ranges over the 19 possible sums S = 0..18 of two
        # digits 0..9; each is wired through as a named module output.
        expected_outputs = {
            ("addition", ("i1",), ("i2",), (str(s),)) for s in range(19)
        }
        assert set(answers) == expected_outputs
        # The compiled module labels each root with the circuit's algebra.
        assert set(module.get_output_shape()) == {
            with_structure(answer, "boolean") for answer in expected_outputs
        }

        # The neural `digit` groundings become the circuit's boolean input leaves:
        # both images over digits 0..9 (20 leaves), regardless of engine order.
        expected_inputs = {
            ("_", ("digit", (img,), (str(d),)), ("boolean",))
            for img in ("i1", "i2")
            for d in range(10)
        }
        assert set(module.get_input_shape()) == expected_inputs
