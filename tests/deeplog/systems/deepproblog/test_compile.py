#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the compile_to_module pipeline."""

import torch

from deeplog.formula import DeepLogModuleFactory
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rules
from deeplog.shape import get_all_symbols
from deeplog.systems.deepproblog import Solver
from deeplog.systems.deepproblog import compile_to_module
from deeplog.systems.deepproblog.compile import build_leaf_mapping
from deeplog.util import as_tuple


def test_compile_produces_module():
    """compile_to_module produces a runnable DeepLogModule."""
    engine = Solver(SimpleGrounder())

    program = tuple(
        str_to_rules(
            """
        0.6::burglary.
        0.1::earthquake.
        alarm :- burglary.
        alarm :- earthquake.
        ?- alarm.
        """
        )
    )

    # The engine builds each query's boolean circuit with a CircuitFactory; a
    # separate DeepLogModuleFactory lowers those CircuitNodes to a module.
    result = engine.get_query_result(program, CircuitFactory())

    module = compile_to_module(result, DeepLogModuleFactory())
    assert module is not None

    output_symbols = list(module.get_output_shape())
    assert len(output_symbols) == 1


def test_multiple_atoms_share_arguments():
    engine = Solver(SimpleGrounder())

    program = tuple(
        str_to_rules("""
        nn1(x1) :: a(x1).
        nn2(x1) :: b(x1).
        c(x1) :- a(x1), b(x1).
        ?- c(x1).
    """)
    )

    result = engine.get_query_result(program, CircuitFactory())

    module = compile_to_module(result, DeepLogModuleFactory())
    assert module is not None


def test_compile_transforms_boolean_lump_to_probability():
    """compile_to_module routes each boolean lump through the WMC transform.

    The result is a probability-backed module (the structural guarantee that
    replaces the old per-call ``create_aggregation`` routing and its runtime
    non-probability check).
    """
    from deeplog.formula.ast import FormulaNode
    from deeplog.systems.deepproblog import EngineResult

    # The engine result is a boolean CircuitNode lump (built with a
    # CircuitFactory); compile_to_module transforms it to probability in a
    # single batch and lowers it with a DeepLogModuleFactory.
    result: EngineResult[FormulaNode] = EngineResult(
        formulas={("a",): CircuitFactory().create_atom(("_", ("a",), ("boolean",)))},
        labels={},
    )

    module = compile_to_module(result, DeepLogModuleFactory())
    assert len(list(module.get_output_shape())) == 1


def test_numeric_constants_are_baked_not_inputs():
    """Numeric-constant facts are pre-filled into the AC, not left as inputs.

    The knowledge-compilation backend recognises each leaf whose label is a
    numeric constant (``0.6::burglary``) and bakes it in, so a program of only
    numeric facts compiles to a module with *no* runtime inputs — ``P(q)``
    evaluates without anyone supplying the probabilities by hand.
    """
    engine = Solver(SimpleGrounder())
    program = tuple(
        str_to_rules(
            """
        0.6::burglary.
        0.3::earthquake.
        alarm :- burglary.
        alarm :- earthquake.
        ?- alarm.
        """
        )
    )
    result = engine.get_query_result(program, CircuitFactory())
    module = compile_to_module(result, DeepLogModuleFactory())

    # Every leaf was a numeric constant, so nothing remains a runtime input --
    # not even the empty channel, since there is no tensor to declare.
    assert list(get_all_symbols(module.get_input_shape())) == []
    assert as_tuple(module.get_input_shape()) == ()
    # P(alarm) = 1 - (1 - 0.6)(1 - 0.3) = 0.72, from the baked constants alone.
    out = module()
    torch.testing.assert_close(float(out.flatten()[0]), 0.72)


def test_constants_bake_but_neural_labels_stay_inputs():
    """Constants bake; non-numeric (neural) labels stay runtime inputs.

    In a mixed program only the numeric fact is pre-filled — the neural-labelled
    leaf is still fed at forward time, so the user supplies just the network
    output, not the hand-written constant.
    """
    engine = Solver(SimpleGrounder())
    program = tuple(
        str_to_rules(
            """
        0.5::a.
        nn1(x1) :: b(x1).
        c(x1) :- a, b(x1).
        ?- c(x1).
        """
        )
    )
    result = engine.get_query_result(program, CircuitFactory())
    module = compile_to_module(result, DeepLogModuleFactory())

    inputs = list(get_all_symbols(module.get_input_shape()))
    # The constant `a` is baked away; the neural leaf `nn1(x1)` stays an input.
    assert all(sym[1] != ("a",) for sym in inputs)
    assert any(sym[1] == ("nn1", ("x1",)) for sym in inputs)

    # P(c) = P(a) * P(b) = 0.5 * 0.8, with only the neural probability supplied.
    args = [
        torch.tensor([[0.8 for _ in get_all_symbols(symtensor)]])
        for symtensor in as_tuple(module.get_input_shape())
    ]
    out = module(*args)
    torch.testing.assert_close(float(as_tuple(out)[0].flatten()[0]), 0.4)


def test_baked_module_is_called_with_no_inputs():
    """An all-constant program bakes every leaf, so its module is callable bare.

    With no runtime inputs the module needs only the batch size, which it
    synthesises itself — ``module()`` evaluates a single constant row.
    """
    engine = Solver(SimpleGrounder())
    program = tuple(
        str_to_rules(
            """
        0.6::a.
        b :- a.
        c :- a.
        ?- b.
        ?- c.
        """
        )
    )
    result = engine.get_query_result(program, CircuitFactory())
    module = compile_to_module(result, DeepLogModuleFactory())

    assert list(get_all_symbols(module.get_input_shape())) == []
    out = module()  # no inputs at all — not even the empty (batch, 0) channel
    values = torch.cat([tensor.flatten() for tensor in as_tuple(out)]).tolist()
    # P(b) = P(c) = P(a) = 0.6.
    assert len(values) == 2
    for value in values:
        torch.testing.assert_close(value, 0.6)


def test_compile_multiple_queries():
    """Multiple queries compile into a module with multiple outputs."""
    engine = Solver(SimpleGrounder())

    program = tuple(
        str_to_rules(
            """
        0.6::a.
        b :- a.
        c :- a.
        ?- b.
        ?- c.
        """
        )
    )

    # Both queries' (structurally identical) proofs co-reside in one boolean
    # source circuit (built with a CircuitFactory), and stay two distinct outputs.
    result = engine.get_query_result(program, CircuitFactory())

    assert len(result.formulas) == 2
    module = compile_to_module(result, DeepLogModuleFactory())
    assert module is not None

    output_symbols = list(module.get_output_shape())
    assert len(output_symbols) == 2


def test_leaf_mapping_maps_a_labeled_atom_to_its_probability_label():
    labels = {
        ("digit", ("i1",), ("0",)): ("classifier", ("i1",), ("0",)),
        ("digit", ("i1",), ("1",)): ("classifier", ("i1",), ("1",)),
    }
    mapping = build_leaf_mapping(labels)
    assert mapping(("digit", ("i1",), ("0",))) == (
        "_",
        ("classifier", ("i1",), ("0",)),
        ("probability",),
    )
    assert mapping(("digit", ("i1",), ("1",))) == (
        "_",
        ("classifier", ("i1",), ("1",)),
        ("probability",),
    )


def test_leaf_mapping_keeps_atoms_sharing_arguments_apart():
    """a(x1) and b(x1) share arguments; the labels disambiguate them.

    A by-arguments heuristic cannot resolve this — both atoms and both
    labels collide on arguments alone — but the label map is exact.
    """
    labels = {
        ("a", ("x1",)): ("nn1", ("x1",)),
        ("b", ("x1",)): ("nn2", ("x1",)),
    }
    mapping = build_leaf_mapping(labels)
    assert mapping(("a", ("x1",))) == (
        "_",
        ("nn1", ("x1",)),
        ("probability",),
    )
    assert mapping(("b", ("x1",))) == (
        "_",
        ("nn2", ("x1",)),
        ("probability",),
    )


def test_leaf_mapping_retags_an_unlabeled_leaf_to_probability():
    mapping = build_leaf_mapping({("a", ("x1",)): ("nn1", ("x1",))})
    # An unlabeled boolean leaf keeps its atom, retagged as probability.
    assert mapping(("fact", ("y",))) == (
        "_",
        ("fact", ("y",)),
        ("probability",),
    )


def test_leaf_mapping_without_labels_retags_everything():
    mapping = build_leaf_mapping({})
    assert mapping(("a",)) == ("_", ("a",), ("probability",))
