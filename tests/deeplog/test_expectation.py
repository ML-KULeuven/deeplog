#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the expectation aggregation operator.

Expectation is circuit-representable: the construction fold absorbs it via the
weighted-model-count fast path (a boolean lump is model-counted into a
probability lump), so it is exercised here through the high-level
``DeepLogModuleFactory.compile`` path over an expectation AST — recognition is
already done, the construction fold builds the probability circuit, lowering
composes the module. The error cases are raised by the fast path during
construction.
"""

import pytest
import torch

from deeplog import CircuitNode
from deeplog import DeepLogModuleFactory
from deeplog import reshape
from deeplog.formula import Aggregation
from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import UnaryOp
from deeplog.formula import lower_circuit_nodes
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.shape import SymTensor


FALSE = ("false",)
TRUE = ("true",)

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)

BURGLARY_BOOL_SYM = ("_", BURGLARY_ATOM, ("boolean",))
EARTHQUAKE_BOOL_SYM = ("_", EARTHQUAKE_ATOM, ("boolean",))

BURGLARY_PROB_SYM = ("_", BURGLARY_ATOM, ("probability",))
EARTHQUAKE_PROB_SYM = ("_", EARTHQUAKE_ATOM, ("probability",))


def _expectation(binders, child, params=()):
    """Build an ``expectation`` AST node."""
    return Aggregation("expectation", tuple(binders), tuple(params), child)


def test_expectation_boolean_disjunction():
    """Expectation of Burglary OR Earthquake compiles to probability semiring."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    exp = _expectation([BURGLARY, EARTHQUAKE], disjunction)

    module = reshape(
        DeepLogModuleFactory().compile(exp),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # Inputs are probability values for each atom
    # With P(B=true)=0.5, P(E=true)=0.5 (uniform):
    # E[B or E] = 0.75
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.75]]))

    # With P(B=true)=0.8, P(E=true)=0.3:
    # E[B or E] = 1 - P(B=false)*P(E=false) = 1 - 0.2*0.7 = 0.86
    result2 = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(result2, torch.tensor([[0.86]]))


def test_expectation_single_variable():
    """Expectation over a single boolean variable."""
    # A trivial circuit by double negation (equivalent to the bare atom).
    b_circuit = UnaryOp("not", UnaryOp("not", Atom(BURGLARY_BOOL_SYM)))
    exp = _expectation([BURGLARY], b_circuit)

    module = reshape(
        DeepLogModuleFactory().compile(exp), input=SymTensor([BURGLARY_PROB_SYM])
    )

    assert module.get_input_shape() == SymTensor([BURGLARY_PROB_SYM])
    assert len(list(module.get_output_shape())) == 1

    # E[B] with P(B=true)=0.5 -> 0.5
    result = module(torch.tensor([[0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.5]]))

    # E[B] with P(B=true)=0.7 -> 0.7
    result2 = module(torch.tensor([[0.7]]))
    torch.testing.assert_close(result2, torch.tensor([[0.7]]))


def test_expectation_conjunction():
    """Expectation of Burglary AND Earthquake."""
    conjunction = BinaryOp("and", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    exp = _expectation([BURGLARY, EARTHQUAKE], conjunction)

    module = reshape(
        DeepLogModuleFactory().compile(exp),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert module.get_input_shape() == SymTensor(
        [BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]
    )
    assert len(list(module.get_output_shape())) == 1

    # E[B and E] with uniform (0.5, 0.5) = 0.25
    result = module(torch.tensor([[0.5, 0.5]]))
    torch.testing.assert_close(result, torch.tensor([[0.25]]))


def test_expectation_rejects_probability_param():
    """Expectation no longer infers leaf mappings from probability formulas."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    pb_sym = ("_", ("prob", BURGLARY, TRUE), ("probability",))
    pe_sym = ("_", ("prob", EARTHQUAKE, TRUE), ("probability",))
    prob_formula = BinaryOp("times", Atom(pb_sym), Atom(pe_sym))

    exp = _expectation([BURGLARY, EARTHQUAKE], disjunction, params=[prob_formula])

    with pytest.raises(ValueError, match="does not accept"):
        DeepLogModuleFactory().compile(exp)


def test_expectation_too_many_params_raises_error():
    """Expectation with more than one param should raise an error."""
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM))
    pb = Atom(("_", ("prob", BURGLARY, TRUE), ("probability",)))
    pe = Atom(("_", ("prob", EARTHQUAKE, TRUE), ("probability",)))

    exp = _expectation([BURGLARY], disjunction, params=[pb, pe])

    with pytest.raises(ValueError, match="does not accept"):
        DeepLogModuleFactory().compile(exp)


def test_multiple_expectations_coreside_via_batched_transform():
    """Several boolean roots transform together into one probability circuit.

    Co-residence is an explicit batch
    (:func:`~deeplog.formula.strategies.transform_expectation_to_probability`):
    the roots share one arithmetic circuit (and dedup their shared sub-circuits)
    and compile together to a multi-output module.
    """
    from deeplog.formula.strategies import transform_expectation_to_probability

    cf = CircuitFactory()
    b = cf.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e = cf.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    or_node = cf.create_binary_node("or", b, e)
    and_node = cf.create_binary_node("and", b, e)

    exp_or, exp_and = transform_expectation_to_probability(or_node, and_node)

    # Co-residence: both roots transformed into the *same* circuit, and the
    # shared boolean leaves map to the same probability nodes.
    assert isinstance(exp_or, CircuitNode) and isinstance(exp_and, CircuitNode)
    assert exp_or.circuit is exp_and.circuit
    assert exp_or.node != exp_and.node

    # lower_circuit_nodes names its outputs positionally, in the order the roots
    # are handed in (or first, and second), so a name-based input reshape pins the
    # probability inputs and the outputs are read by position.
    module = lower_circuit_nodes(DeepLogModuleFactory(), exp_or, exp_and)
    module = reshape(
        module,
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    # E[B or E] = 1 - (1-0.8)(1-0.3) = 0.86; E[B and E] = 0.8*0.3 = 0.24
    out = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(out, torch.tensor([[0.86, 0.24]]))


def test_compiling_two_expectations_shares_one_knowledge_compilation(monkeypatch):
    """Two formulas compiled together count their shared circuit once.

    The same sharing :func:`lower_circuit_nodes` gives the engine, reached from
    plain ASTs: both folds run under one memo, so the two counts land on one
    boolean circuit and are compiled together.
    """
    import deeplog.circuit.knowledge_compile.sdd as sdd

    roots_per_call = []
    original = sdd.compile_sdd

    def counting_compile_sdd(circuit, roots, *args, **kwargs):
        roots_per_call.append(len(roots))
        return original(circuit, roots, *args, **kwargs)

    monkeypatch.setattr(sdd, "compile_sdd", counting_compile_sdd)

    b, e = Atom(BURGLARY_BOOL_SYM), Atom(EARTHQUAKE_BOOL_SYM)
    disjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("or", b, e))
    conjunction = _expectation([BURGLARY, EARTHQUAKE], BinaryOp("and", b, e))

    module = reshape(
        DeepLogModuleFactory().compile(disjunction, conjunction),
        input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]),
    )

    assert roots_per_call == [2]

    # E[B or E] = 1 - (1-0.8)(1-0.3) = 0.86; E[B and E] = 0.8*0.3 = 0.24
    out = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(out, torch.tensor([[0.86, 0.24]]))


def test_expectation_non_boolean_raises_error():
    """Expectation over non-boolean formula should raise an error."""
    X = ("X",)
    prob_pred = ("p", X, ("_", ("0.3",), ("probability",)))
    prob_atom = Atom(("_", prob_pred, ("probability",)))

    exp = _expectation([X], prob_atom)

    with pytest.raises(ValueError, match="boolean"):
        DeepLogModuleFactory().compile(exp)


def test_multiple_expectations_share_one_knowledge_compilation(monkeypatch):
    """Counts over one boolean circuit are compiled together, not one by one.

    Each ``expectation`` defers to a boundary feeder, and the lowering gathers
    the feeders that count the same circuit so a single knowledge compilation
    serves them all — what the DeepProbLog conditional path needs, where N
    numerators and their shared evidence would otherwise be compiled N+1 times.
    """
    import deeplog.circuit.knowledge_compile.sdd as sdd

    roots_per_call = []
    original = sdd.compile_sdd

    def counting_compile_sdd(circuit, roots, *args, **kwargs):
        roots_per_call.append(len(roots))
        return original(circuit, roots, *args, **kwargs)

    monkeypatch.setattr(sdd, "compile_sdd", counting_compile_sdd)

    cf = CircuitFactory()
    b = cf.create_atom(("_", BURGLARY_ATOM, ("boolean",)))
    e = cf.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",)))
    exp_or = cf.create_aggregation(
        "expectation", (), (), cf.create_binary_node("or", b, e)
    )
    exp_and = cf.create_aggregation(
        "expectation", (), (), cf.create_binary_node("and", b, e)
    )

    module = lower_circuit_nodes(DeepLogModuleFactory(), exp_or, exp_and)
    module = reshape(module, input=SymTensor([BURGLARY_PROB_SYM, EARTHQUAKE_PROB_SYM]))

    # One compilation, holding both counts.
    assert roots_per_call == [2]

    out = module(torch.tensor([[0.8, 0.3]]))
    torch.testing.assert_close(out, torch.tensor([[0.86, 0.24]]))


def test_deferred_lump_reads_back_only_what_absorption_minted():
    """The marker the strategy left for itself, recognised where it is minted.

    An ``expectation`` that still carries binders was never absorbed — it lowers
    by enumeration like any other aggregation — so it is not one of these.
    """
    from deeplog.formula.strategies import absorb_aggregation
    from deeplog.formula.strategies import deferred_lump

    cf = CircuitFactory()
    lump = cf.create_binary_node(
        "or",
        cf.create_atom(("_", BURGLARY_ATOM, ("boolean",))),
        cf.create_atom(("_", EARTHQUAKE_ATOM, ("boolean",))),
    )
    absorbed = absorb_aggregation(cf.get_circuit, "expectation", (), (), lump)
    assert absorbed is not None
    ((_, feeder),) = absorbed.feeders

    assert deferred_lump(feeder) is lump
    assert deferred_lump(_expectation([BURGLARY], Atom(BURGLARY_BOOL_SYM))) is None


def test_lower_circuit_nodes_rejects_a_root_that_is_not_a_lump():
    """A symbolic node, like an aggregation no circuit holds, is lowered with ``compile``."""
    symbolic = Aggregation(
        "sum", (BURGLARY,), (), Atom(("_", BURGLARY_ATOM, ("boolean",)))
    )
    with pytest.raises(TypeError, match="lowers raw circuit-node roots"):
        lower_circuit_nodes(DeepLogModuleFactory(), symbolic)
