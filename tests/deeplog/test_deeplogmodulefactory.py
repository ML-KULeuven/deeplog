#  Copyright (c) 2024-2026. KU Leuven
"""Tests for lowering a formula AST to a DeepLogModule.

Circuit *construction* (atoms, operators, casts, the expectation fast path) is
the :class:`CircuitFactory`'s job; :class:`DeepLogModuleFactory` *lowers* the
resulting lumps (and the ``sum`` enumerations a circuit cannot express) into a
runnable module. These tests therefore build lumps with a ``CircuitFactory`` (or
a plain AST) and lower them with ``DeepLogModuleFactory.compile`` (an AST) or
``lower_circuit_nodes`` (raw co-resident nodes).
"""

import math
from functools import partial

import pytest
import torch

from deeplog import AggregationModule
from deeplog import Algebra
from deeplog import AlgebraicStructure
from deeplog import CircuitNode
from deeplog import DeepLogModuleFactory
from deeplog import EqualityPredicate
from deeplog import Predicate
from deeplog import SumsPredicate
from deeplog import SymTensor
from deeplog import get_network_predicate
from deeplog import parse_formula_to_module
from deeplog import reshape
from deeplog import sole_structure
from deeplog import with_structure
from deeplog.formula import Aggregation
from deeplog.formula import Atom
from deeplog.formula import BinaryOp
from deeplog.formula import Transformation
from deeplog.formula import UnaryOp
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.formula.deeplogmodulefactory.transform_cast import build_transform
from deeplog.module import WrappedModule
from deeplog.shape import get_all_symbols
from deeplog.variable import Domain

from .testing_modules import IndexClassifier


FALSE = ("false",)
TRUE = ("true",)
BOOLEAN = [FALSE, TRUE]

BURGLARY = ("Burglary",)
EARTHQUAKE = ("Earthquake",)

BURGLARY_ATOM = ("=", BURGLARY, TRUE)
EARTHQUAKE_ATOM = ("=", EARTHQUAKE, TRUE)

BURGLARY_BOOL = ("_", BURGLARY_ATOM, ("boolean",))
EARTHQUAKE_BOOL = ("_", EARTHQUAKE_ATOM, ("boolean",))


def test_model_count():
    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL), Atom(EARTHQUAKE_BOOL))
    summed = Aggregation("sum", (BURGLARY, EARTHQUAKE), (), disjunction)
    module = DeepLogModuleFactory().compile(summed)
    result = module()

    torch.testing.assert_close(result, torch.tensor([[3.0]], dtype=result.dtype))


def test_a_weighted_model_count_reads_back_in_log_space():
    """A probability-space result casts into log space, as the last step.

    The count itself stays where it is written — ``sum`` totals the weighted
    models arithmetically — and the cast reports that total as its logarithm.
    """
    module = parse_formula_to_module("""
        (
            sum(B): sum(E):
                (=(B,true)_boolean or =(E,true)_boolean)_probability
                times
                (p(B,0.8)_probability times p(E,0.3)_probability)
        )_logprobability
    """)

    assert float(module()) == pytest.approx(math.log(0.8 + 0.2 * 0.3))


def test_missing_predicate_exposed_as_input():
    atom1 = ("_", ("nn", ("0",)), ("probability",))
    atom2 = ("_", ("nn", ("1",)), ("probability",))

    module = DeepLogModuleFactory().compile(BinaryOp("times", Atom(atom1), Atom(atom2)))

    assert module is not None
    assert set(get_all_symbols(module.get_input_shape())) == {atom1, atom2}

    input = torch.rand(5, 2)
    output = module(input)
    torch.testing.assert_close(output, (input[:, 0] * input[:, 1]).unsqueeze(1))


def test_weighted_model_count():
    burglary_prob = ("p", BURGLARY, ("_", ("0.8",), ("probability",)))
    earthquake_prob = ("p", EARTHQUAKE, ("_", ("0.3",), ("probability",)))

    disjunction = BinaryOp("or", Atom(BURGLARY_BOOL), Atom(EARTHQUAKE_BOOL))
    transformed = Transformation("probability", disjunction)
    joint = BinaryOp(
        "times",
        Atom(("_", burglary_prob, ("probability",))),
        Atom(("_", earthquake_prob, ("probability",))),
    )
    weighted = BinaryOp("times", transformed, joint)
    summed = Aggregation("sum", (BURGLARY, EARTHQUAKE), (), weighted)
    module = DeepLogModuleFactory().compile(summed)

    assert module is not None

    result = module()
    torch.testing.assert_close(result, torch.tensor([[1 - 0.2 * 0.7]]))


def test_mnist_addition():
    I1, I2 = ("I1",), ("I2",)
    N1, N2 = ("N1",), ("N2",)
    Sum = ("Sum",)

    sums_atom = ("sums", N1, N2, Sum)
    digit1_atom = ("digit", I1, N1)
    digit2_atom = ("digit", I2, N2)

    digit_domain = Domain.of_tensor(torch.arange(10))
    sum_domain = Domain.of_tensor(torch.arange(19))

    variable_domains = {
        N1: digit_domain,
        N2: digit_domain,
        Sum: sum_domain,
    }

    atom_builders = {
        SumsPredicate.get_predicate(): SumsPredicate,
        ("digit", 2, "probability"): get_network_predicate(
            "digit", 2, "probability", module=IndexClassifier(num_classes=10)
        ),
    }

    digit_joint = BinaryOp(
        "times",
        Atom(("_", digit1_atom, ("probability",))),
        Atom(("_", digit2_atom, ("probability",))),
    )
    sums_probability = Transformation(
        "probability", Atom(("_", sums_atom, ("boolean",)))
    )
    root = BinaryOp("times", sums_probability, digit_joint)
    summed = Aggregation("sum", (N1, N2), (), root)

    factory = DeepLogModuleFactory(variable_domains, atom_builders=atom_builders)
    module = factory.compile(summed)
    module = reshape(module, input=SymTensor([I1, I2, Sum]))

    for i1_gt in range(10):
        for i2_gt in range(10):
            expected_result = torch.zeros(19, 1)
            for i1 in range(10):
                p1 = 0.9 if i1 == i1_gt else 0.1 / 9
                for i2 in range(10):
                    p2 = 0.9 if i2 == i2_gt else 0.1 / 9
                    expected_result[i1 + i2] += p1 * p2

            result = module(torch.tensor([[i1_gt, i2_gt, s] for s in range(19)]))
            torch.testing.assert_close(result, expected_result)


def test_an_atom_builder_passed_to_the_factory_replaces_the_default():
    """The caller's builder for a predicate is used, not the default one."""

    class AlwaysTrue(EqualityPredicate):
        def forward_predicate(self, lhs, rhs):
            return torch.ones_like(lhs, dtype=torch.get_default_dtype())

    factory = DeepLogModuleFactory(
        {("Burglary",): Domain.of([("false",), ("true",)])},
        atom_builders={
            AlwaysTrue.get_predicate(): partial(
                AlwaysTrue, domain=[("false",), ("true",)]
            )
        },
    )

    module = parse_formula_to_module("sum(Burglary): =(Burglary,true)_boolean", factory)

    torch.testing.assert_close(module(), torch.tensor([[2.0]]))


def test_create_unary_node_negation_boolean():
    burglary = ("Burglary",)
    burglary_atom = ("=", burglary, TRUE)

    factory = CircuitFactory()

    # Create leaf node and negate it
    burglary_node = factory.create_atom(("_", burglary_atom, ("boolean",)))
    negated_burglary = factory.create_unary_node("not", burglary_node)

    # Verify node and structure
    assert isinstance(negated_burglary, CircuitNode)
    assert negated_burglary.circuit.structure.name == "boolean"

    # Aggregate and convert to module
    summed = Aggregation("sum", (burglary,), (), negated_burglary)
    module = DeepLogModuleFactory().compile(summed)
    assert module is not None
    result = module()

    # NOT(True) + NOT(False) = False + True = 1
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_negation_probability():
    burglary = ("Burglary",)
    burglary_prob = ("p", burglary, ("_", ("0.7",), ("probability",)))

    factory = CircuitFactory()

    # Create probability leaf and negate it
    prob_node = factory.create_atom(("_", burglary_prob, ("probability",)))
    negated_prob = factory.create_unary_node("negate", prob_node)

    # Verify structure
    assert isinstance(negated_prob, CircuitNode)
    assert negated_prob.circuit.structure.name == "probability"

    # Sum over all values: (1-0.3) + (1-0.7) = 0.7 + 0.3 = 1.0
    summed = Aggregation("sum", (burglary,), (), negated_prob)
    module = DeepLogModuleFactory().compile(summed)
    assert module is not None
    result = module()

    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_with_binary_operations():
    burglary = ("Burglary",)
    earthquake = ("Earthquake",)

    burglary_atom = ("=", burglary, TRUE)
    earthquake_atom = ("=", earthquake, TRUE)

    # Create: Burglary AND NOT(Earthquake)
    conjunction = BinaryOp(
        "and",
        Atom(("_", burglary_atom, ("boolean",))),
        UnaryOp("not", Atom(("_", earthquake_atom, ("boolean",)))),
    )

    # Sum over all combinations
    summed = Aggregation("sum", (burglary, earthquake), (), conjunction)
    module = DeepLogModuleFactory().compile(summed)
    assert module is not None
    result = module()

    # Count models where Burglary=True AND NOT(Earthquake)
    # B=0, E=0: False AND True = False
    # B=0, E=1: False AND False = False
    # B=1, E=0: True AND True = True ✓
    # B=1, E=1: True AND False = False
    # Result: 1 model
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_create_unary_node_double_negation():
    burglary_atom = ("=", BURGLARY, TRUE)

    # Create NOT(NOT(Burglary))
    negated_twice = UnaryOp(
        "not", UnaryOp("not", Atom(("_", burglary_atom, ("boolean",))))
    )

    # Sum over all values - should be same as original
    summed = Aggregation("sum", (BURGLARY,), (), negated_twice)
    module = DeepLogModuleFactory().compile(summed)
    assert module is not None
    result = module()

    # NOT(NOT(True)) + NOT(NOT(False)) = True + False = 1
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_custom_algebraic_structure():
    """Test that the lowering pipeline works with a custom AlgebraicStructure."""
    # Define a custom structure (same as probability but with a different name)
    custom_structure = Algebra(
        name="custom_prob",
        product="times",
        product_fn=lambda a, b: a * b,
        sum="plus",
        sum_fn=lambda a, b: a + b,
        negation="negate",
        negation_fn=lambda x: 1.0 - x,
    )

    atom1 = ("_", ("nn", ("0",)), ("custom_prob",))
    atom2 = ("_", ("nn", ("1",)), ("custom_prob",))

    factory = DeepLogModuleFactory(structures={"custom_prob": custom_structure})
    module = factory.compile(BinaryOp("times", Atom(atom1), Atom(atom2)))

    assert module is not None

    input_data = torch.rand(5, 2)
    output = module(input_data)
    torch.testing.assert_close(
        output, (input_data[:, 0] * input_data[:, 1]).unsqueeze(1)
    )


def test_aggregation_defaults_to_boolean_domain():
    summed = Aggregation("sum", (BURGLARY,), (), Atom(BURGLARY_BOOL))
    module = DeepLogModuleFactory().compile(summed)
    assert module is not None
    result = module()

    # Should enumerate both False/True assignments even without explicit domain mapping
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_subroot_materialization_only_uses_reachable_leaves():
    """Materializing a sub-root ignores unrelated leaves of the shared circuit.

    Circuits are shared per structure, so a second formula's atom coexists in
    the graph; it must contribute neither a feeder nor an external input to a
    module that cannot use it.
    """
    factory = CircuitFactory()
    a = factory.create_atom(("_", ("=", ("A",), TRUE), ("boolean",)))
    factory.create_atom(
        ("_", ("=", ("B",), TRUE), ("boolean",))
    )  # unrelated, co-resident

    summed = Aggregation("sum", (("A",),), (), a)
    module = DeepLogModuleFactory().compile(summed)

    assert set(get_all_symbols(module.get_input_shape())) == set()
    result = module()
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_expectation_rejects_boundary_leaves():
    """An expectation child with a cross-structure boundary leaf is refused.

    The boolean→probability transform rewrites leaf symbols, which would
    silently detach a boundary leaf from its recorded feeder — better loud.
    """
    factory = CircuitFactory()
    base = factory.create_atom(("_", ("q",), ("boolean",)))
    # A boolean lump carrying a cross-structure cast feeder on its leaf.
    leaf = ("_", ("q",), ("boolean",))
    child = CircuitNode(
        base.circuit, base.node, ((leaf, Transformation("boolean", base)),)
    )
    with pytest.raises(ValueError, match="boundary"):
        factory.create_aggregation("expectation", [], [], child)


def test_cast_defers_to_a_symbolic_transform_leaf():
    """``create_transformation`` defers: a ``transform`` circuit leaf, no spine.

    The cast is a ``transform`` leaf in the target circuit whose single boundary
    child is the *symbolic* :class:`Transformation` of its source; the cast
    module is built only at a closing point (lowering).
    """
    factory = CircuitFactory()
    leaf_x = ("_", ("=", ("X",), TRUE), ("boolean",))
    source = factory.create_atom(leaf_x)
    cast_x = factory.create_transformation("probability", source)

    assert isinstance(cast_x, CircuitNode)
    assert cast_x.circuit.structure.name == "probability"
    # The boundary child is the source, cast — and its own boundary is the leaf.
    assert cast_x.children == (Transformation("probability", source),)
    (feeder,) = cast_x.children
    assert feeder.child.children == (Atom(leaf_x),)


def test_predicate_across_two_casts_lowered_per_leaf():
    """A predicate feeding two casts is lowered once *per leaf* (transparent fold).

    Each cast re-leafs its own boolean sub-circuit; the transparent fold lowers
    every leaf independently through ``create_atom``, so the shared ``=``
    predicate is instantiated once per atom (sharing the builder's weights, not
    one batched forward pass — the accepted per-leaf tradeoff). Correctness is
    unchanged: ``X ∧ Y`` is the single model of the four assignments.
    """
    factory = CircuitFactory()
    leaf_x = ("_", ("=", ("X",), TRUE), ("boolean",))
    leaf_y = ("_", ("=", ("Y",), TRUE), ("boolean",))
    cast_x = factory.create_transformation("probability", factory.create_atom(leaf_x))
    cast_y = factory.create_transformation("probability", factory.create_atom(leaf_y))
    product = factory.create_binary_node("times", cast_x, cast_y)

    summed = Aggregation("sum", (("X",), ("Y",)), (), product)
    module = DeepLogModuleFactory().compile(summed)

    result = module()  # one model (X ∧ Y) out of four assignments
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))

    # Per-leaf lowering: one EqualityPredicate instance per atom (recursively, as
    # each is nested inside its cast spine), not a single batched feeder.
    eq_instances = [
        m for m in module.modules() if type(m).__name__ == "EqualityPredicate"
    ]
    assert len(eq_instances) == 2


def test_unreachable_boundary_spine_not_dragged_in():
    """Lowering one cast's subtree excludes the other cast's spine and feeders.

    Both casts re-leaf into the shared probability circuit, but only the
    leaves reachable from the compiled root contribute modules or inputs.
    """
    factory = CircuitFactory()
    leaf_x = ("_", ("=", ("X",), TRUE), ("boolean",))
    leaf_y = ("_", ("=", ("Y",), TRUE), ("boolean",))
    cast_x = factory.create_transformation("probability", factory.create_atom(leaf_x))
    cast_y = factory.create_transformation("probability", factory.create_atom(leaf_y))
    negated_x = factory.create_unary_node("negate", cast_x)
    factory.create_unary_node("negate", cast_y)  # co-resident, unreachable

    summed = Aggregation("sum", (("X",),), (), negated_x)
    module = DeepLogModuleFactory().compile(summed)

    assert set(get_all_symbols(module.get_input_shape())) == set()
    result = module()  # (1 - [true=true]) + (1 - [false=true]) = 0 + 1
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


def test_shared_circuit_node_handle_keeps_circuit_linear():
    """Reusing one boolean ``CircuitNode`` handle keeps the circuit linear.

    The circuit dedups leaves but not operator nodes, so what bounds size is
    *handle reuse*, not a fold-level memo: conjoining the same node object
    repeatedly references the existing node id each time, so a depth-``d``
    doubling chain yields ``d`` AND nodes with no per-path re-expansion. This is
    how the DeepProbLog engine — which reuses one sub-proof ``CircuitNode`` across
    a conjunction — stays linear on heavily-shared (SDD-shaped) proofs.
    """
    factory = CircuitFactory()
    node = factory.create_atom(("_", ("x",), ("boolean",)))
    depth = 40
    for _ in range(depth):
        node = factory.create_binary_node("and", node, node)  # same handle twice

    assert isinstance(node, CircuitNode)
    circuit = node.circuit
    and_nodes = sum(
        1
        for nid in circuit.iter_topological([node.node])
        if circuit._get_node(nid).node_type == "and"
    )
    assert and_nodes == depth


# --- Connectives over non-circuit children (LTN) ---------------------------


def _ltn_factory():
    """The LTN grounding setup: a fuzzy algebra and two p-mean quantifiers.

    Mirrors ``examples/ltn/ltn.ipynb``. The structure is a plain
    ``AlgebraicStructure`` — its product t-norm does not distribute over its
    probabilistic sum, so it is deliberately not a ``Semiring``.
    """
    fuzzy = AlgebraicStructure(
        name="fuzzy",
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "implies": lambda a, b: 1 - a + a * b,
            "not": lambda x: 1.0 - x,
        },
    )

    def generalized_mean(x, p):
        return torch.mean(x**p, dim=1) ** (1 / p)

    def aggregator(name, op):
        return lambda child, binders, _params, domains: AggregationModule(
            child, binders, domains, name=name, op=op
        )

    class EqualityPredicate(Predicate):
        functor, arity, structure = "eq", 2, "fuzzy"

        def forward_predicate(self, x, y):
            return torch.exp(-torch.norm(x - y, dim=1))

    torch.manual_seed(0)
    return fuzzy, DeepLogModuleFactory(
        structures={"fuzzy": fuzzy},
        aggregators={
            "forall": aggregator("forall", lambda x: 1 - generalized_mean(1 - x, 4)),
            "exists": aggregator("exists", lambda x: generalized_mean(x, 6)),
        },
        variables={
            ("x",): Domain.of_tensor(torch.randn(10, 2)),
            ("y",): Domain.of_tensor(torch.randn(5, 2) * 2),
        },
        atom_builders={("eq", 2, "fuzzy"): EqualityPredicate},
    )


def _eq(*variables):
    """The fuzzy ``eq`` atom over the given variables."""
    return Atom(with_structure(("eq", *variables), "fuzzy"))


def test_binary_connective_over_two_quantifiers():
    """``And(Forall x . φ, Forall y . φ)`` lowers over the symbol union.

    Neither operand is a circuit node — a quantifier binds nothing a circuit can
    express — so the ``and`` survives the construction fold and is applied to the
    two already-reduced outputs. The operands consume different free variables,
    so the module's input is their union.
    """
    _, factory = _ltn_factory()
    x, y = ("x",), ("y",)

    module = factory.compile(
        BinaryOp(
            "and",
            Aggregation("forall", (x,), (), _eq(x, y)),
            Aggregation("forall", (y,), (), _eq(x, y)),
        )
    )

    assert list(get_all_symbols(module.get_input_shape())) == [y, x]
    assert torch.isfinite(module(torch.full((1, 2, 2), 0.25))).all()


@pytest.mark.parametrize("connective", ["and", "or", "implies"])
def test_connective_over_quantifiers_matches_the_algebra(connective):
    """The lowered connective equals its operator function on the two scalars."""
    fuzzy, factory = _ltn_factory()
    x, y = ("x",), ("y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))
    existential = Aggregation("exists", (x, y), (), _eq(x, y))

    combined = factory.compile(BinaryOp(connective, universal, existential))
    expected = fuzzy.get_operator_fn(connective)(
        factory.compile(universal)(), factory.compile(existential)()
    )

    torch.testing.assert_close(combined(), expected)


def test_unary_connective_over_a_quantifier_matches_the_algebra():
    """``Not(Forall …)`` applies the structure's negation to the reduced output."""
    fuzzy, factory = _ltn_factory()
    x, y = ("x",), ("y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))

    negated = factory.compile(UnaryOp("not", universal))
    expected = fuzzy.get_operator_fn("not")(factory.compile(universal)())

    torch.testing.assert_close(negated(), expected)


def test_connective_the_structure_does_not_define_is_rejected():
    """An operator missing from the algebra names itself and what is available."""
    _, factory = _ltn_factory()
    x, y = ("x",), ("y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))
    lowered = factory.compile(universal)

    with pytest.raises(NotImplementedError, match="has no operator 'xor'"):
        factory.create_binary_node("xor", lowered, lowered)


# --- Inside a formula, every output is labelled ------------------------------


def _bare_module(symbol=("q",)) -> WrappedModule:
    """A module whose output names no algebra — built outside the formula layer."""
    return WrappedModule(lambda x: x, SymTensor([symbol]), SymTensor([symbol]))


def test_operator_over_an_unlabelled_operand_is_refused():
    """A missing label is not a default algebra; guessing one would return
    plausible wrong numbers instead of an error."""
    factory = DeepLogModuleFactory()
    bare = _bare_module()

    with pytest.raises(NotImplementedError, match="carry no algebraic structure"):
        factory.create_binary_node("times", bare, bare)


def test_cast_of_an_unlabelled_child_is_refused_not_treated_as_real():
    """The ``real -> probability`` sigmoid is selected by a *declared* ``real``
    label, never by the absence of one."""
    factory = DeepLogModuleFactory()

    with pytest.raises(NotImplementedError, match="carry no algebraic structure"):
        factory.create_transformation("probability", _bare_module())


def test_cast_of_a_declared_real_child_applies_the_sigmoid():
    """Labelling a network output ``real`` is what opts into the cast."""
    factory = DeepLogModuleFactory(
        transformations={
            ("real", "probability"): partial(
                build_transform, from_structure="real", to_structure="probability"
            )
        }
    )
    logits = ("logit",)
    network = WrappedModule(
        lambda x: x,
        SymTensor([logits]),
        SymTensor([with_structure(logits, "real")]),
    )

    cast = factory.create_transformation("probability", network)

    x = torch.tensor([[2.0]])
    torch.testing.assert_close(cast(x), torch.sigmoid(x))
    assert sole_structure(cast.get_output_shape()) == "probability"


def test_cast_without_a_registered_builder_names_what_is_available():
    """A labelled child with no ``(from, to)`` builder fails by name, not KeyError."""
    factory = DeepLogModuleFactory()
    network = WrappedModule(
        lambda x: x,
        SymTensor([("logit",)]),
        SymTensor([with_structure(("logit",), "real")]),
    )

    with pytest.raises(
        NotImplementedError, match="no cast from 'real' to 'probability'"
    ):
        factory.create_transformation("probability", network)


def test_aggregation_over_an_unlabelled_child_is_refused():
    """The same rule at the binder: an enumeration reduces *within* an algebra,
    so there has to be one."""
    factory = DeepLogModuleFactory(
        variables={("X",): Domain.of_tensor(torch.tensor([0.0, 1.0]))}
    )
    child = WrappedModule(lambda x: x, SymTensor([("X",)]), SymTensor([("q",)]))

    with pytest.raises(NotImplementedError, match="carry no algebraic structure"):
        factory.create_aggregation("sum", [("X",)], (), child)


# --- Compiling several formulas at once ---


def _probability_product(*names: str) -> BinaryOp:
    """``p(a) times p(b)`` over probability atoms."""
    first, second = (Atom(("_", ("p", (name,)), ("probability",))) for name in names)
    return BinaryOp("times", first, second)


def test_compile_gives_one_column_per_formula():
    """Each root is an output, in the order given, and means what it did alone."""
    shared, other = _probability_product("x", "y"), _probability_product("x", "z")

    together = DeepLogModuleFactory().compile(shared, other)
    together = reshape(
        together,
        input=SymTensor(
            [("_", ("p", (name,)), ("probability",)) for name in "xyz"],
        ),
    )

    out = together(torch.tensor([[0.5, 0.4, 0.2]]))
    torch.testing.assert_close(out, torch.tensor([[0.2, 0.1]]))


def test_compile_shares_the_circuit_and_the_atoms_between_formulas():
    """Roots are co-resident: one compiled circuit, and a shared atom is one slot."""
    module = DeepLogModuleFactory().compile(
        _probability_product("x", "y"), _probability_product("x", "z")
    )

    compiled = [m for m in module.modules() if isinstance(m, WrappedModule)]
    assert len(compiled) == 1
    # p(x) feeds both roots, so it is asked for once.
    assert len(list(get_all_symbols(module.get_input_shape()))) == 3


def test_compile_of_one_formula_is_unchanged():
    """One root still lowers to what the single-root path produced."""
    formula = _probability_product("x", "y")

    module = DeepLogModuleFactory().compile(formula)

    (column,) = get_all_symbols(module.get_output_shape())
    assert sole_structure(module.get_output_shape()) == "probability"
    assert column[1][0].startswith("circuit_probability_")


def test_compile_composes_roots_that_are_not_co_resident():
    """Roots in different algebras cannot share a circuit, and still compose."""
    probability = _probability_product("x", "y")
    boolean = BinaryOp(
        "and",
        Atom(("_", ("=", ("B",), ("true",)), ("boolean",))),
        Atom(("_", ("=", ("E",), ("true",)), ("boolean",))),
    )

    module = DeepLogModuleFactory().compile(probability, boolean)

    assert len(list(get_all_symbols(module.get_output_shape()))) == 2


def test_compile_refuses_two_formulas_that_name_one_column():
    """Structurally equal roots are one node, so they cannot be two columns."""
    formula = _probability_product("x", "y")

    with pytest.raises(ValueError, match="same output column"):
        DeepLogModuleFactory().compile(formula, formula)


def test_compile_requires_a_formula():
    """There is no module with no columns."""
    with pytest.raises(ValueError, match="At least one formula"):
        DeepLogModuleFactory().compile()
