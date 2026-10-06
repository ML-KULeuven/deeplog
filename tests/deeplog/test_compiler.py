#  Copyright (c) 2024-2026. KU Leuven
"""Tests for compiling a formula AST to a DeepLogModule.

Circuit *construction* (atoms, operators, casts) is the :class:`CircuitFactory`'s
job; :class:`Compiler` constructs formulas with it and *lowers* the resulting
lumps, and every aggregation, top-down into one runnable module. These tests
build formulas as a plain AST, or lumps with a ``CircuitFactory``, and compile
them.
"""

import math
from functools import partial

import pytest
import torch

from deeplog import PROBABILITY
from deeplog import REAL
from deeplog import Aggregation
from deeplog import Algebra
from deeplog import AlgebraicStructure
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Compiler
from deeplog import EqualityPredicate
from deeplog import NetworkPredicate
from deeplog import Predicate
from deeplog import SymTensor
from deeplog import Transformation
from deeplog import UnaryOp
from deeplog import WrappedModule
from deeplog import parse_formula
from deeplog import parse_formula_to_module
from deeplog import reduction
from deeplog import reshape
from deeplog import with_structure
from deeplog.circuit import Circuit
from deeplog.formula.ast import CircuitNode
from deeplog.formula.circuit_factory import CircuitFactory
from deeplog.formula.lowering.lowering import Lowering
from deeplog.shape import get_all_symbols
from deeplog.shape import sole_structure
from deeplog.symbol import without_structure
from deeplog.variable import Domain

from .testing_modules import VALUES
from .testing_modules import Forecast
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
    summed = Aggregation(
        "sum", (BURGLARY, EARTHQUAKE), (), Transformation("real", disjunction)
    )
    module = Compiler().compile(summed)
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

    module = Compiler().compile(BinaryOp("times", Atom(atom1), Atom(atom2)))

    assert module is not None
    assert set(get_all_symbols(module.get_input_shape())) == {atom1, atom2}

    input = torch.rand(5, 2)
    output = module(input)
    torch.testing.assert_close(output, (input[:, 0] * input[:, 1]).unsqueeze(1))


def test_an_atom_no_builder_computes_is_read_at_each_value_of_its_binder():
    """An aggregation substitutes each value, so the atom is its ground atoms."""
    module = Compiler().compile(parse_formula("sum(X): w(X)_real"))
    ground = SymTensor(["w(false) _ real", "w(true) _ real"])

    assert set(get_all_symbols(module.get_input_shape())) == set(
        get_all_symbols(ground)
    )
    torch.testing.assert_close(
        reshape(module, input=ground)(torch.tensor([[0.1, 0.5]])), torch.tensor([[0.6]])
    )


def test_an_atom_no_builder_computes_is_read_at_the_values_of_every_binder():
    """``w(X,Y)`` reads ``X`` from the outer sum and ``Y`` from the inner one."""
    compiler = Compiler(
        variables={("X",): Domain.of(["a", "b"]), ("Y",): Domain.of(["c", "d", "e"])}
    )
    module = compiler.compile(
        parse_formula("sum(X): sum(Y): w(X,Y)_real times v(Y)_real")
    )
    w = SymTensor([f"w({x},{y}) _ real" for x in "ab" for y in "cde"])
    v = SymTensor([f"v({y}) _ real" for y in "cde"])
    module = reshape(module, input=(w, v))

    # w(a,c)..w(b,e) are 0..5 and v(c), v(d), v(e) are 1, 10, 100.
    torch.testing.assert_close(
        module(torch.arange(6.0).reshape(1, 6), torch.tensor([[1.0, 10.0, 100.0]])),
        torch.tensor([[3.0 * 1 + 5.0 * 10 + 7.0 * 100]]),
    )


def test_an_atom_no_builder_computes_over_a_free_variable_is_one_input():
    """The caller gives a free variable its value, and the atom's label there."""
    module = Compiler().compile(parse_formula("w(Y)_real times (sum(X): w(X)_real)"))

    assert set(get_all_symbols(module.get_input_shape())) == set(
        get_all_symbols(SymTensor(["w(Y) _ real", "w(false) _ real", "w(true) _ real"]))
    )


def test_an_equality_reads_the_same_whichever_side_its_variable_is_on():
    """``=(a,X)`` is ``=(X,a)``: the label of ``X``'s value ``a``, not ground over ``X``."""
    compiler = Compiler(variables={("X",): Domain.of(["a", "b"])})
    written, swapped = (
        compiler.compile(parse_formula(f"sum(X): {test}_probability"))
        for test in ("=(X,a)", "=(a,X)")
    )

    assert swapped.get_input_shape() == written.get_input_shape()
    x = torch.tensor([[0.25]])
    torch.testing.assert_close(swapped(x), written(x))


def test_a_lump_leaf_reads_the_equality_feeding_it_whichever_its_spelling():
    """A circuit's leaf ``=(a,X)`` reads the atom ``=(X,a)`` its lump's child is."""
    circuit = Circuit("boolean")
    root = circuit.get_operator("and")(
        circuit.get_leaf_node(("=", ("a",), ("X",))), circuit.get_leaf_node(("q",))
    )
    compiler = Compiler(variables={("X",): Domain.of(["a", "b"])})

    module = reshape(
        compiler.compile(CircuitNode(circuit, root)),
        input=(SymTensor([("X",)]), SymTensor(["q _ boolean"])),
    )

    torch.testing.assert_close(
        module(torch.tensor([[0], [1], [0]]), torch.tensor([[1.0], [1.0], [0.0]])),
        torch.tensor([[1.0], [0.0], [0.0]]),
    )


def test_an_atom_no_builder_computes_over_unnamed_values_is_refused():
    """Its ground atoms would have no names to be inputs under."""
    compiler = Compiler(variables={("X",): Domain.of_tensor(torch.arange(3))})

    with pytest.raises(ValueError, match="register a builder for w/1"):
        compiler.compile(parse_formula("sum(X): w(X)_real"))


#: ``forecast``, which reads the truth values.
_FORECAST = {("forecast", 1, "probability"): Forecast}


@pytest.mark.parametrize(
    "text",
    [
        "forecast(true)_probability",
        "expectation(Rain; forecast(Rain)_probability): =(Rain,true)_boolean",
    ],
    ids=["written", "substituted or fed"],
)
@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_a_written_value_reads_as_a_variable_holding_it(text, builder):
    """Definition 13: a predicate at an assignment is the label of the substituted atom."""
    from deeplog import enumeration

    compiler = Compiler(
        atom_builders=_FORECAST,
        aggregation_builders={"enumerated": {"expectation": enumeration}}.get(
            builder, {}
        ),
    )
    module = compiler.compile(parse_formula(text))

    assert list(get_all_symbols(module.get_input_shape())) == []
    torch.testing.assert_close(module(), torch.tensor([[0.2]]))


def test_a_binder_where_a_predicate_declares_another_domain_is_refused():
    """``forecast`` reads truth values, so a variable over colours would be misread."""
    compiler = Compiler(
        variables={("C",): Domain.of(["red", "green", "blue"])},
        atom_builders=_FORECAST,
    )

    with pytest.raises(
        ValueError,
        match="C is declared over {red, green, blue}, but forecast/1 reads it",
    ):
        compiler.compile(parse_formula("sum(C): forecast(C)_probability"))


@pytest.mark.parametrize("builder", ["counted", "enumerated"])
def test_p_weighs_only_truth_values(builder):
    """``p(X, 0.8)`` over three colours read two of them as true."""
    from deeplog import enumeration

    compiler = Compiler(
        variables={("X",): Domain.of(["a", "b", "c"])},
        aggregation_builders={"enumerated": {"expectation": enumeration}}.get(
            builder, {}
        ),
    )

    with pytest.raises(
        ValueError, match="X is declared over {a, b, c}, but p/2 reads it"
    ):
        compiler.compile(
            parse_formula("expectation(X; p(X,0.8)_probability): =(X,b)_boolean")
        )


def test_a_test_outside_boolean_is_the_label_of_its_value_at_every_value():
    """``=(B, true)`` in probability names ``B``'s value ``true``; it reads no ``B``."""
    module = Compiler().compile(
        parse_formula(
            "sum(B): (=(B,true)_boolean)_probability times =(B,true)_probability"
        )
    )

    assert list(get_all_symbols(module.get_input_shape())) == [
        with_structure(("=", ("B",), ("true",)), "probability")
    ]


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
    module = Compiler().compile(summed)

    assert module is not None

    result = module()
    torch.testing.assert_close(result, torch.tensor([[1 - 0.2 * 0.7]]))


class _Sums(Predicate[torch.Tensor, torch.Tensor, torch.Tensor]):
    """``sums(X, Y, Z)``: whether ``X + Y == Z``."""

    def __init__(self, atoms):
        super().__init__(atoms, (VALUES, VALUES, VALUES))

    def forward_predicate(self, x, y, z):
        return (x + y == z).to(torch.get_default_dtype())


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
        ("sums", 3, "boolean"): _Sums,
        ("digit", 2, "probability"): partial(
            NetworkPredicate, module=IndexClassifier(num_classes=10)
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

    compiler = Compiler(variable_domains, atom_builders=atom_builders)
    module = compiler.compile(summed)
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


class _Counted(torch.nn.Module):
    """``module``, counting how often it runs."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.runs = 0

    def forward(self, x):
        self.runs += 1
        return self.module(x)


def _digit(image, value):
    """The probability-valued leaf ``digit(image, value)``."""
    return Atom(("_", ("digit", (image,), value), ("probability",)))


def test_a_lump_runs_a_network_once_for_all_its_leaves():
    """Six leaves of one network predicate, over two images, are one run of it.

    The rows of each image's distribution sum to one, so their sum over both
    images is two.
    """
    classifier = _Counted(IndexClassifier(num_classes=3))
    compiler = Compiler(
        atom_builders={
            ("digit", 2, "probability"): partial(NetworkPredicate, module=classifier)
        }
    )
    leaves = [_digit(image, (str(v),)) for image in ("i1", "i2") for v in range(3)]
    root = leaves[0]
    for leaf in leaves[1:]:
        root = BinaryOp("plus", root, leaf)

    module = reshape(compiler.compile(root), input=SymTensor([("i1",), ("i2",)]))
    result = module(torch.tensor([[0.0, 2.0]]))

    assert classifier.runs == 1
    torch.testing.assert_close(result, torch.tensor([[2.0]]))


def test_a_lump_under_an_aggregation_runs_its_network_once():
    """``sum(N): digit(i1, N) times digit(i2, N)`` runs the classifier once per call.

    Both images' leaves feed one lump, which the aggregation evaluates over every
    ``N`` at once. With images 0 and 2 of three classes, the two agree on no
    class they both favour: 0.9 * 0.05 + 0.05 * 0.05 + 0.05 * 0.9.
    """
    classifier = _Counted(IndexClassifier(num_classes=3))
    compiler = Compiler(
        {("N",): Domain.of_tensor(torch.arange(3))},
        atom_builders={
            ("digit", 2, "probability"): partial(NetworkPredicate, module=classifier)
        },
    )
    body = BinaryOp("times", _digit("i1", ("N",)), _digit("i2", ("N",)))

    module = reshape(
        compiler.compile(Aggregation("sum", (("N",),), (), body)),
        input=SymTensor([("i1",), ("i2",)]),
    )
    result = module(torch.tensor([[0.0, 2.0]]))

    assert classifier.runs == 1
    expected = 0.9 * 0.05 + 0.05 * 0.05 + 0.05 * 0.9
    torch.testing.assert_close(result, torch.tensor([[expected]]))


def test_leaves_reach_their_atom_builder_in_one_call_across_lumps():
    """Every leaf of a predicate in a level is built by one builder call.

    ``digit(i1, 0)`` goes to log space and back, so its lump and the lump
    multiplying it by ``digit(i2, 0)`` are two lumps of the probability circuit,
    one reading the other. Their leaves still reach the builder together, while
    the two lumps compile into separate cores: one core cannot read its own
    output. With images 0 and 2 the product is 0.9 * 0.05. The type check
    builds the predicate once before, to read its sorts.
    """
    calls = []
    network = partial(NetworkPredicate, module=IndexClassifier(num_classes=3))

    def recording(atoms):
        calls.append(sorted(map(str, atoms)))
        return network(atoms)

    compiler = Compiler(atom_builders={("digit", 2, "probability"): recording})
    there_and_back = Transformation(
        "probability", Transformation("logprobability", _digit("i1", ("0",)))
    )
    formula = BinaryOp("times", there_and_back, _digit("i2", ("0",)))

    module = reshape(compiler.compile(formula), input=SymTensor([("i1",), ("i2",)]))

    leaves = [_digit(image, ("0",)).atom for image in ("i1", "i2")]
    assert calls == [sorted(map(str, leaves))] * 2
    torch.testing.assert_close(
        module(torch.tensor([[0.0, 2.0]])), torch.tensor([[0.9 * 0.05]])
    )


def test_an_operand_read_twice_runs_once():
    """A value two operators read is one module in the graph, run once.

    ``sum(N): digit(i1, N)`` totals a distribution, so it is one, and so is its
    square.
    """
    classifier = _Counted(IndexClassifier(num_classes=3))
    compiler = Compiler(
        {("N",): Domain.of_tensor(torch.arange(3))},
        atom_builders={
            ("digit", 2, "probability"): partial(NetworkPredicate, module=classifier)
        },
    )
    total = Aggregation("sum", (("N",),), (), _digit("i1", ("N",)))

    module = compiler.compile(BinaryOp("times", total, total))
    result = module(torch.tensor([[1.0]]))

    assert classifier.runs == 1
    torch.testing.assert_close(result, torch.tensor([[1.0]]))


def test_an_atom_builder_passed_to_the_compiler_replaces_the_default():
    """The caller's builder for a predicate is used, not the default one."""

    class AlwaysTrue(EqualityPredicate):
        def forward_predicate(self, lhs, rhs):
            return torch.ones_like(lhs, dtype=torch.get_default_dtype())

    compiler = Compiler(
        {("Burglary",): Domain.of([("false",), ("true",)])},
        atom_builders={
            ("=", 2, "boolean"): partial(
                AlwaysTrue, domain_of=lambda _: Domain.of([("false",), ("true",)])
            )
        },
    )

    module = parse_formula_to_module(
        "sum(Burglary): (=(Burglary,true)_boolean)_real", compiler
    )

    torch.testing.assert_close(module(), torch.tensor([[2.0]]))


class _Over(WrappedModule):
    """A module whose atoms' arguments range over the values."""

    def domains_of(self, arguments):
        return (VALUES,) * len(arguments)


def test_a_builder_without_a_column_for_its_atom_is_refused():
    """Its module would be dropped, and the atom made an input in its place."""

    def build(atoms):
        name = with_structure(("oops",), "probability")
        return _Over(lambda: torch.full((1, 1), 0.5), (), SymTensor([name]))

    compiler = Compiler(atom_builders={("p", 1, "probability"): build})

    with pytest.raises(ValueError, match=r"no column for p\(x\) _ probability"):
        compiler.compile(parse_formula("p(x)_probability times q_probability"))


def test_one_predicate_serves_several_predicates_and_algebras():
    """A predicate names its columns by the atoms it is given, not by a key of its own."""

    class Half(Predicate):
        def __init__(self, atoms):
            super().__init__(atoms, (VALUES,))

        def forward_predicate(self, x):
            return x / 2

    compiler = Compiler(
        atom_builders={("p", 1, "probability"): Half, ("q", 1, "real"): Half}
    )

    module = compiler.compile(
        parse_formula("p(X)_probability"), parse_formula("q(X)_real")
    )

    torch.testing.assert_close(
        module(torch.tensor([[1.0]])), torch.tensor([[0.5, 0.5]])
    )


def test_a_builder_may_return_columns_it_was_not_asked_for():
    def build(atoms):
        names = list(atoms)
        extra = with_structure(("p", ("unused",)), "probability")
        return _Over(
            lambda: torch.tensor([[0.5] * len(names) + [0.9]]),
            (),
            SymTensor([*names, extra]),
        )

    module = Compiler(atom_builders={("p", 1, "probability"): build}).compile(
        parse_formula("p(x)_probability times p(y)_probability")
    )

    torch.testing.assert_close(module(), torch.tensor([[0.25]]))


def test_a_chain_of_casts_thousands_deep_compiles():
    node = Atom(with_structure(("a",), "probability"))
    for i in range(2_000):
        node = Transformation("logprobability" if i % 2 == 0 else "probability", node)

    module = Compiler().compile(node)

    torch.testing.assert_close(module(torch.tensor([[0.3]])), torch.tensor([[0.3]]))


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
    summed = Aggregation(
        "sum", (burglary,), (), Transformation("real", negated_burglary)
    )
    module = Compiler().compile(summed)
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
    module = Compiler().compile(summed)
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
    summed = Aggregation(
        "sum", (burglary, earthquake), (), Transformation("real", conjunction)
    )
    module = Compiler().compile(summed)
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
    summed = Aggregation("sum", (BURGLARY,), (), Transformation("real", negated_twice))
    module = Compiler().compile(summed)
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

    compiler = Compiler(structures={"custom_prob": custom_structure})
    module = compiler.compile(BinaryOp("times", Atom(atom1), Atom(atom2)))

    assert module is not None

    input_data = torch.rand(5, 2)
    output = module(input_data)
    torch.testing.assert_close(
        output, (input_data[:, 0] * input_data[:, 1]).unsqueeze(1)
    )


def test_aggregation_defaults_to_boolean_domain():
    summed = Aggregation(
        "sum", (BURGLARY,), (), Transformation("real", Atom(BURGLARY_BOOL))
    )
    module = Compiler().compile(summed)
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

    summed = Aggregation("sum", (("A",),), (), Transformation("real", a))
    module = Compiler().compile(summed)

    assert set(get_all_symbols(module.get_input_shape())) == set()
    result = module()
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))


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


def test_a_predicate_under_two_casts_is_built_once():
    """A predicate whose leaves sit in two lumps is one module for both.

    Each cast re-leafs its own boolean sub-circuit, so ``=(X, true)`` and
    ``=(Y, true)`` are leaves of two lumps; both lumps are in the sum's body, so
    the ``=`` predicate is built once for the two of them. ``X ∧ Y`` is the
    single model of the four assignments.
    """
    factory = CircuitFactory()
    leaf_x = ("_", ("=", ("X",), TRUE), ("boolean",))
    leaf_y = ("_", ("=", ("Y",), TRUE), ("boolean",))
    cast_x = factory.create_transformation("probability", factory.create_atom(leaf_x))
    cast_y = factory.create_transformation("probability", factory.create_atom(leaf_y))
    product = factory.create_binary_node("times", cast_x, cast_y)

    summed = Aggregation("sum", (("X",), ("Y",)), (), product)
    module = Compiler().compile(summed)

    result = module()  # one model (X ∧ Y) out of four assignments
    torch.testing.assert_close(result, torch.tensor([[1.0]], dtype=result.dtype))

    eq_instances = [
        m for m in module.modules() if type(m).__name__ == "EqualityPredicate"
    ]
    assert len(eq_instances) == 1


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
    module = Compiler().compile(summed)

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
        for nid in circuit._iter_topological([node.node])
        if circuit._get_node(nid).node_type == "and"
    )
    assert and_nodes == depth


# --- Connectives over non-circuit children (LTN) ---------------------------


def _ltn_compiler(**aggregation_builders):
    """The LTN grounding setup: a fuzzy algebra with two p-mean quantifiers.

    Mirrors ``examples/ltn.md``, plus a ``below`` predicate reading a scalar
    radius beside ``eq``'s points, a ``weighted`` mean aggregator, and the
    ``aggregation_builders`` given. The structure is a plain
    ``AlgebraicStructure`` — its product t-norm does not distribute over its
    probabilistic sum, so it is deliberately not a ``Semiring``.
    """

    def generalized_mean(x, p):
        return torch.mean(x**p, dim=1) ** (1 / p)

    fuzzy = AlgebraicStructure(
        name="fuzzy",
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "implies": lambda a, b: 1 - a + a * b,
            "not": lambda x: 1.0 - x,
        },
        aggregation_fns={
            "forall": lambda x: 1 - generalized_mean(1 - x, 4),
            "exists": lambda x: generalized_mean(x, 6),
            "weighted": _weighted_mean,
        },
    )

    class EqualityPredicate(Predicate):
        def __init__(self, atoms):
            super().__init__(atoms, (VALUES, VALUES))

        def forward_predicate(self, x, y):
            return torch.exp(-torch.norm(x - y, dim=1))

    class BelowPredicate(EqualityPredicate):
        def forward_predicate(self, point, radius):
            return torch.sigmoid(radius - torch.norm(point, dim=1))

    torch.manual_seed(0)
    return fuzzy, Compiler(
        structures={"fuzzy": fuzzy},
        aggregation_builders=aggregation_builders,
        variables={
            ("X",): Domain.of_tensor(torch.randn(10, 2)),
            ("Y",): Domain.of_tensor(torch.randn(5, 2) * 2),
        },
        atom_builders={
            ("eq", 2, "fuzzy"): EqualityPredicate,
            ("below", 2, "fuzzy"): BelowPredicate,
        },
    )


def _eq(*variables):
    """The fuzzy ``eq`` atom over the given variables."""
    return Atom(with_structure(("eq", *variables), "fuzzy"))


def test_binary_connective_over_two_quantifiers():
    """``And(Forall x . φ, Forall y . φ)`` lowers over the symbol union.

    Neither operand is a circuit node — a quantifier binds nothing a circuit can
    express — so the ``and`` survives the construction fold and is applied to the
    two already-reduced outputs. The operands consume different free variables,
    so the module takes both, one tensor each.
    """
    _, compiler = _ltn_compiler()
    x, y = ("X",), ("Y",)

    module = compiler.compile(
        BinaryOp(
            "and",
            Aggregation("forall", (x,), (), _eq(x, y)),
            Aggregation("forall", (y,), (), _eq(x, y)),
        )
    )

    assert module.get_input_shape() == (SymTensor([y]), SymTensor([x]))
    point = torch.full((1, 1, 2), 0.25)
    assert torch.isfinite(module(point, point)).all()


@pytest.mark.parametrize("connective", ["and", "or", "implies"])
def test_connective_over_quantifiers_matches_the_algebra(connective):
    """The lowered connective equals its operator function on the two scalars."""
    fuzzy, compiler = _ltn_compiler()
    x, y = ("X",), ("Y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))
    existential = Aggregation("exists", (x, y), (), _eq(x, y))

    combined = compiler.compile(BinaryOp(connective, universal, existential))
    expected = fuzzy.get_operator_fn(connective)(
        compiler.compile(universal)(), compiler.compile(existential)()
    )

    torch.testing.assert_close(combined(), expected)


def test_connective_over_quantifiers_reading_inputs_of_different_shapes():
    """Operands reading a point and a scalar take one tensor each.

    ``Forall x . eq(x, y)`` reads the point ``y`` and ``Exists x . below(x, r)``
    the radius ``r``, which no single tensor can hold together.
    """
    fuzzy, compiler = _ltn_compiler()
    x, y, r = ("X",), ("Y",), ("R",)
    universal = Aggregation("forall", (x,), (), _eq(x, y))
    existential = Aggregation(
        "exists", (x,), (), Atom(with_structure(("below", x, r), "fuzzy"))
    )

    module = compiler.compile(BinaryOp("and", universal, existential))

    assert module.get_input_shape() == (SymTensor([y]), SymTensor([r]))
    point, radius = torch.full((1, 1, 2), 0.25), torch.full((1, 1), 1.5)
    expected = fuzzy.get_operator_fn("and")(
        compiler.compile(universal)(point), compiler.compile(existential)(radius)
    )
    torch.testing.assert_close(module(point, radius), expected)


def test_unary_connective_over_a_quantifier_matches_the_algebra():
    """``Not(Forall …)`` applies the structure's negation to the reduced output."""
    fuzzy, compiler = _ltn_compiler()
    x, y = ("X",), ("Y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))

    negated = compiler.compile(UnaryOp("not", universal))
    expected = fuzzy.get_operator_fn("not")(compiler.compile(universal)())

    torch.testing.assert_close(negated(), expected)


def test_connective_the_structure_does_not_define_is_rejected():
    """An operator missing from the algebra names itself and what is available."""
    _, compiler = _ltn_compiler()
    x, y = ("X",), ("Y",)
    universal = Aggregation("forall", (x, y), (), _eq(x, y))

    with pytest.raises(NotImplementedError, match="has no operator 'xor'"):
        compiler.compile(BinaryOp("xor", universal, universal))


# --- Aggregation builders -----------------------------------------------------


def _weighted_mean(truth, weight):
    """The mean of ``truth`` over the assignments, each weighted by ``weight``."""
    return (truth * weight).sum(dim=1) / weight.sum(dim=1)


def test_an_aggregator_receives_the_body_and_then_each_param():
    """It gets the body's values and the params', stacked over the domain."""
    _, compiler = _ltn_compiler()
    x, y, r = ("X",), ("Y",), ("R",)
    below = Atom(with_structure(("below", x, r), "fuzzy"))

    module = compiler.compile(Aggregation("weighted", (x,), (below,), _eq(x, y)))

    assert module.get_input_shape() == (SymTensor([y]), SymTensor([r]))
    point, radius = torch.full((1, 1, 2), 0.25), torch.full((1, 1), 1.5)
    points = compiler.variables[x].as_tensor()
    truth = torch.exp(-torch.norm(points - point[0], dim=1))
    weight = torch.sigmoid(1.5 - torch.norm(points, dim=1))
    expected = (truth * weight).sum() / weight.sum()
    torch.testing.assert_close(module(point, radius), expected.reshape(1, 1))


def test_a_reduction_reads_one_formula_in_several_places():
    """A param equal to the body is one column, which the aggregator receives twice."""
    _, compiler = _ltn_compiler()
    x, y = ("X",), ("Y",)

    module = compiler.compile(Aggregation("weighted", (x,), (_eq(x, y),), _eq(x, y)))

    point = torch.full((1, 1, 2), 0.25)
    truth = torch.exp(-torch.norm(compiler.variables[x].as_tensor() - point[0], dim=1))
    expected = (truth * truth).sum() / truth.sum()
    torch.testing.assert_close(module(point), expected.reshape(1, 1))


def _recording_forall():
    """The LTN compiler, whose ``forall`` builder records the nodes of each call."""
    calls = []

    def recording(nodes, lowering):
        calls.append([node.binders for node in nodes])
        return reduction(nodes, lowering)

    _, compiler = _ltn_compiler(forall=recording)
    return compiler, calls


def test_a_builder_gets_every_aggregation_of_its_operation_in_one_call():
    """The aggregations of one operation in a level reach their builder together."""
    compiler, calls = _recording_forall()
    x, y, point = ("X",), ("Y",), ("P",)
    first = Aggregation("forall", (x,), (), _eq(x, point))
    second = Aggregation("forall", (y,), (), _eq(y, point))

    compiler.compile(BinaryOp("and", first, second))

    assert calls == [[(x,), (y,)]]


def test_aggregations_binding_what_another_reads_free_reach_their_builder_apart():
    """``Y`` is free in the first and bound by the second: two variables, two calls."""
    compiler, calls = _recording_forall()
    x, y = ("X",), ("Y",)
    first = Aggregation("forall", (x,), (), _eq(x, y))
    second = Aggregation("forall", (y,), (), _eq(x, y))

    compiler.compile(BinaryOp("and", first, second))

    assert calls == [[(x,)], [(y,)]]


def _probabilities(*atoms: str) -> set:
    """The symbols of ``atoms`` in probability."""
    return set(get_all_symbols(SymTensor([f"{atom} _ probability" for atom in atoms])))


def test_an_atom_free_in_one_aggregation_is_not_read_at_another_s_binder():
    """``r(X)`` is free in the first sum and bound by the second."""
    free = parse_formula("sum(Y): (s(Y)_probability times r(X)_probability)")
    bound = parse_formula("sum(X): r(X)_probability")

    module = Compiler().compile(free, bound)

    assert set(get_all_symbols(module.get_input_shape())) == _probabilities(
        "s(false)", "s(true)", "r(X)", "r(false)", "r(true)"
    )


def test_an_aggregation_reached_where_its_variable_is_bound_and_free_is_built_for_each():
    """Inside ``sum(X)`` the inner sum reads ``r(X)`` at ``X``'s values; outside, as given."""
    inner = "(sum(Y): (s(Y)_probability times r(X)_probability))"
    module = Compiler().compile(parse_formula(f"{inner} plus (sum(X): {inner})"))
    names = ["s(false)", "s(true)", "r(X)", "r(false)", "r(true)"]
    module = reshape(
        module, input=SymTensor([f"{name} _ probability" for name in names])
    )

    s_false, s_true, r_x, r_false, r_true = 0.1, 0.2, 0.3, 0.4, 0.5
    expected = (s_false + s_true) * r_x + (s_false + s_true) * (r_false + r_true)
    torch.testing.assert_close(
        module(torch.tensor([[s_false, s_true, r_x, r_false, r_true]])),
        torch.tensor([[expected]]),
    )


def _identity(evaluations):
    """A builder whose aggregations are their bodies, all computed by one module.

    The module appends to ``evaluations`` each time it runs.
    """

    def build(nodes, lowering):
        bodies = lowering.lower(*(node.child for node in nodes))

        def forward(*inputs):
            evaluations.append(None)
            return bodies(*inputs)

        module = WrappedModule(
            forward, bodies.get_input_shape(), bodies.get_output_shape()
        )
        return [(module, i) for i in range(len(nodes))]

    return build


def test_aggregations_reported_on_one_module_evaluate_it_once():
    """Two aggregations whose builder computes both in one module run it once.

    The module names its columns for the bodies, which are also the compiled
    module's inputs; the walk names each aggregation's column itself, so the
    two never meet.
    """
    evaluations = []
    compiler = Compiler(aggregation_builders={"sum": _identity(evaluations)})
    a, b = (Atom(with_structure((name,), "probability")) for name in "ab")

    module = compiler.compile(
        BinaryOp("times", Aggregation("sum", (), (), a), Aggregation("sum", (), (), b))
    )
    assert module.get_input_shape() == (SymTensor([a.atom]), SymTensor([b.atom]))
    output = module(torch.tensor([[0.5]]), torch.tensor([[0.4]]))

    torch.testing.assert_close(output, torch.tensor([[0.2]]))
    assert len(evaluations) == 1


def test_aggregations_over_one_formula_in_two_algebras_name_two_columns():
    """An operand outside the aggregation's algebra keeps its label in the name.

    Both aggregations are ``real``, and their params are the atom ``a`` in two
    algebras, so a name holding the params bare would be one column for two
    values.
    """

    def body(nodes, lowering):
        return [(lowering.lower(node.child), 0) for node in nodes]

    compiler = Compiler(aggregation_builders={"sum": body})
    boolean, probability = (
        Aggregation(
            "sum",
            (),
            (Atom(with_structure(("a",), structure)),),
            Atom(with_structure(("v",), "real")),
        )
        for structure in ("boolean", "probability")
    )

    module = Lowering(compiler).lower(boolean, probability)

    columns = list(get_all_symbols(module.get_output_shape()))
    assert columns[0] != columns[1]
    assert sole_structure(module.get_output_shape()) == "real"


def test_an_aggregation_is_named_after_operands_its_builder_never_lowered():
    """A builder that lowers nothing still gets a column named for what it computes."""

    def one(nodes, lowering):
        module = WrappedModule(
            lambda x: torch.ones(x.shape[0], 1),
            SymTensor([]),
            SymTensor([with_structure(("one",), "boolean")]),
        )
        return [(module, 0) for _ in nodes]

    inner = Aggregation("exists", (BURGLARY,), (), Atom(BURGLARY_BOOL))
    compiler = Compiler(aggregation_builders={"forall": one})

    module = Lowering(compiler).lower(Aggregation("forall", (), (), inner))

    torch.testing.assert_close(module(), torch.tensor([[1.0]]))
    (column,) = get_all_symbols(module.get_output_shape())
    assert without_structure(column)[2][0] == "exists"


def test_naming_an_operand_its_builder_never_lowered_builds_nothing():
    """A column is named from its formula alone, so naming runs no builder."""
    calls = []

    def one(nodes, lowering):
        module = WrappedModule(
            lambda x: torch.ones(x.shape[0], 1),
            SymTensor([]),
            SymTensor([with_structure(("one",), "boolean")]),
        )
        return [(module, 0) for _ in nodes]

    def recorded(nodes, lowering):
        calls.append(nodes)
        return _identity([])(nodes, lowering)

    compiler = Compiler(aggregation_builders={"forall": one, "exists": recorded})
    inner = Aggregation("exists", (), (), Atom(BURGLARY_BOOL))

    compiler.compile(Aggregation("forall", (), (), UnaryOp("not", inner)))

    assert calls == []


def test_a_builder_reports_a_column_for_every_aggregation():
    compiler = Compiler(aggregation_builders={"exists": lambda nodes, lowering: []})

    with pytest.raises(ValueError, match="reported 0 columns for 1 aggregations"):
        compiler.compile(Aggregation("exists", (), (), Atom(BURGLARY_BOOL)))


def test_a_builder_reports_a_column_its_module_computes():
    def past_the_end(nodes, lowering):
        return [(lowering.lower(node.child), 1) for node in nodes]

    compiler = Compiler(aggregation_builders={"exists": past_the_end})

    with pytest.raises(ValueError, match="reported column 1 of a module computing 1"):
        compiler.compile(Aggregation("exists", (), (), Atom(BURGLARY_BOOL)))


# --- Inside a formula, every value names its algebra --------------------------


@pytest.mark.parametrize(
    "column",
    [("q",), with_structure(("q",), "real")],
    ids=["no algebra", "another algebra"],
)
def test_an_aggregation_builder_computing_outside_its_formulas_algebra_is_refused(
    column,
):
    """An aggregation's algebra is its formula's; a builder only computes it.

    An ``exists`` over a boolean body is boolean, so a builder reporting a value
    labelled otherwise, or labelled with nothing, has computed something else.
    """

    def outside(nodes, lowering):
        module = WrappedModule(lambda x: x, SymTensor([]), SymTensor([column]))
        return [(module, 0) for _ in nodes]

    compiler = Compiler(aggregation_builders={"exists": outside})
    exists = Aggregation("exists", (BURGLARY,), (), Atom(BURGLARY_BOOL))

    with pytest.raises(ValueError, match="makes it a value of 'boolean'"):
        compiler.compile(UnaryOp("not", exists))


def test_an_aggregation_its_formula_gives_no_algebra_is_refused():
    """A body of no algebra gives its aggregation none, and no column is guessed.

    Construction refuses such an atom, so it reaches the walk only from a
    builder lowering a formula of its own.
    """
    lowering = Lowering(Compiler())

    with pytest.raises(NotImplementedError, match="is a value of no algebra"):
        lowering.lower(Aggregation("sum", (), (), Atom(("bare",))))


def _probability_of_logit():
    """The ``real``-valued leaf ``logit``, cast into probability."""
    return Transformation("probability", Atom(with_structure(("logit",), "real")))


def _sigmoid(shape):
    """The ``real -> probability`` cast a caller registers: ``torch.sigmoid``."""
    (source,) = shape
    target = with_structure(without_structure(source), "probability")
    return WrappedModule(torch.sigmoid, shape, SymTensor([target]))


def test_cast_of_a_declared_real_child_applies_the_registered_builder():
    """Labelling a value ``real`` selects the ``(real, probability)`` builder."""
    compiler = Compiler(
        structures={"real": REAL},
        transformation_builders={("real", "probability"): _sigmoid},
    )

    cast = compiler.compile(_probability_of_logit())

    x = torch.tensor([[2.0]])
    torch.testing.assert_close(cast(x), torch.sigmoid(x))
    assert sole_structure(cast.get_output_shape()) == "probability"


def test_cast_without_a_registered_builder_names_what_is_available():
    """A labelled value with no ``(from, to)`` builder fails by name, not KeyError."""
    compiler = Compiler(structures={"real": REAL})

    with pytest.raises(
        NotImplementedError, match="no cast from 'real' to 'probability'"
    ):
        compiler.compile(_probability_of_logit())


# --- Compiling several formulas at once ---


def _probability_product(*names: str) -> BinaryOp:
    """``p(a) times p(b)`` over probability atoms."""
    first, second = (Atom(("_", ("p", (name,)), ("probability",))) for name in names)
    return BinaryOp("times", first, second)


def test_compile_gives_one_column_per_formula():
    """Each root is an output, in the order given, and means what it did alone."""
    shared, other = _probability_product("x", "y"), _probability_product("x", "z")

    together = Compiler().compile(shared, other)
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
    module = Compiler().compile(
        _probability_product("x", "y"), _probability_product("x", "z")
    )

    inside = list(module.modules())[1:]
    compiled = [m for m in inside if isinstance(m, WrappedModule)]
    assert len(compiled) == 1
    # p(x) feeds both roots, so it is asked for once.
    assert len(list(get_all_symbols(module.get_input_shape()))) == 3


def test_an_unnamed_formula_is_named_by_its_position():
    """The ``i``-th formula given without a name computes ``@i``, in its algebra."""
    module = Compiler().compile(
        _probability_product("x", "y"), _probability_product("x", "z")
    )

    assert module.get_output_shape() == SymTensor(
        ["@0 _ probability", "@1 _ probability"]
    )


def test_a_named_formula_computes_its_name_in_its_algebra():
    module = Compiler().compile({"both": _probability_product("x", "y")})

    assert module.get_output_shape() == SymTensor(["both _ probability"])
    torch.testing.assert_close(
        module(torch.tensor([[0.5, 0.4]])), torch.tensor([[0.2]])
    )


def test_a_name_labelled_with_another_algebra_is_refused():
    with pytest.raises(ValueError, match="labelled boolean"):
        Compiler().compile({"both _ boolean": _probability_product("x", "y")})


def test_compile_refuses_two_formulas_of_one_name():
    """``both`` and ``both _ probability`` name one column."""
    with pytest.raises(ValueError, match=r"named \(both _ probability\)"):
        Compiler().compile(
            {
                "both": _probability_product("x", "y"),
                "both _ probability": _probability_product("x", "z"),
            }
        )


def test_compile_refuses_a_mapping_beside_formulas():
    formula = _probability_product("x", "y")

    with pytest.raises(TypeError, match="not both"):
        Compiler().compile(formula, {"both": formula})


def test_compile_composes_roots_that_are_not_co_resident():
    """Roots in different algebras cannot share a circuit, and still compose."""
    probability = _probability_product("x", "y")
    boolean = BinaryOp(
        "and",
        Atom(("_", ("=", ("B",), ("true",)), ("boolean",))),
        Atom(("_", ("=", ("E",), ("true",)), ("boolean",))),
    )

    module = Compiler().compile(probability, boolean)

    assert len(list(get_all_symbols(module.get_output_shape()))) == 2


def test_one_formula_under_two_names_is_two_columns_computed_once():
    """Equal roots compile once, and each name reads the one column."""
    formula = _probability_product("x", "y")

    module = Compiler().compile({"first": formula, "second": formula})

    assert len(list(get_all_symbols(module.get_input_shape()))) == 2
    torch.testing.assert_close(
        module(torch.tensor([[0.5, 0.4]])), torch.tensor([[0.2, 0.2]])
    )


def test_compile_requires_a_formula():
    """There is no module with no columns."""
    with pytest.raises(ValueError, match="At least one formula"):
        Compiler().compile()


# --- Atoms ------------------------------------------------------------------


def test_an_unlabelled_atom_is_refused():
    """An atom carries the structure it lives in; a bare symbol is not one.

    The same refusal :class:`~deeplog.formula.circuit_factory.CircuitFactory`
    makes during construction, so the two halves agree on what an atom is.
    """
    with pytest.raises(ValueError, match="Invalid atom"):
        Lowering(Compiler()).lower(Atom(("plain_atom",)))


_P = "p(B,0.8)_probability times p(E,0.3)_probability"
_SAVED = [
    "sum(C): (=(C,red)_boolean)_probability",
    f"expectation(B, E; {_P}): =(B,true)_boolean or =(E,true)_boolean",
    f"(expectation(B, E; {_P}): =(B,true)_boolean and =(E,true)_boolean) divide "
    f"(expectation(E; p(E,0.3)_probability): =(E,true)_boolean)",
]


@pytest.mark.parametrize("text", _SAVED)
@pytest.mark.parametrize("builder", ["counted", "enumerated", "sampled"])
def test_a_compiled_module_saves_and_loads(text, builder, tmp_path):
    """A compiled module keeps no compiler and no closure, so it pickles."""
    from deeplog import enumeration
    from deeplog import sampling

    builders = {
        "counted": {},
        "enumerated": {"expectation": enumeration},
        "sampled": {"expectation": sampling(10)},
    }[builder]
    compiler = Compiler(
        variables={("C",): Domain.of(["red", "blue"])}, aggregation_builders=builders
    )
    module = compiler.compile(parse_formula(text))

    torch.save(module, tmp_path / "module.pt")
    loaded = torch.load(tmp_path / "module.pt", weights_only=False)

    torch.manual_seed(0)
    expected = module()
    torch.manual_seed(0)
    torch.testing.assert_close(loaded(), expected)


def test_a_registered_algebra_loads_as_itself(tmp_path):
    torch.save(PROBABILITY, tmp_path / "algebra.pt")

    assert torch.load(tmp_path / "algebra.pt", weights_only=False) is PROBABILITY


def test_an_aggregation_inside_another_of_its_level_is_built_once():
    """``sum(Y)`` is an operand of the product and inside the outer sum's body."""
    built = []

    def counting(nodes, lowering):
        built.extend(str(node) for node in nodes)
        return reduction(nodes, lowering)

    inner = "sum(Y): p(Y,0.5)_probability"
    formula = parse_formula(
        f"(sum(X): ({inner}) times p(X,0.5)_probability) times ({inner})"
    )

    module = Compiler(aggregation_builders={"sum": counting}).compile(formula)

    assert [text.startswith("sum(Y)") for text in built].count(True) == 1
    torch.testing.assert_close(module(), torch.tensor([[1.0]]))


def test_an_aggregation_two_formulas_write_is_built_once():
    """Each parse is its own DAG; compiled together, their equal sums are one."""
    built = []

    def counting(nodes, lowering):
        built.extend(str(node) for node in nodes)
        return reduction(nodes, lowering)

    inner = "sum(Y): p(Y,0.5)_probability"
    module = Compiler(aggregation_builders={"sum": counting}).compile(
        parse_formula(inner), parse_formula(f"({inner}) times q_probability")
    )

    assert len(built) == 1
    torch.testing.assert_close(
        module(torch.tensor([[0.5]])), torch.tensor([[1.0, 0.5]])
    )


# --- An aggregation is its algebra's aggregator -------------------------------


def test_a_sum_over_a_boolean_body_is_refused():
    """``boolean`` has no ``sum``; a count casts the body into ``real`` first."""
    summed = Aggregation("sum", (BURGLARY,), (), Atom(BURGLARY_BOOL))

    with pytest.raises(NotImplementedError, match="'boolean' has no aggregator 'sum'"):
        Compiler().compile(summed)


def test_a_refused_aggregation_names_the_algebras_to_cast_its_body_into():
    """Those with the aggregator and a cast from the body's algebra."""
    formula = parse_formula("sum(B, E): =(B,true)_boolean or =(E,true)_boolean")

    with pytest.raises(NotImplementedError) as refused:
        Compiler().compile(formula)

    assert "sum(B, E): (φ)_probability or sum(B, E): (φ)_real" in str(refused.value)


def test_a_refused_aggregation_no_cast_can_reach_says_where_aggregators_live():
    formula = parse_formula("max(X): (=(X,true)_boolean)_probability")

    with pytest.raises(NotImplementedError, match="declares its aggregators"):
        Compiler().compile(formula)


def test_a_builder_cannot_give_an_operation_its_algebra_lacks_a_meaning():
    """A builder computes an aggregator, so it is not called for one that is missing."""
    calls = []

    def recorded(nodes, lowering):
        calls.append(nodes)
        return reduction(nodes, lowering)

    compiler = Compiler(aggregation_builders={"sum": recorded})
    summed = Aggregation("sum", (BURGLARY,), (), Atom(BURGLARY_BOOL))

    with pytest.raises(NotImplementedError, match="'boolean' has no aggregator 'sum'"):
        compiler.compile(summed)
    assert calls == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("exists(X1): =(X0,true)_boolean", [[0.0], [1.0]]),
        ("forall(X1): =(X0,true)_boolean", [[0.0], [1.0]]),
        ("sum(X1): (=(X0,true)_boolean)_real", [[0.0], [2.0]]),
    ],
)
def test_a_binder_its_body_does_not_read_repeats_the_body(text, expected):
    """The body has one value at both of ``X1``'s: ``sum`` counts it twice."""
    module = parse_formula_to_module(text)

    torch.testing.assert_close(
        module(torch.tensor([[0.0], [1.0]])), torch.tensor(expected)
    )


def test_an_aggregation_binding_a_read_and_an_unread_binder_compiles():
    module = parse_formula_to_module("sum(X, Y): (=(Y,true)_boolean)_real")

    torch.testing.assert_close(module(), torch.tensor([[2.0]]))


@pytest.mark.parametrize(("operation", "expected"), [("exists", 1.0), ("forall", 0.0)])
def test_boolean_quantifies_with_exists_and_forall(operation, expected):
    """``Burglary`` is true at one of its two values."""
    quantified = Aggregation(operation, (BURGLARY,), (), Atom(BURGLARY_BOOL))

    result = Compiler().compile(quantified)()

    torch.testing.assert_close(result, torch.tensor([[expected]]))


def test_a_sum_in_logprobability_is_a_sum_of_probabilities():
    """The two values of ``Burglary`` have probabilities 0.3 and 0.7, so log 1."""
    formula = parse_formula("sum(Burglary): logp(Burglary,0.3)_logprobability")

    result = Compiler().compile(formula)()

    torch.testing.assert_close(result, torch.tensor([[0.0]]))
