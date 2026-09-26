#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the MV-SDD knowledge-compilation pass (the categorical compiler).

Annotated-disjunction branches are not independent variables, and only this pass
makes their mutual exclusivity structural. Every case grounds a plain program,
declares its multi-valued variables and labels alongside it, then
knowledge-compiles the boolean proofs and counts them in probability.
"""

import importlib
import sys

import pytest
import torch

from deeplog import OPEN
from deeplog import CircuitFactory
from deeplog import Domain
from deeplog import Variable
from deeplog import to_module
from deeplog import with_structure
from deeplog.formula.strategies import transform_expectation_to_probability
from deeplog.grounding import ProofBuilder
from deeplog.grounding import SimpleGrounder
from deeplog.grounding import str_to_rules
from deeplog.grounding.prolog import is_query


pymvsdd = pytest.importorskip("pymvsdd")


def _ground(code, open_predicates):
    """The proof of every query answer in ``code``, all in one boolean circuit."""
    program = tuple(str_to_rules(code))
    builder = ProofBuilder(CircuitFactory())
    grounder = SimpleGrounder()
    proofs = {}
    for query in filter(is_query, program):
        proofs.update(grounder.ground(program, query[2], builder, open_predicates))
    return proofs


def _variable(name, occurrence, values):
    """A multi-valued variable over ``values``, occurring in ``occurrence``."""
    return {Variable(name, Domain.of(values)): (occurrence,)}


def _compile_program(code, open_predicates, labels, variables):
    """Knowledge-compile ``code``'s proofs and count them in probability.

    ``variables`` name the boolean atoms as their domain values, which is exactly
    what the pass consumes, and ``labels`` become the leaf mapping the transform
    applies — a numeric label folding to a constant node.
    """
    answers, nodes = zip(*_ground(code, open_predicates).items(), strict=True)
    counted = transform_expectation_to_probability(
        *nodes,
        leaf_mapping=lambda leaf: with_structure(labels.get(leaf, leaf), "probability"),
        variables=variables,
    )
    return to_module(*counted, names=answers)


_ABC = {("a", 0), ("b", 0), ("c", 0)}
_CHOICE = _variable(("choice",), OPEN, ["a", "b", "c"])


def test_constant_ad_bakes_branches_into_circuit():
    # Every AD branch's label is a numeric constant — the compiler
    # bakes those values into the AC, so the resulting torch module
    # takes NO runtime inputs. The output is exactly each branch's
    # declared probability.
    mod = _compile_program(
        "a.\nb.\nc.\n?- a.\n?- b.\n?- c.",
        _ABC,
        {("a",): ("0.3",), ("b",): ("0.5",), ("c",): ("0.2",)},
        _CHOICE,
    )
    # No input slots — declared constants 0.3/0.5/0.2 live inside
    # the AC.
    assert list(mod.get_input_shape()) == []
    # Root outputs are labelled with the algebra they were compiled in — the
    # override here, since a boolean circuit compiled with the probability
    # semiring produces probabilities.
    assert list(mod.get_output_shape()) == [
        with_structure(sym, "probability") for sym in (("a",), ("b",), ("c",))
    ]
    # WrappedModule with vmap=True needs at least a batch axis; pass
    # an empty trailing dim.
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.3, 0.5, 0.2]]), atol=1e-5)


def test_the_multi_valued_compiler_names_the_extra_that_installs_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymvsdd", None)

    with pytest.raises(ImportError, match=r"pydeeplog\[mvsdd\]"):
        _compile_program(
            "a.\nb.\nc.\n?- a.\n?- b.\n?- c.",
            _ABC,
            {("a",): ("0.3",), ("b",): ("0.5",), ("c",): ("0.2",)},
            _CHOICE,
        )


def test_constant_ad_in_rule_body_outputs_marginal():
    # Rule body composes an AD branch into a derived predicate. With
    # both `heads` and `tails` carrying numeric labels, no runtime
    # input is needed; `?- win.`'s marginal is just P(heads) = 0.4.
    mod = _compile_program(
        "heads.\ntails.\nwin :- heads.\n?- win.",
        {("heads", 0), ("tails", 0)},
        {("heads",): ("0.4",), ("tails",): ("0.6",)},
        _variable(("coin",), OPEN, ["heads", "tails"]),
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.4]]), atol=1e-5)


def test_unresolved_labels_remain_input_slots():
    # The compiler bakes a label into the AC iff it's a numeric
    # constant. Non-numeric labels (e.g. ``net(0)``) have no value
    # at compile time, so the AC keeps them as free leaves whose
    # truth indicator the caller fills at evaluation time. In real
    # DeepProbLog use this is the slot a neural network's per-class
    # softmax flows into — exercised end-to-end by the MNIST tests
    # below; here we just pin down the input-slot contract.
    mod = _compile_program(
        "a.\nb.\nc.\n?- a.\n?- b.\n?- c.",
        _ABC,
        {(atom,): ("net", (str(i),)) for i, atom in enumerate("abc")},
        _CHOICE,
    )
    # Each free slot is named by its AD-branch leaf symbol, labelled with the
    # algebra the caller feeds it: under the probability override these slots
    # take softmax outputs, not boolean truth values.
    # Each free slot is named by the *label* the branch carries, which is what
    # fills it: the network predicate ``net(i)`` produces that column. A branch
    # whose label is itself the branch atom (MNIST's ``digit(i,d)``) is named by
    # it either way; the two only differ when a program names them separately.
    assert list(mod.get_input_shape()) == [
        with_structure(("net", (str(i),)), "probability") for i in range(3)
    ]
    # The AC isn't computable without runtime inputs — there's no
    # numeric value associated with `net(0)`/`net(1)`/`net(2)`, so the
    # caller has to feed truth indicators (or neural softmax outputs)
    # for each (cat_id, value) slot.
    out = mod(torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]]))
    out = mod(torch.tensor([[0.3, 0.5, 0.2]]))
    assert torch.allclose(out, torch.tensor([[0.3, 0.5, 0.2]]), atol=1e-5)


def test_independent_constant_variables_conjoin():
    # Two independent constant-labeled ADs; query their conjunction.
    # With both ADs baked in and the probability semiring (AND=product,
    # OR=sum), the output is the joint probability of the two chosen
    # branches: P(a) · P(c).
    mod = _compile_program(
        "a.\nb.\nc.\nd.\nwin :- a, c.\n?- win.",
        _ABC | {("d", 0)},
        {("a",): ("0.3",), ("b",): ("0.7",), ("c",): ("0.4",), ("d",): ("0.6",)},
        _variable(("ab",), OPEN, ["a", "b"]) | _variable(("cd",), OPEN, ["c", "d"]),
    )
    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.3 * 0.4]]), atol=1e-5)


#: The classic n=2 MNIST addition program, one digit fact per image and value.
_MNIST_ADDITION = (
    "".join(f"digit({img},{n}).\n" for img in ("i1", "i2") for n in range(10))
    + "addition(I1,I2,S) :- between(0,9,N1), between(0,9,N2), "
    "digit(I1,N1), digit(I2,N2), is(S,+(N1,N2)).\n"
    "?- addition(i1,i2,S).\n"
)

#: One digit variable per image, occurring in ``digit(img,_)``.
_DIGITS = _variable(("digit", ("i1",)), ("digit", ("i1",), OPEN), range(10)) | (
    _variable(("digit", ("i2",)), ("digit", ("i2",), OPEN), range(10))
)


def _addition_inputs(mod, p1, p2):
    """Build the 20-slot input tensor matching mod's declared input order."""
    inputs = [0.0] * len(list(mod.get_input_shape()))
    for i, sym in enumerate(mod.get_input_shape()):
        # Each slot is named by its AD-branch leaf symbol,
        # ("_", ("digit", (img,), (n,)), ("boolean",)).
        _functor, (img,), (n,) = sym[1]
        inputs[i] = (p1 if img == "i1" else p2)[int(n)]
    return torch.tensor([inputs])


def test_mnist_addition_n2_one_hot():
    # n=2 MNIST addition. With one-hot inputs at digits i1=3, i2=5, the
    # AC should output 1.0 only at addition(i1,i2,8) and 0.0 elsewhere.
    answers, nodes = zip(*_ground(_MNIST_ADDITION, {("digit", 2)}).items(), strict=True)
    # 19 possible sums (0..18).
    assert len(answers) == 19

    mod = to_module(
        *transform_expectation_to_probability(*nodes, variables=_DIGITS),
        names=answers,
    )

    i1_gt, i2_gt = 3, 5
    p1 = [1.0 if d == i1_gt else 0.0 for d in range(10)]
    p2 = [1.0 if d == i2_gt else 0.0 for d in range(10)]
    out = mod(_addition_inputs(mod, p1, p2))
    out_by_sum = {
        int(name[3][0]): float(v)
        for name, v in zip(answers, out[0].tolist(), strict=True)
    }
    for s in range(19):
        expected = 1.0 if s == i1_gt + i2_gt else 0.0
        assert pytest.approx(out_by_sum[s], abs=1e-5) == expected


def test_mnist_addition_n2_soft_distribution():
    # Same program, with peaked-but-not-saturated softmax inputs. The
    # addition's marginal must equal the convolution
    # P(sum=s) = Σ_{n1+n2=s} P_classifier_1(n1) · P_classifier_2(n2).
    answers, nodes = zip(*_ground(_MNIST_ADDITION, {("digit", 2)}).items(), strict=True)

    mod = to_module(
        *transform_expectation_to_probability(*nodes, variables=_DIGITS),
        names=answers,
    )

    # Two simulated softmax outputs.
    p1 = [0.9 if d == 3 else 0.1 / 9 for d in range(10)]
    p2 = [0.9 if d == 5 else 0.1 / 9 for d in range(10)]
    out = mod(_addition_inputs(mod, p1, p2))
    out_by_sum = {
        int(name[3][0]): float(v)
        for name, v in zip(answers, out[0].tolist(), strict=True)
    }

    # Expected: classic discrete convolution.
    for s in range(19):
        expected = sum(p1[n1] * p2[s - n1] for n1 in range(10) if 0 <= s - n1 <= 9)
        assert pytest.approx(out_by_sum[s], abs=1e-5) == expected
    # And the marginal sums to (sum p1) * (sum p2) = 1.
    assert pytest.approx(sum(out_by_sum.values()), abs=1e-5) == 1.0


def test_declaring_the_variable_is_what_makes_the_branches_exclusive():
    # The declared variable selects the multi-valued compiler, which bakes mutual
    # exclusivity into the circuit, so a∧b of two values of one variable is
    # unsatisfiable. Compiled without it the atoms are independent variables and
    # the conjunction is satisfiable.
    proofs = _ground("a.\nb.\nwin :- a, b.\n?- win.", {("a", 0), ("b", 0)})
    answers, nodes = zip(*proofs.items(), strict=True)
    variables = _variable(("ab",), OPEN, ["a", "b"])

    mod_cats = to_module(
        *transform_expectation_to_probability(*nodes, variables=variables),
        names=answers,
    )
    mod_no_cats = to_module(
        *transform_expectation_to_probability(*nodes), names=answers
    )

    # With the branches declared exclusive the conjunction is unsatisfiable, so
    # the compiled circuit is the constant false and has no inputs left at all.
    assert list(mod_cats.get_input_shape()) == []
    assert mod_cats().item() == pytest.approx(0.0)

    # Without them, two independent variables that can both be true.
    leaves = [("_", ("a",), ("probability",)), ("_", ("b",), ("probability",))]
    assert sorted(mod_no_cats.get_input_shape()) == leaves
    assert mod_no_cats(torch.tensor([[1.0, 1.0]])).item() == pytest.approx(1.0)


def test_a_value_no_proof_reaches_is_read_back_as_its_own_atom():
    """``not(a), not(b)`` holds exactly when ``choice`` takes the unreached ``c``.

    ``c`` has an atom although no proof reaches it, so the compiled circuit reads
    the residual as that atom, whose label fills it.
    """
    mod = _compile_program(
        "a.\nb.\nc.\nq :- not(a), not(b).\n?- q.\n?- a.",
        _ABC,
        {(atom,): ("net", (str(i),)) for i, atom in enumerate("abc")},
        _CHOICE,
    )

    net = {with_structure(("net", (str(i),)), "probability"): i for i in (0, 2)}
    assert sorted(mod.get_input_shape()) == sorted(net)
    weights = [0.2, 0.3, 0.5]
    x = torch.tensor([[weights[net[symbol]] for symbol in mod.get_input_shape()]])
    assert torch.allclose(mod(x), torch.tensor([[0.5, 0.2]]), atol=1e-5)


def test_a_fact_outside_every_variable_is_negated_as_itself():
    """A leaf no variable declares is two-valued, so its ``false`` is its complement."""
    mod = _compile_program(
        "a.\nb.\nrain.\nq :- a, not(rain).\n?- q.\n?- b.",
        {("a", 0), ("b", 0), ("rain", 0)},
        {("a",): ("0.4",), ("b",): ("0.6",), ("rain",): ("0.3",)},
        _variable(("ab",), OPEN, ["a", "b"]),
    )

    assert list(mod.get_input_shape()) == []
    out = mod(torch.zeros((1, 0)))
    assert torch.allclose(out, torch.tensor([[0.4 * 0.7, 0.6]]), atol=1e-5)


# --- The compiler follows from the variables, not from a caller's choice ------


_DIGIT_I1 = _variable(("digit", ("i1",)), ("digit", ("i1",), OPEN), range(3))


def _compiler_used(monkeypatch, circuit, roots, variables):
    """Which knowledge compiler ``knowledge_compile`` dispatches to."""
    from deeplog.circuit.knowledge_compile import dispatch as dispatch_module

    used: list[str] = []
    for name in ("sdd", "mvsdd"):
        module = importlib.import_module(f"deeplog.circuit.knowledge_compile.{name}")
        original = getattr(module, f"compile_{name}")
        monkeypatch.setattr(
            module,
            f"compile_{name}",
            lambda *args, _n=name, _f=original, **kw: (
                used.append(_n),
                _f(*args, **kw),
            )[1],
        )
    dispatch_module.knowledge_compile(circuit, roots, variables=variables)
    return used


def test_a_variable_with_one_asserted_value_compiles_as_a_plain_sdd_variable(
    monkeypatch,
):
    """Dispatch is derived from the variables, not chosen by the caller.

    A plain SDD's variables are two-valued, so a variable at most one of whose
    values the formula asserts already *is* one. Only a variable with two or
    more asserted values needs the multi-valued compiler, or two of its atoms
    could hold at once.
    """
    nodes = list(
        _ground(
            "digit(i1,0).\ndigit(i1,1).\ndigit(i1,2).\nrain.\n"
            "wet :- rain.\none :- digit(i1,1).\n?- wet.\n?- one.",
            {("digit", 2), ("rain", 0)},
        ).values()
    )
    circuit, roots = nodes[0].circuit, [node.node for node in nodes]

    # Only `digit(i1,1)` is proved, so the variable asserts one value: plain SDD.
    assert _compiler_used(monkeypatch, circuit, roots, _DIGIT_I1) == ["sdd"]
    # Nothing declared at all is the same case, reached the same way.
    assert _compiler_used(monkeypatch, circuit, roots, None) == ["sdd"]


def test_a_variable_with_two_asserted_values_needs_the_multi_valued_compiler(
    monkeypatch,
):
    """Two atoms of one variable can hold at once under a plain SDD; MV-SDD is picked."""
    nodes = list(
        _ground(
            "digit(i1,0).\ndigit(i1,1).\ndigit(i1,2).\n"
            "both :- digit(i1,0), digit(i1,1).\n?- both.",
            {("digit", 2)},
        ).values()
    )
    circuit, roots = nodes[0].circuit, [node.node for node in nodes]

    assert _compiler_used(monkeypatch, circuit, roots, _DIGIT_I1) == ["mvsdd"]


def test_compiling_into_another_algebra_builds_one_circuit():
    """Knowledge compilation emits into the algebra that will count it.

    The rewrite preserves roles, so reading them in the probability semiring is
    what transforming the compiled circuit afterwards would have done — and the
    boolean d-DNNF it would have transformed is never built. ``leaf_mapping``
    renames each atom on the way, as that transform would have, so a numeric
    label lands as a constant node rather than a free input.
    """
    from deeplog.circuit import knowledge_compile as kc_module
    from deeplog.circuit.circuit import Circuit

    source = Circuit("boolean")
    a, b = source.get_leaf_node(("a",)), source.get_leaf_node(("b",))
    root = source.get_operator("or")(a, b)

    built: list[str] = []
    original = kc_module.dispatch.Circuit

    class _Counting(original):
        def __init__(self, structure):
            super().__init__(structure)
            built.append(self.structure.name)

    kc_module.dispatch.Circuit = _Counting
    try:
        counted, node_map = kc_module.knowledge_compile(
            source,
            [root],
            structure="probability",
            leaf_mapping=lambda sym: with_structure(
                ("0.5",) if sym == ("a",) else sym, "probability"
            ),
        )
    finally:
        kc_module.dispatch.Circuit = original

    # One circuit, in the target algebra — no boolean d-DNNF along the way.
    assert built == ["probability"]
    assert counted.structure.name == "probability"
    # ``a``'s label folded to a constant; ``b`` stayed a probability input.
    assert list(counted.reachable_leaves([node_map[root]])) == [
        with_structure(("b",), "probability")
    ]
