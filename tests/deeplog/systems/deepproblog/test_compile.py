#  Copyright (c) 2024-2026. KU Leuven
"""Tests for compile_to_module."""

import pytest
import torch

from deeplog import Compiler
from deeplog import Domain
from deeplog import Predicate
from deeplog import SymTensor
from deeplog import enumeration
from deeplog import reshape
from deeplog import sampling
from deeplog.grounding.prolog import SimpleGrounder
from deeplog.grounding.prolog import str_to_rules
from deeplog.shape import get_all_symbols
from deeplog.systems.deepproblog import Solver
from deeplog.systems.deepproblog import compile_to_module
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

    # The engine grounds each query to a boolean formula; compile_to_module
    # lifts it to an expectation and the Compiler constructs and lowers it.
    result = engine.get_query_result(program)

    module = compile_to_module(result, Compiler())
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

    result = engine.get_query_result(program)

    module = reshape(
        compile_to_module(result, Compiler()),
        input=SymTensor(
            [("_", (name, ("x1",)), ("probability",)) for name in ("nn1", "nn2")]
        ),
    )

    # Each atom reads its own label: P(c) = P(a) * P(b).
    torch.testing.assert_close(
        module(torch.tensor([[0.8, 0.5]])), torch.tensor([[0.4]])
    )


def test_compile_transforms_a_boolean_formula_to_probability():
    """compile_to_module counts each boolean formula into probability.

    The result is a probability-backed module (the structural guarantee that
    replaces the old per-call ``create_aggregation`` routing and its runtime
    non-probability check).
    """
    from deeplog import Atom
    from deeplog.systems.deepproblog import EngineResult

    result = EngineResult(formulas={("a",): Atom(("_", ("a",), ("boolean",)))})

    module = compile_to_module(result, Compiler())
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
    result = engine.get_query_result(program)
    module = compile_to_module(result, Compiler())

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
    result = engine.get_query_result(program)
    module = compile_to_module(result, Compiler())

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
    result = engine.get_query_result(program)
    module = compile_to_module(result, Compiler())

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

    # Both queries' (structurally identical) proofs stay two distinct outputs.
    result = engine.get_query_result(program)

    assert len(result.formulas) == 2
    module = compile_to_module(result, Compiler())
    assert module is not None

    output_symbols = list(module.get_output_shape())
    assert len(output_symbols) == 2


class _Constant(Predicate):
    """``nn1(X)`` in probability, 0.25 for every ``X``."""

    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of_values(),))

    def resolve_argument(self, symbol, index):
        return 0.0

    def forward_predicate(self, x):
        return torch.full(x.shape[:1], 0.25)


def test_a_label_naming_an_atom_with_a_builder_is_its_value():
    """A label is an atom's value: computed by its builder when there is one."""
    program = tuple(str_to_rules("nn1(x1) :: a(x1).\n?- a(x1)."))
    result = Solver(SimpleGrounder()).get_query_result(program)

    module = compile_to_module(
        result, Compiler(atom_builders={("nn1", 1, "probability"): _Constant})
    )

    assert list(get_all_symbols(module.get_input_shape())) == []
    torch.testing.assert_close(module(), torch.tensor([[0.25]]))


def test_an_unlabelled_atom_weighs_by_its_own_value_in_probability():
    """An atom without a label is a fact whose probability is supplied, as itself."""
    from deeplog import Atom
    from deeplog.systems.deepproblog import EngineResult

    result = EngineResult(formulas={("a",): Atom(("_", ("a",), ("boolean",)))})

    module = compile_to_module(result, Compiler())

    assert list(get_all_symbols(module.get_input_shape())) == [
        ("_", ("a",), ("probability",))
    ]
    torch.testing.assert_close(module(torch.tensor([[0.3]])), torch.tensor([[0.3]]))


def test_a_random_atom_whose_predicate_has_a_builder_is_refused():
    """The program's labels are its random atoms' labelling function; a caller's
    builder for the same predicate would be a second one."""
    program = tuple(str_to_rules("0.5::a(x1).\n?- a(x1)."))
    result = Solver(SimpleGrounder()).get_query_result(program)

    with pytest.raises(ValueError, match="already registered for a/2"):
        compile_to_module(
            result, Compiler(atom_builders={("a", 2, "probability"): _Constant})
        )


_ALARM = """
0.1::burglary. 0.2::earthquake.
0.3::c(1); 0.5::c(2); 0.2::c(3).
alarm :- burglary. alarm :- earthquake, c(2).
?- alarm.
"""


def test_every_value_of_a_disjunction_is_labelled():
    """The proofs reach ``c(2)`` alone, yet its variable ranges over all three."""
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(_ALARM)))

    assert {("c", ("1",)), ("c", ("3",))} <= result.labels.keys()


@pytest.mark.parametrize(
    "builder", [enumeration, sampling(20_000)], ids=["enumeration", "sampling"]
)
def test_a_disjunction_the_proofs_reach_in_part_is_weighed_whole(builder):
    """Enumerating or sampling reads every value's label, reached or not."""
    torch.manual_seed(0)
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(_ALARM)))

    module = compile_to_module(
        result, Compiler(aggregation_builders={"expectation": builder})
    )

    assert module.get_input_shape() == ()
    # 0.1 + 0.9 * 0.2 * 0.5
    torch.testing.assert_close(module(), torch.tensor([[0.19]]), atol=0.01, rtol=0)


@pytest.mark.parametrize(
    "builder",
    [None, enumeration, sampling(40_000)],
    ids=["counted", "enumeration", "sampling"],
)
def test_a_disjunction_whose_labels_sum_below_one_may_choose_no_branch(builder):
    """``0.2::c(r); 0.5::c(g).`` leaves 0.3 to neither, where ``c(r)`` is false."""
    torch.manual_seed(0)
    program = "0.2::c(r); 0.5::c(g). q :- not(c(r)). ?- q."
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(program)))
    builders = {} if builder is None else {"expectation": builder}

    module = compile_to_module(result, Compiler(aggregation_builders=builders))

    torch.testing.assert_close(module(), torch.tensor([[0.8]]), atol=0.01, rtol=0)


def test_a_program_predicate_shadows_a_default_builder():
    """``0.3::p(a).`` extends ``p/1`` to ``p/2``, the default ``p`` builder's key."""
    program = "0.3::p(a). q :- p(a). ?- q."
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(program)))

    torch.testing.assert_close(
        compile_to_module(result, Compiler())(), torch.tensor([[0.3]])
    )


def test_a_compiled_program_saves_and_loads(tmp_path):
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(_ALARM)))
    module = compile_to_module(result, Compiler())

    torch.save(module, tmp_path / "module.pt")

    torch.testing.assert_close(
        torch.load(tmp_path / "module.pt", weights_only=False)(), module()
    )


_CONSTRAINED = """
0.3::a. 0.6::b. 0.2::c(1); 0.5::c(2); 0.3::c(3).
x :- a, c(1). x :- b, c(2). y :- a; b.
:- a, b.
?- x. ?- y. ?- c(2).
"""


@pytest.mark.parametrize(
    "builder", [enumeration, sampling(40_000)], ids=["enumeration", "sampling"]
)
def test_a_constraint_over_several_atoms_is_enumerated_and_sampled(builder):
    """The evidence ``not(a and b)`` negates a conjunction, which Klay could not."""
    torch.manual_seed(0)
    result = Solver(SimpleGrounder()).get_query_result(
        tuple(str_to_rules(_CONSTRAINED))
    )

    counted = compile_to_module(result, Compiler())()
    other = compile_to_module(
        result, Compiler(aggregation_builders={"expectation": builder})
    )()

    torch.testing.assert_close(other, counted, atol=0.02, rtol=0)


def test_a_program_is_counted(monkeypatch):
    """The labelling declares each variable's domain, so counting reads its values."""
    pytest.importorskip("pymvsdd")
    import deeplog.circuit.knowledge_compilation.mvsdd as mvsdd
    import deeplog.circuit.knowledge_compilation.sdd as sdd

    calls = []
    for module, name in [(sdd, "compile_sdd"), (mvsdd, "compile_mvsdd")]:
        original = getattr(module, name)
        monkeypatch.setattr(
            module,
            name,
            lambda *args, original=original, **kwargs: (
                calls.append(1) or original(*args, **kwargs)
            ),
        )
    program = str_to_rules(
        "0.6::burglary.\n0.3::earthquake.\nalarm :- burglary.\n"
        "alarm :- earthquake.\n?- alarm."
    )
    result = Solver(SimpleGrounder()).get_query_result(program)

    module = compile_to_module(result, Compiler())

    assert calls == [1]
    torch.testing.assert_close(module(), torch.tensor([[1 - 0.4 * 0.7]]))
