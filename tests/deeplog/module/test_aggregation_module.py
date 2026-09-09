#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import AggregationModule
from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import sole_structure
from deeplog import with_structure


a, b, c = ((x,) for x in "abc")


def test_aggregation_module():
    aggregated_module = WrappedModule(
        lambda x: x,
        SymTensor([a, b, c]),
        SymTensor([a, b, c]),
    )
    aggregation_module = AggregationModule(
        aggregated_module, [b], [torch.tensor([0, 1])], "sum"
    )

    assert aggregation_module.get_input_shape() == SymTensor([a, c])
    assert aggregation_module.get_output_shape() == SymTensor(
        [
            ("sum", ("binders", b), a),
            ("sum", ("binders", b), b),
            ("sum", ("binders", b), c),
        ]
    )

    output = aggregation_module(torch.tensor([[0.2, 0.8], [0.3, 0.7]]))
    expected = torch.tensor([[0.4, 1.0, 1.6], [0.6, 1.0, 1.4]])

    torch.testing.assert_close(output, expected)


def _aggregation_over(child: WrappedModule) -> AggregationModule:
    """Aggregate ``child`` over its binder ``b``."""
    return AggregationModule(child, [b], [torch.tensor([0, 1])], "sum")


def test_structure_is_carried_by_the_minted_output_symbols():
    """An aggregation reduces within one algebra, so each minted name keeps it.

    The tag goes on the *outside* of the new name — the reduction of a
    probability is a probability — rather than staying buried on the child
    symbol it wraps.
    """
    labelled = [with_structure(sym, "probability") for sym in (a, b, c)]
    child = WrappedModule(lambda x: x, SymTensor([a, b, c]), SymTensor(labelled))

    aggregation = _aggregation_over(child)

    assert aggregation.get_output_shape() == SymTensor(
        [
            with_structure(("sum", ("binders", b), sym), "probability")
            for sym in (a, b, c)
        ]
    )
    assert sole_structure(aggregation.get_output_shape()) == "probability"


def test_an_unlabelled_child_yields_an_unlabelled_aggregation():
    """The module layer invents nothing: no label in, no label out.

    Bare outputs are the outside-the-formula-layer case. The lowering factory
    is what refuses them; ``AggregationModule`` on its own stays usable.
    """
    aggregation = _aggregation_over(
        WrappedModule(lambda x: x, SymTensor([a, b, c]), SymTensor([a, b, c]))
    )

    assert sole_structure(aggregation.get_output_shape()) is None
