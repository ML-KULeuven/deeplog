#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import with_structure
from deeplog.module.aggregation_modules import AggregationModule
from deeplog.shape import sole_structure


a, b, c = ((x,) for x in "abc")


def _sum(*columns: torch.Tensor) -> torch.Tensor:
    return sum(columns).sum(dim=1)


def test_aggregation_module():
    """The inner module is evaluated at every value of ``b`` and ``op`` reduces it."""
    aggregated_module = WrappedModule(
        lambda x: x,
        SymTensor([a, b, c]),
        SymTensor([a, b, c]),
    )
    aggregation_module = AggregationModule(
        aggregated_module,
        [b],
        [torch.tensor([0, 1])],
        "sum",
        op=lambda a, b, c: (a * b + c).sum(dim=1),
    )

    assert aggregation_module.get_input_shape() == SymTensor([a, c])
    assert aggregation_module.get_output_shape() == SymTensor(
        [("sum", ("binders", b), a, b, c)]
    )

    output = aggregation_module(torch.tensor([[0.2, 0.8], [0.3, 0.7]]))

    torch.testing.assert_close(output, torch.tensor([[1.8], [1.7]]))


def test_op_receives_each_column_stacked_over_the_domain():
    """Every column arrives as ``(batch, assignments, 1)``, in column order."""
    received = []

    def op(*columns):
        received.extend(column.clone() for column in columns)
        return columns[0].sum(dim=1)

    child = WrappedModule(lambda x: x.flip(1), SymTensor([a, b]), SymTensor([b, a]))
    AggregationModule(child, [b], [torch.tensor([0, 1, 2])], "sum", op=op)(
        torch.tensor([[5.0]])
    )

    torch.testing.assert_close(received[0], torch.tensor([[[0.0], [1.0], [2.0]]]))
    torch.testing.assert_close(received[1], torch.tensor([[[5.0], [5.0], [5.0]]]))


def _aggregation_over(child: WrappedModule) -> AggregationModule:
    """Aggregate ``child`` over its binder ``b``."""
    return AggregationModule(child, [b], [torch.tensor([0, 1])], "sum", op=_sum)


def test_structure_is_carried_by_the_minted_output_symbol():
    """The reduction is a value of the first column's algebra.

    The tag goes on the *outside* of the new name — the reduction of a
    probability is a probability — rather than staying buried on the child
    symbols it wraps.
    """
    labelled = [with_structure(sym, "probability") for sym in (a, b, c)]
    child = WrappedModule(lambda x: x, SymTensor([a, b, c]), SymTensor(labelled))

    aggregation = _aggregation_over(child)

    assert aggregation.get_output_shape() == SymTensor(
        [with_structure(("sum", ("binders", b), a, b, c), "probability")]
    )
    assert sole_structure(aggregation.get_output_shape()) == "probability"


def test_an_unlabelled_child_yields_an_unlabelled_aggregation():
    """The module layer invents nothing: no label in, no label out.

    Bare outputs are the outside-the-formula-layer case. The lowering
    is what refuses them; ``AggregationModule`` on its own stays usable.
    """
    aggregation = _aggregation_over(
        WrappedModule(lambda x: x, SymTensor([a, b, c]), SymTensor([a, b, c]))
    )

    assert sole_structure(aggregation.get_output_shape()) is None
