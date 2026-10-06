#  Copyright (c) 2024-2026. KU Leuven
"""Tests for :class:`~deeplog.module.columnwise.ColumnwiseModule`.

It combines column groups of one module's output with a tensor callable,
evaluating that module once.
"""

import pytest
import torch

from deeplog import SymTensor
from deeplog import with_structure
from deeplog.module.columnwise import ColumnwiseModule

from ._utils import DummyModule


A = ("a",)


def _p(symbol):
    """Label ``symbol`` as probability-valued, the way a lowered module names it."""
    return with_structure(symbol, "probability")


def _operand(inputs, outputs, forward):
    """A DummyModule whose outputs are labelled, as the lowering names them."""
    outputs = [_p(symbol) for symbol in outputs]
    return DummyModule(SymTensor(list(inputs)), SymTensor(list(outputs)), forward)


def test_columnwise_combines_column_groups_of_one_module():
    """Column groups of a single output are combined without re-running it."""
    inner = _operand(
        [A], [("q1",), ("q2",), ("z",)], lambda x: torch.tensor([[0.2, 0.6, 0.8]])
    )

    # Groups are resolved by symbol, so they are spelled the way the inner
    # module names its columns — labelled.
    module = ColumnwiseModule(
        lambda n, d: n / d,
        inner,
        (_p(("q1",)), _p(("q2",))),
        (_p(("z",)),),
        name="divide",
    )

    assert list(module.get_output_shape()) == [_p(("q1",)), _p(("q2",))]
    torch.testing.assert_close(
        module(torch.tensor([[1.0]])), torch.tensor([[0.25, 0.75]])
    )


def test_columnwise_evaluates_the_inner_module_once():
    """One forward, however many column groups are selected."""
    calls = []

    def counting(x):
        calls.append(1)
        return torch.tensor([[0.2, 0.6, 0.8]])

    inner = _operand([A], [("q1",), ("q2",), ("z",)], counting)
    module = ColumnwiseModule(
        lambda n, d: n / d,
        inner,
        (_p(("q1",)), _p(("q2",))),
        (_p(("z",)),),
        name="divide",
    )

    module(torch.tensor([[1.0]]))

    assert sum(calls) == 1


def test_columnwise_rejects_a_column_it_does_not_have():
    """A group naming an absent column is a construction error."""
    inner = _operand([A], [("q1",), ("z",)], lambda x: x)

    with pytest.raises(ValueError, match="not among the module's outputs"):
        ColumnwiseModule(
            lambda n, d: n / d, inner, (_p(("q1",)),), (_p(("nope",)),), name="divide"
        )


def test_columnwise_needs_a_column_group():
    """Selecting nothing to combine is a construction error."""
    inner = _operand([A], [("q1",)], lambda x: x)

    with pytest.raises(ValueError, match="at least one column group"):
        ColumnwiseModule(lambda n: n, inner, name="divide")


def test_columnwise_names_its_own_output_columns():
    """A caller computing a new value names it, instead of borrowing group one."""
    inner = _operand([A], [("q",), ("z",)], lambda x: torch.tensor([[0.2, 0.8]]))

    module = ColumnwiseModule(
        lambda n, d: n / d,
        inner,
        (_p(("q",)),),
        (_p(("z",)),),
        names=(_p(("quotient",)),),
        name="divide",
    )

    assert list(module.get_output_shape()) == [_p(("quotient",))]
    torch.testing.assert_close(module(torch.tensor([[1.0]])), torch.tensor([[0.25]]))
