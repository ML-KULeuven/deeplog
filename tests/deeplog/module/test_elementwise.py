#  Copyright (c) 2024-2026. KU Leuven
"""Tests for :class:`~deeplog.module.ElementwiseModule`.

The combinator for operators with no circuit form. It takes the tensor callable
from the algebra, reshapes every operand onto the union of their inputs, and
combines the operands' already-reduced outputs column-wise — broadcasting a
single column across the rest.
"""

import pytest
import torch

from deeplog import SymTensor
from deeplog import with_structure
from deeplog.module import ColumnwiseModule
from deeplog.module import ElementwiseModule

from ._utils import DummyModule


A, B, C = ("a",), ("b",), ("c",)


def _p(symbol):
    """Label ``symbol`` as probability-valued, the way a lowered module names it."""
    return with_structure(symbol, "probability")


def _operand(inputs, outputs, forward, structure="probability"):
    """A DummyModule whose outputs are labelled, as the lowering factory hands over.

    ``structure=None`` gives bare outputs — a module built outside the formula
    layer, where no algebra is declared.
    """
    if structure is not None:
        outputs = [with_structure(symbol, structure) for symbol in outputs]
    return DummyModule(SymTensor(list(inputs)), SymTensor(list(outputs)), forward)


def test_operands_consuming_different_symbols_keep_their_input_tensors():
    """Each operand's input tensor becomes one of the union's."""
    lhs = _operand([A], [("lhs",)], lambda x: x * 2)
    rhs = _operand([B], [("rhs",)], lambda x: x + 1)

    module = ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")

    assert module.get_input_shape() == (SymTensor([A]), SymTensor([B]))
    # a=3 -> 6, b=5 -> 6, product 36.
    torch.testing.assert_close(
        module(torch.tensor([[3.0]]), torch.tensor([[5.0]])), torch.tensor([[36.0]])
    )


def test_operands_whose_inputs_differ_in_feature_shape_are_fed_apart():
    """A vector input and a scalar input are never stacked into one tensor."""
    lhs = _operand([A], [("lhs",)], lambda x: x.sum(-1))
    rhs = _operand([B], [("rhs",)], lambda x: x)

    module = ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")

    # a=(1,2) -> 3, b=5 -> 5, product 15.
    torch.testing.assert_close(
        module(torch.tensor([[[1.0, 2.0]]]), torch.tensor([[5.0]])),
        torch.tensor([[15.0]]),
    )


def test_shared_symbols_are_unioned_once():
    """A symbol both operands consume appears once in the union input."""
    lhs = _operand([A, B], [("lhs",)], lambda x: x.sum(-1, keepdim=True))
    rhs = _operand([B, C], [("rhs",)], lambda x: x.sum(-1, keepdim=True))

    module = ElementwiseModule(lambda a, b: a + b, lhs, rhs, name="plus")

    assert module.get_input_shape() == (SymTensor([A, B]), SymTensor([C]))
    # (1+2) + (2+3) = 8
    torch.testing.assert_close(
        module(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0]])),
        torch.tensor([[8.0]]),
    )


def test_operands_reading_one_input_tensor_share_it():
    """Operands over the same symbols take the one tensor both declare."""
    lhs = _operand([A, B], [("lhs",)], lambda x: x.sum(-1, keepdim=True))
    rhs = _operand([A, B], [("rhs",)], lambda x: x.prod(-1, keepdim=True))

    module = ElementwiseModule(lambda a, b: a + b, lhs, rhs, name="plus")

    assert module.get_input_shape() == SymTensor([A, B])
    # (2+3) + (2*3) = 11
    torch.testing.assert_close(
        module(torch.tensor([[2.0, 3.0]])), torch.tensor([[11.0]])
    )


def test_single_column_broadcasts_across_the_other_operand():
    """N columns over 1 column is the posterior / normalisation shape."""
    joints = _operand(
        [A], [("q1",), ("q2",), ("q3",)], lambda x: x * torch.tensor([1.0, 2.0, 5.0])
    )
    total = _operand([A], [("z",)], lambda x: x * 8.0)

    module = ElementwiseModule(lambda n, d: n / d, joints, total, name="divide")

    torch.testing.assert_close(
        module(torch.tensor([[1.0]])), torch.tensor([[0.125, 0.25, 0.625]])
    )


def test_output_columns_are_named_after_the_operator_and_operands():
    """Each output column names the operator and the symbols it joins."""
    joints = _operand([A], [("q1",), ("q2",)], lambda x: x.expand(-1, 2))
    total = _operand([A], [("z",)], lambda x: x)

    module = ElementwiseModule(lambda n, d: n / d, joints, total, name="divide")

    # The label sits on the outside of each minted name, not nested once per
    # operand symbol.
    assert list(module.get_output_shape()) == [
        with_structure(("divide", ("q1",), ("z",)), "probability"),
        with_structure(("divide", ("q2",), ("z",)), "probability"),
    ]


def test_equal_width_operands_combine_column_wise():
    """Two N-column operands pair up rather than broadcast."""
    lhs = _operand([A], [("l1",), ("l2",)], lambda x: torch.tensor([[2.0, 3.0]]))
    rhs = _operand([A], [("r1",), ("r2",)], lambda x: torch.tensor([[5.0, 7.0]]))

    module = ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")

    torch.testing.assert_close(
        module(torch.tensor([[1.0]])), torch.tensor([[10.0, 21.0]])
    )


def test_incompatible_widths_raise():
    """Widths that neither match nor broadcast are a construction error."""
    lhs = _operand([A], [("l1",), ("l2",)], lambda x: x)
    rhs = _operand([A], [("r1",), ("r2",), ("r3",)], lambda x: x)

    with pytest.raises(ValueError, match="broadcasting column"):
        ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")


def test_structure_mismatch_raises():
    """Operands from different algebras cannot be combined."""
    lhs = _operand([A], [("l",)], lambda x: x, structure="probability")
    rhs = _operand([A], [("r",)], lambda x: x, structure="boolean")

    with pytest.raises(ValueError, match="share a structure"):
        ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")


def test_an_operand_mixing_structures_is_rejected():
    """One operand naming two algebras is as inadmissible as two that disagree."""
    mixed = DummyModule(
        SymTensor([A]),
        SymTensor(
            [
                with_structure(("l",), "probability"),
                with_structure(("r",), "boolean"),
            ]
        ),
        lambda x: x,
    )
    other = _operand([A], [("z",)], lambda x: x)

    with pytest.raises(ValueError, match="share a structure"):
        ElementwiseModule(lambda a, b: a * b, mixed, other, name="times")


def test_unlabelled_operands_combine_without_a_structure():
    """Outside the formula layer the caller supplies ``op``; no algebra is read.

    The module layer invents no label, so the minted columns stay bare too.
    """
    lhs = _operand([A], [("l",)], lambda x: x * 2, structure=None)
    rhs = _operand([A], [("r",)], lambda x: x + 1, structure=None)

    module = ElementwiseModule(lambda a, b: a * b, lhs, rhs, name="times")

    assert list(module.get_output_shape()) == [("times", ("l",), ("r",))]


def test_no_operands_raises():
    """An operator with no operands is meaningless."""
    with pytest.raises(ValueError, match="at least one operand"):
        ElementwiseModule(lambda: torch.tensor(0.0), name="times")


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
