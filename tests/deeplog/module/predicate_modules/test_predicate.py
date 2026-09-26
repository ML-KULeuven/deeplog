#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import Predicate
from deeplog import SymTensor


class IgnoringPredicate(Predicate[torch.Tensor]):
    functor = "ignoring"
    arity = 2
    structure = "boolean"

    def __init__(self, all_arguments):
        self.resolve_argument_calls: list[int] = []
        super().__init__(all_arguments, ignore_arguments=(1,))

    def resolve_argument(self, symbol, index, /):
        self.resolve_argument_calls.append(index)
        return symbol

    def forward_predicate(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.float32) + 1.0


class SymbolRedirectPredicate(Predicate[torch.Tensor, torch.Tensor]):
    functor = "redirect"
    arity = 2
    structure = "boolean"

    def __init__(self, all_arguments):
        self.resolve_argument_calls: list[tuple[int, tuple[str, ...]]] = []
        super().__init__(all_arguments)

    def resolve_argument(self, symbol, index, /):
        self.resolve_argument_calls.append((index, symbol))
        if symbol == ("alias",):
            return ("redirect",)
        return symbol

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


def test_a_predicate_defining_the_former_name_of_resolve_argument_is_refused():
    """An override under the old name would never be called, so it is an error."""
    with pytest.raises(TypeError, match="rename the method"):

        class Stale(Predicate[torch.Tensor]):
            functor, arity, structure = "stale", 1, "boolean"

            def _resolve_argument(self, symbol, index, /):
                return symbol

            def forward_predicate(self, tensor):
                return tensor


def test_predicate_skips_ignored_argument():
    predicate = IgnoringPredicate([(("_",), ("ignored",))])

    x = torch.tensor([[1.0], [2.0]])
    result = predicate(x)

    torch.testing.assert_close(result, torch.tensor([[2.0], [3.0]]))
    assert predicate.resolve_argument_calls == [0]
    assert predicate.get_input_shape() == (SymTensor([("_",)]),)


def test_predicate_treats_returned_symbol_as_variable():
    predicate = SymbolRedirectPredicate([(("x",), ("alias",))])

    left = torch.tensor([[2.0]])
    right = torch.tensor([[3.0]])

    result = predicate(left, right)

    torch.testing.assert_close(result, torch.tensor([[5.0]]))
    assert predicate.resolve_argument_calls == [(0, ("x",)), (1, ("alias",))]
    assert predicate.get_input_shape() == (
        SymTensor([("x",)]),
        SymTensor([("redirect",)]),
    )


class AddingPredicate(Predicate[torch.Tensor, torch.Tensor]):
    functor = "adding"
    arity = 2
    structure = "boolean"

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


class NumericFirstArgument(Predicate[torch.Tensor, torch.Tensor]):
    functor = "numeric"
    arity = 2
    structure = "boolean"
    distinct_arguments = (0,)

    def resolve_argument(self, symbol, index, /):
        return float(symbol[0]) if index == 0 else symbol

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


def test_predicate_asks_for_each_distinct_symbol_once():
    predicate = AddingPredicate([(("x",), ("a",)), (("x",), ("b",)), (("y",), ("a",))])

    assert predicate.get_input_shape() == (
        SymTensor([("x",), ("y",)]),
        SymTensor([("a",), ("b",)]),
    )


def test_predicate_expands_distinct_inputs_over_evaluations():
    predicate = AddingPredicate([(("x",), ("a",)), (("x",), ("b",)), (("y",), ("a",))])

    left = torch.tensor([[1.0, 10.0]])  # x, y
    right = torch.tensor([[100.0, 200.0]])  # a, b

    result = predicate(left, right)

    # (x + a), (x + b), (y + a)
    torch.testing.assert_close(result, torch.tensor([[101.0, 201.0, 110.0]]))


def test_predicate_expands_repeated_symbols_alongside_constants():
    class HalfConstant(AddingPredicate):
        functor = "half_constant"

        def resolve_argument(self, symbol, index, /):
            if index == 1 and symbol == ("two",):
                return 2.0
            return symbol

    predicate = HalfConstant([(("x",), ("two",)), (("x",), ("a",))])

    assert predicate.get_input_shape() == (SymTensor([("x",)]), SymTensor([("a",)]))

    result = predicate(torch.tensor([[3.0]]), torch.tensor([[5.0]]))

    # x + 2 (constant row), then x + a
    torch.testing.assert_close(result, torch.tensor([[5.0, 8.0]]))


def test_predicate_rejects_unexpanded_position_carrying_constants():
    with pytest.raises(ValueError, match="distinct_arguments"):
        NumericFirstArgument([(("0.5",), ("a",))])


class Distance(Predicate[torch.Tensor, torch.Tensor]):
    functor = "distance"
    arity = 2
    structure = "real"

    def resolve_argument(self, symbol, index, /):
        if symbol == ("origin",):
            return torch.tensor([0.0, 0.0])
        return symbol

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.norm(lhs - rhs, dim=1)


def test_predicate_reads_a_vector_constant_at_a_position_with_no_symbol():
    predicate = Distance([(("x",), ("origin",))])

    assert predicate.get_input_shape() == (SymTensor([("x",)]), SymTensor([]))

    result = predicate(torch.tensor([[[3.0, 4.0]]]), torch.empty(1, 0))

    torch.testing.assert_close(result, torch.tensor([[5.0]]))


def test_predicate_of_vector_constants_alone():
    predicate = Distance([(("origin",), ("origin",))])

    result = predicate(torch.empty(2, 0), torch.empty(2, 0))

    torch.testing.assert_close(result, torch.zeros(2, 1))
