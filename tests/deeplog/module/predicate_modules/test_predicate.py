#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import Predicate
from deeplog import SymTensor


class IgnoringPredicate(Predicate[torch.Tensor]):
    functor = "ignoring"
    arity = 2
    structure = "boolean"

    def __init__(self, all_arguments):
        self.resolve_argument_calls: list[int] = []
        super().__init__(all_arguments, ignore_argument=(1,))

    def _resolve_argument(self, symbol, index, /):
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

    def _resolve_argument(self, symbol, index, /):
        self.resolve_argument_calls.append((index, symbol))
        if symbol == ("alias",):
            return ("redirect",)
        return symbol

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


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
