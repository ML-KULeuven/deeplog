#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import Domain
from deeplog import Predicate
from deeplog import SymTensor
from deeplog import with_structure


VALUES = Domain.of_values()


class IgnoringPredicate(Predicate[torch.Tensor]):
    def __init__(self, atoms):
        self.resolve_argument_calls: list[int] = []
        super().__init__(atoms, (VALUES, VALUES), ignore_arguments=(1,))

    def resolve_argument(self, symbol, index, /):
        self.resolve_argument_calls.append(index)
        return symbol

    def forward_predicate(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.float32) + 1.0


class SymbolRedirectPredicate(Predicate[torch.Tensor, torch.Tensor]):
    def __init__(self, atoms):
        self.resolve_argument_calls: list[tuple[int, tuple[str, ...]]] = []
        super().__init__(atoms, (VALUES, VALUES))

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
            def _resolve_argument(self, symbol, index, /):
                return symbol

            def forward_predicate(self, tensor):
                return tensor


@pytest.mark.parametrize("name", ["functor", "arity", "structure"])
def test_a_predicate_declaring_its_predicate_is_refused(name):
    """A compiler registers a predicate under its key, so a declaration would be ignored."""
    with pytest.raises(TypeError, match=f"declares {name}"):
        type(
            "Stale",
            (Predicate,),
            {name: "stale", "forward_predicate": lambda self, tensor: tensor},
        )


def test_predicate_skips_ignored_argument():
    predicate = IgnoringPredicate([("ignoring", ("_",), ("ignored",))])

    x = torch.tensor([[1.0], [2.0]])
    result = predicate(x)

    torch.testing.assert_close(result, torch.tensor([[2.0], [3.0]]))
    assert predicate.resolve_argument_calls == [0]
    assert predicate.get_input_shape() == (SymTensor([("_",)]),)


def test_predicate_treats_returned_symbol_as_variable():
    predicate = SymbolRedirectPredicate([("redirect", ("x",), ("alias",))])

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
    def __init__(self, atoms):
        super().__init__(atoms, (VALUES, VALUES))

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return lhs + rhs


class NumericFirstArgument(AddingPredicate):
    distinct_arguments = (0,)

    def resolve_argument(self, symbol, index, /):
        return float(symbol[0]) if index == 0 else symbol


def test_predicate_names_its_columns_by_the_atoms_it_is_given():
    atoms = [
        with_structure(("adding", ("x",), ("a",)), "real"),
        with_structure(("adding", ("y",), ("a",)), "real"),
    ]

    predicate = AddingPredicate(atoms)

    assert predicate.get_output_shape() == SymTensor(atoms)


def test_predicate_refuses_atoms_of_different_arities():
    with pytest.raises(ValueError, match="atoms of one arity"):
        AddingPredicate([("adding", ("x",), ("a",)), ("adding", ("x",))])


def test_predicate_asks_for_each_distinct_symbol_once():
    predicate = AddingPredicate(
        [
            ("adding", ("x",), ("a",)),
            ("adding", ("x",), ("b",)),
            ("adding", ("y",), ("a",)),
        ]
    )

    assert predicate.get_input_shape() == (
        SymTensor([("x",), ("y",)]),
        SymTensor([("a",), ("b",)]),
    )


def test_predicate_expands_distinct_inputs_over_evaluations():
    predicate = AddingPredicate(
        [
            ("adding", ("x",), ("a",)),
            ("adding", ("x",), ("b",)),
            ("adding", ("y",), ("a",)),
        ]
    )

    left = torch.tensor([[1.0, 10.0]])  # x, y
    right = torch.tensor([[100.0, 200.0]])  # a, b

    result = predicate(left, right)

    # (x + a), (x + b), (y + a)
    torch.testing.assert_close(result, torch.tensor([[101.0, 201.0, 110.0]]))


def test_predicate_expands_repeated_symbols_alongside_constants():
    class HalfConstant(AddingPredicate):
        def resolve_argument(self, symbol, index, /):
            if index == 1 and symbol == ("two",):
                return 2.0
            return symbol

    predicate = HalfConstant(
        [("half_constant", ("x",), ("two",)), ("half_constant", ("x",), ("a",))]
    )

    assert predicate.get_input_shape() == (SymTensor([("x",)]), SymTensor([("a",)]))

    result = predicate(torch.tensor([[3.0]]), torch.tensor([[5.0]]))

    # x + 2 (constant row), then x + a
    torch.testing.assert_close(result, torch.tensor([[5.0, 8.0]]))


def test_predicate_rejects_unexpanded_position_carrying_constants():
    with pytest.raises(ValueError, match="distinct_arguments"):
        NumericFirstArgument([("numeric", ("0.5",), ("a",))])


class Distance(Predicate[torch.Tensor, torch.Tensor]):
    def __init__(self, atoms):
        super().__init__(atoms, (VALUES, VALUES))

    def resolve_argument(self, symbol, index, /):
        if symbol == ("origin",):
            return torch.tensor([0.0, 0.0])
        return symbol

    def forward_predicate(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.norm(lhs - rhs, dim=1)


def test_predicate_reads_a_vector_constant_at_a_position_with_no_symbol():
    predicate = Distance([("distance", ("x",), ("origin",))])

    assert predicate.get_input_shape() == (SymTensor([("x",)]), SymTensor([]))

    result = predicate(torch.tensor([[[3.0, 4.0]]]), torch.empty(1, 0))

    torch.testing.assert_close(result, torch.tensor([[5.0]]))


def test_predicate_of_vector_constants_alone():
    predicate = Distance([("distance", ("origin",), ("origin",))])

    result = predicate(torch.empty(2, 0), torch.empty(2, 0))

    torch.testing.assert_close(result, torch.zeros(2, 1))


class Weight(Predicate[torch.Tensor]):
    """``w(X)``: the value's position, plus one."""

    def forward_predicate(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.get_default_dtype()) + 1.0


_COLOURS = Domain.of(["red", "green", "blue"])


def test_a_name_written_at_a_position_with_a_domain_is_its_position():
    """Where a variable would hold 1, the written ``green`` reads 1 too."""
    predicate = Weight([("w", ("green",)), ("w", ("X",))], (_COLOURS,))

    assert predicate.get_input_shape() == (SymTensor([("X",)]),)
    torch.testing.assert_close(
        predicate(torch.tensor([[1]])), torch.tensor([[2.0, 2.0]])
    )


def test_a_number_written_at_a_position_over_unnamed_values_is_that_number():
    digits = Domain.of_tensor(torch.arange(10))
    predicate = Weight([("w", ("4",))], (digits,))

    torch.testing.assert_close(predicate(torch.empty(1, 0)), torch.tensor([[5.0]]))


@pytest.mark.parametrize(
    ("domain", "value"),
    [(_COLOURS, "purple"), (Domain.of_tensor(torch.arange(10)), "11")],
    ids=["named", "unnamed"],
)
def test_a_word_that_is_no_value_of_its_positions_domain_is_refused(domain, value):
    with pytest.raises(
        ValueError, match=f"{value} is not a value of the domain of w/1"
    ):
        Weight([("w", (value,))], (domain,))


def test_a_word_where_an_argument_ranges_over_the_inputs_names_an_input():
    predicate = Weight([("w", ("green",))], (VALUES,))

    assert predicate.get_input_shape() == (SymTensor([("green",)]),)


def _shades(arguments):
    """``shade(Kind, Value)``: the value ranges over the domain the kind chooses."""
    kinds = {("colour",): _COLOURS, ("grey",): Domain.of(["light", "dark"])}
    return VALUES, kinds[arguments[0]]


class Shade(Predicate[torch.Tensor]):
    """``shade(Kind, Value)``: the value's position."""

    def __init__(self, atoms):
        super().__init__(atoms, _shades, ignore_arguments=(0,))

    def forward_predicate(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.get_default_dtype())


def test_a_predicate_can_give_each_atom_its_own_domain():
    """``blue`` is position 2 of the colours, ``dark`` position 1 of the greys."""
    predicate = Shade(
        [("shade", ("colour",), ("blue",)), ("shade", ("grey",), ("dark",))]
    )

    torch.testing.assert_close(predicate(torch.empty(1, 0)), torch.tensor([[2.0, 1.0]]))


def test_a_predicate_gives_one_domain_per_argument():
    with pytest.raises(ValueError, match="gives 2 domains for an atom of 1 arguments"):
        Weight([("w", ("green",))], (VALUES, VALUES))
