#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import with_structure
from deeplog.module.aggregation_modules import AggregationModule
from deeplog.module.aggregation_modules import AggregationOpModule
from deeplog.module.module_circuit import ModuleCircuit
from deeplog.shape import sole_structure

from ._utils import Counted
from ._utils import values_along


a, b, c, x = ((name,) for name in "abcx")
p, q, r = ((name,) for name in "pqr")
X, Y = ("X",), ("Y",)
p_x = ("p", X)


def _sum(*columns: torch.Tensor) -> torch.Tensor:
    return sum(columns).sum(dim=1)


def _scaled(column, variable, scale: float = 1.0) -> WrappedModule:
    """``column`` as ``a`` times ``scale`` times the value of ``variable``."""
    return WrappedModule(
        lambda v, a: scale * v * a,
        (SymTensor([variable]), SymTensor([a])),
        SymTensor([column]),
    )


def _body(*modules, output) -> ModuleCircuit:
    return ModuleCircuit(list(modules), SymTensor(output))


def test_aggregation_module():
    """The body gives ``p(X)`` along ``X``, and the aggregator reduces it."""
    body = _body(
        values_along(X, torch.tensor([1.0, 2.0, 3.0])), _scaled(p_x, X), output=[p_x]
    )
    aggregation = AggregationModule([(X, 3)], body, [("sum", _sum, [p_x])])

    assert aggregation.get_input_shape() == (SymTensor([a]),)
    assert aggregation.get_output_shape() == SymTensor([("sum", ("binders", X), p_x)])
    torch.testing.assert_close(
        aggregation(torch.tensor([[2.0], [1.0]])), torch.tensor([[12.0], [6.0]])
    )


def test_a_reduction_receives_each_column_over_the_assignments():
    """Every column arrives as ``(batch, assignments, 1)``, in the order it reads them."""
    received = []

    def op(*columns):
        received.extend(column.clone() for column in columns)
        return columns[0].sum(dim=1)

    body = _body(
        values_along(X, torch.tensor([0.0, 1.0, 2.0])),
        _scaled(p_x, X),
        output=[p_x, a],
    )
    AggregationModule([(X, 3)], body, [("sum", op, [p_x, a])])(torch.tensor([[5.0]]))

    torch.testing.assert_close(received[0], torch.tensor([[[0.0], [5.0], [10.0]]]))
    torch.testing.assert_close(received[1], torch.tensor([[[5.0], [5.0], [5.0]]]))


def test_the_assignments_are_in_row_major_order_of_the_binders():
    """``r(Y, X)`` names ``Y`` first, yet ``X``, the first binder, changes slowest.

    Its values are ``X + Y``, with ``X`` in ``{0, 1}`` and ``Y`` in ``{0, 10, 20}``,
    and a binder no column holds keeps its place among the others.
    """
    received = []

    def op(column):
        received.append(column.clone())
        return column.sum(dim=1)

    Z = ("Z",)
    r_yx = ("r", Y, X)
    body = _body(
        values_along(X, torch.tensor([0.0, 1.0])),
        values_along(Y, torch.tensor([0.0, 10.0, 20.0])),
        WrappedModule(
            lambda y, x: y + x, (SymTensor([Y]), SymTensor([X])), SymTensor([r_yx])
        ),
        output=[r_yx],
    )
    AggregationModule([(X, 2), (Z, 2), (Y, 3)], body, [("sum", op, [r_yx])])()

    expected = torch.tensor([0.0, 10.0, 20.0] * 2 + [1.0, 11.0, 21.0] * 2)
    torch.testing.assert_close(received[0], expected.reshape(1, 12, 1))


def test_a_binder_no_column_holds_ranges_over_its_domain_all_the_same():
    """``sum(X, Y): p(X)`` adds ``p(X)`` once for each of the 2 values of ``Y``."""
    body = _body(
        values_along(X, torch.tensor([1.0, 2.0, 3.0])), _scaled(p_x, X), output=[p_x]
    )
    aggregation = AggregationModule([(X, 3), (Y, 2)], body, [("sum", _sum, [p_x])])

    torch.testing.assert_close(
        aggregation(torch.tensor([[1.0]])), torch.tensor([[12.0]])
    )


def test_reductions_of_one_body_are_a_column_each():
    """Two reductions read the columns of one body, which is evaluated once."""
    counted = Counted(lambda v: torch.cat([v, 2 * v], dim=1))
    body = _body(
        values_along(X, torch.tensor([1.0, 2.0, 3.0])),
        WrappedModule(counted, SymTensor([X]), SymTensor([p_x, ("q", X)])),
        output=[p_x, ("q", X)],
    )
    aggregation = AggregationModule(
        [(X, 3)],
        body,
        [
            ("sum", _sum, [p_x]),
            ("sum", _sum, [("q", X)]),
            ("max", lambda v: v.amax(1), [("q", X)]),
        ],
    )

    assert aggregation.get_output_shape() == SymTensor(
        [
            ("sum", ("binders", X), p_x),
            ("sum", ("binders", X), ("q", X)),
            ("max", ("binders", X), ("q", X)),
        ]
    )
    torch.testing.assert_close(aggregation(), torch.tensor([[6.0, 12.0, 6.0]]))
    assert counted.rows == [3]


def _aggregation_over(*columns) -> AggregationModule:
    """Sum ``columns``, each ``a`` times the value of ``X``, over ``X``."""
    body = _body(
        values_along(X, torch.tensor([0.0, 1.0])),
        *(_scaled(column, X) for column in columns),
        output=list(columns),
    )
    return AggregationModule([(X, 2)], body, [("sum", _sum, list(columns))])


def test_structure_is_carried_by_the_minted_output_symbol():
    """The reduction is a value of the first column's algebra.

    The tag goes on the *outside* of the new name — the reduction of a
    probability is a probability — rather than staying buried on the child
    symbols it wraps.
    """
    labelled = [with_structure(sym, "probability") for sym in (p_x, ("q", X))]

    aggregation = _aggregation_over(*labelled)

    assert aggregation.get_output_shape() == SymTensor(
        [with_structure(("sum", ("binders", X), p_x, ("q", X)), "probability")]
    )
    assert sole_structure(aggregation.get_output_shape()) == "probability"


def test_an_unlabelled_child_yields_an_unlabelled_aggregation():
    """The module layer invents nothing: no label in, no label out.

    Bare outputs are the outside-the-formula-layer case. The lowering
    is what refuses them; ``AggregationModule`` on its own stays usable.
    """
    aggregation = _aggregation_over(p_x)

    assert sole_structure(aggregation.get_output_shape()) is None


def test_an_aggregation_needs_a_variable():
    body = WrappedModule(lambda x: x, SymTensor([a]), SymTensor([p]))

    with pytest.raises(ValueError, match="at least one binder"):
        AggregationModule([], body, [("sum", _sum, [p])])


def test_an_aggregation_of_operands_aggregates_its_columns_in_order():
    """Each output aggregates the input columns it names, repeats included."""
    module = AggregationOpModule(
        [a, b, c],
        [(p, _sum, [0, 1, 0]), (q, lambda v: v.amax(dim=1), [2, 1]), (r, _sum, [2])],
    )
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert module.get_output_shape() == SymTensor([p, q, r])
    torch.testing.assert_close(
        module(x), torch.tensor([[4.0, 3.0, 3.0], [13.0, 6.0, 6.0]])
    )


def test_aggregations_of_operands_read_their_input_in_one_operation():
    """The input reaches the gradient through one node, however many outputs read it."""
    module = AggregationOpModule(
        [a, b, c], [(p, _sum, [0, 1]), (q, _sum, [1, 2]), (r, _sum, [2, 0, 1])]
    )
    x = torch.rand(2, 3, requires_grad=True)

    seen, stack, readers = set(), [module(x).grad_fn], 0
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        feeding = [fn for fn, _ in node.next_functions]
        readers += any(getattr(fn, "variable", None) is x for fn in feeding)
        stack.extend(feeding)

    assert readers == 1


def test_aggregations_of_one_aggregator_over_as_many_operands_are_one_call():
    """Four sums of two columns are one call, over four times the batch's rows."""
    counted = Counted(lambda v: v.sum(dim=1))
    module = AggregationOpModule(
        [a, b, c],
        [
            (p, counted, [0, 1]),
            (q, counted, [1, 2]),
            (r, counted, [2, 0]),
            (x, counted, [0, 0]),
        ],
    )

    out = module(torch.tensor([[1.0, 2.0, 4.0]]))

    assert counted.rows == [4]
    torch.testing.assert_close(out, torch.tensor([[3.0, 6.0, 5.0, 2.0]]))
