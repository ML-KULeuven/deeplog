#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import compose_modules
from deeplog.module.module_circuit import ModuleCircuit

from ._utils import Counted
from ._utils import DummyModule
from ._utils import values_along


class TestModuleCircuit:
    def test_simple_chain(self):
        modules = [
            DummyModule(SymTensor("a"), SymTensor("b"), lambda x: x**2),
            DummyModule(SymTensor("b"), SymTensor("c"), lambda x: 2 * x),
            DummyModule(
                (SymTensor("a"), SymTensor("c")), SymTensor("d"), lambda x, y: x + y
            ),
        ]
        module = ModuleCircuit(modules, SymTensor("d"))
        assert pytest.approx(10.0) == float(module(torch.FloatTensor([[2.0]])))

    def test_inputs_follow_the_order_the_modules_first_read_them(self):
        """A circuit's inputs are one tensor per symbol no module produces.

        They come in the order the modules, as given, first read them, so the
        signature does not depend on how symbols hash.
        """
        modules = [
            DummyModule(SymTensor(["z", "a"]), SymTensor("b"), lambda x: x.sum(1)),
            DummyModule(SymTensor(["m", "b"]), SymTensor("c"), lambda x: x.sum(1)),
        ]
        circuit = ModuleCircuit(modules, SymTensor("c"))
        assert circuit.get_input_shape() == (
            SymTensor(["z"]),
            SymTensor(["a"]),
            SymTensor(["m"]),
        )

    def test_missing_transformation_combines_outputs(self):
        """ModuleCircuit inserts a transformation when a module needs [b,c] but
        only [b] and [c] are produced separately."""
        modules = [
            DummyModule(SymTensor("a"), SymTensor("b"), lambda x: x**2),
            DummyModule(SymTensor("a"), SymTensor("c"), lambda x: x * 3),
            # Needs [b, c] as a single tensor – no module produces it directly.
            DummyModule(
                SymTensor(["b", "c"]),
                SymTensor("d"),
                lambda bc: bc.sum(dim=1, keepdim=True),
            ),
        ]
        circuit = ModuleCircuit(modules, SymTensor("d"))
        # input a=2  →  b=4, c=6  →  transformation merges into [4,6]  →  sum=10
        result = circuit(torch.FloatTensor([[2.0]]))
        assert pytest.approx(10.0) == float(result)

    def test_a_transformation_takes_each_output_it_reads_from_once(self):
        """Reading three symbols of one output takes that output once, not three times.

        The grid holds a=1, b=2, c=3 and d=4, so the reader gives 100d + 10a + c = 413.
        """
        grid = SymTensor([["a", "b"], ["c", "d"]])
        producer = DummyModule(
            SymTensor(["w", "x", "y", "z"]), grid, lambda t: t.reshape(-1, 2, 2)
        )
        reader = DummyModule(
            SymTensor(["d", "a", "c"]),
            SymTensor("e"),
            lambda t: 100 * t[:, :1] + 10 * t[:, 1:2] + t[:, 2:],
        )
        circuit = ModuleCircuit(
            [producer, reader], SymTensor("e"), SymTensor(["w", "x", "y", "z"])
        )

        (transformation,) = (
            m for m in circuit._sub_modules if m is not producer and m is not reader
        )
        assert transformation.get_input_shape() == grid
        assert float(circuit(torch.FloatTensor([[1, 2, 3, 4]]))) == pytest.approx(413)

    def test_no_external_inputs_evaluates_single_row(self):
        """A circuit whose leaves are all baked has no inputs; it yields one row.

        With no runtime tensor to read the batch size from, ModuleCircuit must
        evaluate a single constant row rather than an empty (batch-0) output —
        the shape a baked-constant DeepProbLog program compiles to.
        """
        const_b = DummyModule(
            SymTensor([]), SymTensor("b"), lambda e: torch.full((e.shape[0], 1), 0.6)
        )
        const_c = DummyModule(
            SymTensor([]), SymTensor("c"), lambda e: torch.full((e.shape[0], 1), 0.3)
        )
        circuit = ModuleCircuit([const_b, const_c], (SymTensor("b"), SymTensor("c")))
        assert circuit.get_input_shape() == ()  # no external inputs
        out = circuit()  # called with no runtime tensors
        assert [tensor.shape for tensor in out] == [
            (1, 1),
            (1, 1),
        ]  # one row, not empty
        assert pytest.approx([0.6, 0.3]) == [float(tensor) for tensor in out]

    def test_a_module_reading_no_input_is_given_its_empty_row_on_the_inputs_device(
        self,
    ):
        """The empty input stands in for the batch, so it sits where the batch does."""
        devices = []

        def constant(empty):
            devices.append(empty.device)
            return empty.new_ones(empty.shape[0], 1)

        modules = [
            DummyModule(SymTensor([]), SymTensor("b"), constant),
            DummyModule(
                (SymTensor("a"), SymTensor("b")), SymTensor("c"), lambda a, b: a + b
            ),
        ]
        circuit = ModuleCircuit(modules, SymTensor("c"))

        circuit(torch.zeros(2, 1, device="meta"))

        assert devices == [torch.device("meta")]

    def test_output_shape_needs_transformation(self):
        """ModuleCircuit inserts a transformation when the circuit's output_shape
        is not directly produced by any submodule."""
        modules = [
            DummyModule(
                SymTensor("a"),
                SymTensor(["b", "c"]),
                lambda x: torch.cat([x * 2, x * 3], dim=1),
            ),
        ]
        # Request output [c, b] — a reordering that no module produces directly.
        circuit = ModuleCircuit(modules, SymTensor(["c", "b"]))
        # a=5 → [b,c]=[10,15] → transformation reorders to [c,b]=[15,10]
        result = circuit(torch.FloatTensor([[5.0]]))
        assert result.shape == (1, 2)
        assert pytest.approx(15.0) == float(result[0, 0])
        assert pytest.approx(10.0) == float(result[0, 1])


def test_compose_one_module_produces_the_requested_shape():
    module = DummyModule(
        SymTensor("a"), SymTensor(["b", "c"]), lambda x: torch.cat([2 * x, 3 * x], 1)
    )

    composed = compose_modules([module], SymTensor(["c"]))

    assert composed.get_output_shape() == SymTensor(["c"])
    assert float(composed(torch.FloatTensor([[2.0]]))) == pytest.approx(6.0)


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_compose_refuses_two_modules_producing_one_symbol(order):
    """Which module a symbol is read from would depend on the order given."""
    modules = [
        DummyModule(SymTensor("a"), SymTensor(["b", "c"]), lambda x: x),
        DummyModule(SymTensor("d"), SymTensor(["c"]), lambda x: x),
    ]

    with pytest.raises(ValueError, match="Two modules produce c"):
        compose_modules([modules[i] for i in order], SymTensor(["c"]))


def test_compose_one_module_that_already_fits_returns_it():
    module = DummyModule(SymTensor("a"), SymTensor("b"), lambda x: x)

    assert compose_modules([module], SymTensor("b")) is module


def test_compose_hands_the_given_input_to_the_module_reading_it():
    """A module reading ``input_shape`` receives the caller's tensor as it is."""
    seen = []
    modules = [
        DummyModule(
            SymTensor(["a", "b"]),
            SymTensor("c"),
            lambda x: seen.append(x) or x.sum(1, keepdim=True),
        ),
        DummyModule(SymTensor("c"), SymTensor("d"), lambda x: 2 * x),
    ]
    batch = torch.FloatTensor([[2.0, 3.0]])

    composed = compose_modules(
        modules, SymTensor("d"), input_shape=SymTensor(["a", "b"])
    )

    assert composed.get_input_shape() == SymTensor(["a", "b"])
    assert float(composed(batch)) == pytest.approx(10.0)
    assert seen[0] is batch


def test_compose_takes_from_the_given_input_what_each_module_reads():
    modules = [
        DummyModule(SymTensor("a"), SymTensor("c"), lambda x: x**2),
        DummyModule(
            (SymTensor("b"), SymTensor("c")), SymTensor("d"), lambda x, y: x + y
        ),
    ]

    composed = compose_modules(
        modules, SymTensor("d"), input_shape=SymTensor(["b", "a"])
    )

    # b = 3, a = 2  →  c = 4  →  d = 3 + 4
    assert float(composed(torch.FloatTensor([[3.0, 2.0]]))) == pytest.approx(7.0)


def test_compose_one_module_takes_the_requested_input():
    module = DummyModule(
        (SymTensor("a"), SymTensor("b")), SymTensor("c"), lambda x, y: x - y
    )

    composed = compose_modules(
        [module], SymTensor("c"), input_shape=SymTensor(["b", "a"])
    )

    assert composed.get_input_shape() == SymTensor(["b", "a"])
    assert float(composed(torch.FloatTensor([[1.0, 10.0]]))) == pytest.approx(9.0)


def test_compose_refuses_an_input_shape_missing_a_symbol_no_module_produces():
    modules = [
        DummyModule((SymTensor("a"), SymTensor("b")), SymTensor("c"), lambda x, y: x),
        DummyModule(SymTensor("c"), SymTensor("d"), lambda x: x),
    ]

    with pytest.raises(ValueError, match="not produced by any module"):
        compose_modules(modules, SymTensor("d"), input_shape=SymTensor(["a"]))


@pytest.mark.parametrize("count", [1, 2])
def test_compose_refuses_an_input_shape_holding_a_produced_symbol(count):
    modules = [
        DummyModule(SymTensor("a"), SymTensor("b"), lambda x: x),
        DummyModule(SymTensor("b"), SymTensor("c"), lambda x: x),
    ][:count]

    with pytest.raises(ValueError, match="which the modules produce"):
        compose_modules(modules, SymTensor("b"), input_shape=SymTensor(["a", "b"]))


# --- A composition over variables ------------------------------------------------

a = ("a",)
X, Y = ("X",), ("Y",)
p_x, q_y, r_xy = ("p", X), ("q", Y), ("r", X, Y)


def test_a_value_holding_a_variable_has_its_axis_behind_its_symbols():
    """``p(X)`` over the 3 values of ``X`` is given as ``(batch, 1, 3)``."""
    circuit = ModuleCircuit(
        [
            values_along(X, torch.tensor([1.0, 2.0, 3.0])),
            WrappedModule(
                lambda v, a: 10 * v * a,
                (SymTensor([X]), SymTensor([a])),
                SymTensor([p_x]),
            ),
        ],
        SymTensor([p_x]),
    )

    torch.testing.assert_close(
        circuit(torch.tensor([[1.0], [2.0]])),
        torch.tensor([[[10.0, 20.0, 30.0]], [[20.0, 40.0, 60.0]]]),
    )


def test_each_module_is_called_once_per_assignment_of_the_variables_it_reads():
    """``p(X)`` at the 3 values of ``X``, ``q(Y)`` at the 2 of ``Y``, ``r(X,Y)`` at the 6 pairs."""
    scale, shift, add = (
        Counted(lambda v: 10 * v),
        Counted(lambda v, a: v + a),
        Counted(lambda p, q: p + q),
    )
    circuit = ModuleCircuit(
        [
            values_along(X, torch.tensor([1.0, 2.0, 3.0])),
            values_along(Y, torch.tensor([0.1, 0.2])),
            WrappedModule(scale, SymTensor([X]), SymTensor([p_x])),
            WrappedModule(shift, (SymTensor([Y]), SymTensor([a])), SymTensor([q_y])),
            WrappedModule(add, (SymTensor([p_x]), SymTensor([q_y])), SymTensor([r_xy])),
        ],
        SymTensor([r_xy]),
    )

    result = circuit(torch.tensor([[5.0]]))

    assert (scale.rows, shift.rows, add.rows) == ([3], [2], [6])
    expected = 10 * torch.tensor([1.0, 2.0, 3.0])[:, None] + torch.tensor([5.1, 5.2])
    torch.testing.assert_close(result, expected.reshape(1, 1, 3, 2))


def test_a_value_over_no_variable_is_computed_once_per_batch_row():
    """``q`` reads no variable, so it runs on the batch alone, however many ``X`` it meets."""
    once = Counted(lambda a: 2 * a)
    circuit = ModuleCircuit(
        [
            values_along(X, torch.tensor([1.0, 2.0, 3.0])),
            WrappedModule(once, SymTensor([a]), SymTensor([("q",)])),
            WrappedModule(
                lambda v, q: v * q,
                (SymTensor([X]), SymTensor([("q",)])),
                SymTensor([p_x]),
            ),
        ],
        SymTensor([p_x]),
    )

    torch.testing.assert_close(
        circuit(torch.tensor([[1.0]])), torch.tensor([[[2.0, 4.0, 6.0]]])
    )
    assert once.rows == [1]


def test_a_module_giving_a_variable_it_does_not_read_gives_its_axis_behind_its_symbols():
    """Ground atoms ``t(0) .. t(2)`` read as ``t(X)`` along ``X``: the same tensor, reshaped."""
    ground = SymTensor([("t", (str(i),)) for i in range(3)])
    t_x = ("t", X)
    relabel = WrappedModule(
        lambda table: table.reshape(table.shape[0], 1, 3), ground, SymTensor([t_x])
    )
    times = WrappedModule(
        lambda t, a: t * a, (SymTensor([t_x]), SymTensor([a])), SymTensor([p_x])
    )
    circuit = ModuleCircuit(
        [relabel, times],
        SymTensor([p_x]),
        input_shape=(ground, SymTensor([a])),
    )

    torch.testing.assert_close(
        circuit(
            torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]),
            torch.tensor([[1.0], [2.0]]),
        ),
        torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 2.0, 2.0]]]),
    )


def test_a_module_binding_a_variable_without_its_axis_raises():
    """``Score1`` is a variable, so a head giving it as a plain column binds it with no axis."""
    extractor = DummyModule(SymTensor("raw"), SymTensor("hidden"), lambda x: 2 * x)
    head = DummyModule(SymTensor("hidden"), SymTensor(["Score1"]), lambda x: x + 1)
    circuit = ModuleCircuit([extractor, head], SymTensor(["Score1"]))

    with pytest.raises(ValueError, match=r"binds Score1.*lowercase \(score1\)"):
        circuit(torch.tensor([[1.0]]))
