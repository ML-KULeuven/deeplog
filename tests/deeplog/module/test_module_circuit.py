#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import ModuleCircuit
from deeplog import SymTensor

from ._utils import DummyModule


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
