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
