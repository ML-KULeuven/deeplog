#  Copyright (c) 2024-2026. KU Leuven
import pytest
import torch

from deeplog import Sequential
from deeplog import SymTensor
from deeplog.graph_backend import is_backend_available

from ._utils import DummyModule


class TestSequential:
    def test_matching_shapes_chain(self):
        mod1 = DummyModule(SymTensor("a"), SymTensor("b"), lambda x: x + 1)
        mod2 = DummyModule(SymTensor("b"), SymTensor("c"), lambda x: 2 * x)
        seq = Sequential(mod1, mod2)
        assert seq.get_input_shape() == SymTensor("a")
        assert seq.get_output_shape() == SymTensor("c")
        torch.testing.assert_close(seq(torch.tensor([[1.0]])), torch.tensor([[4.0]]))

    def test_nested_sequential_flattens(self):
        mod1 = DummyModule(SymTensor("a"), SymTensor("b"))
        mod2 = DummyModule(SymTensor("b"), SymTensor("c"))
        mod3 = DummyModule(SymTensor("c"), SymTensor("d"))
        seq = Sequential(Sequential(mod1, mod2), mod3)
        assert seq.get_input_shape() == SymTensor("a")
        assert seq.get_output_shape() == SymTensor("d")

    def test_empty_sequential_raises(self):
        with pytest.raises(ValueError, match="at least one submodule"):
            Sequential()

    @pytest.mark.skipif(
        not is_backend_available(),
        reason="pygraphviz not available",
    )
    def test_to_graph_connects_submodules_in_order(self):
        mod1 = DummyModule(SymTensor("a"), SymTensor("b"))
        mod2 = DummyModule(SymTensor("b"), SymTensor("c"))
        mod3 = DummyModule(SymTensor("c"), SymTensor("d"))
        seq = Sequential(mod1, mod2, mod3)

        graph, node_in, node_out = seq.to_graph()
        assert node_in == id(mod1)
        assert node_out == id(mod3)
        edges = {(str(e[0]), str(e[1])) for e in graph.edges()}
        assert (str(id(mod1)), str(id(mod2))) in edges
        assert (str(id(mod2)), str(id(mod3))) in edges
