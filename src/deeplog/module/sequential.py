#  Copyright (c) 2024-2026. KU Leuven
"""Sequential container for chaining DeepLogModules."""

from collections.abc import Iterable
from typing import cast

import torch.nn

from ..algebraic import HasStructure
from ..graph_backend import ensure_graph
from ..util import as_tuple
from .deeplog_module import DeepLogModule


class Sequential(DeepLogModule):
    """
    Chains multiple DeepLogModules so each submodule consumes the previous output.
    Functions similarly to torch.nn.Sequential.
    """

    submodules: torch.nn.ModuleList

    def __init__(self, *submodules: DeepLogModule):
        """Chain the provided submodules, flattening nested Sequential containers."""
        submodules = tuple(self._flatten_submodules(submodules))
        if len(submodules) == 0:
            raise ValueError("Sequential requires at least one submodule.")
        super().__init__(
            submodules[0].get_input_shape(), submodules[-1].get_output_shape()
        )
        self.submodules = torch.nn.ModuleList(submodules)

    def _iter_submodules(self) -> Iterable[DeepLogModule]:
        """Yield submodules as DeepLogModules.

        torch.nn.ModuleList is typed as holding generic ``Module``, so iteration
        and indexing require a cast to recover our narrower element type.
        """
        return cast(Iterable[DeepLogModule], self.submodules)

    @staticmethod
    def _flatten_submodules(
        modules: Iterable[DeepLogModule],
    ) -> Iterable[DeepLogModule]:
        """Yield submodules while flattening nested Sequential containers."""
        for module in modules:
            if isinstance(module, Sequential):
                yield from module._iter_submodules()
            else:
                yield module

    def to_graph(self, graph=None):
        """Render the composed modules into a graph, chaining edges in order."""
        graph = ensure_graph(graph)
        modules = list(self._iter_submodules())
        graph, first_node_in, first_node_out = modules[0].to_graph(graph)
        node = first_node_out
        for module in modules[1:]:
            graph, new_node_in, new_node_out = module.to_graph(graph)
            graph.add_edge(node, new_node_in, label=str(module.get_input_shape()))
            node = new_node_out
        return graph, first_node_in, node

    def forward(self, *inputs):
        """Run the input tensors through each submodule sequentially."""
        output = inputs
        for submodule in self._iter_submodules():
            output = submodule(*as_tuple(output))
        return output

    def __repr__(self):
        """Represent the sequence by chaining contained module reprs with arrows."""
        return " -> ".join(repr(m) for m in self._iter_submodules())

    def get_structure(self) -> str:
        """Return the structure of the final module in the chain."""
        last = cast(HasStructure, self.submodules[-1])
        return last.get_structure()
