"""
A module containing the DeepLogModule class and functions for constructing DeepLogModules.
"""

#  Copyright (c) 2024-2026. KU Leuven
from __future__ import annotations

import os
from typing import Protocol

from torch import nn

from ..graph_backend import ensure_graph
from ..shape import Shape
from ..shape import ShapeMismatchException
from ..shape import correct_shape
from ..util import as_tuple


def _env_validation_enabled() -> bool:
    """Return whether shape validation should be wired into new instances."""
    flag = os.getenv("DEEPLOG_VALIDATE", "1")
    return str(flag).lower() not in {"0", "false", "off"}


def _validate_input_shape(module: DeepLogModule, args: tuple) -> None:
    """Forward-pre-hook: assert ``args`` matches the declared input shape."""
    if not correct_shape(args, module.get_input_shape()):  # pyright: ignore[reportArgumentType]
        raise ShapeMismatchException(module.get_input_shape(), args, (module, "input"))


def _validate_output_shape(module: DeepLogModule, args: tuple, output: object) -> None:
    """Forward-hook: assert ``output`` matches the declared output shape."""
    out = as_tuple(output)
    if not correct_shape(out, module.get_output_shape()):  # pyright: ignore[reportArgumentType]
        raise ShapeMismatchException(module.get_output_shape(), out, (module, "output"))  # pyright: ignore[reportArgumentType]


class SupportsToModule(Protocol):
    """A protocol that specifies that this can be turned into a DeepLogModule"""

    def to_module(self) -> DeepLogModule:
        """Return a DeepLogModule that implements the functionality of this object."""
        ...


class DeepLogModule(nn.Module, SupportsToModule):
    """
    A class that extends Torch modules to include information about its input and output shape.
    """

    structure: str

    def __init__(self, input_shape: Shape, output_shape: Shape):
        """Capture the symbolic input and output shapes for the module.

        Registers shape-validation forward hooks when ``DEEPLOG_VALIDATE``
        is enabled (the default). Disable via ``DEEPLOG_VALIDATE=0`` in the
        environment.

        Validation adds ~1.5 µs per forward call (a small constant for the
        input/output ``correct_shape`` checks). For typical workloads where
        each forward does real computation, this is well under 1% overhead;
        disabling matters only in tight micro-benchmarks over many trivial
        modules. The env var is consulted in ``__init__``, so changing it
        after import only affects newly-constructed instances.
        """
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape
        if _env_validation_enabled():
            self.register_forward_pre_hook(_validate_input_shape)
            self.register_forward_hook(_validate_output_shape)

    def get_input_shape(self) -> Shape:
        """
        Returns the input shape of this module.
        """
        return self._input_shape

    def get_output_shape(self) -> Shape:
        """
        Returns the output shape of this module.
        """
        return self._output_shape

    def __repr__(self):
        """Render the module with its input and output symbolic shapes."""
        return f"{self.__class__.__name__}({','.join(str(i) for i in as_tuple(self.get_input_shape()))}) -> {self.get_output_shape()}"

    def to_graph(self, graph=None):
        """Add this module to the provided Graphviz graph and return connection nodes."""
        graph = ensure_graph(graph)
        graph.add_node(id(self), label=repr(self), shape="box")

        return graph, id(self), id(self)

    def replace_input_shape(self, new_input_shape: Shape) -> None:
        """Replace the declared input shape without adding transformations."""
        self._input_shape = new_input_shape

    def replace_output_shape(self, new_output_shape: Shape) -> None:
        """Replace the declared output shape without adding transformations."""
        self._output_shape = new_output_shape

    def to_module(self) -> DeepLogModule:
        """Returns the module itself"""
        return self
