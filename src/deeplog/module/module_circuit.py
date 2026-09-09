#  Copyright (c) 2024-2026. KU Leuven
"""ModuleCircuit implementation and helpers."""

import operator
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from graphlib import TopologicalSorter
from typing import cast

import torch
import torch.nn

from ..shape import Shape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..util import as_tuple
from .deeplog_module import DeepLogModule
from .reshape import construct_transformation


def _get_missing_transformations(
    input_tensors: tuple[SymTensor, ...],
    modules: Iterable[DeepLogModule],
    required_shapes: Iterable[SymTensor] = (),
) -> list[DeepLogModule]:
    """Determine extra transformation modules needed so every symbolic input has a producer."""
    all_inputs: set[SymTensor] = reduce(
        operator.or_, (set(as_tuple(m.get_input_shape())) for m in modules)
    ) | set(required_shapes)
    all_outputs: set[SymTensor] = reduce(
        operator.or_, (set(as_tuple(m.get_output_shape())) for m in modules)
    ) | set(input_tensors)
    missing_symtensors = all_inputs - all_outputs

    all_transformation_inputs: list[tuple[list[SymTensor], SymTensor]] = []
    transformations = []
    for symtensor in missing_symtensors:
        transformation_inputs: list[SymTensor] = []
        for symbol in symtensor:
            for output in all_outputs:
                if symbol in output:
                    transformation_inputs.append(output)
                    break
            else:
                raise ValueError(
                    f"Symbol {symbol} required by {symtensor} is not produced by "
                    f"any module or provided as an external input."
                )
        all_transformation_inputs.append((transformation_inputs, symtensor))

    for inputs, tensor in all_transformation_inputs:
        if len(inputs) == 1:
            transformations.append(construct_transformation(inputs[0], tensor))
        if len(inputs) > 1:
            transformations.append(construct_transformation(tuple(inputs), tensor))

    return transformations


def _external_shape(symbol: Symbol, modules: Iterable[DeepLogModule]) -> SymTensor:
    """The shape to declare for ``symbol`` as an external input of a circuit.

    A one-symbol tensor has two spellings — ``SymTensor([symbol])`` and the
    scalar ``SymTensor(symbol)``, which differ because a symbol is itself a
    tuple (the ambiguity :class:`~deeplog.shape.SymTensor` documents) — and both
    occur among modules. Declaring the one its consumer declared is what lets the
    circuit hand the caller's tensor straight to it; declaring the other earns a
    transformation that exists only to reconcile the two spellings, and one of
    them changes the tensor's rank. Where nothing consumes the symbol alone
    there is no consumer to agree with, and the one-element spelling stands.
    """
    for module in modules:
        for shape in as_tuple(module.get_input_shape()):
            if tuple(get_all_symbols(shape)) == (symbol,):
                return shape
    return SymTensor([symbol])


def build_module_graph(modules: list[DeepLogModule]) -> dict[int, set[int]]:
    """Build a predecessor map from module index to the indices producing its inputs."""
    inputs_by_shape: dict[Shape, list[int]] = defaultdict(list)
    outputs_by_shape: dict[Shape, list[int]] = defaultdict(list)

    for i, module in enumerate(modules):
        for s in as_tuple(module.get_input_shape()):
            inputs_by_shape[s].append(i)
        for s in as_tuple(module.get_output_shape()):
            outputs_by_shape[s].append(i)

    predecessors: dict[int, set[int]] = {i: set() for i in range(len(modules))}
    for s, producers in outputs_by_shape.items():
        for j in inputs_by_shape.get(s, []):
            predecessors[j].update(p for p in producers if p != j)

    return predecessors


class ModuleCircuit(DeepLogModule):
    """Module that evaluates a circuit of DeepLogModules."""

    _sub_modules: torch.nn.ModuleList

    def __init__(
        self,
        modules: Iterable[DeepLogModule],
        output_shape: Shape,
    ):
        """Assemble a circuit of modules and compute ordering/transformations."""
        all_modules = list(modules)

        input_symbols = set(
            get_all_symbols(m.get_input_shape() for m in all_modules)
        ) - set(get_all_symbols(m.get_output_shape() for m in all_modules))
        input_tensors = tuple(_external_shape(i, all_modules) for i in input_symbols)
        transform_modules = _get_missing_transformations(
            input_tensors, all_modules, as_tuple(output_shape)
        )
        super().__init__(input_tensors, output_shape)
        all_modules += transform_modules

        all_outputs: set[SymTensor] = set(input_tensors) | {SymTensor([])}
        for m in all_modules:
            all_outputs |= set(as_tuple(m.get_output_shape()))
        all_inputs: set[SymTensor] = set()
        for m in all_modules:
            all_inputs |= set(as_tuple(m.get_input_shape()))
        missing = all_inputs - all_outputs
        if missing:
            raise ValueError(
                f"The following tensors are required as input but have no producer: "
                f"{missing}"
            )

        sorted_indices = list(
            TopologicalSorter(build_module_graph(all_modules)).static_order()
        )
        all_modules = [all_modules[i] for i in sorted_indices]
        self._sub_modules = torch.nn.ModuleList(all_modules)

    def forward(self, *inputs: torch.Tensor):
        """Evaluate the circuit by propagating cacheable tensors through each submodule."""
        cache = {}
        # With no runtime inputs every leaf was baked to a constant, so there is
        # no tensor to read the batch size from; evaluate a single constant row.
        batch = inputs[0].shape[0] if len(inputs) > 0 else 1
        cache[SymTensor([])] = torch.zeros(batch, 0)
        for shape, tensor in zip(as_tuple(self.get_input_shape()), inputs, strict=True):
            cache[shape] = tensor

        for module in cast(Iterable[DeepLogModule], self._sub_modules):
            output = module(*(cache[i] for i in as_tuple(module.get_input_shape())))
            for shape, o in zip(
                as_tuple(module.get_output_shape()), as_tuple(output), strict=True
            ):
                cache[shape] = o

        if isinstance(self.get_output_shape(), tuple):
            return tuple(cache[s] for s in self.get_output_shape())
        return cache[self.get_output_shape()]


def compose_modules(
    modules: Iterable[DeepLogModule],
    output_shape: Shape,
) -> DeepLogModule:
    """Compose modules into one module producing ``output_shape``.

    A single module is returned as it is; several become a
    :class:`ModuleCircuit`, which orders them by what each one's inputs need
    and inserts the transformations between them.
    """
    modules = list(modules)
    if len(modules) == 0:
        raise ValueError("At least one module is required to compose_modules.")
    if len(modules) == 1:
        return modules[0]
    # TODO: potential optimization. If dependency graph is linear, return a sequential.
    return ModuleCircuit(modules, output_shape)
