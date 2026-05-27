#  Copyright (c) 2024-2026. KU Leuven
"""Pure-PyTorch generic circuit evaluator for custom operator structures."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ..module.deeplog_module import DeepLogModule
from ..shape import SymTensor


if TYPE_CHECKING:
    from ..symbol import Symbol
    from .circuit import Circuit


_LEAF = 0
_TRUE = 1
_FALSE = 2
_OP = 3
_CONST = 4


# (kind, leaf_idx, children, op_fn)
_Step = tuple[int, int, tuple[int, ...], Callable[..., torch.Tensor] | None]


class GenericCircuitModule(DeepLogModule):
    """Pure-PyTorch circuit evaluator that dispatches via operator_fns."""

    def __init__(
        self,
        steps: list[_Step],
        root_slots: list[int],
        constant_values: dict[int, float],
        input_shape: SymTensor,
        output_shape: SymTensor,
        name: str,
    ):
        """Initialize the generic circuit module with a pre-computed evaluation plan."""
        super().__init__(input_shape, output_shape)
        self._steps = steps
        self._root_slots = root_slots
        self._constant_values = constant_values
        self._name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit on input tensor x."""
        cache: list[torch.Tensor] = [torch.empty(0)] * len(self._steps)
        for i, (kind, leaf_idx, children, op_fn) in enumerate(self._steps):
            if kind == _LEAF:
                cache[i] = x[..., leaf_idx]
            elif kind == _TRUE:
                cache[i] = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype)
            elif kind == _FALSE:
                cache[i] = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
            elif kind == _CONST:
                cache[i] = torch.full(
                    x.shape[:-1],
                    self._constant_values[i],
                    device=x.device,
                    dtype=torch.float32,
                )
            else:
                assert op_fn is not None
                if len(children) == 1:
                    cache[i] = op_fn(cache[children[0]])
                else:
                    result = cache[children[0]]
                    for j in range(1, len(children)):
                        result = op_fn(result, cache[children[j]])
                    cache[i] = result

        if len(self._root_slots) == 1:
            return cache[self._root_slots[0]].unsqueeze(-1)
        return torch.stack([cache[s] for s in self._root_slots], dim=-1)

    def __repr__(self):  # noqa: D105
        return f"{self._name}({self.get_input_shape()})-> {self.get_output_shape()}"


def compile_generic(
    circuit: Circuit,
    roots: dict[int, Symbol],
    structure_override: str | None = None,
) -> DeepLogModule:
    """Compile circuit to a pure-PyTorch module using operator_fns directly."""
    if structure_override is not None:
        raise ValueError(
            "structure_override is not supported for circuits with custom operators. "
            "Define operator_fns on the AlgebraicStructure instead."
        )
    structure = circuit.algebraic_structure

    leaf_nodes = circuit.leaf_nodes
    constant_nodes = circuit.constant_nodes
    constant_values = circuit.constant_values
    root_ids = list(roots.keys())

    topo_order = list(circuit.iter_topological(root_ids))
    leaf_id_to_idx = {lid: i for i, lid in enumerate(leaf_nodes.values())}

    true_ids = {nid for role, nid in constant_nodes.items() if role == "one"}
    false_ids = {nid for role, nid in constant_nodes.items() if role == "zero"}

    # Map node_ids to sequential slot indices
    node_to_slot: dict[int, int] = {}
    steps: list[_Step] = []
    slot_constant_values: dict[int, float] = {}

    for node_id in topo_order:
        slot = len(steps)
        node_to_slot[node_id] = slot
        node = circuit.get_node(node_id)

        if node.node_type == "leaf":
            steps.append((_LEAF, leaf_id_to_idx[node_id], (), None))
        elif node_id in true_ids:
            steps.append((_TRUE, -1, (), None))
        elif node_id in false_ids:
            steps.append((_FALSE, -1, (), None))
        elif node_id in constant_values:
            slot_constant_values[slot] = constant_values[node_id]
            steps.append((_CONST, -1, (), None))
        else:
            op_fn = structure.operator_fns.get(node.node_type)
            if op_fn is None:
                raise ValueError(
                    f"No operator_fn for '{node.node_type}' "
                    f"in structure '{structure.name}'"
                )
            children = tuple(node_to_slot[c] for c in node.children)
            steps.append((_OP, -1, children, op_fn))

    root_slots = [node_to_slot[r] for r in root_ids]

    return GenericCircuitModule(
        steps=steps,
        root_slots=root_slots,
        constant_values=slot_constant_values,
        input_shape=SymTensor(list(leaf_nodes)),
        output_shape=SymTensor(list(roots.values())),
        name=circuit.name,
    )
