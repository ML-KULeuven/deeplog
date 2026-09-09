#  Copyright (c) 2024-2026. KU Leuven
"""Pure-PyTorch generic circuit evaluator for custom operator structures."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ...formula.deeplogformulafactory import DeepLogFormulaFactory
from ...module.deeplog_module import DeepLogModule
from ...shape import SymTensor
from ...symbol import is_structure_wrapped
from ...symbol import unwrap_structure


if TYPE_CHECKING:
    from ...algebraic import AlgebraicStructure
    from ...formula.ast import CircuitNode
    from ...symbol import Symbol
    from ..circuit import Circuit


_LEAF = 0
_OP = 1
_CONST = 2


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
            elif kind == _CONST:
                cache[i] = torch.full(
                    x.shape[:-1],
                    self._constant_values[i],
                    device=x.device,
                    dtype=x.dtype,
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


def lower_generic(
    circuit: Circuit,
    roots: dict[int, Symbol],
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Lower circuit to a pure-PyTorch module using operator_fns directly."""
    root_ids = list(roots.keys())
    leaf_nodes = circuit.reachable_leaves(root_ids, frontier)

    algebra = _StepAlgebra(circuit.structure, leaf_nodes)
    node_to_slot = circuit.fold(root_ids, algebra, frontier=frontier)

    root_slots = [node_to_slot[r] for r in root_ids]

    return GenericCircuitModule(
        steps=algebra.steps,
        root_slots=root_slots,
        constant_values=algebra.constant_values,
        input_shape=SymTensor(list(leaf_nodes)),
        output_shape=SymTensor(list(roots.values())),
        name=circuit.name,
    )


class _StepAlgebra(DeepLogFormulaFactory[int]):
    """Flatten a circuit's nodes into an evaluation plan, one step per node.

    The circuit fragment of the formula algebra
    (:func:`~deeplog.circuit.fold.fold_circuit`), carried by the slot index a
    node's value lands in. Every node appends one step, so the fold's
    children-before-parents order *is* the evaluation order.
    """

    def __init__(
        self, structure: AlgebraicStructure, leaf_nodes: dict[Symbol, int]
    ) -> None:
        """Plan over ``structure``'s operators, reading inputs from ``leaf_nodes``."""
        self._structure = structure
        self._input_index = {symbol: index for index, symbol in enumerate(leaf_nodes)}
        #: The plan, in evaluation order; a step's index is its slot.
        self.steps: list[_Step] = []
        #: Slots pre-filled with a numeric constant rather than read from input.
        self.constant_values: dict[int, float] = {}

    def create_atom(self, atom: Symbol) -> int:
        """A slot reading ``atom``: an input, or a value baked in.

        An identity is baked in like any other constant: the structure resolves
        the symbol naming it to the value it has *there*, which ``one`` in log
        space and ``one`` in probability space do not share.
        """
        index = self._input_index.get(atom)
        if index is not None:
            return self._step((_LEAF, index, (), None))
        name = unwrap_structure(atom) if is_structure_wrapped(atom) else atom
        value = self._structure.get_constant_value(name)
        if value is None:
            raise ValueError(
                f"Circuit atom {atom} is neither an input of this lowering nor a "
                f"constant of structure '{self._structure.name}'."
            )
        slot = self._step((_CONST, -1, (), None))
        self.constant_values[slot] = value
        return slot

    def create_unary_node(self, operator: str, operand: int) -> int:
        """A slot applying ``operator`` to ``operand``."""
        return self._step((_OP, -1, (operand,), self._operator_fn(operator)))

    def create_binary_node(self, operator: str, lhs: int, rhs: int) -> int:
        """A slot applying ``operator`` to ``lhs`` and ``rhs``."""
        return self._step((_OP, -1, (lhs, rhs), self._operator_fn(operator)))

    def _step(self, step: _Step) -> int:
        self.steps.append(step)
        return len(self.steps) - 1

    def _operator_fn(self, operator: str) -> Callable[..., torch.Tensor]:
        operator_fn = self._structure.operator_fns.get(operator)
        if operator_fn is None:
            raise ValueError(
                f"No operator_fn for '{operator}' in structure '{self._structure.name}'"
            )
        return operator_fn

    # A circuit has no node for the three eliminators below.

    def create_transformation(self, structure: str, child: int) -> int:
        """Unreachable: a circuit holds no cross-structure boundary."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"transformation, only the structure it was built in."
        )

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[int],
        child: int,
    ) -> int:
        """Unreachable: a circuit holds no aggregation."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"aggregation, only the ground formula one was expanded into."
        )

    def embed_circuit(self, node: CircuitNode, children: tuple[int, ...] = ()) -> int:
        """Unreachable: the fold is already inside a circuit."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; it is already inside "
            f"the circuit a lump would embed."
        )
