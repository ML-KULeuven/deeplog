#  Copyright (c) 2024-2026. KU Leuven
"""Pure-PyTorch generic circuit evaluator for custom operator structures.

The evaluator works a layer at a time. A node's depth is one more than its
deepest operand's, and the nodes of one depth that apply one operator function
are a group, which one call of that function computes, on the group's operands
gathered along the last dimension: an operator function applies element by
element (:data:`~deeplog.algebraic.OperatorFn`). The input, the constants and
the values of each group are read in one operation each, and split among the
groups and the output that read them.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ...formula.deeplogformulafactory import DeepLogFormulaFactory
from ...module.deeplog_module import DeepLogModule
from ...module.reshape import _per_source
from ...shape import SymTensor
from ...symbol import is_structure_wrapped
from ...symbol import unwrap_structure
from ..fold import fold_circuit


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

# The chunks a group's operand, or the output, is put together from, as (source,
# chunk) pairs, and the slice of the indices ordering their columns, if any.
_Read = tuple[tuple[tuple[int, int], ...], slice | None]

# The slice of the indices a source is read at and the sizes of the chunks the
# read is split into; ``None`` where its one chunk is the source as it is.
_Gather = tuple[slice, tuple[int, ...]] | None


class GenericCircuitModule(DeepLogModule):
    """Pure-PyTorch circuit evaluator that dispatches via operator_fns, a group at a time."""

    _indices: torch.Tensor

    def __init__(
        self,
        steps: list[_Step],
        root_slots: list[int],
        constant_values: dict[int, float],
        input_shape: SymTensor,
        output_shape: SymTensor,
        name: str,
    ):
        """Plan the evaluation of ``steps``, whose ``root_slots`` are its output."""
        super().__init__(input_shape, output_shape)
        constants, groups, output, gathers, indices = _plan(
            steps, root_slots, constant_values, int(input_shape.array.size)
        )
        self._constants = constants
        self._groups = groups
        self._output = output
        self._gathers = gathers
        self.register_buffer(
            "_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        self._name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the circuit on input tensor x."""
        constants = torch.tensor(self._constants, dtype=x.dtype, device=x.device)
        chunks = [
            self._split(0, x),
            self._split(1, constants.expand(*x.shape[:-1], -1)),
        ]
        for operator_fn, reads in self._groups:
            operands = [self._assemble(chunks, read) for read in reads]
            chunks.append(self._split(len(chunks), operator_fn(*operands)))
        return self._assemble(chunks, self._output)

    def _split(self, source: int, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The chunks of ``value``, the values of ``source``, that are read."""
        gather = self._gathers[source]
        if gather is None:
            return (value,)
        at, sizes = gather
        if not sizes:
            return ()
        read = value.index_select(-1, self._indices[at])
        return read.split(sizes, dim=-1) if len(sizes) > 1 else (read,)

    def _assemble(
        self, chunks: list[tuple[torch.Tensor, ...]], read: _Read
    ) -> torch.Tensor:
        """The values ``read`` puts together from ``chunks``."""
        pieces, order = read
        parts = [chunks[source][chunk] for source, chunk in pieces]
        value = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return value if order is None else value.index_select(-1, self._indices[order])

    def __repr__(self):  # noqa: D105
        return f"{self._name}({self.get_input_shape()})-> {self.get_output_shape()}"


def _plan(
    steps: list[_Step],
    root_slots: list[int],
    constant_values: dict[int, float],
    inputs: int,
) -> tuple[
    tuple[float, ...],
    tuple[tuple[Callable[..., torch.Tensor], tuple[_Read, ...]], ...],
    _Read,
    tuple[_Gather, ...],
    list[int],
]:
    """The groups of ``steps``, in evaluation order, and how each source is read.

    Source 0 is the input, of ``inputs`` columns, source 1 the constants, and
    each group a source after them. Returns the constants' values, each group's
    operator function and the reads of its operands, the read of the output,
    each source's gather, and the indices the reads and the gathers slice.
    """
    # Each step's source and its position there.
    place = [(0, 0)] * len(steps)
    constants: list[float] = []
    depth = [0] * len(steps)
    grouped: dict[tuple[int, Callable[..., torch.Tensor], int], list[int]] = {}
    for i, (kind, leaf_idx, children, op_fn) in enumerate(steps):
        if kind == _LEAF:
            place[i] = (0, leaf_idx)
        elif kind == _CONST:
            place[i] = (1, len(constants))
            constants.append(constant_values[i])
        else:
            assert op_fn is not None
            depth[i] = 1 + max(depth[child] for child in children)
            grouped.setdefault((depth[i], op_fn, len(children)), []).append(i)
    layers = sorted(grouped.items(), key=lambda item: item[0][0])
    for group, (_, members) in enumerate(layers):
        for position, i in enumerate(members):
            place[i] = (2 + group, position)

    sizes = [inputs, len(constants), *(len(members) for _, members in layers)]
    # The positions each read of a source takes there, in the order of the reads.
    taken: list[list[list[int]]] = [[] for _ in sizes]
    indices: list[int] = []

    def indexed(values: list[int]) -> slice:
        indices.extend(values)
        return slice(len(indices) - len(values), len(indices))

    def read(slots: list[int]) -> _Read:
        by_source, order = _per_source(
            [place[slot][0] for slot in slots], [place[slot][1] for slot in slots]
        )
        pieces = []
        for source, positions in by_source:
            pieces.append((source, len(taken[source])))
            taken[source].append(positions)
        return tuple(pieces), None if order is None else indexed(order)

    groups = tuple(
        (op_fn, tuple(read([steps[i][2][j] for i in members]) for j in range(arity)))
        for (_, op_fn, arity), members in layers
    )
    output = read(root_slots)
    gathers: list[_Gather] = []
    for size, reads in zip(sizes, taken, strict=True):
        if len(reads) == 1 and reads[0] == list(range(size)):
            gathers.append(None)
        else:
            at = indexed([position for positions in reads for position in positions])
            gathers.append((at, tuple(len(positions) for positions in reads)))
    return tuple(constants), groups, output, tuple(gathers), indices


def lower_generic(
    circuit: Circuit,
    roots: dict[int, Symbol],
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Lower circuit to a pure-PyTorch module using operator_fns directly."""
    root_ids = list(roots.keys())
    leaf_nodes = circuit._reachable_leaves(root_ids, frontier)

    algebra = _StepAlgebra(circuit.structure, leaf_nodes)
    node_to_slot = fold_circuit(circuit, root_ids, algebra, frontier=frontier)

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
    children-before-parents order puts each step after its operands'.
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

    # A circuit has no node for the four eliminators below.

    def create_transformation(self, structure: str, child: int) -> int:
        """Unreachable: a circuit holds no cross-structure boundary."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"transformation, only the structure it was built in."
        )

    def create_aggregation_op(self, operation: str, operands: Sequence[int]) -> int:
        """Unreachable: a circuit holds no aggregation."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; a circuit has no "
            f"aggregation, only operators."
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

    def embed_circuit(self, node: CircuitNode) -> int:
        """Unreachable: the fold is already inside a circuit."""
        raise NotImplementedError(
            f"{type(self).__name__} folds circuit nodes; it is already inside "
            f"the circuit a lump would embed."
        )
