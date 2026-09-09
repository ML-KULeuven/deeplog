#  Copyright (c) 2024-2026. KU Leuven
"""Klay-direct lowering backend.

Used for a circuit whose structure a Klay semiring implements. An operator Klay
has no node for does not disqualify the circuit: the graph is cut there
(:mod:`deeplog.circuit.split`) and the rest still lowers here. Chains of
same-type associative ops are flattened into n-ary Klay nodes and
identity-constant children dropped, producing a shallow circuit.

Klay's only constant primitives are ``true``/``false``, so an arbitrary numeric
constant is carried as an input slot pre-filled with its value rather than as a
node — see :func:`lower_klay`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import klay
import torch
from torch import nn

from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from ..backends import circuit_roles
from ..backends import klay_semiring


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


class ConstantPrefillModule(nn.Module):
    """Expose a subset of an inner module's input slots as runtime inputs.

    The wrapped module expects ``n_total_slots`` inputs. The user-visible
    forward only takes a tensor whose last dim has size ``len(free_slot_indices)``
    — those values are scattered into the free slots, and the remaining
    slots are filled from ``constant_slot_values`` on every call.

    Useful whenever some inputs of an inner torch module are statically
    known constants (e.g. baked numeric labels, ablated features, pinned
    test inputs) and you don't want to rebuild the inner module to drop
    them.
    """

    # Declared so pyright sees the buffers as Tensors instead of
    # nn.Module's broad Tensor | Module union.
    _const_template: torch.Tensor
    _free_idx_buf: torch.Tensor

    def __init__(
        self,
        inner: nn.Module,
        n_total_slots: int,
        free_slot_indices: tuple[int, ...],
        constant_slot_values: dict[int, float],
    ):
        """Wrap ``inner`` so only ``free_slot_indices`` are user inputs."""
        super().__init__()
        self._inner = inner
        self._n_total = n_total_slots
        self._free = free_slot_indices
        # Stash the constant values as a dense tensor template so the
        # forward pass is a couple of indexed writes.
        const_template = torch.zeros(n_total_slots)
        for slot, value in constant_slot_values.items():
            const_template[slot] = value
        # Register as buffer so it follows .to(device) / .cuda()
        # without needing manual moves.
        self.register_buffer("_const_template", const_template)
        self._free_idx = torch.tensor(free_slot_indices, dtype=torch.long)
        self.register_buffer("_free_idx_buf", self._free_idx)

    def forward(self, x):
        """Scatter ``x`` into the free slots, prefill the rest with constants."""
        # x has shape (..., len(free_slots)) per the WrappedModule's
        # vmap convention. Build a (..., n_total) tensor by starting
        # from the constants and overwriting free slots. Out-of-place
        # ops only — vmap doesn't support in-place index_copy_.
        out_shape = x.shape[:-1] + (self._n_total,)
        full = self._const_template.to(dtype=x.dtype).expand(out_shape)
        idx = self._free_idx_buf.expand(out_shape[:-1] + (self._free_idx_buf.shape[0],))
        return self._inner(full.scatter(-1, idx, x))


def lower_klay(
    circuit: Circuit,
    roots: dict[int, Symbol],
    frontier: Mapping[int, Symbol] | None = None,
) -> DeepLogModule:
    """Lower a circuit to a Klay-backed torch module.

    Every reachable leaf gets an input slot. So does every reachable
    arbitrary-value constant node (``0.6`` in ``times(p, 0.6)``), because Klay
    has no constant primitive beyond ``true``/``false`` — but those slots are
    pre-filled with their values and dropped from the module's input shape, so
    they never surface as runtime inputs. Klay evaluates the resulting circuit
    arithmetically, so a constant node shared by several parents (they are
    deduplicated by value) multiplies once per use, as written.
    """
    klay_circuit = klay.Circuit()
    node_to_klay: dict[int, klay.NodePtr] = {}
    root_ids = list(roots.keys())

    # Map leaf nodes to klay literals. Track the signed literal id alongside
    # the NodePtr so we can negate via klay's documented `literal_node(-id)`
    # path; klay has no negate primitive for compound nodes.
    leaf_nodes = circuit.reachable_leaves(root_ids, frontier)
    leaf_ids = list(leaf_nodes.values())
    lit_id_of: dict[int, int] = {}
    for i, leaf_id in enumerate(leaf_ids):
        lit_id_of[leaf_id] = i + 1
        node_to_klay[leaf_id] = klay_circuit.literal_node(i + 1)

    # The identities are the constants klay has primitives for.
    zero_id = circuit.zero_node
    one_id = circuit.one_node
    if zero_id is not None:
        node_to_klay[zero_id] = klay_circuit.false_node()
    if one_id is not None:
        node_to_klay[one_id] = klay_circuit.true_node()

    # Every other constant takes a slot of its own, after the leaves so that
    # leaf slot indices are unaffected. Each is pre-filled below.
    bound = {zero_id, one_id}
    constants = {
        symbol: node_id
        for symbol, node_id in circuit.reachable_constants(root_ids, frontier).items()
        if node_id not in bound
    }
    constant_slot_values: dict[int, float] = {}
    for j, const_id in enumerate(constants.values()):
        slot = len(leaf_ids) + j
        lit_id_of[const_id] = slot + 1
        node_to_klay[const_id] = klay_circuit.literal_node(slot + 1)
        constant_slot_values[slot] = circuit.constant_values[const_id]

    # By role, not by name: the structure's product is Klay's ``and`` node and
    # its sum Klay's ``or``, however they are spelled.
    roles = circuit_roles(circuit.structure)
    _OR_TYPES = frozenset({roles["sum"]} if "sum" in roles else ())
    _AND_TYPES = frozenset({roles["product"]} if "product" in roles else ())
    negation = roles.get("negation")
    absorbed, flat_children = circuit.flatten_chains(
        root_ids,
        frontier=frontier,
        chain_groups=[
            (_OR_TYPES, frozenset({zero_id} if zero_id is not None else ())),
            (_AND_TYPES, frozenset({one_id} if one_id is not None else ())),
        ],
    )

    for node_id in circuit.iter_topological(root_ids, frontier):
        if node_id in node_to_klay or node_id in absorbed:
            continue

        node = circuit._get_node(node_id)

        if node.node_type in _OR_TYPES:
            klay_children = [node_to_klay[c] for c in flat_children[node_id]]
            if not klay_children:
                node_to_klay[node_id] = klay_circuit.false_node()
            elif len(klay_children) == 1:
                node_to_klay[node_id] = klay_children[0]
            else:
                node_to_klay[node_id] = klay_circuit.or_node(klay_children)
        elif node.node_type in _AND_TYPES:
            klay_children = [node_to_klay[c] for c in flat_children[node_id]]
            if not klay_children:
                node_to_klay[node_id] = klay_circuit.true_node()
            elif len(klay_children) == 1:
                node_to_klay[node_id] = klay_children[0]
            else:
                node_to_klay[node_id] = klay_circuit.and_node(klay_children)
        elif node.node_type == negation:
            (child_id,) = node.children
            if child_id not in lit_id_of:
                child_type = circuit._get_node(child_id).node_type
                raise ValueError(
                    f"Klay backend can only negate literals, but '{node.node_type}' "
                    f"is applied to a '{child_type}' node (circuit node {child_id}). "
                    f"Knowledge-compile it first "
                    f"(deeplog.circuit.knowledge_compile), which produces a "
                    f"d-DNNF, or push negation down to leaves before reaching "
                    f"Klay (glab #138)."
                )
            negated_id = -lit_id_of[child_id]
            lit_id_of[node_id] = negated_id
            node_to_klay[node_id] = klay_circuit.literal_node(negated_id)
        elif node.node_type in ("leaf", "constant"):
            pass
        else:
            raise ValueError(
                f"Klay has no node for '{node.node_type}' in structure "
                f"'{circuit.structure.name}': it builds a circuit out of the "
                f"product, sum and complement, and every other operator is applied "
                f"to lowered operands instead (deeplog.circuit.split)."
            )

    for root_id in root_ids:
        klay_circuit.set_root(node_to_klay[root_id])

    torch_module = klay_circuit.to_torch_module(*klay_semiring(circuit.structure))
    input_symbols = list(leaf_nodes)
    # The constant slots are named like any other atom: the symbol the constant
    # was written as. They are pre-filled, so the names are internal only.
    input_symbols += list(constants)
    return _wrap_with_constant_prefill(
        torch_module,
        input_symbols,
        list(roots.values()),
        constant_slot_values,
        circuit.name,
    )


def _wrap_with_constant_prefill(
    torch_module: torch.nn.Module,
    input_symbols: list[Symbol],
    output_symbols: list[Symbol],
    constant_slot_values: dict[int, float],
    name: str,
) -> DeepLogModule:
    """Wrap a packed-tensor torch module, baking its constant input slots.

    ``input_symbols`` names every input slot of ``torch_module`` (one slot per
    last-dim entry, in order). Slots listed in ``constant_slot_values`` are
    pre-filled by a :class:`ConstantPrefillModule`; the rest
    stay runtime inputs, surfaced — in their original order — as the returned
    module's input shape. With no constants to bake the module is wrapped as-is.
    """
    n_total_slots = len(input_symbols)
    if not constant_slot_values:
        return WrappedModule(
            torch_module,
            SymTensor(input_symbols),
            SymTensor(output_symbols),
            name=name,
            vmap=True,
        )
    free_slot_indices = [
        i for i in range(n_total_slots) if i not in constant_slot_values
    ]
    free_input_symbols = [input_symbols[i] for i in free_slot_indices]
    return WrappedModule(
        ConstantPrefillModule(
            torch_module,
            n_total_slots=n_total_slots,
            free_slot_indices=tuple(free_slot_indices),
            constant_slot_values=constant_slot_values,
        ),
        SymTensor(free_input_symbols),
        SymTensor(output_symbols),
        name=name,
        vmap=True,
    )
