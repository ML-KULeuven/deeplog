#  Copyright (c) 2024-2026. KU Leuven
"""Compile dispatcher + shared helpers used by the Klay/PySDD/MV-SDD backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import klay
import torch

from ..generic import compile_generic


if TYPE_CHECKING:
    from ...module import DeepLogModule
    from ...symbol import Symbol
    from ..circuit import Circuit


SEMIRINGS = {
    "boolean": ("godel", False),
    "probability": ("real", False),
    "logprobability": ("log", False),
}

KLAY_OPS = frozenset({"and", "or", "not", "times", "plus", "negate"})


def cast_to_float(module, input):
    """Pre-hook to cast inputs to float32."""
    return tuple(i.to(torch.float32) for i in input)


def klay_to_torch_module(
    klay_circuit: klay.Circuit,
    semiring_key: str,
    semiring_fallback: tuple[str, bool] = ("real", False),
) -> torch.nn.Module:
    """Final-phase helper: klay graph → torch nn.Module.

    Looks up the semiring tuple (falling back to ``semiring_fallback`` if
    ``semiring_key`` isn't a known SEMIRINGS entry), produces the torch
    module, and attaches the float-cast forward pre-hook.
    """
    torch_module = klay_circuit.to_torch_module(
        *SEMIRINGS.get(semiring_key, semiring_fallback)
    )
    torch_module.register_forward_pre_hook(cast_to_float)
    return torch_module


def walk_and_combine(
    circuit: Circuit, roots: dict[int, Symbol], node_map: dict[int, object]
) -> None:
    """Topologically populate ``node_map`` by combining children with ``&``/``|``/``~``.

    Leaves and constants must already be seeded into ``node_map`` by the caller.
    Dispatches on ``node_type`` (``and``/``times`` → ``&``, ``or``/``plus`` → ``|``,
    ``not``/``negate`` → ``~``) and skips leaf/zero/one placeholders. Used by the
    PySDD and MV-SDD backends, whose backend node types both expose the bitwise
    operators.
    """
    for node_id in circuit.iter_topological(list(roots.keys())):
        if node_id in node_map:
            continue
        node = circuit.get_node(node_id)
        children = [node_map[c] for c in node.children]
        if node.node_type in ("and", "times"):
            result = children[0]
            for child in children[1:]:
                result = result & child  # pyright: ignore[reportOperatorIssue]
            node_map[node_id] = result
        elif node.node_type in ("or", "plus"):
            result = children[0]
            for child in children[1:]:
                result = result | child  # pyright: ignore[reportOperatorIssue]
            node_map[node_id] = result
        elif node.node_type in ("not", "negate"):
            node_map[node_id] = ~children[0]  # pyright: ignore[reportOperatorIssue]
        elif node.node_type in ("leaf", "zero", "one"):
            pass
        else:
            raise ValueError(f"Unknown node_type: {node.node_type}")


def _is_klay_compatible(circuit: Circuit) -> bool:
    """Return True if the circuit can be compiled by the Klay/semiring fast path.

    The Klay backend evaluates ``and``/``or``/``not`` (and their semiring
    aliases) via a fixed semiring; it does not consult the structure's
    ``operator_fns``. So a plain :class:`AlgebraicStructure` carrying custom
    operator functions (e.g. a fuzzy ``or`` of ``a + b - a*b``) must NOT take
    this path — otherwise its ``operator_fns`` would be silently replaced by
    the semiring sum/product. Only genuine :class:`Semiring`/:class:`Algebra`
    structures (BOOLEAN/PROBABILITY/LOGPROBABILITY and friends), whose
    semantics match the semiring the Klay backend implements, are eligible;
    everything else falls through to :func:`compile_generic`, which honors the
    custom ``operator_fns`` directly.
    """
    from ...algebraic import Semiring

    return (
        isinstance(circuit.algebraic_structure, Semiring)
        and circuit.algebraic_structure.operators.issubset(KLAY_OPS)
        and not circuit.constant_values
    )


def to_module(
    circuit: Circuit,
    roots: dict[int, Symbol],
    structure_override: str | None = None,
    categoricals: dict[Symbol, tuple[str, int | str]] | None = None,
    labels: dict[Symbol, Symbol] | None = None,
) -> DeepLogModule:
    """Compile a Circuit to a DeepLogModule.

    Args:
        circuit: The circuit to compile.
        roots: A dictionary mapping node IDs to output names.
        structure_override: If provided, compile using this structure's
                           semiring instead of the circuit's own structure.
        categoricals: For circuits with annotated-disjunction (categorical)
                     leaves, maps each leaf goal Symbol to ``(cat_id, value)``
                     where ``value`` is an int (the AD branch index) or
                     ``"none"`` (the residual outcome). When non-empty,
                     compilation routes through the MV-SDD backend.
        labels: Per-leaf annotations from the engine
                (``EngineResult.labels``). The MV-SDD backend uses these
                to recognise leaves whose label is a numeric constant and
                bake those constants into the AC, so only leaves with
                non-numeric (typically neural-network-backed) labels
                remain as runtime input slots.

    Returns:
        A DeepLogModule wrapping the circuit as a torch module.
    """
    if circuit.deterministic:
        # Knowledge-compilation backends produce deterministic, canonical
        # ACs. MV-SDD is the categorical-aware variant (one multi-valued
        # variable per AD), PySDD is the boolean specialisation; pick the
        # right one by whether any AD leaves are present.
        if categoricals:
            from .mvsdd import compile_mvsdd

            return compile_mvsdd(
                circuit, roots, structure_override, categoricals, labels or {}
            )
        from .pysdd import compile_pysdd

        return compile_pysdd(circuit, roots, structure_override)
    if not _is_klay_compatible(circuit):
        return compile_generic(circuit, roots, structure_override)
    from .klay import compile_klay

    return compile_klay(circuit, roots, structure_override)
