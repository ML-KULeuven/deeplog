#  Copyright (c) 2024-2026. KU Leuven
"""Backend selection and dispatch for knowledge compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...symbol import unwrap_structure
from ...variable import indicated_values
from ..circuit import Circuit


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...algebraic import AlgebraicStructure
    from ...symbol import Symbol
    from ...variable import VariableAtoms


def knowledge_compile(
    circuit: Circuit,
    roots: list[int],
    *,
    variables: VariableAtoms | None = None,
    structure: str | AlgebraicStructure | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
) -> tuple[Circuit, dict[int, int]]:
    """Rewrite ``roots`` into a deterministic, decomposable circuit.

    Returns ``(new_circuit, node_map)`` — the same contract as
    :func:`~deeplog.circuit.transform.transform_circuit`, and composable with it.
    All roots are compiled together, so what they share stays shared.

    ``structure`` is the algebra to emit into, the source's own by default. The
    rewrite preserves roles, so emitting into another algebra applies that
    algebra's product, sum and complement to the same diagram — which is what
    transforming the compiled circuit afterwards would have done, without
    building it. ``leaf_mapping`` renames each atom on the way, as that transform
    would have.

    ``variables`` says where each variable occurs. The compiler follows from
    that rather than from a caller's choice: a plain SDD has only two-valued
    variables, so a variable two or more of whose values the formula *reaches*
    needs the multi-valued one, or two of those atoms could hold at once. A
    variable the formula reaches at most one value of *is* a plain SDD variable,
    its remaining values being the one residual an SDD already reads as the
    complement.
    """
    target = Circuit(structure if structure is not None else circuit.structure)
    # Imported at the call: each compiler is an optional dependency, so only the
    # one actually used has to be installed.
    if variables and _reaches_two_values(circuit, roots, variables):
        from .mvsdd import compile_mvsdd

        return compile_mvsdd(
            circuit, roots, variables, target=target, leaf_mapping=leaf_mapping
        )
    from . import sdd

    return sdd.compile_sdd(circuit, roots, target=target, leaf_mapping=leaf_mapping)


def _reaches_two_values(
    circuit: Circuit, roots: list[int], variables: VariableAtoms
) -> bool:
    """Whether ``roots`` reach two values of one variable, which SDD cannot hold."""
    asserted = indicated_values(variables)
    reached: dict[Symbol, set[int]] = {}
    for leaf_symbol in circuit.reachable_leaves(roots):
        for variable, position in asserted.get(unwrap_structure(leaf_symbol), ()):
            reached.setdefault(variable.name, set()).add(position)
    return any(len(positions) > 1 for positions in reached.values())
