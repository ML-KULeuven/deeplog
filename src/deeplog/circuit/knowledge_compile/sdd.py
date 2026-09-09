#  Copyright (c) 2024-2026. KU Leuven
"""Knowledge compilation via PySDD.

Builds an SDD over the circuit's leaves and reads the canonical result back out
as a circuit, which is therefore deterministic and decomposable.

Every leaf is a two-valued variable of its own, so this is the compiler for a
formula whose leaves are independent. Leaves that indicate the values of one
declared variable need :mod:`~deeplog.circuit.knowledge_compile.mvsdd` instead,
which makes them mutually exclusive as well.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..circuit import Circuit
from .diagram import DiagramAlgebra
from .diagram import DiagramEmitter
from .diagram import seed_constants


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...algebraic import AlgebraicStructure
    from ...symbol import Symbol


def compile_sdd(
    circuit: Circuit,
    roots: list[int],
    *,
    target: Circuit | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
) -> tuple[Circuit, dict[int, int]]:
    """Compile ``roots`` into a d-DNNF circuit.

    Returns ``(new_circuit, node_map)``, the same contract as
    :func:`~deeplog.circuit.transform.transform_circuit`. All roots are compiled
    against one SDD manager, so they share what they have in common. ``target``
    and ``leaf_mapping`` are the emitter's — see
    :class:`~deeplog.circuit.knowledge_compile.diagram.DiagramEmitter`.
    """
    from pysdd.sdd import SddManager

    leaf_nodes = circuit.reachable_leaves(roots)
    manager = SddManager(var_count=max(1, len(leaf_nodes)), auto_gc_and_minimize=False)

    # SDD variables are 1-based, in the circuit's own leaf order.
    symbol_of_variable: dict[int, Symbol] = dict(
        enumerate(leaf_nodes, start=1)  # pyright: ignore[reportArgumentType]
    )
    atoms: dict[Symbol, object] = {
        symbol: manager.literal(variable)
        for variable, symbol in enumerate(leaf_nodes, start=1)
    }
    seed_constants(circuit, atoms, manager.false(), manager.true())

    node_to_sdd = circuit.fold(roots, DiagramAlgebra(circuit.structure, atoms))

    emitter = _SddEmitter(
        target if target is not None else Circuit(circuit.structure),
        circuit.structure,
        symbol_of_variable,
        leaf_mapping,
    )
    return emitter.target, {
        root: emitter.emit(node_to_sdd[root])  # pyright: ignore[reportArgumentType]
        for root in roots
    }


class _SddEmitter(DiagramEmitter):
    """Read an SDD back out as a circuit."""

    def __init__(
        self,
        target: Circuit,
        source_structure: AlgebraicStructure,
        symbol_of_variable: dict[int, Symbol],
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    ) -> None:
        """Record which circuit leaf each SDD variable stands for."""
        super().__init__(target, source_structure, leaf_mapping)
        self._symbol_of_variable = symbol_of_variable

    def emit(self, node) -> int:
        """The circuit node for this SDD node, built once per canonical node.

        A decision node is a disjunction of ``prime ∧ sub`` elements.
        """
        return self.memoised(node.id, lambda: self._build(node))

    def _build(self, node) -> int:
        if node.is_false():
            return self.zero()
        if node.is_true():
            return self.one()
        if node.is_literal():
            literal = node.literal
            leaf = self.leaf(self._symbol_of_variable[abs(literal)])
            return leaf if literal > 0 else self.negate(leaf)
        return self.disjoin(
            *(
                self.conjoin(self.emit(prime), self.emit(sub))
                for prime, sub in node.elements()
            )
        )
