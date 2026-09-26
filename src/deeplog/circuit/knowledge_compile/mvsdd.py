#  Copyright (c) 2024-2026. KU Leuven
"""Knowledge compilation via MV-SDD, for formulas over multi-valued variables.

The multi-valued counterpart of :mod:`~deeplog.circuit.knowledge_compile.sdd`.
The atoms asserting one variable's values are not independent -- exactly one of
``digit(i,0) ... digit(i,9)`` holds -- and a multi-valued compiler makes that
exclusivity structural. One MV variable per DeepLog variable, sized by its
domain; the canonical diagram is read back out as a circuit whose leaves are the
asserting atoms themselves.

A leaf asserting no known variable's value is a two-valued variable of its own,
built here rather than tracked separately: its ``true`` value is the leaf and its
``false`` value is a value no atom asserts, which is read back as the complement.
A declared variable's values all have atoms, so the values no reachable leaf
asserts are read back as those atoms instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...algebraic import BOOLEAN
from ...symbol import symbol_to_str
from ...symbol import unwrap_structure
from ...symbol import with_structure
from ...variable import Domain
from ...variable import Variable
from ...variable import atom_asserting
from ...variable import indicated_values
from ..circuit import Circuit
from .diagram import DiagramAlgebra
from .diagram import DiagramEmitter
from .diagram import seed_constants


if TYPE_CHECKING:
    from collections.abc import Callable

    from ...algebraic import AlgebraicStructure
    from ...symbol import Symbol
    from ...variable import VariableAtoms


def compile_mvsdd(
    circuit: Circuit,
    roots: list[int],
    variables: VariableAtoms,
    *,
    target: Circuit | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
) -> tuple[Circuit, dict[int, int]]:
    """Compile ``roots`` into a d-DNNF circuit whose variables' values are exclusive.

    ``variables`` says where each variable occurs; which atom asserts which
    value follows by substitution. Every reachable leaf it does not account for
    becomes a two-valued variable of its own. A value of ``variables`` no
    reachable leaf asserts is read back as the atoms asserting it, so
    ``leaf_mapping`` names those atoms too. Returns ``(new_circuit, node_map)``,
    the same contract as
    :func:`~deeplog.circuit.transform.transform_circuit`. ``target`` and
    ``leaf_mapping`` are the emitter's — see
    :class:`~deeplog.circuit.knowledge_compile.diagram.DiagramEmitter`.

    Raises:
        ImportError: If mv-sdd, which ``pydeeplog[mvsdd]`` installs, is not
            installed.
    """
    try:
        import pymvsdd
    except ModuleNotFoundError as error:
        if error.name != "pymvsdd":
            raise
        raise ImportError(
            "Compiling a formula that reaches two values of one variable needs "
            "mv-sdd, which pydeeplog[mvsdd] installs."
        ) from None

    leaf_nodes = circuit.reachable_leaves(roots)
    asserted = indicated_values(variables)

    # 1. Every reachable leaf asserts a value of some variable: a known one, or
    # a two-valued variable built for it here.
    reached: dict[Variable, dict[int, tuple[Symbol, int]]] = {}
    for leaf_symbol, leaf_id in leaf_nodes.items():
        entry = asserted.get(unwrap_structure(leaf_symbol), ())
        if len(entry) > 1:
            raise NotImplementedError(
                f"Leaf {leaf_symbol} asserts a value of "
                f"{', '.join(symbol_to_str(v.name) for v, _ in entry)}; MV-SDD "
                f"compiles one variable per leaf."
            )
        variable, position = (
            entry[0]
            if entry
            else (Variable(leaf_symbol, Domain.of_structure(BOOLEAN)), 1)
        )
        reached.setdefault(variable, {})[position] = (leaf_symbol, leaf_id)

    # Variable order is deterministic (by name) so repeated compilation produces
    # identical vtrees.
    order = sorted(reached, key=lambda variable: symbol_to_str(variable.name))

    # 2. Each variable takes one MV variable. Its reached values keep a position
    # each; every value no reachable leaf asserts collapses into a *single*
    # residual position -- the formula cannot tell those apart. mv-sdd requires
    # a domain of at least two values.
    domain_sizes: list[int] = []
    index_of: dict[Variable, int] = {}
    position_of: dict[Variable, dict[int, int]] = {}
    residual_of: dict[Variable, int | None] = {}
    for variable in order:
        found = sorted(reached[variable])
        position_of[variable] = {value: slot for slot, value in enumerate(found)}
        residual_of[variable] = len(found) if len(found) < len(variable) else None
        index_of[variable] = len(domain_sizes)
        domain_sizes.append(max(len(found) + (residual_of[variable] is not None), 2))
    if not domain_sizes:
        # A circuit of nothing but constants still needs a vtree.
        domain_sizes = [2]

    # 3. Build the diagram bottom-up over those variables.
    manager = pymvsdd.Manager(pymvsdd.Vtree.right_linear(domain_sizes))
    atoms: dict[Symbol, object] = {}
    leaf_of_slot: dict[tuple[int, int], Symbol] = {}
    unreached_of_slot: dict[tuple[int, int], tuple[Symbol, ...]] = {}
    complement_of_slot: dict[tuple[int, int], Symbol] = {}
    for variable in order:
        index = index_of[variable]
        for value, (leaf_symbol, _) in reached[variable].items():
            slot = position_of[variable][value]
            atoms[leaf_symbol] = manager.literal(index, slot)
            leaf_of_slot[(index, slot)] = leaf_symbol
        residual = residual_of[variable]
        if residual is None:
            continue
        if variable in variables:
            unreached_of_slot[(index, residual)] = tuple(
                with_structure(
                    atom_asserting(occurrence, value), circuit.structure.name
                )
                for position, value in enumerate(variable.domain.values)
                if position not in reached[variable]
                for occurrence in variables[variable]
            )
        else:
            ((leaf_symbol, _),) = reached[variable].values()
            complement_of_slot[(index, residual)] = leaf_symbol
    seed_constants(circuit, atoms, manager.bot(), manager.top())

    node_to_mv = circuit.fold(roots, DiagramAlgebra(circuit.structure, atoms))

    # 4. Read the canonical diagram back out as a circuit.
    emitter = _MvSddEmitter(
        target if target is not None else Circuit(circuit.structure),
        circuit.structure,
        leaf_of_slot,
        unreached_of_slot,
        complement_of_slot,
        leaf_mapping,
    )
    return emitter.target, {root: emitter.emit(node_to_mv[root]) for root in roots}


class _MvSddEmitter(DiagramEmitter):
    """Read an MV-SDD back out as a circuit."""

    def __init__(
        self,
        target: Circuit,
        source_structure: AlgebraicStructure,
        leaf_of_slot: dict[tuple[int, int], Symbol],
        unreached_of_slot: dict[tuple[int, int], tuple[Symbol, ...]],
        complement_of_slot: dict[tuple[int, int], Symbol],
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    ) -> None:
        """Record what each ``(variable, slot)`` of the diagram asserts."""
        super().__init__(target, source_structure, leaf_mapping)
        self._leaf_of_slot = leaf_of_slot
        self._unreached_of_slot = unreached_of_slot
        self._complement_of_slot = complement_of_slot

    def emit(self, node) -> int:
        """The circuit node for this MV-SDD node, built once per canonical node.

        Memoised on ``node_id`` rather than Python ``id()``: ``node.elements``
        allocates fresh wrappers on every call, so two wrappers around one
        canonical node have different ``id()``s and a shared subdiagram would be
        re-emitted as disjoint copies.
        """
        return self.memoised(node.node_id, lambda: self._build(node))

    def _build(self, node) -> int:
        kind = node.kind
        if kind == "top":
            return self.one()
        if kind == "bot":
            return self.zero()
        if kind == "terminal":
            return self.disjoin(
                *(self._indicator(node.var, value) for value in node.values)
            )
        if kind == "decomp":
            return self.disjoin(
                *(
                    self.conjoin(self.emit(prime), self.emit(sub))
                    for prime, sub in node.elements
                )
            )
        raise ValueError(f"Unknown MV-SDD node kind: {kind}")

    def _indicator(self, variable: int, slot: int) -> int:
        """The circuit node asserting that ``variable`` takes the value at ``slot``.

        A value the formula asserts has an atom of its own, so the compiled
        interface speaks the user's atoms. The residual slot stands for every
        value no reachable leaf asserts. A declared variable's values have atoms
        whether reached or not, so the slot is the disjunction of theirs; a
        two-valued variable built for a leaf has no atom for its ``false`` value,
        so that slot is the complement of the leaf. A slot that is neither is
        padding to mv-sdd's two-value minimum and is false.
        """
        leaf_symbol = self._leaf_of_slot.get((variable, slot))
        if leaf_symbol is not None:
            return self.leaf(leaf_symbol)
        unreached = self._unreached_of_slot.get((variable, slot))
        if unreached is not None:
            return self.disjoin(*(self.leaf(atom) for atom in unreached))
        complemented = self._complement_of_slot.get((variable, slot))
        if complemented is not None:
            return self.negate(self.leaf(complemented))
        return self.zero()
