#  Copyright (c) 2024-2026. KU Leuven
"""Transform a circuit from one algebraic structure to another."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..algebraic import AlgebraicStructure
from ..algebraic import get_algebraic_structure
from ..formula.deeplogformulafactory import DeepLogFormulaFactory
from ..symbol import Symbol
from ..symbol import is_structure_wrapped
from ..symbol import unwrap_structure
from .circuit import Circuit


if TYPE_CHECKING:
    from ..formula.ast import CircuitNode


def _build_operator_mapping(
    source: AlgebraicStructure,
    target: AlgebraicStructure,
    explicit: dict[str, str] | None,
) -> dict[str, str]:
    """The target operator each of ``source``'s operators crosses as.

    An operator crosses by the role it plays
    (:attr:`~deeplog.algebraic.AlgebraicStructure.roles`), so the roles both
    structures declare map and nothing else does. A source operator the target
    has no counterpart for is simply absent here: whether that matters is a
    question about the graph, not about the pair of structures, and it is
    answered at the node that uses it.
    """
    if explicit is not None:
        for target_op in explicit.values():
            if target_op not in target.operators:
                raise ValueError(
                    f"Target operator '{target_op}' not in target structure "
                    f"'{target.name}'. Available: {target.operators}"
                )
        return explicit

    target_roles = target.roles
    return {
        name: target_roles[role]
        for role, name in source.roles.items()
        if role in target_roles
    }


def _build_constant_mapping(
    source: AlgebraicStructure, target: AlgebraicStructure
) -> dict[Symbol, Symbol]:
    """The target symbol each of ``source``'s named constants crosses as.

    A named constant crosses by the role it plays
    (:attr:`~deeplog.algebraic.AlgebraicStructure.identities`), not by its
    value: the additive identity is ``("0",)`` in probability and ``("-inf",)``
    in log space, so carrying it over as a number would change which element it
    is. An identity the target does not name is left out, and crosses as the
    value it is.
    """
    target_identities = target.identities
    return {
        symbol: target_identities[role]
        for role, symbol in source.identities.items()
        if role in target_identities
    }


def transform_circuit(
    source: Circuit,
    target_structure: str | AlgebraicStructure,
    roots: list[int],
    *,
    operator_mapping: dict[str, str] | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    into: tuple[Circuit, dict[int, int]] | None = None,
) -> tuple[Circuit, dict[int, int]]:
    """Transform a circuit to a different algebraic structure.

    Creates a new circuit with the target structure by traversing the source
    circuit and rebuilding each node with mapped operators, leaves, and constants.

    The map is exact only insofar as the source's operators mean in the target
    what they meant at home. Reading a *logical* circuit's ``or`` as a semiring
    sum requires the source to be deterministic and decomposable, which is what
    :func:`~deeplog.circuit.knowledge_compile.knowledge_compile` produces.

    Args:
        source: The circuit to transform_circuit.
        target_structure: The target algebraic structure (name or instance).
        roots: Root node IDs defining the subgraph to transform_circuit.
        operator_mapping: Explicit mapping from source operator names to target
            operator names, replacing the inferred one. If None, the mapping is
            inferred from the roles both structures declare (product→product,
            sum→sum, negation→negation, division→division); an operator the
            target has no role for raises when a node uses it.
        leaf_mapping: Optional callable to remap leaf symbols.
        into: An existing ``(target_circuit, node_map)`` to continue transforming
            into, instead of allocating a fresh target. Source nodes already in
            ``node_map`` are skipped, so repeated calls sharing one target
            accumulate and deduplicate exactly as a single multi-root call
            would. The ``node_map`` is mutated in place and returned. Its target
            structure must match ``target_structure``.

    Returns:
        A tuple of (new_circuit, node_map) where node_map maps source node IDs
        to their corresponding IDs in the new circuit.
    """
    if isinstance(target_structure, str):
        target_struct = get_algebraic_structure(target_structure)
    else:
        target_struct = target_structure

    source_struct = source.structure

    op_mapping = _build_operator_mapping(source_struct, target_struct, operator_mapping)

    if into is not None:
        target_circuit, node_map = into
        if target_circuit.structure.name != target_struct.name:
            raise ValueError(
                f"Cannot continue a transform into a circuit with structure "
                f"'{target_circuit.structure.name}' using target "
                f"structure '{target_struct.name}'."
            )
    else:
        target_circuit = Circuit(target_struct)
        node_map = {}

    algebra = CircuitAlgebra(
        target_circuit,
        source_struct,
        operator_mapping=op_mapping,
        leaf_mapping=leaf_mapping,
    )
    source.fold(roots, algebra, memo=node_map)
    return target_circuit, node_map


class CircuitAlgebra(DeepLogFormulaFactory[int]):
    """Build a formula's nodes in ``target``, mapping operators and atoms.

    A formula algebra whose carrier is a node id of one given circuit, so
    anything the algebra drives -- a circuit fold
    (:func:`~deeplog.circuit.fold.fold_circuit`), an AST fold
    (:func:`~deeplog.formula.ast.fold`), or a walk that calls the eliminators
    itself -- writes into that circuit. It names operators in the *source's*
    spelling and maps them to the target's, so a caller says ``and`` whether the
    target calls it ``and`` or ``times``.

    An atom is a leaf or a constant: a named constant crosses by *role*, so the
    target spells its own; a numeric constant crosses by value, unmapped, since
    it is not one of the source's names; anything else is a leaf and goes
    through ``leaf_mapping``, which sees the leaf's *bare* identity.
    """

    def __init__(
        self,
        target: Circuit,
        source_structure: AlgebraicStructure,
        *,
        operator_mapping: dict[str, str] | None = None,
        leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    ) -> None:
        """Build into ``target``, renaming from ``source_structure``'s spelling."""
        self._target = target
        self._source_structure = source_structure
        self._target_structure = target.structure
        self._operator_mapping = _build_operator_mapping(
            source_structure, target.structure, operator_mapping
        )
        self._constant_mapping = _build_constant_mapping(
            source_structure, target.structure
        )
        self._leaf_mapping = leaf_mapping

    def create_atom(self, atom: Symbol) -> int:
        """The target node for the source leaf or constant ``atom`` names."""
        name = unwrap_structure(atom) if is_structure_wrapped(atom) else atom
        target_symbol = self._constant_mapping.get(name)
        if target_symbol is not None:
            return self._target.get_leaf_node(target_symbol)
        if self._source_structure.get_constant_value(name) is not None:
            # A value, not a name: it crosses as itself, unmapped.
            return self._target.get_leaf_node(name)
        return self._target.get_leaf_node(
            self._leaf_mapping(name) if self._leaf_mapping else name
        )

    def create_unary_node(self, operator: str, operand: int) -> int:
        """Apply ``operator``'s target counterpart to ``operand``."""
        return self._target.get_operator(self._mapped(operator))(operand)

    def create_binary_node(self, operator: str, lhs: int, rhs: int) -> int:
        """Apply ``operator``'s target counterpart to ``lhs`` and ``rhs``."""
        return self._target.get_operator(self._mapped(operator))(lhs, rhs)

    def _mapped(self, operator: str) -> str:
        """The target's name for ``operator``, or raise saying what is missing."""
        target_op = self._operator_mapping.get(operator)
        if target_op is not None:
            return target_op
        role = next(
            (r for r, name in self._source_structure.roles.items() if name == operator),
            None,
        )
        missing = (
            f"'{self._target_structure.name}' declares no {role}"
            if role is not None
            else f"'{operator}' plays no role in '{self._source_structure.name}'"
        )
        raise ValueError(
            f"Cannot map node type '{operator}' from "
            f"'{self._source_structure.name}' to "
            f"'{self._target_structure.name}': {missing}. "
            f"Provide an explicit operator_mapping that includes '{operator}'."
        )

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
