#  Copyright (c) 2024-2026. KU Leuven
"""Which backend compiles a circuit, and which of its nodes that backend can walk.

Backend selection follows the algebra alone: Klay when a Klay semiring implements
the structure, the generic evaluator otherwise. It never consults the graph.

:func:`routed_operators` reports the operators the selected backend has a node
for. An operator outside that set is not a reason to pick another backend:
:mod:`deeplog.circuit.split` cuts the graph there, applies the algebra's own
``operator_fns`` entry to the compiled operands, and hands the value to the
circuit above as a leaf. That is how a semifield's ``divide`` compiles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..algebraic import BOOLEAN
from ..algebraic import LOGPROBABILITY
from ..algebraic import MPE
from ..algebraic import PROBABILITY
from ..algebraic import AlgebraicStructure
from ..algebraic import Semiring


if TYPE_CHECKING:
    from .circuit import Circuit


#: Structures Klay can evaluate, mapped to the Klay semiring implementing them.
#: Keyed by structure object rather than by name, so membership asserts that this
#: structure's product and sum *are* that semiring's. An unregistered structure
#: goes to the generic evaluator.
_KLAY_SEMIRINGS: dict[AlgebraicStructure, tuple[str, bool]] = {}


def register_klay_semiring(
    structure: AlgebraicStructure, semiring: str, log: bool = False
) -> None:
    """Declare that Klay's ``semiring`` implements ``structure``'s semantics.

    Makes it eligible for the Klay fast path. ``structure`` must be a
    :class:`~deeplog.algebraic.Semiring`, since the claim is about its product
    and sum.
    """
    if not isinstance(structure, Semiring):
        raise ValueError(
            f"Cannot register a Klay semiring for '{structure.name}': it declares "
            f"no product and sum, so there is nothing for '{semiring}' to "
            f"implement. Leave it to the generic evaluator, which honours its "
            f"'operator_fns' directly."
        )
    _KLAY_SEMIRINGS[structure] = (semiring, log)


register_klay_semiring(BOOLEAN, "godel")
register_klay_semiring(PROBABILITY, "real")
register_klay_semiring(LOGPROBABILITY, "log")
register_klay_semiring(MPE, "mpe")


def klay_semiring(structure: AlgebraicStructure) -> tuple[str, bool]:
    """Resolve the Klay semiring to evaluate ``structure`` in.

    Raises when none is registered rather than falling back to ``real``, which
    would silently discard the structure's own ``operator_fns``.
    """
    try:
        return _KLAY_SEMIRINGS[structure]
    except KeyError:
        raise ValueError(
            f"No Klay semiring is registered for structure '{structure.name}', so "
            f"Klay cannot evaluate it: Klay applies a fixed semiring and never "
            f"consults 'operator_fns'. Register one with register_klay_semiring() "
            f"if a Klay semiring implements this structure's product and sum, or "
            f"compile through select_backend(), which routes an unregistered "
            f"structure to the generic evaluator and honours 'operator_fns'."
        ) from None


def routed_operators(structure: AlgebraicStructure, backend: str) -> frozenset[str]:
    """Operator names ``backend`` has a circuit node for.

    Klay and the knowledge-compilation walks build one node per semiring role, so
    they route the names this structure spells its product, sum and complement
    with: ``and``/``or``/``not`` in :data:`~deeplog.algebraic.BOOLEAN` and
    ``times``/``plus``/``negate`` in :data:`~deeplog.algebraic.PROBABILITY` are
    the same three nodes. The generic evaluator dispatches out of
    ``operator_fns`` and routes everything the structure defines.

    An operator outside this set is cut
    (:func:`~deeplog.circuit.split.find_cuts`).
    """
    if backend == "generic":
        return structure.operators
    return frozenset(circuit_roles(structure).values())


#: The roles a diagram walk has a node for. Any other operator a structure
#: declares -- a semifield's division -- is cut instead
#: (:mod:`deeplog.circuit.split`).
_NODE_ROLES = frozenset({"product", "sum", "negation"})


def circuit_roles(structure: AlgebraicStructure) -> dict[str, str]:
    """The operator names this structure spells its circuit roles with.

    ``{"product": ..., "sum": ..., "negation": ...}``, each present only when the
    structure declares that axiom. Empty for a bare
    :class:`~deeplog.algebraic.AlgebraicStructure`.
    """
    return {role: name for role, name in structure.roles.items() if role in _NODE_ROLES}


def select_backend(circuit: Circuit) -> str:
    """The backend that will evaluate ``circuit``: ``'klay'`` or ``'generic'``.

    A function of the algebra alone: Klay when a Klay semiring implements the
    structure, otherwise the generic evaluator, which honours the structure's own
    ``operator_fns``.
    """
    if circuit.structure in _KLAY_SEMIRINGS:
        return "klay"
    return "generic"
