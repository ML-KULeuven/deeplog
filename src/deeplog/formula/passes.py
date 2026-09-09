#  Copyright (c) 2024-2026. KU Leuven
"""AST→AST rewrite passes over the formula AST.

Two conservative recognition passes, run in :data:`DEFAULT_PASSES` order:
:func:`recognize_posterior` normalises a ``divide`` of two weighted model counts
into a ratio of two ``expectation`` nodes, and :func:`recognize_expectation`
rewrites a hand-written weighted model count to ``expectation`` — only when the
probability factor uses the exact same atoms as the boolean formula, retagged
from ``boolean`` to ``probability``. Label aliases such as ``nn(...) :: a(...)``
are handled by the DeepProbLog engine's explicit label map, not inferred here.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator

from ..algebraic import BOOLEAN
from ..algebraic import PROBABILITY
from ..symbol import Symbol
from ..symbol import get_term_variables
from ..symbol import unwrap_structure
from ..symbol import with_structure
from .ast import Aggregation
from .ast import Atom
from .ast import BinaryOp
from .ast import FormulaNode
from .ast import Transformation
from .ast import children
from .ast import map_children


def recognize_posterior(node: FormulaNode) -> FormulaNode:
    """Normalize a ``divide`` of two weighted model counts into a posterior.

    A scalar posterior ``P(q | e) = E[q∧e] / E[e]`` is a ratio of two
    expectations, not an aggregation: the division binds nothing. This pass
    matches ``divide(WMC_qe, WMC_e)`` (or a divide of two explicit
    ``expectation`` nodes) and rewrites each operand to an ``expectation``,
    leaving the ``divide`` itself alone: it is an operator of the probability
    :class:`~deeplog.algebraic.Semifield` and lowers like any other, as a circuit
    node the compiling backend cuts at (:mod:`deeplog.circuit.split`).

    Recognition only, like every other pass: a ``divide`` that is not a posterior
    — either operand not an expectation / canonical WMC, or a denominator whose
    evidence is not a conjunct of the numerator ``q ∧ e`` — is left as it is and
    still lowers, as a plain division of its two operands.
    """
    node = map_children(node, recognize_posterior)
    if not (isinstance(node, BinaryOp) and node.operator == PROBABILITY.division):
        return node
    numerator = _posterior_operand(node.lhs)
    denominator = _posterior_operand(node.rhs)
    if numerator is None or denominator is None:
        return node
    num_binders, joint = numerator
    den_binders, evidence = denominator
    if evidence not in set(_and_conjuncts(joint)):
        return node
    return BinaryOp(
        PROBABILITY.division,
        Aggregation("expectation", num_binders, (), joint),
        Aggregation("expectation", den_binders, (), evidence),
    )


def recognize_expectation(node: FormulaNode) -> FormulaNode:
    """Normalize canonical hand-written WMC ``sum`` nodes into ``expectation``.

    The rewrite fires only when every probability leaf is exactly the
    corresponding boolean leaf retagged to the probability structure. This keeps
    the parser path free of argument-overlap heuristics.
    """
    node = map_children(node, recognize_expectation)
    if not (
        isinstance(node, Aggregation) and node.operation == "sum" and not node.params
    ):
        return node
    boolean_child = _canonical_wmc_child(node)
    if boolean_child is None:
        return node
    return Aggregation("expectation", node.binders, (), boolean_child)


DEFAULT_PASSES: tuple[Callable[[FormulaNode], FormulaNode], ...] = (
    recognize_posterior,
    recognize_expectation,
)


def _posterior_operand(
    node: FormulaNode,
) -> tuple[tuple[Symbol, ...], FormulaNode] | None:
    """Return ``(binders, boolean)`` for a WMC sum or explicit expectation."""
    if isinstance(node, Aggregation) and node.operation == "sum" and not node.params:
        boolean_child = _canonical_wmc_child(node)
        if boolean_child is not None:
            return node.binders, boolean_child
    if isinstance(node, Aggregation) and node.operation == "expectation":
        return node.binders, node.child
    return None


def _and_conjuncts(node: FormulaNode) -> Iterator[FormulaNode]:
    """Yield the top-level boolean ``and`` conjuncts of ``node`` (itself if none)."""
    if isinstance(node, BinaryOp) and node.operator == BOOLEAN.product:
        yield from _and_conjuncts(node.lhs)
        yield from _and_conjuncts(node.rhs)
    else:
        yield node


def _canonical_wmc_child(agg: Aggregation) -> FormulaNode | None:
    """Return the boolean child for a canonical WMC sum, otherwise ``None``."""
    child = agg.child
    if not isinstance(child, BinaryOp) or child.operator != PROBABILITY.product:
        return None

    for cast_side, prob_side in ((child.lhs, child.rhs), (child.rhs, child.lhs)):
        boolean_child = _probability_cast_of_boolean(cast_side)
        if boolean_child is None or not _is_factorized_probability_ast(prob_side):
            continue
        bound = set(agg.binders)
        if _free_variables(boolean_child) != bound:
            continue
        boolean_prob_leaves = {
            with_structure(unwrap_structure(atom), PROBABILITY.name)
            for atom in _iter_atoms(boolean_child)
        }
        probability_leaves = set(_iter_atoms(prob_side))
        if boolean_prob_leaves == probability_leaves:
            return boolean_child
    return None


def _probability_cast_of_boolean(node: FormulaNode) -> FormulaNode | None:
    """Return the inner formula if ``node`` casts a boolean formula to probability."""
    if (
        isinstance(node, Transformation)
        and node.structure == PROBABILITY.name
        and node.child.structure == BOOLEAN.name
    ):
        return node.child
    return None


def _iter_atoms(node: FormulaNode) -> Iterator[Symbol]:
    """Yield the structure-wrapped symbol of every atom leaf in ``node``."""
    if isinstance(node, Atom):
        yield node.atom
        return
    for child in children(node):
        yield from _iter_atoms(child)


def _free_variables(node: FormulaNode) -> set[Symbol]:
    """Return logic variables appearing in ``node``'s atoms."""
    return {
        var
        for atom in _iter_atoms(node)
        for var in get_term_variables(unwrap_structure(atom))
    }


def _is_factorized_probability_ast(node: FormulaNode) -> bool:
    """Whether ``node`` is a pure product of probability atoms."""
    match node:
        case Atom():
            return node.structure == PROBABILITY.name
        case BinaryOp(operator, lhs, rhs):
            return (
                operator == PROBABILITY.product
                and _is_factorized_probability_ast(lhs)
                and _is_factorized_probability_ast(rhs)
            )
    return False
