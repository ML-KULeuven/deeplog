#  Copyright (c) 2024-2026. KU Leuven
"""The DeepProbLog solver: drive a grounder, then reattach probabilistic meaning.

A :class:`Solver` wraps a plain :class:`~deeplog.grounding.prolog.PrologGrounder`. It prepares
a DeepProbLog program (splitting annotated disjunctions, moving rule labels to
aux facts, stripping labels to a plain program plus an *open*-predicate set and
label declarations), grounds every query / constraint through the grounder, and
then builds the :class:`EngineResult` (labels / variables / evidence) by
interpreting the ground leaves post-hoc.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from functools import reduce
from typing import TYPE_CHECKING
from typing import TypeVar

from deeplog import Symbol
from deeplog import VariableAtoms
from deeplog import apply_substitution
from deeplog import get_predicate
from deeplog.grounding import ProofBuilder
from deeplog.grounding.prolog import PrologGrounder
from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import calculate_mgu
from deeplog.grounding.prolog import create_rule
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_fact
from deeplog.grounding.prolog import is_query

from .ad import Declaration
from .ad import declare_neural
from .ad import instantiate
from .ad import split_annotated_disjunctions
from .parser import get_atom
from .parser import get_label
from .transformation import remove_labeled_rules


if TYPE_CHECKING:
    from deeplog import DeepLogFormulaFactory


F = TypeVar("F")


@dataclass
class EngineResult[F]:
    """Result of proving a goal: proof formulas and the atom labels for them."""

    #: Maps ground goals to proof formulas (the boolean proof structure).
    formulas: dict[Symbol, F]
    #: Maps ground atom symbols to their probability labels.
    labels: dict[Symbol, Symbol] = field(default_factory=dict)
    #: The annotated disjunctions recognized in the ground atoms — where each
    #: variable occurs. Recognizing these is what makes the branches mutually
    #: exclusive downstream.
    variables: VariableAtoms = field(default_factory=dict)
    #: The shared evidence formula ``e = ⋀ᵢ ¬(bodyᵢ)`` over the program's
    #: integrity constraints, or ``None`` when the program declares none.
    evidence: F | None = None


def _detach_labels(
    program: tuple[RuleType, ...],
) -> tuple[
    tuple[RuleType, ...],
    set[tuple[str, int]],
    dict[tuple[str, int], list[tuple[Symbol, Symbol]]],
    tuple[Declaration, ...],
]:
    """Detach labels from a DeepProbLog program, yielding a plain grounder program.

    The inverse of :func:`_reattach_labels`. Splits annotated disjunctions and
    rule labels, then strips every ``::`` label off, returning
    ``(plain program, open-predicate set, declarations, disjunctions)``: the
    plain program is the grounder's label-free input; ``open`` names the
    predicates whose facts are probabilistic leaves; ``declarations`` maps each
    open predicate to its ``(atom_template, label_template)`` pairs, which
    :func:`_reattach_labels` uses to put the labels back onto the ground leaves;
    ``variables`` is what the program's annotated disjunctions declare, in either
    spelling, which :func:`~deeplog.systems.deepproblog.ad.instantiate` turns
    back into variables once the atoms are ground.
    """
    split, variables = split_annotated_disjunctions(program)
    transformed = list(remove_labeled_rules(split))
    plain: list[RuleType] = []
    open_predicates: set[tuple[str, int]] = set()
    declarations: dict[tuple[str, int], list[tuple[Symbol, Symbol]]] = defaultdict(list)
    for clause in transformed:
        label = get_label(clause[1]) if is_fact(clause) else None
        if label is None:
            plain.append(clause)
            continue
        atom = get_atom(clause[1])
        neural = declare_neural(atom, label)
        if neural is not None:
            branches, declaration = neural
            variables = (*variables, declaration)
        else:
            branches = ((atom, label),)
        for branch_atom, branch_label in branches:
            predicate = get_predicate(branch_atom)
            open_predicates.add(predicate)
            declarations[predicate].append((branch_atom, branch_label))
            plain.append(create_rule([branch_atom], []))
    return tuple(plain), open_predicates, dict(declarations), variables


def _reattach_labels(
    leaves: Iterable[Symbol],
    declarations: dict[tuple[str, int], list[tuple[Symbol, Symbol]]],
) -> dict[Symbol, Symbol]:
    """Reattach labels to the grounder's leaves (the inverse of :func:`_detach_labels`).

    The grounder is semantics-free: it emits open-fact atoms as leaves with no
    notion of a label. This maps each ground leaf back to its probability label
    by matching it against the ``(atom_template, label_template)`` declarations
    :func:`_detach_labels` collected, then grounding the label through the match.
    Because ``remove_labeled_rules`` forces every label variable into the fact's
    arguments, the ground atom fully determines its label, so this reproduces
    exactly what an in-search engine would have recorded.
    """
    labels: dict[Symbol, Symbol] = {}
    for ground_atom in leaves:
        for atom_template, label_template in declarations.get(
            get_predicate(ground_atom), ()
        ):
            mgu = calculate_mgu(atom_template, ground_atom)
            if mgu is None:
                continue
            labels[ground_atom] = apply_substitution(label_template, mgu)
            break
    return labels


class Solver:
    """A DeepProbLog solver over a plain :class:`~deeplog.grounding.prolog.PrologGrounder`."""

    def __init__(self, grounder: PrologGrounder):
        """Wrap ``grounder`` with DeepProbLog's probabilistic semantics."""
        self._grounder = grounder

    def add_builtin(self, functor: str, arity: int, builtin_function) -> None:
        """Register an additional builtin predicate on the underlying grounder."""
        self._grounder.add_builtin(functor, arity, builtin_function)

    def get_result(
        self,
        program: tuple[RuleType, ...],
        goal: Symbol,
        factory: DeepLogFormulaFactory[F],
    ) -> EngineResult[F]:
        """Prove a single ``goal`` and return its formulas + labels."""
        plain, open_predicates, declarations, variables = _detach_labels(program)
        builder: ProofBuilder[F] = ProofBuilder(factory)
        formulas = self._grounder.ground(plain, goal, builder, open_predicates)
        return EngineResult(
            formulas,
            _reattach_labels(builder.leaves, declarations),
            instantiate(variables, builder.leaves),
        )

    def get_query_result(
        self,
        program: tuple[RuleType, ...],
        factory: DeepLogFormulaFactory[F],
    ) -> EngineResult[F]:
        """Evaluate every query in ``program``, conditioned on its constraints.

        Each query's proof is ``P(q)``; when the program declares integrity
        constraints ``:- body.``, the result is conditioned on evidence
        ``e = ⋀ᵢ ¬(bodyᵢ)`` and each query formula becomes ``q ∧ e``. One shared
        :class:`~deeplog.grounding.ProofBuilder` drives every query and constraint
        body, so all leaves co-reside in one source circuit.
        """
        plain, open_predicates, declarations, variables = _detach_labels(program)
        builder: ProofBuilder[F] = ProofBuilder(factory)
        evidence = self._build_evidence(plain, open_predicates, builder)

        all_formulas: dict[Symbol, F] = {}
        for query in filter(is_query, plain):
            for answer, formula in self._grounder.ground(
                plain, query[2], builder, open_predicates
            ).items():
                all_formulas[answer] = (
                    formula if evidence is None else builder.conjoin(formula, evidence)
                )
        return EngineResult(
            all_formulas,
            _reattach_labels(builder.leaves, declarations),
            instantiate(variables, builder.leaves),
            evidence,
        )

    def _build_evidence(
        self,
        plain: tuple[RuleType, ...],
        open_predicates: set[tuple[str, int]],
        builder: ProofBuilder[F],
    ) -> F | None:
        """Build the shared evidence formula ``e = ⋀ᵢ ¬(bodyᵢ)`` over all constraints.

        Each integrity constraint ``:- body.`` contributes ``¬body``: the body is
        grounded with the same ``builder`` (so its leaves co-reside with the query
        proofs) and its proofs are OR-folded before negation. Returns ``None``
        when the program has no constraints.
        """
        constraints = list(filter(is_constraint, plain))
        if not constraints:
            return None
        evidence = builder.get_true()
        for constraint in constraints:
            body = get_constraint_body(constraint)
            proofs = self._grounder.ground(plain, body, builder, open_predicates)
            body_formula = reduce(builder.disjoin, proofs.values(), builder.get_false())
            evidence = builder.conjoin(evidence, builder.negate(body_formula))
        return evidence
