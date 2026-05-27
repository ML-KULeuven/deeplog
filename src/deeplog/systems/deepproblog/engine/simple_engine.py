#  Copyright (c) 2024-2026. KU Leuven
"""A lightweight Prolog-style engine used for DeepProbLog experiments."""

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from functools import reduce
from typing import TypeVar
from typing import cast

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import get_predicate
from deeplog.symbol import get_term_variables

from ..program import Program
from ..program import RuleType
from ..program import expand_annotated_disjunctions
from ..program import is_categorical_label
from ..program import remove_labeled_rules
from ..program import unwrap_categorical_label
from ..program.program import get_atom
from ..program.program import get_label
from ..program.program import is_fact
from .builtins import all_builtins
from .engine import Builtin
from .engine import Engine
from .engine import EngineFactory
from .engine import Result
from .engine import UnknownPredicateException
from .unify import chain_substitution
from .unify import replace_with_fresh_variables
from .unify import unify


type Predicate = tuple[str, int]
type Sub = Mapping[Symbol, Symbol]
F = TypeVar("F")
type ADAssignment = Mapping[str, int]
type Proof[F] = tuple[Sub, F, ADAssignment]
type DictProgram = Mapping[Predicate, Iterable[RuleType]]


def _merge_ad_assignments(
    lhs: ADAssignment, rhs: ADAssignment
) -> dict[str, int] | None:
    """Merge two per-proof AD commitments. Returns None on conflict.

    A conflict means the same categorical variable is committed to two
    different values within a single proof — proofs that exhibit one are
    dropped (mutual exclusivity within an annotated disjunction).
    """
    merged = dict(lhs)
    for cat_id, value_idx in rhs.items():
        if cat_id in merged and merged[cat_id] != value_idx:
            return None
        merged[cat_id] = value_idx
    return merged


class SimpleEngine(Engine):
    """
    A simple memoizing prolog engine without any dependencies.
    """

    def __init__(self):
        """Create an in-process Prolog-like engine."""
        self._counter = 0
        self.builtins = all_builtins

    def _get_result(
        self, program: Program, goal: Symbol, factory: EngineFactory[F]
    ) -> Result[F]:
        prepared_program = self._prepare_program(program)
        result = self._prove(prepared_program, goal, factory)
        # Multiple (substitution, ad_assignment) keys can map to the same
        # ground goal once substitution is applied; OR-fold their formulas.
        # The AD-assignment is private to proof construction and is dropped
        # at the public boundary.
        out: dict[Symbol, F] = {}
        for substitution, formula, _assignment in result:
            ground = apply_substitution(goal, substitution)
            if ground in out:
                out[ground] = factory.disjoin(out[ground], formula)
            else:
                out[ground] = formula
        return out

    @staticmethod
    def _prepare_program(program: Program) -> DictProgram:
        new_program = defaultdict(list)
        for clause in remove_labeled_rules(expand_annotated_disjunctions(program)):
            new_program[get_predicate(get_atom(clause[1]))].append(clause)
        return dict(new_program)

    def _prove(
        self, program: DictProgram, goal: Symbol, factory: EngineFactory[F]
    ) -> Iterable[Proof[F]]:
        predicate = get_predicate(goal)
        if predicate in [(",", 2), (";", 2), ("not", 1), ("true", 0)]:
            results = self._prove_logical_predicates(program, goal, factory)
        elif predicate in self.builtins:
            results = self._prove_builtins(goal, factory)
        elif predicate in program:
            results = self._prove_clauses(program, goal, factory)
        else:
            raise UnknownPredicateException(f"No clauses known for {predicate}.")
        results = list(results)
        # Aggregate proofs that share both the answer substitution AND the
        # AD assignment; proofs with different AD assignments must remain
        # separate so conflicting commitments can still be detected when
        # this result feeds into a parent conjunction.
        result: dict[tuple[frozenset, frozenset], F] = defaultdict(
            lambda: factory.get_false()
        )
        for answer_substitution, formula, assignment in results:
            key = (
                frozenset(answer_substitution.items()),
                frozenset(assignment.items()),
            )
            result[key] = factory.disjoin(result[key], formula)
        yield from ((dict(sub), f, dict(assn)) for (sub, assn), f in result.items())

    def _prove_logical_predicates(
        self, program, goal: Symbol, factory: EngineFactory[F]
    ) -> Iterable[Proof[F]]:
        predicate = get_predicate(goal)
        if predicate == ("true", 0):
            yield {}, factory.get_false(), {}
        elif predicate == (",", 2):
            lhs, rhs = goal[1:]
            for substitution1, formula1, assignment1 in self._prove(
                program, lhs, factory
            ):
                for substitution2, formula2, assignment2 in self._prove(
                    program, apply_substitution(rhs, substitution1), factory
                ):
                    merged = _merge_ad_assignments(assignment1, assignment2)
                    if merged is None:
                        # Conflicting AD commitments — drop this combined proof.
                        continue
                    yield (
                        chain_substitution(substitution1, substitution2),
                        factory.conjoin(formula1, formula2),
                        merged,
                    )
        elif predicate == (";", 2):
            lhs, rhs = goal[1:]
            yield from self._prove(program, lhs, factory)
            yield from self._prove(program, rhs, factory)

        elif predicate == ("not", 1):
            subgoal = cast(Symbol, goal[1])
            # NAF collapses all sub-proof formulas into a single negated
            # disjunction, dropping their per-proof AD assignments. Mutex
            # tracking under negation is out of scope (see plan).
            proof = reduce(
                lambda x, y: factory.disjoin(x, y),
                (r[1] for r in self._prove(program, subgoal, factory)),
                factory.get_false(),
            )
            proof = factory.negate(proof)
            yield {}, proof, {}

    def _prove_builtins(self, goal, factory) -> Iterable[Proof]:
        predicate = get_predicate(goal)
        if predicate in self.builtins:
            for r in self.builtins[predicate](*goal[1:]):
                yield r, factory.get_true(), {}

    def _prove_clauses(
        self, program: DictProgram, goal, factory: EngineFactory[F]
    ) -> Iterable[Proof[F]]:
        for clause in program[get_predicate(goal)]:
            # if is_rule(clause):
            clause, _ = replace_with_fresh_variables(
                clause, lambda: self._get_fresh_variable()
            )
            clause = cast(RuleType, clause)
            head, body = clause[1], clause[2]
            unification = unify(goal, get_atom(head))
            if unification is None:
                continue
            unified, mgu = unification
            goal_variables = set(get_term_variables(goal))
            answer_substitution = {k: v for k, v in mgu.items() if k in goal_variables}
            if is_fact(clause):
                label = get_label(head)
                if label is None:
                    yield answer_substitution, factory.get_true(), {}
                elif is_categorical_label(label):
                    inner_label, cat_id, value_idx = unwrap_categorical_label(label)
                    yield (
                        answer_substitution,
                        factory.get_categorical_value(
                            unified,
                            apply_substitution(inner_label, mgu),
                            cat_id,
                            value_idx,
                        ),
                        {cat_id: value_idx},
                    )
                else:
                    yield (
                        answer_substitution,
                        factory.get_boolean(unified, apply_substitution(label, mgu)),
                        {},
                    )
            else:
                for sub, f, assignment in self._prove(
                    program, apply_substitution(body, mgu), factory
                ):
                    yield (
                        {
                            apply_substitution(k, sub): apply_substitution(v, sub)
                            for k, v in answer_substitution.items()
                        },
                        f,
                        assignment,
                    )

    def _get_fresh_variable(self) -> Symbol:
        self._counter += 1
        return (f"VAR_{self._counter}",)

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register an additional builtin predicate for this engine instance."""
        self.builtins[(functor, arity)] = builtin_function
