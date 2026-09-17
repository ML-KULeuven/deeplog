#  Copyright (c) 2024-2026. KU Leuven
"""A dependency-free Prolog grounder (SLD resolution)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
from functools import reduce
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import get_predicate
from deeplog.symbol import get_term_variables
from deeplog.symbol import is_variable
from deeplog.symbol import symbol_to_pretty_string


if TYPE_CHECKING:
    from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory

from ..builder import ProofBuilder
from .builtins import all_builtins
from .grounder import Builtin
from .grounder import OpenPredicates
from .grounder import PrologGrounder
from .grounder import UnknownPredicateException
from .program import Program
from .program import RuleType
from .program import is_constraint
from .program import is_fact
from .program import is_query
from .unify import calculate_mgu
from .unify import chain_substitution
from .unify import replace_with_fresh_variables
from .unify import unify


type Predicate = tuple[str, int]
type Sub = Mapping[Symbol, Symbol]
T = TypeVar("T")
type Proof[T] = tuple[Sub, T]
type DictProgram = Mapping[Predicate, Iterable[RuleType]]


class SimpleGrounder(PrologGrounder):
    """A simple memoizing Prolog grounder without any dependencies."""

    def __init__(self):
        """Create an in-process Prolog-like grounder."""
        self._counter = 0
        self.builtins = dict(all_builtins)

    def ground(
        self,
        program: Program,
        goal: Symbol,
        factory: DeepLogFormulaFactory[T] | ProofBuilder[T],
        open_predicates: OpenPredicates = frozenset(),
    ) -> dict[Symbol, T]:
        """Prove ``goal`` in ``program`` to one proof formula per ground answer.

        Raises:
            RecursionError: If a goal calls a variant of itself, as the goals of
                ``e(X,Y) :- e(X,Z), e(Z,Y).`` do. Resolution alone cannot prove
                such a program; :class:`~deeplog.grounding.JanusGrounder` tables
                its calls and proves it.
        """
        builder = ProofBuilder.wrapping(factory)
        prepared = self._prepare_program(program)
        # Multiple substitutions can ground ``goal`` to the same atom; OR-fold
        # their proof formulas.
        out: dict[Symbol, T] = {}
        for substitution, formula in self._prove(
            prepared, goal, builder, open_predicates, ()
        ):
            ground = apply_substitution(goal, substitution)
            if ground in out:
                out[ground] = builder.disjoin(out[ground], formula)
            else:
                out[ground] = formula
        return out

    @staticmethod
    def _prepare_program(program: Program) -> DictProgram:
        new_program: dict[Predicate, list[RuleType]] = defaultdict(list)
        for clause in program:
            if is_query(clause) or is_constraint(clause):
                continue
            new_program[get_predicate(clause[1])].append(clause)
        return dict(new_program)

    def _prove(
        self,
        program: DictProgram,
        goal: Symbol,
        builder: ProofBuilder[T],
        open_predicates: OpenPredicates,
        ancestors: tuple[Symbol, ...],
    ) -> Iterable[Proof[T]]:
        predicate = get_predicate(goal)
        if predicate in [(",", 2), (";", 2), ("not", 1), ("true", 0)]:
            results = self._prove_logical(
                program, goal, builder, open_predicates, ancestors
            )
        elif predicate in self.builtins:
            results = self._prove_builtins(goal, builder)
        elif predicate in program:
            results = self._prove_clauses(
                program, goal, builder, open_predicates, ancestors
            )
        else:
            raise UnknownPredicateException(f"No clauses known for {predicate}.")
        # Aggregate proofs that share the answer substitution.
        result: dict[frozenset, T] = defaultdict(builder.get_false)
        for answer_substitution, formula in results:
            key = frozenset(answer_substitution.items())
            result[key] = builder.disjoin(result[key], formula)
        yield from ((dict(sub), f) for sub, f in result.items())

    def _prove_logical(
        self,
        program: DictProgram,
        goal: Symbol,
        builder: ProofBuilder[T],
        open_predicates: OpenPredicates,
        ancestors: tuple[Symbol, ...],
    ) -> Iterable[Proof[T]]:
        predicate = get_predicate(goal)
        if predicate == ("true", 0):
            # `true` succeeds with the identity of conjunction, not its zero:
            # `get_false()` here would poison every body it appears in.
            yield {}, builder.get_true()
        elif predicate == (",", 2):
            lhs, rhs = goal[1:]
            for substitution1, formula1 in self._prove(
                program, lhs, builder, open_predicates, ancestors
            ):
                for substitution2, formula2 in self._prove(
                    program,
                    apply_substitution(rhs, substitution1),
                    builder,
                    open_predicates,
                    ancestors,
                ):
                    yield (
                        chain_substitution(substitution1, substitution2),
                        builder.conjoin(formula1, formula2),
                    )
        elif predicate == (";", 2):
            lhs, rhs = goal[1:]
            yield from self._prove(program, lhs, builder, open_predicates, ancestors)
            yield from self._prove(program, rhs, builder, open_predicates, ancestors)
        elif predicate == ("not", 1):
            subgoal = cast(Symbol, goal[1])
            proof = reduce(
                builder.disjoin,
                (
                    f
                    for _, f in self._prove(
                        program, subgoal, builder, open_predicates, ancestors
                    )
                ),
                builder.get_false(),
            )
            yield {}, builder.negate(proof)

    def _prove_builtins(
        self, goal: Symbol, builder: ProofBuilder[T]
    ) -> Iterable[Proof[T]]:
        predicate = get_predicate(goal)
        if predicate in self.builtins:
            for r in self.builtins[predicate](*goal[1:]):
                yield r, builder.get_true()

    def _prove_clauses(
        self,
        program: DictProgram,
        goal: Symbol,
        builder: ProofBuilder[T],
        open_predicates: OpenPredicates,
        ancestors: tuple[Symbol, ...],
    ) -> Iterable[Proof[T]]:
        _check_terminates(goal, ancestors)
        is_open = get_predicate(goal) in open_predicates
        for clause in program[get_predicate(goal)]:
            clause, _ = replace_with_fresh_variables(
                clause, lambda: self._get_fresh_variable()
            )
            clause = cast(RuleType, clause)
            head, body = clause[1], clause[2]
            unification = unify(goal, head)
            if unification is None:
                continue
            unified, mgu = unification
            goal_variables = set(get_term_variables(goal))
            answer_substitution = {k: v for k, v in mgu.items() if k in goal_variables}
            if is_fact(clause):
                # An open predicate's facts are leaves; a defined predicate's
                # (deterministic) facts contribute ``true`` and collapse away.
                formula = builder.leaf(unified) if is_open else builder.get_true()
                yield answer_substitution, formula
            else:
                for sub, f in self._prove(
                    program,
                    apply_substitution(body, mgu),
                    builder,
                    open_predicates,
                    (*ancestors, goal),
                ):
                    if is_open:
                        # A rule for an open predicate declares its domain:
                        # each derived ground head instance is itself a leaf,
                        # conjoined with the body's proof.
                        head_instance = apply_substitution(unified, sub)
                        if any(get_term_variables(head_instance)):
                            raise ValueError(
                                f"Open predicate instance {head_instance} is "
                                "not ground after proving its rule body."
                            )
                        f = builder.conjoin(f, builder.leaf(head_instance))
                    yield (
                        {
                            apply_substitution(k, sub): apply_substitution(v, sub)
                            for k, v in answer_substitution.items()
                        },
                        f,
                    )

    def _get_fresh_variable(self) -> Symbol:
        self._counter += 1
        return (f"VAR_{self._counter}",)

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register an additional builtin predicate for this grounder instance."""
        self.builtins[(functor, arity)] = builtin_function


def _check_terminates(goal: Symbol, ancestors: tuple[Symbol, ...]) -> None:
    """Check that ``goal`` is not a call its own proof descends from.

    Resolution proves a goal by proving the body of a clause it matches. When
    that body calls the goal again, up to the names of its variables, proving it
    is proving itself: ``e(X,Y) :- e(X,Z), e(Z,Y).`` calls ``e(a,Z)`` while
    proving ``e(a,Z)``, and every call after it is the same call.

    Raises:
        RecursionError: If ``goal`` is a variant of one of ``ancestors``.
    """
    predicate = get_predicate(goal)
    for ancestor in ancestors:
        if get_predicate(ancestor) == predicate and _is_variant(goal, ancestor):
            raise RecursionError(
                f"Proving {symbol_to_pretty_string(ancestor)} calls "
                f"{symbol_to_pretty_string(goal)}, which is the same call, so "
                "resolution alone does not prove this program. JanusGrounder "
                "tables its calls and proves it."
            )


def _is_variant(goal: Symbol, other: Symbol) -> bool:
    """Whether ``goal`` and ``other`` are the same up to the names of their variables."""
    mgu = calculate_mgu(goal, other)
    if mgu is None:
        return False
    renaming = list(mgu.values())
    return all(is_variable(name) for name in renaming) and len(set(renaming)) == len(
        renaming
    )
