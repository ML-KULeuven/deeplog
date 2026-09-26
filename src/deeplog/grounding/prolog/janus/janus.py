#  Copyright (c) 2024-2026. KU Leuven
"""A Prolog grounder based on a meta-prover implemented with SWI-Prolog Janus."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeVar

from deeplog.symbol import Symbol
from deeplog.symbol import get_predicate

from ...builder import ProofBuilder
from ..grounder import Builtin
from ..grounder import OpenPredicates
from ..grounder import PrologGrounder
from ..parser import symbol_to_prolog_str
from ..program import Program
from ..program import is_constraint
from ..program import is_fact
from ..program import is_query
from .prover import JANUS_AVAILABLE
from .prover import JanusProver


if TYPE_CHECKING:
    from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory


T = TypeVar("T")

ROOT = Path(__file__).parent


class JanusGrounder(PrologGrounder):
    """A Prolog grounder based on a meta-prover implemented with SWI-Prolog Janus."""

    def __init__(self):
        """Initialize a Janus-backed grounder and ensure the Prolog code is loaded."""
        super().__init__()
        self._prover = JanusProver(ROOT / "engine.pl")

    def ground(
        self,
        program: Program,
        goal: Symbol,
        factory: DeepLogFormulaFactory[T] | ProofBuilder[T],
        open_predicates: OpenPredicates = frozenset(),
    ) -> dict[Symbol, T]:
        """Prove ``goal`` in ``program`` to one proof formula per ground answer."""
        rows = self._prover.query(
            "prove_query(Program, Prover, Query, Builder, GroundQuery, Formula)",
            {
                "Program": self._prover.consult(
                    _clauses(program, frozenset(open_predicates))
                ),
                "Prover": self._prover,
                "Query": goal,
                "Builder": ProofBuilder.wrapping(factory),
            },
        )
        return {row["GroundQuery"]: row["Formula"] for row in rows}

    @staticmethod
    def is_available() -> bool:
        """Return true if janus_swi is available and the class can be instantiated."""
        return JANUS_AVAILABLE

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register a builtin for the goals this grounder proves."""
        self._prover.add_builtin(functor, arity, builtin_function)


@lru_cache(maxsize=64)
def _clauses(program: Program, open_predicates: OpenPredicates) -> tuple[str, ...]:
    """``program`` as the Prolog clauses the grounder's engine reads.

    The facts of an open predicate become ``leaf/1``; every other clause becomes
    ``rule/2``. Queries and constraints are goals, not clauses. The renderings of
    the 64 most recent programs are cached.
    """
    clauses = [
        f"open_predicate({symbol_to_prolog_str((name,))},{arity})."
        for name, arity in open_predicates
    ]
    for rule in program:
        if is_query(rule) or is_constraint(rule):
            continue
        if is_fact(rule) and get_predicate(rule[1]) in open_predicates:
            clauses.append(f"leaf({symbol_to_prolog_str(rule[1])}).")
        else:
            clauses.append(
                f"rule({symbol_to_prolog_str(rule[1])},"
                f"{symbol_to_prolog_str(rule[2])})."
            )
    return (":- dynamic rule/2, leaf/1, open_predicate/2.", *sorted(clauses))
