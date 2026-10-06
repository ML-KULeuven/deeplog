#  Copyright (c) 2024-2026. KU Leuven
"""Parse DIMACS CNF inputs into DeepLog formulas."""

from __future__ import annotations

from ..algebraic import BOOLEAN
from ..symbol import Symbol
from ..symbol import with_structure
from .ast import Atom
from .ast import BinaryOp
from .ast import FormulaNode
from .ast import UnaryOp


def parse_dimacs_cnf(dimacs: str) -> FormulaNode:
    """Parse ``dimacs`` into the boolean formula of its clauses.

    Variable ``i`` is the variable ``Vi``, and the literal ``i`` its test
    ``=(Vi,true)``, so ``expectation(V1, ..., Vn): φ`` is the probability that
    the formula ``φ`` holds. A negative literal is the negation of its test, a
    clause the disjunction of its literals, and the formula the conjunction of
    its clauses. An empty clause is ``false``, and a clause holding a literal and
    its negation ``true``.

    Raises:
        ValueError: If ``dimacs`` is not well-formed DIMACS CNF.
    """

    def atom(symbol: Symbol) -> FormulaNode:
        return Atom(with_structure(symbol, BOOLEAN.name))

    def literal(value: int) -> FormulaNode:
        test = atom(("=", (f"V{abs(value)}",), ("true",)))
        return test if value > 0 else UnaryOp(BOOLEAN.negation, test)

    def clause(literals: list[int]) -> FormulaNode:
        if not literals:
            return atom(BOOLEAN.zero)
        present = set(literals)
        if any(-value in present for value in present):
            return atom(BOOLEAN.one)
        return _chain(BOOLEAN.sum, [literal(value) for value in literals])

    return _chain(
        BOOLEAN.product, [clause(literals) for literals in _read_dimacs(dimacs)]
    )


def _chain(operator: str, nodes: list[FormulaNode]) -> FormulaNode:
    """``nodes`` combined left to right with ``operator``."""
    result = nodes[0]
    for node in nodes[1:]:
        result = BinaryOp(operator, result, node)
    return result


def _read_dimacs(dimacs: str) -> list[list[int]]:
    num_vars: int | None = None
    num_clauses: int | None = None
    clauses: list[list[int]] = []
    current_clause: list[int] = []

    for line_no, raw_line in enumerate(dimacs.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) < 4 or parts[1] != "cnf":
                raise ValueError(f"Invalid DIMACS header on line {line_no}: {raw_line}")
            num_vars = int(parts[2])
            num_clauses = int(parts[3])
            continue
        for token in line.split():
            try:
                literal = int(token)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid literal '{token}' on line {line_no}"
                ) from exc
            if literal == 0:
                clauses.append(current_clause)
                current_clause = []
            else:
                current_clause.append(literal)

    if current_clause:
        raise ValueError("DIMACS CNF is missing a terminating 0 for the last clause.")

    if num_clauses is not None and len(clauses) != num_clauses:
        raise ValueError(
            f"Expected {num_clauses} clauses but found {len(clauses)} clauses"
        )

    if num_vars is not None:
        max_var = max(
            (abs(literal) for clause in clauses for literal in clause), default=0
        )
        if max_var > num_vars:
            raise ValueError(
                f"Found variable id {max_var} exceeding declared variable count {num_vars}"
            )

    if not clauses:
        raise ValueError("No clauses found in DIMACS CNF input.")

    return clauses
