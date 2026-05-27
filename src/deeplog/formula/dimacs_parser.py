#  Copyright (c) 2024-2026. KU Leuven
"""Parse DIMACS CNF inputs into DeepLog formulas using a factory."""

from __future__ import annotations

from typing import Any

from ..algebraic import Algebra
from ..algebraic import get_algebraic_structure
from ..symbol import Symbol
from .deeplogformulafactory import DeepLogFormulaFactory


def parse_dimacs_cnf[T](
    dimacs: str,
    factory: DeepLogFormulaFactory[T],
    *,
    structure: str | Algebra = "boolean",
) -> T:
    """Parse ``dimacs`` text and emit a DeepLog deeplogfactory via ``factory``."""
    if isinstance(structure, str):
        resolved = get_algebraic_structure(structure)
        if not isinstance(resolved, Algebra):
            raise TypeError(
                f"Structure '{structure}' is not an Algebra (has no negation operator)"
            )
        algebra = resolved
        structure_name = structure
    else:
        algebra = structure
        structure_name = algebra.name

    clauses = _read_dimacs(dimacs)
    and_op = algebra.product
    or_op = algebra.sum
    not_op = algebra.negation

    def literal_to_node(literal: int):
        symbol: Symbol = (f"v{abs(literal)}",)
        leaf = factory.create_atom(("_", symbol, (structure_name,)))
        return leaf if literal > 0 else factory.create_unary_node(not_op, leaf)

    def clause_to_node(literals: list[int]):
        if not literals:
            return factory.create_atom(("_", algebra.zero, (structure_name,)))
        literal_set = set(literals)
        if any(-literal in literal_set for literal in literal_set):
            return factory.create_atom(("_", algebra.one, (structure_name,)))
        nodes = [literal_to_node(literal) for literal in literals]
        return _fold_binary(nodes, factory, or_op)

    clause_nodes = [clause_to_node(clause) for clause in clauses]
    return _fold_binary(clause_nodes, factory, and_op)


def _fold_binary(nodes: list[Any], factory: DeepLogFormulaFactory[Any], operator: str):
    if not nodes:
        raise ValueError("Cannot fold an empty set of nodes")
    result = nodes[0]
    for node in nodes[1:]:
        result = factory.create_binary_node(operator, result, node)
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
