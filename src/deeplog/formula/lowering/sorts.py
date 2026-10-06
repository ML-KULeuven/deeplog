#  Copyright (c) 2024-2026. KU Leuven
"""The type check: each variable's domain, from the sorts of the arguments it is.

Every predicate declares the domain each argument of its atoms ranges over, its
sort (:meth:`~deeplog.formula.predicates.predicate.Predicate.domains_of`). An
argument of sort a named or a tensor domain takes a variable over that domain or
over part of it, and an argument of sort the values
(:meth:`~deeplog.variable.Domain.of_values`) a variable over values without
names. The compiler's ``=`` compares values of any sort, so it gives none.

A variable an aggregation binds is that aggregation's own: it is the arguments
it is in the aggregation's body and params, outside any aggregation there that
binds its name again. A variable no aggregation binds is free, and is one
variable throughout the formulas of a compile.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from typing import NamedTuple

from ...symbol import Symbol
from ...symbol import get_args
from ...symbol import get_predicate
from ...symbol import get_term_variables
from ...symbol import is_variable
from ...symbol import structure_of
from ...symbol import symbol_to_pretty_string
from ...symbol import unwrap_structure
from ...symbol import without_structure
from ...variable import Domain
from ...variable import SymbolicDomain
from ...variable import ValueDomain
from ..ast import Aggregation
from ..ast import Atom
from ..ast import FormulaNode
from ..ast import children
from .builder_protocols import DeclaresDomains


if TYPE_CHECKING:
    from .compiler import Compiler


class Sorts:
    """The domains of the variables of one compile's formulas.

    A variable ranges over its domain in the compiler's ``variables``, which
    each argument it is must include, or else over the named or tensor domain of
    the arguments it is that each of the others includes. A bound variable no
    argument gives a domain ranges over the compiler's default
    (:meth:`~deeplog.formula.lowering.compiler.Compiler.domain`). The sorts are
    read from the modules of the compiler's atom builders, each predicate built
    once for the atoms read together.
    """

    def __init__(self, compiler: Compiler) -> None:
        """Read sorts with ``compiler``'s atom builders and declared domains."""
        self._compiler = compiler
        #: Where each variable of an atom read so far is an argument, by atom.
        self._atoms: dict[Symbol, _Free] = {}
        #: Where each variable free in a node read so far is an argument, by
        #: node id. The node is kept beside it, so its id is not reused.
        self._free: dict[int, tuple[FormulaNode, _Free]] = {}
        #: The domain the arguments give each binder of an aggregation read so
        #: far, or its declared one, or ``None``, by node id.
        self._bound: dict[int, dict[Symbol, Domain | None]] = {}

    def free(self, nodes: Iterable[FormulaNode]) -> dict[Symbol, Domain]:
        """The domain the arguments give each undeclared variable free in ``nodes``.

        A variable no argument gives a domain is left out.

        Raises:
            TypeError: If an atom builder returns a module that declares no sorts.
            ValueError: If no domain of the arguments a variable is lies within all
                of them, or one does not include the variable's declared domain;
                or a variable over named values is an argument of sort the values.
        """
        nodes = list(nodes)
        self._read(nodes)
        found = _merged(self._free[id(node)][1] for node in nodes)
        domains: dict[Symbol, Domain] = {}
        for variable, uses in found.items():
            domain = self._domain(variable, uses, bound=False)
            if domain is not None and variable not in self._compiler.variables:
                domains[variable] = domain
        return domains

    def binders(self, node: Aggregation) -> dict[Symbol, Domain]:
        """The domain of each of ``node``'s binders.

        Raises:
            TypeError: If an atom builder returns a module that declares no sorts.
            ValueError: As :meth:`free`, or if a binder that is only an argument
                of sort the values has no declared domain, or no domain is
                declared or given for one and the compiler has no default.
        """
        self._read([node])
        return {
            binder: domain if domain is not None else self._compiler.domain(binder)
            for binder, domain in self._bound[id(node)].items()
        }

    def variables(self, node: FormulaNode) -> frozenset[Symbol]:
        """The variables free in ``node``.

        Raises:
            TypeError: If an atom builder returns a module that declares no sorts.
            ValueError: As :meth:`binders`, for an aggregation beneath ``node``.
        """
        self._read([node])
        return frozenset(self._free[id(node)][1])

    def _read(self, nodes: list[FormulaNode]) -> None:
        """Read ``nodes`` and every node beneath them not read yet, children first."""
        order: list[FormulaNode] = []
        atoms: dict[Symbol, None] = {}
        seen: set[int] = set()
        stack = [(node, False) for node in nodes]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen or id(node) in self._free:
                continue
            seen.add(id(node))
            stack.append((node, True))
            if isinstance(node, Atom):
                if node.atom not in self._atoms:
                    atoms[node.atom] = None
            else:
                stack.extend((child, False) for child in children(node))
        self._read_atoms(atoms)
        for node in order:
            if isinstance(node, Atom):
                free = self._atoms[node.atom]
            else:
                free = _merged(self._free[id(child)][1] for child in children(node))
            if isinstance(node, Aggregation):
                free = dict(free)
                self._bound[id(node)] = {
                    binder: self._domain(binder, free.pop(binder, _UNUSED), bound=True)
                    for binder in node.binders
                }
            self._free[id(node)] = (node, free)

    def _read_atoms(self, atoms: Iterable[Symbol]) -> None:
        """Read where the variables of ``atoms`` are arguments, building each predicate once.

        Raises:
            TypeError: If an atom builder returns a module that declares no sorts.
        """
        grouped: dict[tuple[str, int, str], list[Symbol]] = {}
        for atom in atoms:
            self._atoms[atom] = {
                variable: _UNUSED
                for argument in get_args(without_structure(atom))
                for variable in get_term_variables(argument)
            }
            structure = structure_of(atom)
            if structure is None:
                continue
            key = (*get_predicate(unwrap_structure(atom)), structure)
            builder = self._compiler._atom_builders.get(key)
            if builder is not None and builder is not self._compiler._equality:
                grouped.setdefault(key, []).append(atom)
        for key, members in grouped.items():
            module = self._compiler._atom_builders[key](members)
            name = f"{key[0]}/{key[1]}"
            if not isinstance(module, DeclaresDomains):
                raise TypeError(
                    f"The builder for {name} in {key[2]!r} returned a module that "
                    "declares no sorts; an atom builder's module declares the "
                    "domain of each argument, as every Predicate does "
                    "(DeclaresDomains)."
                )
            for atom in members:
                arguments = get_args(unwrap_structure(atom))
                uses = self._atoms[atom]
                for argument, sort in zip(
                    arguments, module.domains_of(arguments), strict=True
                ):
                    if is_variable(argument):
                        uses[argument] = uses[argument].joined(_Uses.of(sort, name))

    def _domain(self, variable: Symbol, uses: _Uses, *, bound: bool) -> Domain | None:
        """The domain ``variable`` ranges over where it is the arguments of ``uses``.

        Its declared domain, else the one its arguments give it, else ``None``.

        Raises:
            ValueError: If no domain of the arguments lies within all of them, or
                one does not include the declared domain; ``variable`` ranges over
                named values and is an argument of sort the values; or it is
                ``bound``, only an argument of sort the values, and undeclared.
        """
        declared = self._compiler.variables.get(variable)
        if declared is None:
            domain = _innermost(variable, uses.sorts) if uses.sorts else None
        else:
            domain = declared
            for sort, name in uses.sorts:
                if not within(declared, sort):
                    raise ValueError(
                        f"{_pretty(variable)} is declared over {describe(declared)}, "
                        f"but {name} reads it over {describe(sort)}, which does not "
                        "include it."
                    )
        if uses.valued is None:
            return domain
        if isinstance(domain, SymbolicDomain):
            raise ValueError(
                f"{_pretty(variable)} ranges over {describe(domain)}, but "
                f"{uses.valued} reads it as a value without a name."
            )
        if domain is None and bound:
            raise ValueError(
                f"{_pretty(variable)} is bound, and {uses.valued} reads it as a "
                "value, but nothing gives it a domain; declare one, as "
                "Compiler(variables={...})."
            )
        return domain


class _Uses(NamedTuple):
    """The sorts of the arguments a variable is, each with its predicate, and the
    first predicate reading it as values without names."""

    sorts: tuple[tuple[Domain, str], ...] = ()
    valued: str | None = None

    @staticmethod
    def of(sort: Domain, name: str) -> _Uses:
        """A variable that is an argument of ``name`` of sort ``sort``."""
        if isinstance(sort, ValueDomain):
            return _Uses(valued=name)
        return _Uses(sorts=((sort, name),))

    def joined(self, other: _Uses) -> _Uses:
        """The arguments of both."""
        if not other.sorts and (other.valued is None or self.valued is not None):
            return self
        return _Uses(
            self.sorts
            + tuple(found for found in other.sorts if found not in self.sorts),
            self.valued if self.valued is not None else other.valued,
        )


_UNUSED = _Uses()

#: Each variable free in a node, with the arguments it is there.
_Free = dict[Symbol, _Uses]


def _merged(parts: Iterable[_Free]) -> _Free:
    """The variables free in any of ``parts``, with the arguments they are in all.

    A part is never changed, so one with every variable is returned as it is.
    """
    merged: _Free | None = None
    for part in parts:
        if not part:
            continue
        if merged is None:
            merged = part
            continue
        if merged is part:
            continue
        joined = dict(merged)
        for variable, uses in part.items():
            known = joined.get(variable)
            joined[variable] = uses if known is None else known.joined(uses)
        merged = joined
    return merged if merged is not None else {}


def within(domain: Domain, sort: Domain) -> bool:
    """Whether every value of ``domain`` is a value of ``sort``, both named or both not."""
    if isinstance(domain, SymbolicDomain) and isinstance(sort, SymbolicDomain):
        return set(domain.names) <= set(sort.names)
    if isinstance(domain, (SymbolicDomain, ValueDomain)) or isinstance(
        sort, (SymbolicDomain, ValueDomain)
    ):
        return False
    rows, among = domain.as_tensor(), sort.as_tensor()
    if rows.shape[1:] != among.shape[1:]:
        return False
    found = rows.unsqueeze(1) == among.unsqueeze(0)
    return bool(found.reshape(*found.shape[:2], -1).all(dim=-1).any(dim=1).all())


def _innermost(variable: Symbol, found: tuple[tuple[Domain, str], ...]) -> Domain:
    """The domain among ``found`` that every other one includes.

    Raises:
        ValueError: If none does: ``variable``'s domain is then the caller's to
            declare.
    """
    for sort, _ in found:
        if all(within(sort, other) for other, _ in found):
            return sort
    (first, by), (second, also) = next(
        (a, b)
        for a in found
        for b in found
        if not within(a[0], b[0]) and not within(b[0], a[0])
    )
    raise ValueError(
        f"{_pretty(variable)} ranges over {describe(first)} as an argument of "
        f"{by}, and over {describe(second)} as one of {also}, and neither "
        "includes the other; declare the domain it ranges over."
    )


def describe(domain: Domain) -> str:
    """``domain`` in words, for an error message."""
    if isinstance(domain, SymbolicDomain):
        names = ", ".join(map(_pretty, domain.names[:6]))
        return "{" + names + (", …}" if len(domain.names) > 6 else "}")
    if isinstance(domain, ValueDomain):
        return "the values"
    return f"{len(domain)} unnamed values"


def _pretty(symbol: Symbol) -> str:
    return symbol_to_pretty_string(symbol)
