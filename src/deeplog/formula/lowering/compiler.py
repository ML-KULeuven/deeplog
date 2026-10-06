#  Copyright (c) 2024-2026. KU Leuven
"""The compiler: formulas in, one module out."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import overload

from ...algebraic import BOOLEAN
from ...algebraic import AlgebraicStructure
from ...algebraic import get_algebraic_structure
from ...module.deeplog_module import DeepLogModule
from ...module.reshape import reshape
from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from ...shape import get_all_symbols
from ...symbol import Symbol
from ...symbol import parse_symbol
from ...symbol import structure_of
from ...symbol import symbol_to_pretty_string
from ...symbol import with_structure
from ...variable import Domain
from ..ast import FormulaNode
from ..ast import fold
from ..ast import hash_cons
from ..circuit_factory import CircuitFactory
from ..predicates.builtin_predicates import EqualityPredicate
from .builder_protocols import AggregationBuilder
from .builder_protocols import AtomBuilder
from .builder_protocols import TransformationBuilder
from .default_builders import default_aggregation_builders
from .default_builders import default_atom_builders
from .default_builders import default_transformation_builders
from .lowering import Lowering


class Compiler:
    """Compiles formulas to one module, with the domains, algebras and builders to use.

    ``variables`` declares the domain each binder ranges over; a binder it does
    not declare is a reification variable, ranging over the values of
    ``reification``. ``structures`` names the algebras a formula may use beside
    the registered ones, and wins over them. ``atom_builders``,
    ``aggregation_builders`` and ``transformation_builders`` add builders, keyed
    by predicate, operation and pair of algebras, and each wins over the default
    registered for its key. An aggregation whose operation has no builder is computed by
    :func:`~deeplog.formula.lowering.reduction.reduction`.

    The value test ``=/2`` is the compiler's own: it reads a variable's value as
    its position in the variable's domain (:meth:`domain`), which only the
    compiler knows.

    An atom no builder computes is an input of the compiled module, named by the
    atom. Where it reads variables an aggregation binds, it is its ground atoms
    instead, an input each, named by substituting those variables' values, and
    read at the values the aggregation gives them; such a variable must range
    over named values. A test ``=(V, v)`` outside ``boolean`` does not read
    ``V``: it is the label of ``V``'s value ``v``.

    Every argument of an atom has a sort, the domain its predicate declares for
    it (:meth:`~deeplog.formula.predicates.predicate.Predicate.domains_of`). A
    variable ranges over its domain in ``variables``, or else over the named or
    tensor domain of the arguments it is that each of the others includes; each
    must include the variable's domain, and a variable over part of a named one
    reaches its predicate as a position in it. An argument of sort the values
    takes a variable over values without names. A variable that no argument
    gives a domain ranges over its declared one, or, if it is bound and only
    ``=`` tests it, over the values of ``reification``. A variable an
    aggregation binds is that aggregation's own, so two aggregations binding
    one name may give it different domains; a free variable is one variable
    in every formula compiled together
    (:class:`~deeplog.formula.lowering.sorts.Sorts`).
    """

    def __init__(
        self,
        variables: Mapping[Symbol, Domain] | None = None,
        reification: AlgebraicStructure = BOOLEAN,
        structures: Mapping[str, AlgebraicStructure] | None = None,
        aggregation_builders: Mapping[str, AggregationBuilder] | None = None,
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder] | None = None,
        transformation_builders: Mapping[tuple[str, str], TransformationBuilder]
        | None = None,
    ) -> None:
        """Configure the compiler with the domains, algebras and builders to use."""
        self.variables: dict[Symbol, Domain] = dict(variables or {})
        self.reification = reification
        self._structures = dict(structures or {})
        self._given_atom_builders = dict(atom_builders or {})
        #: The builder of ``=``, which reads its variables' domains.
        self._equality = partial(EqualityPredicate, domain_of=self.domain)
        self._atom_builders: dict[tuple[str, int, str], AtomBuilder] = {
            **default_atom_builders,
            ("=", 2, "boolean"): self._equality,
            **self._given_atom_builders,
        }
        self._aggregation_builders = {
            **default_aggregation_builders,
            **(aggregation_builders or {}),
        }
        #: Casts between algebras, keyed ``(from, to)``. A module whose outputs
        #: take the ``probability -> logprobability`` cast declares
        #: ``probability`` on its output symbols; the source structure is never
        #: inferred from an absent label.
        self._transformation_builders = {
            **default_transformation_builders,
            **(transformation_builders or {}),
        }

    @overload
    def compile(
        self, formulas: Mapping[str | Symbol, FormulaNode], /
    ) -> DeepLogModule: ...

    @overload
    def compile(self, *formulas: FormulaNode) -> DeepLogModule: ...

    def compile(
        self, *formulas: FormulaNode | Mapping[str | Symbol, FormulaNode]
    ) -> DeepLogModule:
        """Compile ``formulas`` to one module with a column per formula, in order.

        Given as one mapping, each formula's column is named by its key, a symbol
        or the text of one; given as formulas, the ``i``-th column is named
        ``@i``. A name is labelled with the algebra of its column, as
        ``rain_implies_wet _ probability``, and equal formulas compile once,
        whatever their names.

        Each formula is constructed bottom-up by folding it through a
        :class:`~deeplog.formula.circuit_factory.CircuitFactory`, which rewrites
        every region a circuit can hold into a lump, and the results are lowered
        together, top-down (:class:`~deeplog.formula.lowering.lowering.Lowering`).
        Equal subformulas of all the formulas are first made one object, and
        construction runs under one memo, so a subformula is built once however
        many formulas write it, their shared atoms are one circuit leaf, and
        counts over one boolean circuit take one knowledge compilation.

        Raises:
            TypeError: If a mapping is given beside formulas.
            ValueError: If no formula is given, two formulas are given one name,
                or a name is labelled with an algebra its formula's value is not
                in.
        """
        if len(formulas) == 1 and isinstance(formulas[0], Mapping):
            named = [
                (parse_symbol(name) if isinstance(name, str) else name, formula)
                for name, formula in formulas[0].items()
            ]
        else:
            named = []
            for i, formula in enumerate(formulas):
                if isinstance(formula, Mapping):
                    raise TypeError(
                        "compile takes formulas, or one mapping from name to "
                        "formula, not both."
                    )
                named.append(((f"@{i}",), formula))
        if not named:
            raise ValueError("At least one formula is required.")
        circuits = CircuitFactory(self._structures)
        interned: dict[FormulaNode, FormulaNode] = {}
        built: dict[int, tuple[FormulaNode, FormulaNode]] = {}
        roots = [
            fold(hash_cons(formula, interned), circuits, memo=built)
            for _, formula in named
        ]
        distinct = {root: i for i, root in enumerate(dict.fromkeys(roots))}
        module = Lowering(self).lower(*distinct)
        columns = list(get_all_symbols(module.get_output_shape()))
        outputs: dict[Symbol, Symbol] = {}
        for (name, _), root in zip(named, roots, strict=True):
            column = columns[distinct[root]]
            output = _labelled(name, column)
            if output in outputs:
                raise ValueError(
                    f"Two formulas are named {symbol_to_pretty_string(output)}."
                )
            outputs[output] = column
        selected = reshape(module, output=SymTensor(list(outputs.values())))
        return WrappedModule(
            selected,
            selected.get_input_shape(),
            SymTensor(list(outputs)),
            name="compiled",
        )

    def domain(self, binder: Symbol) -> Domain:
        """The domain ``binder`` ranges over.

        A binder the compiler declares in its ``variables`` uses that domain. Any
        other binder is a reification variable, and inherits the values of the
        compiler's ``reification`` algebra as its domain.

        Raises:
            ValueError: If the binder is undeclared and the reification algebra
                declares no enumerable values.
        """
        declared = self.variables.get(binder)
        if declared is not None:
            return declared
        try:
            return Domain.of_structure(self.reification)
        except ValueError as error:
            raise ValueError(f"No domain for binder {binder}: {error}") from None

    def algebra(self, name: str) -> AlgebraicStructure:
        """The algebra named ``name``: the compiler's own, else the registered one.

        Raises:
            ValueError: If ``name`` is neither the compiler's own nor registered.
        """
        if name in self._structures:
            return self._structures[name]
        return get_algebraic_structure(name)

    def atom_builder(
        self, functor: str, arity: int, structure: str
    ) -> AtomBuilder | None:
        """The builder of the ``functor``/``arity`` atoms in ``structure``, or ``None``."""
        return self._atom_builders.get((functor, arity, structure))

    def declaring(
        self,
        variables: Mapping[Symbol, Domain] | None = None,
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder] | None = None,
    ) -> Compiler:
        """A copy of this compiler that also declares ``variables`` and ``atom_builders``.

        For a front end that introduces binders and predicates of its own, whose
        caller cannot know them. Its builders shadow the compiler's defaults, such
        as ``p/2`` in probability for a program's own ``p/1`` extended with a
        binder. This compiler is left as it is.

        Raises:
            ValueError: If a binder is already declared with another domain, or
                this compiler was given a builder for one of the keys.
        """
        variables = dict(variables or {})
        atom_builders = dict(atom_builders or {})
        for binder, domain in variables.items():
            declared = self.variables.get(binder)
            if declared is not None and declared != domain:
                raise ValueError(
                    f"{symbol_to_pretty_string(binder)} is already declared over "
                    "another domain."
                )
        for key in atom_builders:
            if key in self._given_atom_builders:
                raise ValueError(
                    f"A builder is already registered for {key[0]}/{key[1]} in "
                    f"{key[2]}."
                )
        return self._binding(variables, atom_builders)

    def _binding(
        self,
        variables: Mapping[Symbol, Domain],
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder] | None = None,
    ) -> Compiler:
        """A copy of this compiler whose ``variables`` range over the domains given.

        A variable declared here already ranges over the one given instead.
        """
        return Compiler(
            variables={**self.variables, **variables},
            reification=self.reification,
            structures=self._structures,
            aggregation_builders=self._aggregation_builders,
            atom_builders={**self._given_atom_builders, **(atom_builders or {})},
            transformation_builders=self._transformation_builders,
        )


def _labelled(name: Symbol, column: Symbol) -> Symbol:
    """``name``, labelled with the algebra of ``column``.

    Raises:
        ValueError: If ``name`` is labelled with another algebra.
    """
    structure = structure_of(column)
    given = structure_of(name)
    if given is None:
        return name if structure is None else with_structure(name, structure)
    if given != structure:
        raise ValueError(
            f"{symbol_to_pretty_string(name)} is labelled {given}, but names a "
            f"value of {structure}."
        )
    return name
