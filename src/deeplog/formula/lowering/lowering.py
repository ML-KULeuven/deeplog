#  Copyright (c) 2024-2026. KU Leuven
"""The top-down walk that lowers constructed formulas to one module.

After construction a formula is lumps (:class:`~deeplog.formula.ast.CircuitNode`)
and the symbolic nodes between them. :meth:`Lowering.lower` lowers formulas a
*level* at a time: everything reachable from them without passing through an
:class:`~deeplog.formula.ast.Aggregation`, the aggregations included. Every node
of a level computes one column, named from the node: an atom's column is its
symbol, a lump's its :func:`~deeplog.formula.circuit_node.lump_name`, a cast's
its :func:`~deeplog.formula.circuit_node.cast_name`, an operator's the operator
over its operands' names, and an aggregation's its operation over its binders
and its operands' names. One handler per AST type builds the modules computing
the columns of every node of that type in the level, and the modules are
composed into one module graph wired by those names, in which each runs once per
forward.

An aggregation's builder lowers what lies beneath it through
:meth:`Lowering.lower`, which starts the next level down.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from functools import partial
from itertools import product
from typing import TYPE_CHECKING
from typing import assert_never

import torch
from torch import Tensor

from ...module.aggregation_modules import AggregationOpModule
from ...module.deeplog_module import DeepLogModule
from ...module.module_circuit import ModuleCircuit
from ...module.module_circuit import compose_modules
from ...module.reshape import construct_transformation
from ...module.reshape import reshape
from ...module.wrappers import WrappedModule
from ...shape import SymTensor
from ...shape import get_all_symbols
from ...symbol import Symbol
from ...symbol import apply_substitution
from ...symbol import free_variables
from ...symbol import get_args
from ...symbol import get_predicate
from ...symbol import get_term_variables
from ...symbol import is_variable
from ...symbol import retag
from ...symbol import structure_of
from ...symbol import symbol_to_str
from ...symbol import unwrap_structure
from ...symbol import with_structure
from ...symbol import without_structure
from ...variable import Domain
from ...variable import SymbolicDomain
from ..ast import Aggregation
from ..ast import AggregationOp
from ..ast import Atom
from ..ast import BinaryOp
from ..ast import CircuitNode
from ..ast import FormulaNode
from ..ast import Transformation
from ..ast import UnaryOp
from ..ast import children
from ..ast import operands
from ..circuit_node import cast_name
from ..circuit_node import lump_name
from ..circuit_node import to_module as circuit_to_module
from ..predicates.predicate import Predicate
from .builder_protocols import DeclaresDomains
from .reduction import aggregator
from .reduction import reduction
from .sorts import Sorts


if TYPE_CHECKING:
    from .compiler import Compiler


class Lowering:
    """One compile's walk: lowers constructed formulas top-down, a level at a time.

    An aggregation builder receives the walk, and lowers through :meth:`lower`
    what its method needs, with the domains and builders of its :attr:`compiler`,
    which declares the domain of each binder of the aggregations it builds.

    The walk's handlers mirror the construction factory's eliminators, one per
    AST type, but each receives every node of its type in a level, unlowered:

    - atoms go to their atom builders, one call per predicate;
    - unary and binary operators become the algebra's operator functions, read
      over their operands' columns;
    - aggregations of operands become the algebra's aggregators, one module
      reading all their operands' columns;
    - casts go to the builder registered for their pair of algebras;
    - aggregations go to the builder registered for their operation, one call
      per operation, apart from those where one binds a variable another
      reads free, or two bind one name over different domains;
    - lumps of one circuit compile into one core, apart from a lump that reads
      another of its circuit through a cast.
    """

    def __init__(self, compiler: Compiler) -> None:
        """Lower with ``compiler``'s domains, algebras and builders."""
        self._compiler = compiler
        self._sorts = Sorts(compiler)
        #: The column of every node named so far, by node id. The node is kept
        #: beside it, so its id is not reused while the walk runs.
        self._columns: dict[int, tuple[FormulaNode, Symbol]] = {}
        #: The module computing each aggregation lowered so far, by node id and
        #: what its free variables are (:meth:`_context`). It reads no column of
        #: the level it sits in, so a node reached in several levels where its
        #: free variables are the same is built once.
        self._aggregations: dict[tuple, tuple[Aggregation, DeepLogModule]] = {}
        #: The variables the aggregations being built bind.
        self._binders: frozenset[Symbol] = frozenset()
        #: The variables the innermost aggregations being built bind, which the
        #: level being lowered ranges over: a column holding one has its axis.
        self._own: frozenset[Symbol] = frozenset()
        #: While aggregations are built, each atom of their formulas that no
        #: builder computes and that reads variables they bind, with those
        #: variables.
        self._bound: dict[Symbol, tuple[Symbol, ...]] = {}
        #: While :meth:`_lower_roots` runs, the atoms computed before any
        #: aggregation, which every level reads as inputs.
        self._once: frozenset[Symbol] | None = None
        #: The symbols each module of :meth:`_held` reads and holds, by its id.
        #: The module is kept beside them, so its id is not reused.
        self._symbols: dict[int, tuple[DeepLogModule, frozenset, frozenset]] = {}

    @property
    def compiler(self) -> Compiler:
        """The compiler whose domains, algebras and builders this walk lowers with."""
        return self._compiler

    def lower(self, *nodes: FormulaNode) -> DeepLogModule:
        """One module with a column per node, in the order given.

        A variable free in ``nodes`` that the :attr:`compiler` does not declare
        ranges over the domain its arguments there give it, from then on.

        Raises:
            ValueError: If no node is given, or if two nodes name one column.
        """
        return self._lower_above(nodes, ())

    def _lower_roots(self, *nodes: FormulaNode) -> DeepLogModule:
        """:meth:`lower` the formulas of a compile, computing first what reads no variable.

        The atoms beneath ``nodes``, at any depth, whose builder's module reads
        none of their variables are computed first, by one module per predicate,
        and every level reads them as inputs: a network fed no variable runs
        once, however many aggregations read its rows and however deep.
        """
        first = self._computed_once(nodes)
        self._once = frozenset(
            symbol for module in first for symbol in self._produced(module)
        )
        try:
            main = self.lower(*nodes)
        finally:
            self._once = None
        read = set(get_all_symbols(main.get_input_shape()))
        used = [module for module in first if read & self._produced(module)]
        if not used:
            return main
        return compose_modules([*used, main], main.get_output_shape())

    def _computed_once(self, nodes: Sequence[FormulaNode]) -> list[DeepLogModule]:
        """The modules computing the atoms beneath ``nodes`` that read no variable, one per predicate.

        An atom is among them when the module its builder built for its sorts
        (:meth:`~deeplog.formula.lowering.sorts.Sorts.module`) reads none of its
        variables.
        """
        # Reading the sorts of all of them first builds each predicate once.
        self._sorts.free(nodes)
        grouped: dict[tuple[str, int, str], dict[Symbol, None]] = {}
        for node in _atoms_beneath(nodes):
            module = self._sorts.module(node)
            if module is None or self._sorts.variables(node) & self._reads(module):
                continue
            structure = structure_of(node.atom)
            assert structure is not None
            key = (*get_predicate(unwrap_structure(node.atom)), structure)
            grouped.setdefault(key, {})[node.atom] = None
        builders = self._compiler._atom_builders
        return [builders[key](list(atoms)) for key, atoms in grouped.items()]

    def _held(self, node: Atom) -> frozenset[Symbol]:
        """The variables of ``node`` at which its builder's module holds its ground atoms.

        Those of its variables the module built for its sorts does not read,
        where that module holds not ``node`` itself but, in its place, its
        ground atoms (:meth:`_lower_atoms`). None for an atom no builder gives
        sorts.
        """
        module = self._sorts.module(node)
        if module is None:
            return frozenset()
        unread = self._sorts.variables(node) - self._reads(module)
        if not unread or node.atom in self._produced(module):
            return frozenset()
        return unread

    def _reads(self, module: DeepLogModule) -> frozenset[Symbol]:
        """The symbols ``module`` reads."""
        return self._symbols_of(module)[1]

    def _produced(self, module: DeepLogModule) -> frozenset[Symbol]:
        """The symbols ``module`` holds among its outputs."""
        return self._symbols_of(module)[2]

    def _symbols_of(
        self, module: DeepLogModule
    ) -> tuple[DeepLogModule, frozenset[Symbol], frozenset[Symbol]]:
        if id(module) not in self._symbols:
            self._symbols[id(module)] = (
                module,
                frozenset(get_all_symbols(module.get_input_shape())),
                frozenset(get_all_symbols(module.get_output_shape())),
            )
        return self._symbols[id(module)]

    def _lower_above(
        self, nodes: Sequence[FormulaNode], given: Collection[FormulaNode]
    ) -> DeepLogModule:
        """:meth:`lower` ``nodes``, reading the column of each node of ``given`` as an input.

        Nothing at or beneath a node of ``given`` is built. No node of ``nodes``
        is among them.
        """
        if not nodes:
            raise ValueError("At least one formula is required.")
        undeclared = {
            variable: domain
            for variable, domain in self._sorts.free(nodes).items()
            if variable not in self._compiler.variables
        }
        if undeclared:
            self._compiler = self._compiler._binding(undeclared)
        level = _gather(nodes, given)
        self._name_all(level)
        vertices = self._lower_aggregations(_of_type(level, Aggregation))
        vertices += self._lower_atoms(_of_type(level, Atom))
        vertices += self._lower_circuits(
            _of_type(level, CircuitNode), _lump_heights(level)
        )
        vertices += self._lower_transformations(
            _one_per_column(_of_type(level, Transformation), self._column)
        )
        vertices += self._lower_unary_nodes(
            _one_per_column(_of_type(level, UnaryOp), self._column)
        )
        vertices += self._lower_binary_nodes(
            _one_per_column(_of_type(level, BinaryOp), self._column)
        )
        vertices += self._lower_aggregation_ops(
            _one_per_column(_of_type(level, AggregationOp), self._column)
        )
        outputs = _distinct([self._column(node) for node in nodes])
        if not self._own:
            return _compose(vertices, outputs)
        read = set(get_all_symbols(v.get_input_shape() for v in vertices))
        made = set(get_all_symbols(v.get_output_shape() for v in vertices))
        over = sorted(self._own, key=symbol_to_str)
        vertices += [
            self._values(binder)
            for binder in over
            if binder in read and binder not in made
        ]
        unique = list({id(v): v for v in vertices}.values())
        return ModuleCircuit(unique, SymTensor(outputs))

    def _lower_over(
        self, binders: Sequence[Symbol], nodes: Sequence[FormulaNode]
    ) -> DeepLogModule:
        """:meth:`lower` ``nodes`` over ``binders``: a column holding one has its axis."""
        enclosing = self._own
        self._own = frozenset(binders)
        try:
            return self.lower(*nodes)
        finally:
            self._own = enclosing

    def _values(self, binder: Symbol) -> DeepLogModule:
        """The module giving ``binder``'s values along its axis, as its domain enumerates them."""
        values = self._compiler.domain(binder).as_tensor()
        return WrappedModule(
            partial(_spread, values), SymTensor([]), SymTensor([binder]), name="values"
        )

    def _name_all(self, nodes: list[FormulaNode]) -> None:
        """Name the column of each of ``nodes`` not named yet, in order.

        ``nodes`` come children first, so each is named from named children.
        """
        for node in nodes:
            if id(node) not in self._columns:
                self._columns[id(node)] = (node, _name(node, self._column))

    def _column(self, node: FormulaNode) -> Symbol:
        """The column ``node`` computes, naming its level first if it is unnamed.

        Naming builds nothing, so an operand an aggregation's builder did not
        lower is named here too, since the aggregation's column is named after
        it.
        """
        if id(node) not in self._columns:
            self._name_all(_gather([node]))
        return self._columns[id(node)][1]

    # -- One handler per AST type ---------------------------------------------

    def _lower_atoms(self, atoms: list[Atom]) -> list[DeepLogModule]:
        """Build every atom's predicate, one builder call per predicate.

        A builder gets all the atoms of its predicate and makes one module
        holding each, by name, anywhere among its outputs and in any shape, so a
        network whose rows give several atoms runs once for them. A module may
        hold an atom's ground atoms in its place, at the variables of the atom
        it does not read, and the atom is then read from them at those
        variables' values (:meth:`_ground`). An atom with no builder, a baked
        constant among them, is an input of the module the walk returns, or,
        where it reads variables an aggregation binds, its ground atoms are
        (:class:`~deeplog.formula.lowering.compiler.Compiler`). A variable whose
        named domain is part of its argument's reaches the predicate as a
        position in the argument's (:meth:`_translated`).

        The symbols the atoms of one predicate give an argument position share
        that position's input tensor, so they must share a feature shape.

        Raises:
            ValueError: If an atom carries no structure label, or a builder's
                module holds neither an atom it was given nor its ground atoms.
            NotImplementedError: If a module other than a predicate reads a
                variable over part of its argument's named domain.
        """
        builders = self._compiler._atom_builders
        grouped: dict[tuple, dict[Symbol, None]] = {}
        grounded: dict[Symbol, tuple[Symbol, ...]] = {}
        for node in atoms:
            if self._once is not None and node.atom in self._once:
                continue
            structure = structure_of(node.atom)
            if structure is None:
                raise ValueError(f"Invalid atom: {node.atom}")
            key = (
                *get_predicate(unwrap_structure(node.atom)),
                structure,
                frozenset(free_variables(node.atom)) & self._own,
            )
            if (variables := self._read_from_once(node)) is not None:
                grounded[node.atom] = variables
            elif key[:3] in builders:
                grouped.setdefault(key, {})[node.atom] = None
            elif node.atom in self._bound:
                grounded[node.atom] = self._bound[node.atom]
        modules = [
            self._ground(atom, variables) for atom, variables in grounded.items()
        ]
        for key, members in grouped.items():
            module = builders[key[:3]](list(members))
            produced = set(get_all_symbols(module.get_output_shape()))
            read = set(get_all_symbols(module.get_input_shape()))
            held: dict[Symbol, tuple[Symbol, ...]] = {}
            for atom in members:
                if atom not in produced:
                    held[atom] = tuple(
                        dict.fromkeys(
                            variable
                            for argument in get_args(unwrap_structure(atom))
                            for variable in get_term_variables(argument)
                            if variable not in read
                        )
                    )
            if missing := [
                atom
                for atom, variables in held.items()
                if not set(self._ground_atoms(atom, variables) or [atom]) <= produced
            ]:
                functor, arity, structure, _ = key
                raise ValueError(
                    f"The builder for {functor}/{arity} in {structure!r} returned a "
                    f"module holding no {', '.join(map(symbol_to_str, missing))}; it "
                    f"holds {module.get_output_shape()}. A builder's module holds "
                    "each atom it is given among its outputs, or its ground atoms "
                    "at the variables it does not read."
                )
            modules.append(self._translated(module, list(members)))
            modules += [
                self._ground(atom, variables) for atom, variables in held.items()
            ]
        return modules

    def _read_from_once(self, node: Atom) -> tuple[Symbol, ...] | None:
        """The variables at which ``node`` is read from the atoms computed once, if it is.

        It is where its ground atoms at the variables its builder's module
        holds it at (:meth:`_held`) are all among the atoms
        :meth:`_lower_roots` computed before any aggregation; ``None`` where
        they are not.
        """
        if self._once is None:
            return None
        held = self._held(node)
        variables = tuple(
            dict.fromkeys(
                variable
                for argument in get_args(unwrap_structure(node.atom))
                for variable in get_term_variables(argument)
                if variable in held
            )
        )
        ground = self._ground_atoms(node.atom, variables)
        if not variables or ground is None or not set(ground) <= self._once:
            return None
        return variables

    def _translated(self, module: DeepLogModule, atoms: list[Symbol]) -> DeepLogModule:
        """``module``, reading each variable it reads over part of its argument's named domain there.

        A variable is fed its value's position in its own domain; where its
        argument's sort is a larger named domain, the position is translated
        into that domain's before ``module`` reads it.

        Raises:
            NotImplementedError: If ``module`` is not a predicate and needs a
                translation, since which input an argument is is a predicate's.
            ValueError: If one input of ``module`` needs two translations.
        """
        if not isinstance(module, DeclaresDomains):
            return module
        read = set(get_all_symbols(module.get_input_shape()))
        tables: dict[tuple[int, int], torch.Tensor] = {}
        for atom in atoms:
            arguments = get_args(unwrap_structure(atom))
            for index, (argument, sort) in enumerate(
                zip(arguments, module.domains_of(arguments), strict=True)
            ):
                if (
                    not is_variable(argument)
                    or argument not in read
                    or not isinstance(sort, SymbolicDomain)
                ):
                    continue
                own = self._compiler.domain(argument)
                if own == sort:
                    continue
                if not isinstance(module, Predicate):
                    raise NotImplementedError(
                        f"{symbol_to_str(atom)} reads {symbol_to_str(argument)} "
                        "over part of its argument's domain, which only a "
                        "Predicate can be fed translated."
                    )
                key = module._input_of(argument, index)
                table = torch.tensor([sort.index(value) for value in own.values])
                if key in tables and not torch.equal(tables[key], table):
                    raise ValueError(
                        f"{symbol_to_str(argument)} is read over two domains at "
                        f"argument {index + 1} of {module}."
                    )
                tables[key] = table
        return _Translated(module, tables) if tables else module

    def _ground(self, atom: Symbol, binders: tuple[Symbol, ...]) -> DeepLogModule:
        """The module reading ``atom`` from its ground atoms at its ``binders``' values.

        A binder the level ranges over gives the atom its axis; for each other,
        an input gives its value per row, a named value as its position in its
        domain and a number as itself, and the atom is read at that value. The
        ground atoms are the last input (:meth:`_ground_atoms`).

        Raises:
            ValueError: If a binder's domain does not write its values, so the
                ground atoms have no names.
        """
        ground = self._ground_atoms(atom, binders)
        if ground is None:
            raise ValueError(
                f"{symbol_to_str(atom)} has no builder and reads a variable of "
                f"{', '.join(map(symbol_to_str, binders))} whose values have no "
                "names, so it has no ground atoms to read; register a builder for "
                f"{'/'.join(map(str, get_predicate(unwrap_structure(atom))))} "
                f"in {structure_of(atom)!r}."
            )
        domains = [self._compiler.domain(binder) for binder in binders]
        given = tuple(binder for binder in binders if binder not in self._own)
        axes = tuple(v for v in free_variables(atom) if v in self._own and v in binders)
        return WrappedModule(
            partial(
                _look_up,
                tuple(map(len, domains)),
                tuple(
                    None if isinstance(domain, SymbolicDomain) else domain.as_tensor()
                    for domain, binder in zip(domains, binders, strict=True)
                    if binder in given
                ),
                tuple(binders.index(binder) for binder in given),
                tuple(binders.index(binder) for binder in axes),
            ),
            (*(SymTensor([binder]) for binder in given), SymTensor(ground)),
            SymTensor([atom]),
            name="ground",
        )

    def _ground_atoms(
        self, atom: Symbol, variables: tuple[Symbol, ...]
    ) -> list[Symbol] | None:
        """``atom`` at each combination of ``variables``' values, the last changing fastest.

        A value is substituted as its domain writes it
        (:attr:`~deeplog.variable.Domain.values`). ``None`` if a variable's
        domain does not write its values.
        """
        try:
            names = [self._compiler.domain(variable).values for variable in variables]
        except ValueError:
            return None
        bare = unwrap_structure(atom)
        return [
            retag(
                apply_substitution(bare, dict(zip(variables, values, strict=True))),
                atom,
            )
            for values in product(*names)
        ]

    def _lower_unary_nodes(self, nodes: list[UnaryOp]) -> list[DeepLogModule]:
        """Apply each operator's function to its operand's column."""
        return [
            self._operator(
                node.operator, self._column(node), self._column(node.operand)
            )
            for node in nodes
        ]

    def _lower_binary_nodes(self, nodes: list[BinaryOp]) -> list[DeepLogModule]:
        """Apply each operator's function to its operands' columns.

        ``divide`` is not special here: a :class:`~deeplog.algebraic.Semifield`
        defines it like any other operator.
        """
        return [
            self._operator(
                node.operator,
                self._column(node),
                self._column(node.lhs),
                self._column(node.rhs),
            )
            for node in nodes
        ]

    def _lower_aggregation_ops(self, nodes: list[AggregationOp]) -> list[DeepLogModule]:
        """Aggregate each node's operands' columns with its algebra's aggregator, in one module.

        Raises:
            NotImplementedError: If a node's algebra has no aggregator for its
                operation.
        """
        groups: dict[frozenset[Symbol], list[AggregationOp]] = {}
        for node in nodes:
            over = frozenset(free_variables(self._column(node))) & self._own
            groups.setdefault(over, []).append(node)
        modules = []
        for members in groups.values():
            inputs = list(
                dict.fromkeys(
                    self._column(o) for node in members for o in node.operands
                )
            )
            position = {column: index for index, column in enumerate(inputs)}
            modules.append(
                AggregationOpModule(
                    inputs,
                    [
                        (
                            self._column(node),
                            aggregator(node, self._compiler),
                            [position[self._column(o)] for o in node.operands],
                        )
                        for node in members
                    ],
                )
            )
        return modules

    def _lower_transformations(
        self, nodes: list[Transformation]
    ) -> list[DeepLogModule]:
        """Cast each source column with the builder for its pair of algebras.

        Raises:
            NotImplementedError: If no cast is registered for the pair.
        """
        builders = self._compiler._transformation_builders
        vertices = []
        for node in nodes:
            source = self._column(node.child)
            pair = (_structure(source, "cast"), node.structure)
            if pair not in builders:
                raise NotImplementedError(
                    f"no cast from {pair[0]!r} to {pair[1]!r}; registered: "
                    f"{sorted(builders)}."
                )
            module = builders[pair](SymTensor([source]))
            vertices.append(_named(module, self._column(node), f"The {pair} cast"))
        return vertices

    def _lower_aggregations(self, nodes: list[Aggregation]) -> list[DeepLogModule]:
        """Build each aggregation's module.

        The aggregations go to the builder registered for their operation, one
        call per operation and set of them whose variables agree
        (:meth:`_aggregate`, :meth:`_agreeing`). A node lowered before where its
        free variables are the same is not built again, and one inside another
        of ``nodes`` is left to the lowering of the other, which reaches it
        first, and built here only if that lowering did not.
        """
        keys = {id(node): self._context(node) for node in nodes}
        pending = [node for node in nodes if keys[id(node)] not in self._aggregations]
        inside = _reached(pending)
        for batch in (
            [node for node in pending if id(node) not in inside],
            [node for node in pending if id(node) in inside],
        ):
            operations: dict[str, list[Aggregation]] = {}
            for node in batch:
                if keys[id(node)] not in self._aggregations:
                    operations.setdefault(node.operation, []).append(node)
            for operation, members in operations.items():
                for agreeing in self._agreeing(members):
                    self._aggregate(operation, agreeing, keys)
        return [self._aggregations[keys[id(node)]][1] for node in nodes]

    def _context(self, node: Aggregation) -> tuple:
        """``node``'s id, with the domain of each variable free in it and whether it is bound."""
        return (
            id(node),
            *sorted(
                (
                    (
                        variable,
                        self._compiler.variables.get(variable),
                        variable in self._binders,
                    )
                    for variable in self._sorts.variables(node)
                ),
                key=lambda found: symbol_to_str(found[0]),
            ),
        )

    def _agreeing(self, nodes: list[Aggregation]) -> list[list[Aggregation]]:
        """``nodes``, in order, in sets whose variables agree.

        No node of a set binds a variable another reads free, or a name another
        binds over another domain.
        """
        sets: list[tuple[list[Aggregation], dict[Symbol, Domain], set[Symbol]]] = []
        for node in nodes:
            binds = self._sorts.binders(node)
            reads = self._sorts.variables(node)
            for members, bound, free in sets:
                if (
                    reads.isdisjoint(bound)
                    and free.isdisjoint(binds)
                    and all(
                        bound.get(b, domain) == domain for b, domain in binds.items()
                    )
                ):
                    members.append(node)
                    bound.update(binds)
                    free.update(reads)
                    break
            else:
                sets.append(([node], dict(binds), set(reads)))
        return [members for members, _, _ in sets]

    def _lower_circuits(
        self, lumps: list[CircuitNode], heights: dict[int, int]
    ) -> list[DeepLogModule]:
        """Compile the lumps of each circuit into one core.

        Lumps of one circuit where one reaches the other, through a cast and
        another circuit, have different heights (:func:`_lump_heights`) and go
        to different cores, since a core cannot read its own output. Each leaf
        of a lump reads the column of the child that feeds it.
        """
        groups: dict[tuple, dict[Symbol, CircuitNode]] = {}
        feeds: dict[Symbol, Symbol] = {}
        for lump in lumps:
            over = frozenset(free_variables(self._column(lump))) & self._own
            group = groups.setdefault((id(lump.circuit), heights[id(lump)], over), {})
            group.setdefault(self._column(lump), lump)
            feeds.update((leaf, self._column(fed)) for leaf, fed in lump.boundary)
        cores = [
            circuit_to_module(*group.values(), names=tuple(group))
            for group in groups.values()
        ]
        return cores + [
            WrappedModule(
                torch.nn.Identity(), SymTensor([column]), SymTensor([leaf]), name="feed"
            )
            for leaf, column in feeds.items()
            if column != leaf
        ]

    # -- What the handlers share ----------------------------------------------

    def _operator(
        self, operator: str, column: Symbol, *operands: Symbol
    ) -> DeepLogModule:
        """The module applying ``operator``'s function to the ``operands`` columns.

        The operator function comes from the operands' algebra
        (:attr:`~deeplog.algebraic.AlgebraicStructure.operator_fns`).

        Raises:
            NotImplementedError: If the algebra does not define ``operator``.
        """
        algebra = self._compiler.algebra(_shared_structure(operator, operands))
        operator_fn = algebra.get_operator_fn(operator)
        if operator_fn is None:
            raise NotImplementedError(
                f"structure {algebra.name!r} has no operator {operator!r}; "
                f"available: {sorted(algebra.operators)}."
            )
        inputs = tuple(SymTensor([operand]) for operand in operands)
        return WrappedModule(
            operator_fn,
            inputs[0] if len(inputs) == 1 else inputs,
            SymTensor([column]),
            name=operator,
        )

    def _aggregate(
        self, operation: str, nodes: list[Aggregation], keys: dict[int, tuple]
    ) -> None:
        """Build ``nodes`` with the builder registered for ``operation``, or by reduction.

        An aggregation other than an expectation needs an aggregator in its
        algebra, whichever builder computes it. Each module the builder reports
        becomes a vertex computing the columns of the nodes it serves, under the
        names the walk gives them, and none of its other columns, and is kept
        under the node's key in ``keys``. While the builder runs, the
        :attr:`compiler` declares the domain of each of the nodes' binders, and
        an atom that no builder computes and that reads a bound variable lowers
        to its ground atoms (:meth:`_ground`).

        Raises:
            NotImplementedError: If an algebra has no aggregator for
                ``operation``.
            ValueError: If the builder reports other than one column per node,
                a column its module does not compute, or a column labelled
                with other than its node's algebra.
        """
        if operation != "expectation":
            for node in nodes:
                aggregator(node, self._compiler)
        builder = self._compiler._aggregation_builders.get(operation, reduction)
        domains = {
            binder: domain
            for node in nodes
            for binder, domain in self._sorts.binders(node).items()
        }
        enclosing = self._compiler, self._binders, self._bound, self._own
        self._compiler = self._compiler._binding(domains)
        self._binders = self._binders | domains.keys()
        self._own = frozenset()
        self._bound = self._reading(nodes, self._bound)
        try:
            reports = list(builder(nodes, self))
        finally:
            self._compiler, self._binders, self._bound, self._own = enclosing
        if len(reports) != len(nodes):
            raise ValueError(
                f"The {operation!r} builder reported {len(reports)} columns for "
                f"{len(nodes)} aggregations; it reports one per aggregation."
            )
        served: dict[int, tuple[DeepLogModule, dict[Symbol, Symbol]]] = {}
        outputs: dict[int, list[Symbol]] = {}
        for node, (module, index) in zip(nodes, reports, strict=True):
            if id(module) not in outputs:
                outputs[id(module)] = list(get_all_symbols(module.get_output_shape()))
            source = _reported_column(outputs[id(module)], index, operation)
            name = self._column(node)
            if structure_of(source) != structure_of(name):
                raise ValueError(
                    f"The {operation!r} builder reported the column {source!r} for "
                    f"an aggregation whose formula makes it a value of "
                    f"{structure_of(name)!r}. A builder computes an aggregation "
                    "in its formula's algebra; to aggregate in another, cast the "
                    "body into it."
                )
            served.setdefault(id(module), (module, {}))[1].setdefault(name, source)
        vertices = {
            key: _renamed(module, names) for key, (module, names) in served.items()
        }
        for node, (module, _) in zip(nodes, reports, strict=True):
            self._aggregations[keys[id(node)]] = (node, vertices[id(module)])

    def _reading(
        self, nodes: list[Aggregation], bound: dict[Symbol, tuple[Symbol, ...]]
    ) -> dict[Symbol, tuple[Symbol, ...]]:
        """``bound``, joined by the atoms of ``nodes``' formulas that read their binders.

        Only atoms no builder computes are kept, each with the binders it
        reads, those it is kept with in ``bound`` included. The walk enters
        aggregations beneath ``nodes``, which their binders still bind.
        """
        joined = dict(bound)
        builders = self._compiler._atom_builders
        for node in nodes:
            seen: set[int] = set()
            stack = list(operands(node))
            while stack:
                below = stack.pop()
                if id(below) in seen:
                    continue
                seen.add(id(below))
                if isinstance(below, CircuitNode):
                    stack.extend(below.children)
                    continue
                stack.extend(operands(below))
                if not isinstance(below, Atom):
                    continue
                bare = unwrap_structure(below.atom)
                key = (*get_predicate(bare), structure_of(below.atom))
                read = _read(bare, node.binders)
                if read and key not in builders:
                    known = joined.get(below.atom, ())
                    joined[below.atom] = known + tuple(
                        b for b in read if b not in known
                    )
        return joined


def _reached(aggregations: Sequence[Aggregation]) -> set[int]:
    """The ids of ``aggregations`` that lie beneath another of them.

    A lump holds an aggregation only in what feeds it, since a leaf fed by its
    own atom holds none, so a lump's graph is never walked.
    """
    wanted = {id(node) for node in aggregations}
    reached: set[int] = set()
    seen: set[int] = set()
    stack = [below for node in aggregations for below in children(node)]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if id(node) in wanted:
            reached.add(id(node))
        if isinstance(node, CircuitNode):
            stack.extend(fed for _, fed in node.feeders)
        else:
            stack.extend(children(node))
    return reached


def _read(atom: Symbol, binders: Sequence[Symbol]) -> tuple[Symbol, ...]:
    """The ``binders`` among ``atom``'s arguments, in the order they first occur."""
    arguments = get_args(atom)
    return tuple(
        dict.fromkeys(
            variable
            for argument in arguments
            for variable in get_term_variables(argument)
            if variable in binders
        )
    )


class _Translated(DeepLogModule):
    """A module whose inputs are translated before it reads them.

    ``tables`` maps an input's index and a column of it to the table that
    translates the column's values: a value ``v`` becomes ``table[v]``.
    """

    def __init__(
        self, module: DeepLogModule, tables: dict[tuple[int, int], Tensor]
    ) -> None:
        """Translate ``module``'s inputs by ``tables``."""
        super().__init__(module.get_input_shape(), module.get_output_shape())
        self._module = module
        self._columns = list(tables)
        for i, table in enumerate(tables.values()):
            self.register_buffer(f"_table_{i}", table)

    def forward(self, *x: Tensor) -> Tensor:
        """``module`` at ``x``, its translated columns translated."""
        inputs = list(x)
        for i, (index, column) in enumerate(self._columns):
            table = getattr(self, f"_table_{i}")
            values = inputs[index].clone()
            values[:, column] = table[values[:, column].long()].to(values.dtype)
            inputs[index] = values
        return self._module(*inputs)


def _look_up(
    sizes: tuple[int, ...],
    numbers: tuple[Tensor | None, ...],
    given: tuple[int, ...],
    axes: tuple[int, ...],
    *x: Tensor,
) -> Tensor:
    """The last of ``x``, a column per combination of values, at the values the others give.

    The columns are the combinations of values of variables of the sizes
    ``sizes``, row-major. The variables at ``given`` are read at a value per
    row, each from one of the others of ``x``: a position, or, where
    ``numbers`` gives the domain's numbers, one of them. The result has an axis
    for each variable at ``axes``, in that order, behind its one column.

    Raises:
        ValueError: If a number is not one of its domain's.
    """
    *values, table = x
    batch = table.shape[0]
    table = table.reshape(batch, *sizes).permute(
        0, *(1 + k for k in given), *(1 + k for k in axes)
    )
    if given:
        index = torch.zeros(batch, dtype=torch.long, device=table.device)
        for value, k, domain in zip(values, given, numbers, strict=True):
            position = value.reshape(-1, 1)
            if domain is not None:
                matches = position == domain.to(position.device).reshape(1, -1)
                found = matches.any(dim=1)
                if not bool(found.all()):
                    raise ValueError(
                        f"A variable is given {position[~found][0].item()}, which is "
                        "not a value of its domain."
                    )
                position = matches.int().argmax(dim=1, keepdim=True)
            index = index * sizes[k] + position.reshape(-1).long()
        table = table.reshape(batch, -1, *table.shape[1 + len(given) :])
        table = table[torch.arange(batch, device=table.device), index]
    return table.unsqueeze(1)


def _spread(values: Tensor, x: Tensor) -> Tensor:
    """``values``, one row per value, along an axis behind one column, for every row of ``x``."""
    return (
        values.to(x.device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(x.shape[0], 1, *values.shape)
    )


def _atoms_beneath(roots: Sequence[FormulaNode]) -> list[Atom]:
    """Every atom beneath ``roots``, at any depth, aggregations entered, each once."""
    found: list[Atom] = []
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Atom):
            found.append(node)
        stack.extend(children(node))
    return found


def _gather(
    roots: Sequence[FormulaNode], given: Collection[FormulaNode] = ()
) -> list[FormulaNode]:
    """The level under ``roots``: every node reachable without entering an aggregation.

    Each node once, children before parents, in fold order. A node of ``given``
    is left out, with what only it reaches.
    """
    level: list[FormulaNode] = []
    seen: set[int] = {id(node) for node in given}
    # An explicit stack rather than recursion, as in fold: a chain of casts can
    # be thousands deep.
    stack: list[tuple[FormulaNode, bool]] = [(root, False) for root in reversed(roots)]
    while stack:
        node, ready = stack.pop()
        if ready:
            level.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        if not isinstance(node, Aggregation):
            stack.extend((child, False) for child in reversed(children(node)))
    return level


def _of_type[N](level: list[FormulaNode], kind: type[N]) -> list[N]:
    """The nodes of ``level`` of AST type ``kind``, in level order."""
    return [node for node in level if isinstance(node, kind)]


def _one_per_column[N: FormulaNode](
    nodes: list[N], column: Callable[[FormulaNode], Symbol]
) -> list[N]:
    """``nodes`` less those computing a column an earlier one computes."""
    kept: dict[Symbol, N] = {}
    for node in nodes:
        kept.setdefault(column(node), node)
    return list(kept.values())


def _lump_heights(level: list[FormulaNode]) -> dict[int, int]:
    """The number of lumps on the longest path down from each node, itself included.

    A lump that reaches another has the greater height, so the lumps of one
    circuit at one height never read each other. A child outside ``level``,
    whose column the level reads as an input, counts none.
    """
    heights: dict[int, int] = {}
    for node in level:
        below = () if isinstance(node, Aggregation) else children(node)
        heights[id(node)] = max(
            (heights.get(id(child), 0) for child in below), default=0
        )
        heights[id(node)] += isinstance(node, CircuitNode)
    return heights


def _name(node: FormulaNode, column: Callable[[FormulaNode], Symbol]) -> Symbol:
    """The column ``node`` computes, named from the ``column`` of each child."""
    match node:
        case Atom(atom):
            return atom
        case CircuitNode():
            return lump_name(node)
        case Transformation(structure, child):
            return cast_name(structure, column(child))
        case UnaryOp(operator, operand):
            return _operator_name(operator, column(operand))
        case BinaryOp(operator, lhs, rhs):
            return _operator_name(operator, column(lhs), column(rhs))
        case AggregationOp(operation, operands):
            return _aggregation_op_name(operation, [column(o) for o in operands])
        case Aggregation():
            return _aggregation_name(node, column)
    assert_never(node)


def _aggregation_name(
    node: Aggregation, column: Callable[[FormulaNode], Symbol]
) -> Symbol:
    """The operation over the binders and the operands' names, in ``node``'s algebra.

    An operand in that algebra is named bare, as an operator's operands are; one
    in another keeps its label, as a cast's source does, so aggregations over
    formulas that differ only in their algebra name different columns.

    Raises:
        NotImplementedError: If the formula gives ``node`` no algebra.
    """
    structure = node.structure
    if structure is None:
        raise NotImplementedError(
            f"The {node.operation!r} aggregation over {node.child!r} is a value "
            "of no algebra: its formula gives it none."
        )
    operands = [column(operand) for operand in (node.child, *node.params)]
    return with_structure(
        (
            node.operation,
            ("binders", *node.binders),
            *(
                without_structure(operand)
                if structure_of(operand) == structure
                else operand
                for operand in operands
            ),
        ),
        structure,
    )


def _aggregation_op_name(operation: str, operands: Sequence[Symbol]) -> Symbol:
    """The operation over ``("operands", ...)`` of the operands' names, labelled with their algebra."""
    return with_structure(
        (operation, ("operands", *map(without_structure, operands))),
        _shared_structure(operation, operands),
    )


def _operator_name(operator: str, *operands: Symbol) -> Symbol:
    """The operator over the operands' names, labelled with their algebra."""
    return with_structure(
        (operator, *map(without_structure, operands)),
        _shared_structure(operator, operands),
    )


def _shared_structure(operator: str, operands: Sequence[Symbol]) -> str:
    """The one algebra the ``operands`` columns are values of.

    Raises:
        ValueError: If they are values of different algebras.
    """
    found = {_structure(operand, f"apply {operator!r} to") for operand in operands}
    if len(found) > 1:
        raise ValueError(
            f"The operands of {operator!r} must share a structure, got "
            f"{sorted(found)!r}."
        )
    return found.pop()


def _structure(column: Symbol, action: str) -> str:
    """The algebra ``column`` is labelled with.

    Raises:
        NotImplementedError: If it is labelled with none: guessing an algebra
            would return plausible wrong numbers.
    """
    structure = structure_of(column)
    if structure is None:
        raise NotImplementedError(
            f"cannot {action} {column!r}: it carries no algebraic structure. Every "
            "module in a formula must label its output symbols, e.g. "
            "with_structure(symbol, 'probability')."
        )
    return structure


def _reported_column(outputs: list[Symbol], index: int, operation: str) -> Symbol:
    """The column at ``index`` among a reported module's ``outputs``.

    Raises:
        ValueError: If the module computes no column at ``index``.
    """
    if not 0 <= index < len(outputs):
        raise ValueError(
            f"The {operation!r} builder reported column {index} of a module "
            f"computing {len(outputs)}."
        )
    return outputs[index]


def _named(module: DeepLogModule, column: Symbol, what: str) -> DeepLogModule:
    """``module``, its one output column named ``column``.

    Raises:
        ValueError: If ``module`` computes more or fewer than one column.
    """
    outputs = list(get_all_symbols(module.get_output_shape()))
    if len(outputs) != 1:
        raise ValueError(f"{what} computes {len(outputs)} columns for one value.")
    return _renamed(module, {column: outputs[0]})


def _renamed(module: DeepLogModule, columns: dict[Symbol, Symbol]) -> DeepLogModule:
    """The ``columns`` values among ``module``'s outputs, named for their keys.

    ``module``'s other output columns are left out, so a name only its builder
    chose never reaches the module graph.
    """
    output = SymTensor(list(columns))
    if module.get_output_shape() == output:
        return module
    selected = reshape(module, output=SymTensor(list(columns.values())))
    return WrappedModule(selected, selected.get_input_shape(), output, name="rename")


def _distinct(names: list[Symbol]) -> list[Symbol]:
    """Return ``names``, or raise naming the column two roots would collide on."""
    seen: set[Symbol] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"Two formulas compile to the same output column {name}; "
                f"compile them separately, or relabel one."
            )
        seen.add(name)
    return names


def _compose(vertices: list[DeepLogModule], outputs: list[Symbol]) -> DeepLogModule:
    """One module graph over ``vertices``, computing the ``outputs`` columns.

    Each vertex appears once, however many columns of the level it computes.
    With no vertex, every output is an input: atoms no builder computes.
    """
    output = SymTensor(outputs)
    if not vertices:
        return construct_transformation(output, output)
    return compose_modules(list({id(v): v for v in vertices}.values()), output)
