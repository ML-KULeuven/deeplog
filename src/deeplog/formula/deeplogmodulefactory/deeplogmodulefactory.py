#  Copyright (c) 2024-2026. KU Leuven
"""Factory that lowers a *materialized* formula AST to a DeepLogModule.

The second half of lowering: this fold runs over an AST whose
circuit-representable regions the construction fold has already compiled to
:class:`~deeplog.formula.ast.CircuitNode` lumps, so the only symbolic nodes it
still sees are the aggregations that fold could not absorb, standing between
lumps.

A lump is *transparent* to the fold
(:attr:`~deeplog.formula.deeplogformulafactory.DeepLogFormulaFactory.lowers_circuit_children`
is set): its boundary is folded through the ordinary eliminators while its
interior stays compiled, and ``embed_circuit`` composes the two. Every
eliminator returns a :class:`~deeplog.module.DeepLogModule`.

Callers holding raw co-resident lumps — the DeepProbLog engine, the DIMACS
parser — use :func:`~deeplog.formula.deeplogmodulefactory.lower_circuit_nodes`.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from ...algebraic import BOOLEAN
from ...algebraic import AlgebraicStructure
from ...algebraic import get_algebraic_structure
from ...module import DeepLogModule
from ...module import ElementwiseModule
from ...module import Sequential
from ...module import compose_modules
from ...shape import SymTensor
from ...shape import get_all_symbols
from ...shape import sole_structure
from ...symbol import Symbol
from ...symbol import get_args
from ...symbol import get_predicate
from ...symbol import structure_of
from ...symbol import unwrap_structure
from ...variable import Domain
from ..ast import CircuitNode
from ..ast import FormulaNode
from ..ast import fold
from ..circuit_node import lump_name
from ..deeplogformulafactory import DeepLogFormulaFactory
from .builder_protocols import AggregationBuilder
from .builder_protocols import AtomBuilder
from .builder_protocols import TransformationBuilder
from .lower import _compose_circuit_core
from .lower import batched_lowering
from .lower import lower_circuit_nodes
from .registry import default_aggregation_builders
from .registry import default_atom_builders
from .registry import default_transformation_builders


def _distinct(names: tuple[Symbol, ...]) -> tuple[Symbol, ...]:
    """Return ``names``, or raise naming the column two roots would collide on.

    Two roots naming one column is not a draw: a shape-keyed composition keeps
    whichever producer ran last, so a column would be silently dropped.
    """
    seen: set[Symbol] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"Two formulas compile to the same output column {name}; "
                f"compile them separately, or relabel one."
            )
        seen.add(name)
    return names


def _compose_roots(lowered: Sequence[DeepLogModule]) -> DeepLogModule:
    """Compose one module per root into one module with their columns side by side.

    Roots that lowered to the same module — the fold memo is shared, so two that
    are the same node do — are composed once, and the result carries every root's
    output symbols in root order.
    """
    outputs = _distinct(
        tuple(
            symbol
            for module in lowered
            for symbol in get_all_symbols(module.get_output_shape())
        )
    )
    fed = list({id(module): module for module in lowered}.values())
    return compose_modules(fed, SymTensor(list(outputs)))


def _required_structure(operand: DeepLogModule, action: str) -> str:
    """Return the algebra ``operand``'s outputs are labelled with, or raise.

    An unlabelled output is refused rather than defaulted: guessing an algebra
    here would return plausible wrong numbers.
    """
    structure = sole_structure(operand.get_output_shape())
    if structure is None:
        raise NotImplementedError(
            f"cannot {action} {operand!r}: its outputs carry no algebraic "
            "structure. Every module in a formula must label its output "
            "symbols, e.g. with_structure(symbol, 'probability')."
        )
    return structure


class DeepLogModuleFactory(DeepLogFormulaFactory[DeepLogModule]):
    """Lower a *materialized* formula AST into a DeepLogModule."""

    lowers_circuit_children = True

    def __init__(
        self,
        variables: Mapping[Symbol, Domain] | None = None,
        reification: AlgebraicStructure = BOOLEAN,
        structures: Mapping[str, AlgebraicStructure] | None = None,
        aggregators: Mapping[str, AggregationBuilder] | None = None,
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder] | None = None,
        transformations: Mapping[tuple[str, str], TransformationBuilder] | None = None,
    ) -> None:
        """Configure the module factory with available domains and default builders."""
        self.variables: dict[Symbol, Domain] = dict(variables or {})
        self.reification = reification

        self._atom_builders = dict(atom_builders or {})
        self._atom_builders.update(default_atom_builders)

        #: Custom algebraic structures, handed to the ``CircuitFactory`` the
        #: construction fold runs over (see :meth:`compile`).
        self._structures = dict(structures or {})

        self._aggregation_builders: dict[str, AggregationBuilder] = dict(
            default_aggregation_builders
        )
        self._aggregation_builders.update(aggregators or {})

        #: Casts between algebras, keyed ``(from, to)``. A module that wants the
        #: ``real -> probability`` cast declares ``real`` on its output symbols;
        #: the source structure is never inferred from an absent label.
        self._transformation_builders: dict[tuple[str, str], TransformationBuilder] = (
            dict(default_transformation_builders)
        )
        self._transformation_builders.update(transformations or {})

    # -- The eliminators (fold calls these bottom-up) -----------------------

    def embed_circuit(
        self, node: CircuitNode, children: tuple[DeepLogModule | None, ...] = ()
    ) -> DeepLogModule:
        """Lower a lump: compose its compiled interior with its folded boundary.

        ``children`` are the fold results of
        :attr:`~deeplog.formula.ast.CircuitNode.children`: a predicate module per
        leaf (``None`` where the leaf has no builder, or is a baked constant) and
        a cast spine per cross-structure boundary.

        The lump's output is named by node id (``<circuit>_n<node>``), the
        ``child_name`` a cross-structure cast embeds in its ``transform`` leaf,
        so two sub-roots of one shared circuit get distinct boundary names.
        """
        name: Symbol = lump_name(node)
        return _compose_circuit_core([node], children, names=(name,))

    def create_atom(self, atom: Symbol) -> DeepLogModule | None:
        """Lower a single circuit-leaf to its predicate module, or ``None``.

        The real atom handler. A leaf ``("_", literal, (structure,))`` with a
        registered builder becomes that predicate's module for this one
        arg-tuple. A leaf with no builder is ``None`` and stays an external
        input of the composed lump — a baked constant among them, since no
        predicate is registered for a numeral.

        Raises:
            ValueError: If ``atom`` carries no structure label.
        """
        structure = structure_of(atom)
        if structure is None:
            raise ValueError(f"Invalid atom: {atom}")
        literal = unwrap_structure(atom)
        functor, arity = get_predicate(literal)
        builder = self._atom_builders.get((functor, arity, structure))
        if builder is None:
            return None
        return builder([get_args(literal)]).to_module()

    def _structure_of(self, operand: DeepLogModule) -> AlgebraicStructure:
        """Resolve an operand's structure to the object carrying its operator functions.

        Read off the operand's *output symbols*, so an operand that labels none
        of them, or labels them inconsistently, is rejected rather than defaulted.
        """
        return self._resolve(_required_structure(operand, "lower an operator over"))

    def _resolve(self, name: str) -> AlgebraicStructure:
        """Resolve a structure *name* to the object carrying its operator functions.

        This factory's own ``_structures`` wins over the global registry, since a
        custom structure handed to the constructor (the LTN fuzzy algebra) is
        deliberately not registered globally.
        """
        if name in self._structures:
            return self._structures[name]
        return get_algebraic_structure(name)

    def _elementwise(self, operator: str, *operands: DeepLogModule) -> DeepLogModule:
        """Combine already-lowered operands with the structure's operator function.

        An operator survives symbolically this far only when an operand could not
        be circuit-represented, so its operands are already reduced to numbers
        and it becomes a tensor operation over them, with the algebra supplying
        which one (:mod:`deeplog.module.elementwise`).
        """
        structure = self._structure_of(operands[0])
        operator_fn = structure.get_operator_fn(operator)
        if operator_fn is None:
            raise NotImplementedError(
                f"structure {structure.name!r} has no operator {operator!r}; "
                f"available: {sorted(structure.operators)}."
            )
        return ElementwiseModule(operator_fn, *operands, name=operator)

    def create_unary_node(self, operator: str, operand: DeepLogModule) -> DeepLogModule:
        """Apply a symbolic unary operator elementwise to its lowered operand."""
        return self._elementwise(operator, operand)

    def create_binary_node(
        self, operator: str, lhs: DeepLogModule, rhs: DeepLogModule
    ) -> DeepLogModule:
        """Apply a symbolic binary operator elementwise to its lowered operands.

        ``divide`` is not special here: a :class:`~deeplog.algebraic.Semifield`
        defines it like any other operator.
        """
        return self._elementwise(operator, lhs, rhs)

    def create_transformation(
        self, structure: str, child: DeepLogModule
    ) -> DeepLogModule:
        """Cast a *module* ``child`` into ``structure`` with an elementwise spine.

        Every cast reaches here as an already-lowered module: a cast *inside* a
        circuit is a boundary child whose source lump the fold lowered first. The
        registered ``from -> structure`` builder is chained onto it.
        """
        from_structure = _required_structure(child, "cast")
        try:
            builder = self._transformation_builders[(from_structure, structure)]
        except KeyError:
            raise NotImplementedError(
                f"no cast from {from_structure!r} to {structure!r}; registered: "
                f"{sorted(self._transformation_builders)}."
            ) from None
        return Sequential(child, builder(child.get_output_shape()).to_module())

    def create_aggregation(
        self,
        operation: str,
        binders: list[Symbol],
        params: Sequence[DeepLogModule],
        child: DeepLogModule,
    ) -> DeepLogModule:
        """Lower an aggregation by enumerating ``child`` over ``binders``.

        ``expectation`` is absorbed by the construction fold and never reaches
        this factory; the aggregations with no circuit form do. The child is
        already a closed module, so the binder variables are its free inputs and
        the registered builder enumerates over them.
        """
        if operation == "expectation":
            raise NotImplementedError(
                "expectation is absorbed into a probability circuit by the "
                "construction fold before lowering; the lowering factory only "
                "enumerates aggregations such as 'sum'."
            )
        _required_structure(child, f"aggregate with {operation!r} over")
        domains = [self._binder_domain(binder) for binder in binders]
        builder = self._aggregation_builders[operation]
        return builder(
            child, binders, params, [domain.as_tensor() for domain in domains]
        ).to_module()

    def _binder_domain(self, binder: Symbol) -> Domain:
        """The domain ``binder`` ranges over.

        A binder the model declares in :attr:`variables` uses that domain.
        Any other binder is a reification variable, and inherits the values of
        :attr:`reification` as its domain.

        Raises:
            ValueError: If the binder is undeclared and :attr:`reification`
                declares no enumerable values.
        """
        declared = self.variables.get(binder)
        if declared is not None:
            return declared
        try:
            return Domain.of_structure(self.reification)
        except ValueError as error:
            raise ValueError(f"No domain for binder {binder}: {error}") from None

    # -- Terminal ----------------------------------------------------------

    def compile(self, *nodes: FormulaNode) -> DeepLogModule:
        """Lower formula ASTs to one module in two folds: construct, then lower.

        The first fold runs each ``node`` through a
        :class:`~deeplog.formula.circuit_factory.CircuitFactory` configured with
        this factory's custom ``_structures``; the second runs the results through
        this factory, lowering each lump and enumeration to a module. The fold is
        the whole lowering — there is no separate closing step.

        Every root becomes one output column, in the order given. Both folds run
        over all of them under one memo, so a subformula they share *as an object*
        is built once, the atoms they share are one circuit leaf, and counts over
        one boolean circuit take one knowledge compilation. Roots that all
        constructed to lumps of a single circuit are lowered together, so their
        boundary is folded once and each predicate module is evaluated once; any
        other mix is lowered root by root and composed column-wise, where a shared
        sub-module is built once but still evaluated once per root that encloses
        it.

        Raises:
            ValueError: If no formula is given, or if two roots would name the
                same output column.
        """
        from ..circuit_factory import CircuitFactory

        if not nodes:
            raise ValueError("At least one formula is required.")

        circuits = CircuitFactory(self._structures)
        built: dict[int, FormulaNode] = {}
        constructed = [fold(node, circuits, memo=built) for node in nodes]

        lumps = [node for node in constructed if isinstance(node, CircuitNode)]
        if len(lumps) == len(constructed) and all(
            lump.circuit is lumps[0].circuit for lump in lumps
        ):
            return lower_circuit_nodes(
                self, *lumps, names=_distinct(tuple(lump_name(lump) for lump in lumps))
            )

        # Weighted model counts are gathered before the fold so that counts over
        # one boolean circuit share a single knowledge compilation; the fold
        # itself is unchanged, it just finds them already lowered.
        memo = batched_lowering(self, *constructed)
        return _compose_roots([fold(node, self, memo=memo) for node in constructed])
