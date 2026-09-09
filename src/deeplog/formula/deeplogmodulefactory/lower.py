#  Copyright (c) 2024-2026. KU Leuven
"""Lower raw co-resident CircuitNode roots into a DeepLogModule.

Callers that build circuits *directly* - the DeepProbLog engine and the DIMACS
parser - hold raw :class:`~deeplog.formula.ast.CircuitNode` objects that never
went through a factory's fold.
:func:`~deeplog.formula.deeplogmodulefactory.lower_circuit_nodes` is their entry
point: it compiles the shared arithmetic circuit once (deduped) and folds the
unioned boundary through the factory's ordinary eliminators. The factory's own
:meth:`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory.embed_circuit`
reuses ``_compose_circuit_core`` for the single-lump case; a formula AST is
lowered with
:meth:`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory.compile`.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING

from ...module import DeepLogModule
from ...module import compose_modules
from ...symbol import Symbol
from ..ast import Atom
from ..ast import CircuitNode
from ..ast import FormulaNode
from ..ast import children
from ..ast import fold
from ..circuit_node import to_module as circuit_to_module
from ..strategies import deferred_lump
from ..strategies import expectation_symbol
from ..strategies import transform_expectation_to_probability


if TYPE_CHECKING:
    from ...variable import VariableAtoms
    from .deeplogmodulefactory import DeepLogModuleFactory


def lower_circuit_nodes(
    factory: DeepLogModuleFactory,
    *nodes: CircuitNode,
    names: tuple[Symbol, ...] | None = None,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    variables: VariableAtoms | None = None,
) -> DeepLogModule:
    """Lower raw co-resident :class:`~deeplog.formula.ast.CircuitNode` roots.

    The entry point for callers that build circuits directly and so hold raw
    :class:`~deeplog.formula.ast.CircuitNode` objects that never went through
    ``factory``'s fold: the DeepProbLog engine and the DIMACS parser. The roots
    share one circuit, so they compile together into one module with an output
    per root, and their unioned boundary is folded once through ``factory``'s
    ordinary eliminators.

    A formula AST is lowered with
    :meth:`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory.compile`
    instead, since its fold already yields a closed module. Output naming is
    left to the default (``<circuit>_<i>``) unless ``names`` is given; callers
    that need query-specific names reshape the result.

    ``leaf_mapping`` and ``variables`` reach every weighted model count in the
    lowered boundary: how a boolean leaf is named in the probability semiring
    (DeepProbLog's Definition 12 α, as a mapping built from the atom labels) and
    where each multi-valued variable occurs. Both default to the general reading
    -- retag the leaf, no declared variables -- which is what a formula written
    in the textual language means.
    """
    if not nodes:
        raise ValueError("At least one circuit-node root is required.")
    if not all(isinstance(n, CircuitNode) for n in nodes):
        raise TypeError(
            "lower_circuit_nodes lowers raw circuit-node roots; lower a formula "
            "AST with factory.compile()."
        )
    roots = list(nodes)

    # The raw roots were never folded, so fold their boundary here. The union of
    # the roots' reachable leaves / casts — deduped by symbol across roots (a
    # symbol's feeder is a deterministic function of the symbol, so colliding
    # entries agree) — is lowered through the ordinary eliminators under one memo.
    circuit = roots[0].circuit
    overrides: dict[Symbol, FormulaNode] = {}
    for node in roots:
        overrides.update(node.feeders)
    boundary = [
        overrides.get(name, Atom(name))
        for name in circuit.reachable_symbol_names([node.node for node in roots])
    ]
    memo = batched_lowering(
        factory, *boundary, leaf_mapping=leaf_mapping, variables=variables
    )
    children = [fold(child, factory, memo=memo) for child in boundary]

    return _compose_circuit_core(roots, children, names=names)


def _compose_circuit_core(
    nodes: list[CircuitNode],
    children: Sequence[DeepLogModule | None],
    names: tuple[Symbol, ...] | None = None,
) -> DeepLogModule:
    """Compose co-resident lump cores with their lowered boundary children.

    ``circuit_to_module`` compiles the co-resident ``nodes`` together — one
    shared arithmetic circuit, deduped — into a core module whose inputs are the
    reachable leaves. ``children`` are the already-lowered boundary modules
    (``None`` where a leaf has no builder or is a baked constant — those stay
    external inputs). Composing the core with the non-``None`` children feeds
    each predicate module / cast spine onto its leaf slot *by shape*; un-fed
    leaves remain the result's external inputs.

    """
    core = circuit_to_module(*nodes, names=names)
    # One module can feed several leaves at once — a batched count does, with a
    # column per expectation — so identical children are composed once.
    fed = list({id(child): child for child in children if child is not None}.values())
    return compose_modules([core, *fed], core.get_output_shape())


def batched_lowering(
    factory: DeepLogModuleFactory,
    *boundary: FormulaNode,
    leaf_mapping: Callable[[Symbol], Symbol] | None = None,
    variables: VariableAtoms | None = None,
) -> dict[int, DeepLogModule]:
    """Lower the deferred work in ``boundary``, one compilation per circuit.

    The single weighted-model-count site. Counts over the *same* boolean circuit
    share its knowledge compilation: compiling the DeepProbLog conditional path's
    N numerators and their shared evidence separately would knowledge-compile
    N+1 times and run each predicate module once per answer.

    ``leaf_mapping`` and ``variables`` are the model facts the count needs and
    cannot derive — see :func:`lower_circuit_nodes`.

    Returns a fold memo keyed by ``id(node)``, which
    :func:`~deeplog.formula.ast.fold` consults first, so the ordinary
    eliminators never see a deferred count. Every node of a group maps to the
    *same* module.
    """
    groups: _Groups = {}
    seen: set[int] = set()
    for node in boundary:
        _collect_deferred(node, seen, groups)

    memo: dict[int, DeepLogModule] = {}
    for group in groups.values():
        lumps = [lump for _, lump in group]
        counted = transform_expectation_to_probability(
            *lumps, leaf_mapping=leaf_mapping, variables=variables
        )
        module = lower_circuit_nodes(
            factory, *counted, names=tuple(expectation_symbol(lump) for lump in lumps)
        )
        for node, _ in group:
            memo[id(node)] = module
    return memo


#: Deferred nodes and the lump each defers, grouped by ``id`` of that lump's circuit.
type _Groups = dict[int, list[tuple[FormulaNode, CircuitNode]]]


def _collect_deferred(node: FormulaNode, seen: set[int], groups: _Groups) -> None:
    """Gather the deferred counts in ``node``, grouped by the circuit counted."""
    if id(node) in seen:
        return
    seen.add(id(node))
    lump = deferred_lump(node)
    if lump is not None:
        groups.setdefault(id(lump.circuit), []).append((node, lump))
        return
    for child in children(node):
        _collect_deferred(child, seen, groups)
