#  Copyright (c) 2024-2026. KU Leuven
"""Compile DeepProbLog engine results into DeepLogModules."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import cast

from deeplog import PROBABILITY
from deeplog import CircuitFactory
from deeplog import CircuitNode
from deeplog import Symbol
from deeplog import SymTensor
from deeplog import WrappedModule
from deeplog import get_all_symbols
from deeplog import sole_structure
from deeplog import with_structure
from deeplog.formula import lower_circuit_nodes
from deeplog.module import ColumnwiseModule

from .solver import EngineResult


if TYPE_CHECKING:
    from deeplog import DeepLogModule
    from deeplog import DeepLogModuleFactory
    from deeplog import FormulaNode


def compile_to_module(
    result: EngineResult[FormulaNode],
    factory: DeepLogModuleFactory,
) -> DeepLogModule:
    """Compile an engine result to a DeepLogModule.

    ``result.formulas`` are boolean ``CircuitNode`` lumps sharing one source
    circuit. Each is lowered as the *expectation* of it, so every query is
    counted at the lowering's single weighted-model-count site, co-resides in
    one arithmetic circuit, and ``factory`` lowers them into one multi-output
    module. Outputs are relabelled from positional
    names to their query answer atom; two queries with structurally identical
    proofs still get one output each.

    With evidence (``result.evidence`` set) each output is the posterior
    ``P(q | e) = E[q∧e] / E[e]`` instead — see :func:`_compile_conditional`.
    Either way the result is a single ``SymTensor`` with one column per answer.

    Args:
        result: The engine result; ``formulas`` are boolean circuit lumps and
            ``labels`` their probability annotations. ``evidence``, when present,
            is the shared boolean ``e`` lump to condition on.
        factory: The module factory that lowers the transformed lumps.

    Returns:
        A composed DeepLogModule whose outputs are named by query answer —
        ``P(q)`` without evidence, ``P(q | e)`` with it.
    """
    if result.evidence is not None:
        return _compile_conditional(result, factory)

    answers = tuple(result.formulas.keys())

    # The engine already built each query into one shared boolean source circuit.
    boolean_lumps = _as_circuit_nodes(
        result.formulas.values(),
        "Engine result formulas must be boolean circuit lumps.",
    )

    module = _lower(factory, boolean_lumps, result)
    return _relabel_to_answers(module, _tag_answers(module, answers))


def _compile_conditional(
    result: EngineResult[FormulaNode],
    factory: DeepLogModuleFactory,
) -> DeepLogModule:
    """Compile a conditional result (``result.evidence`` set) to ``P(q | e)``.

    Each query answer carries its joint proof ``q∧e`` in ``result.formulas`` and
    the shared evidence ``e`` in ``result.evidence``.

    The numerators and the denominator go through one batched
    weighted-model-count and one call to
    :func:`~deeplog.formula.deeplogmodulefactory.lower_circuit_nodes`, so each
    predicate module appears once no matter how many answers there are. The
    resulting N+1 outputs are divided by
    :class:`~deeplog.module.ColumnwiseModule` using the probability
    :class:`~deeplog.algebraic.Semifield`'s ``divide`` (clamped, so impossible
    evidence stays finite). Only the answer columns are relabelled to their query
    atoms, so the denominator is told apart by the positional name it kept.
    """
    answers = tuple(result.formulas.keys())
    lumps = _as_circuit_nodes(
        (*result.formulas.values(), result.evidence),
        "Conditional result formulas must be boolean circuit lumps.",
    )

    # The denominator is the last root, so it is the last output. Relabel only
    # the answers; the evidence keeps the positional name the lowering gave it,
    # which is already outside the user's atom space, so no reserved output name
    # is needed to tell it apart.
    lowered = _lower(factory, lumps, result)
    # ``evidence`` is a root name minted by the lowering, so it is already
    # labelled; the answers are bare grounder atoms and are labelled to match.
    # ``ColumnwiseModule`` resolves its column groups by symbol, so the two
    # spellings have to agree exactly.
    evidence = list(get_all_symbols(lowered.get_output_shape()))[-1]
    tagged = _tag_answers(lowered, answers)
    joint = _relabel_to_answers(lowered, (*tagged, evidence))
    return ColumnwiseModule(
        PROBABILITY.division_fn,
        joint,
        tagged,
        (evidence,),
        name=PROBABILITY.division,
    )


def _as_circuit_nodes(formulas: object, message: str) -> tuple[CircuitNode, ...]:
    """Narrow engine-result formulas to circuit lumps, or raise ``message``."""
    lumps = tuple(cast("tuple[FormulaNode, ...]", formulas))
    if not all(isinstance(lump, CircuitNode) for lump in lumps):
        raise TypeError(message)
    return cast("tuple[CircuitNode, ...]", lumps)


def _lower(
    factory: DeepLogModuleFactory,
    boolean_lumps: tuple[CircuitNode, ...],
    result: EngineResult[FormulaNode],
) -> DeepLogModule:
    """Lower each boolean lump as the expectation of it — one compilation.

    The count is *deferred*, exactly as it is for an ``expectation`` the textual
    language writes: each lump becomes a leaf of one probability circuit fed by
    the aggregation counting it, and the lowering does every count at its single
    site. What the engine knows and the count cannot derive — Definition 12's α
    as a leaf mapping, and where each variable occurs — is handed to it there.
    """
    # One factory, so every count's leaf lands in its one probability circuit.
    counts = CircuitFactory()
    deferred = [
        counts.create_aggregation("expectation", [], (), lump) for lump in boolean_lumps
    ]
    return lower_circuit_nodes(
        factory,
        *deferred,
        leaf_mapping=build_leaf_mapping(result.labels),
        variables=result.variables,
    )


def build_leaf_mapping(
    labels: Mapping[Symbol, Symbol],
) -> Callable[[Symbol], Symbol]:
    """Build a boolean-to-probability leaf mapping directly from atom labels.

    ``labels`` maps each labeled boolean atom (e.g. ``("a", ("x1",))``) to its
    probability label atom (e.g. ``("nn1", ("x1",))``). The returned callable
    rewrites a *bare* boolean leaf (the canonical identity a circuit exposes via
    :meth:`~deeplog.circuit.circuit.Circuit.get_leaf_name`) to its matching
    probability leaf. Leaves without an atom label are retagged to the
    probability structure unchanged (they become probability inputs or
    builder-backed leaves).

    Building the mapping straight from ``labels`` keeps it unambiguous when
    distinct atoms share arguments — ``a(x1)`` and ``b(x1)`` labeled by
    ``nn1(x1)`` and ``nn2(x1)`` — which arguments alone cannot resolve.

    A numeric label (e.g. ``0.6 :: fact``) maps to its constant symbol like any
    other, and the target circuit folds it to a constant node.

    Args:
        labels: Maps boolean atoms to their probability label atoms.

    Returns:
        A callable mapping boolean leaf symbols to probability leaf symbols.
    """
    mapping: dict[Symbol, Symbol] = {
        bool_atom: with_structure(prob_atom, "probability")
        for bool_atom, prob_atom in labels.items()
    }

    def leaf_mapping(sym: Symbol) -> Symbol:
        # ``sym`` is a bare boolean leaf; map it to its label or carry it into
        # the probability structure unchanged.
        return mapping.get(sym) or with_structure(sym, "probability")

    return leaf_mapping


def _tag_answers(
    module: DeepLogModule, answers: tuple[Symbol, ...]
) -> tuple[Symbol, ...]:
    """Label each query answer atom with the algebra ``module``'s outputs carry.

    The answers arrive as bare ground atoms from the grounder, while the columns
    they rename are probability-valued. The renamed shape and every later
    by-symbol lookup (``ColumnwiseModule``'s column groups, ``to_dict``) must
    agree on the labelled spelling, so it is derived once, here.
    """
    structure = sole_structure(module.get_output_shape())
    if structure is None:
        raise ValueError("Lowered query module carries no algebraic structure.")
    return tuple(with_structure(answer, structure) for answer in answers)


def _relabel_to_answers(
    module: DeepLogModule, answers: tuple[Symbol, ...]
) -> DeepLogModule:
    """Relabel ``module``'s positional outputs to their query answer atoms.

    :func:`~deeplog.formula.deeplogmodulefactory.lower_circuit_nodes` names the N
    co-resident roots positionally, in ``result.formulas`` order, which is
    ``answers`` order; output ``i`` is redeclared as ``answers[i]`` so consumers
    can index by name. The forward pass is unchanged.

    ``answers`` must already be structure-labelled (see :func:`_tag_answers`),
    since the relabel replaces the output symbols wholesale and a bare answer
    would strip the algebra off a probability column.
    """
    if len(answers) != len(list(module.get_output_shape())):
        raise ValueError("Expected one answer per lowered output.")
    return WrappedModule(
        module,
        module.get_input_shape(),
        SymTensor(list(answers)),
        name="query_answers",
    )
