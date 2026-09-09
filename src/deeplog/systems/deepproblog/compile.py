#  Copyright (c) 2024-2026. KU Leuven
"""Compile DeepProbLog engine results into DeepLogModules."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...algebraic import PROBABILITY
from ...circuit.circuit import Circuit
from ...formula.ast import CircuitNode
from ...formula.deeplogmodulefactory import lower_circuit_nodes
from ...formula.distribution import build_leaf_mapping
from ...formula.strategies import absorb_aggregation
from ...module import ColumnwiseModule
from ...module import WrappedModule
from ...shape import SymTensor
from ...shape import get_all_symbols
from ...shape import sole_structure
from ...symbol import Symbol
from ...symbol import with_structure
from .solver import EngineResult


if TYPE_CHECKING:
    from ...formula.ast import FormulaNode
    from ...formula.deeplogmodulefactory import DeepLogModuleFactory
    from ...module import DeepLogModule


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
    target = Circuit(structure=PROBABILITY)
    # Total on these: the lumps are circuit nodes (``_as_circuit_nodes``) and the
    # operation is ``expectation``, which is the pair ``absorb_aggregation``
    # answers ``None`` to anything else for. A non-boolean lump raises there.
    deferred = cast(
        "list[CircuitNode]",
        [
            absorb_aggregation(lambda _structure: target, "expectation", (), (), lump)
            for lump in boolean_lumps
        ],
    )
    return lower_circuit_nodes(
        factory,
        *deferred,
        leaf_mapping=build_leaf_mapping(result.labels),
        variables=result.variables,
    )


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
