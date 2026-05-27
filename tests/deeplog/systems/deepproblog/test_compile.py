#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the compile_to_module pipeline."""

import pytest

from deeplog.formula import DeepLogModuleFactory
from deeplog.systems.deepproblog import compile_to_module
from deeplog.systems.deepproblog.engine import SimpleEngine
from deeplog.systems.deepproblog.program import str_to_rules


def test_compile_produces_module():
    """compile_to_module produces a runnable DeepLogModule."""
    engine = SimpleEngine()

    program = tuple(
        str_to_rules(
            """
        0.6::burglary.
        0.1::earthquake.
        alarm :- burglary.
        alarm :- earthquake.
        ?- alarm.
        """
        )
    )

    factory = DeepLogModuleFactory()
    result = engine.get_query_result(program, factory)

    module = compile_to_module(result, factory)
    assert module is not None

    output_symbols = list(module.get_output_shape())
    assert len(output_symbols) == 1


def test_build_modules_for_leaves_ignores_non_wrapped_symbols():
    """Leaves whose shape isn't the ``("_", atom, (structure,))`` wrapper are skipped."""
    factory = DeepLogModuleFactory()
    # A bare atom (len != 3 / sym[0] != "_") shouldn't produce any module.
    assert factory.build_modules_for_leaves([("plain_atom",)]) == []


def test_build_modules_for_leaves_skips_unregistered_predicates():
    """Leaves whose predicate isn't registered are skipped without error."""
    factory = DeepLogModuleFactory()
    leaf = ("_", ("unknown_pred", ("a",), ("b",)), ("probability",))
    assert factory.build_modules_for_leaves([leaf]) == []


def test_build_modules_for_leaves_invokes_registered_builder():
    """When a matching builder exists, build_modules_for_leaves returns its module."""
    factory = DeepLogModuleFactory()
    # ("=", 2, "boolean") is registered by default (EqualityPredicate).
    leaf = ("_", ("=", ("true",), ("false",)), ("boolean",))
    modules = factory.build_modules_for_leaves([leaf])
    assert len(modules) == 1


def test_compile_rejects_non_expectation_aggregations():
    """compile_to_module defends against aggregations that aren't expectations."""
    from deeplog.systems.deepproblog.engine.engine import EngineResult

    factory = DeepLogModuleFactory()

    # Build a minimal boolean child node so create_aggregation has something to wrap.
    atom = factory.create_atom(("_", ("a",), ("boolean",)))
    result = EngineResult(formulas={("a",): atom}, labels={})

    # Intercept create_aggregation to return something other than ExpectationNode.
    factory.create_aggregation = lambda *_a, **_k: atom  # type: ignore[assignment]

    with pytest.raises(TypeError, match="ExpectationNode"):
        compile_to_module(result, factory)


def test_compile_multiple_queries():
    """Multiple queries compile into a module with multiple outputs."""
    engine = SimpleEngine()

    program = tuple(
        str_to_rules(
            """
        0.6::a.
        b :- a.
        c :- a.
        ?- b.
        ?- c.
        """
        )
    )

    factory = DeepLogModuleFactory()
    result = engine.get_query_result(program, factory)

    assert len(result.formulas) == 2
    module = compile_to_module(result, factory)
    assert module is not None

    output_symbols = list(module.get_output_shape())
    assert len(output_symbols) == 2
