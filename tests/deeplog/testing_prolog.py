#  Copyright (c) 2024-2026. KU Leuven
"""Helpers for tests that run SWI-Prolog through Janus."""

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def prolog_stack_limit(limit: int) -> Iterator[None]:
    """Run the body with Prolog's stack limit at ``limit`` bytes, then restore it."""
    from janus_swi import janus

    previous = janus.query_once("current_prolog_flag(stack_limit, Limit)")["Limit"]
    janus.query_once("set_prolog_flag(stack_limit, Limit)", {"Limit": limit})
    try:
        yield
    finally:
        janus.query_once("set_prolog_flag(stack_limit, Limit)", {"Limit": previous})
