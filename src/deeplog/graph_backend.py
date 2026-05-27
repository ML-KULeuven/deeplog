#  Copyright (c) 2024-2026. KU Leuven
"""
Utilities for working with the optional PyGraphviz dependency.
"""

from __future__ import annotations

from typing import Any


class GraphBackendUnavailable(RuntimeError):
    """Raised when graph rendering is requested without PyGraphviz installed."""


try:
    import pygraphviz as _pygraphviz  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _pygraphviz = None


_DEFAULT_GRAPH_KWARGS = {"directed": True, "rankdir": "BT"}


def is_backend_available() -> bool:
    """Return True when PyGraphviz is available."""
    return _pygraphviz is not None


def ensure_graph(graph: Any | None = None, **kwargs: Any):
    """
    Return a PyGraphviz graph, creating one if necessary.

    :param graph: Existing graph instance or ``None``.
    :param kwargs: Additional kwargs passed to ``pygraphviz.AGraph`` when creating a graph.
    :raises GraphBackendUnavailable: When no backend is available and a new graph is needed.
    """
    if graph is not None:
        return graph
    if _pygraphviz is None:
        raise GraphBackendUnavailable(
            "PyGraphviz is not installed. Install it to enable graph visualisation."
        )
    graph_kwargs = dict(_DEFAULT_GRAPH_KWARGS)
    graph_kwargs.update(kwargs)
    return _pygraphviz.AGraph(**graph_kwargs)


def draw_graph(graph: Any, output_path: str, *, prog: str = "dot") -> str:
    """
    Render the given graph to ``output_path`` using PyGraphviz.

    :param graph: A PyGraphviz ``AGraph`` instance.
    :param output_path: Path where the graph should be written.
    :param prog: Layout engine to use. Defaults to ``dot``.
    :raises GraphBackendUnavailable: When PyGraphviz is not available.
    :return: The string path that was written.
    """
    if _pygraphviz is None:
        raise GraphBackendUnavailable(
            "PyGraphviz is not installed. Install it to render graphs."
        )
    graph.layout(prog=prog)
    graph.draw(output_path)
    return output_path


__all__ = [
    "GraphBackendUnavailable",
    "ensure_graph",
    "draw_graph",
    "is_backend_available",
]
