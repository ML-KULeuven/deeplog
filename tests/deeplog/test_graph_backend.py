#  Copyright (c) 2024-2026. KU Leuven
"""Tests for the optional PyGraphviz backend wrapper."""

import pytest

from deeplog import graph_backend
from deeplog.graph_backend import GraphBackendUnavailable
from deeplog.graph_backend import draw_graph
from deeplog.graph_backend import ensure_graph
from deeplog.graph_backend import is_backend_available


requires_pygraphviz = pytest.mark.skipif(
    not is_backend_available(),
    reason="pygraphviz not available",
)


def test_is_backend_available_true_when_imported(monkeypatch):
    monkeypatch.setattr(graph_backend, "_pygraphviz", object())
    assert is_backend_available() is True


def test_is_backend_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(graph_backend, "_pygraphviz", None)
    assert is_backend_available() is False


def test_ensure_graph_returns_provided_graph():
    sentinel = object()
    assert ensure_graph(sentinel) is sentinel


def test_ensure_graph_raises_when_backend_missing(monkeypatch):
    monkeypatch.setattr(graph_backend, "_pygraphviz", None)
    with pytest.raises(GraphBackendUnavailable, match="PyGraphviz is not installed"):
        ensure_graph()


def test_ensure_graph_creates_new_graph_with_defaults(monkeypatch):
    created = {}

    class FakePygraphviz:
        def AGraph(self, **kwargs):  # noqa: N802 - mirrors pygraphviz API
            created.update(kwargs)
            return "new-graph"

    monkeypatch.setattr(graph_backend, "_pygraphviz", FakePygraphviz())
    result = ensure_graph()
    assert result == "new-graph"
    assert created == {"directed": True, "rankdir": "BT"}


def test_ensure_graph_forwards_kwargs(monkeypatch):
    seen = {}

    class FakePygraphviz:
        def AGraph(self, **kwargs):  # noqa: N802
            seen.update(kwargs)
            return "g"

    monkeypatch.setattr(graph_backend, "_pygraphviz", FakePygraphviz())
    ensure_graph(rankdir="TB", strict=True)
    # User kwargs override defaults.
    assert seen == {"directed": True, "rankdir": "TB", "strict": True}


@requires_pygraphviz
def test_draw_graph_invokes_layout_and_draw(tmp_path):
    calls = []

    class FakeGraph:
        def layout(self, prog):
            calls.append(("layout", prog))

        def draw(self, path):
            calls.append(("draw", path))

    output = str(tmp_path / "graph.svg")
    result = draw_graph(FakeGraph(), output)
    assert result == output
    assert calls == [("layout", "dot"), ("draw", output)]


@requires_pygraphviz
def test_draw_graph_respects_prog_argument(tmp_path):
    calls = []

    class FakeGraph:
        def layout(self, prog):
            calls.append(prog)

        def draw(self, path):
            pass

    draw_graph(FakeGraph(), str(tmp_path / "g.svg"), prog="neato")
    assert calls == ["neato"]


def test_draw_graph_raises_when_backend_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_backend, "_pygraphviz", None)
    with pytest.raises(GraphBackendUnavailable, match="render graphs"):
        draw_graph(object(), str(tmp_path / "g.svg"))
