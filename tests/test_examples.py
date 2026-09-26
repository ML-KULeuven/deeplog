#  Copyright (c) 2026. KU Leuven
"""Properties of the example notebooks that executing them does not check."""

from pathlib import Path

import jupytext
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("examples/*.md"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_cell_passes_whatever_it_raises(path: Path):
    tagged = [
        i
        for i, cell in enumerate(jupytext.read(path).cells)
        if "raises-exception" in cell.metadata.get("tags", [])
    ]
    assert not tagged, (
        f"cells {tagged} are tagged raises-exception, so running the notebook passes "
        "them whatever they raise; catch the exception the cell demonstrates instead"
    )
