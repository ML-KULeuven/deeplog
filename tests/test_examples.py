#  Copyright (c) 2026. KU Leuven
"""Properties of the example notebooks that executing them does not check."""

import json
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("examples/*/*.ipynb"))
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
COLAB = f"https://colab.research.google.com/github/ML-KULeuven/deeplog/blob/v{VERSION}/"
INSTALL = (
    "try:\n"
    "    import deeplog\n"
    "except ImportError:\n"
    f'    %pip install -q "pydeeplog[examples]=={VERSION}"'
)


def _cells(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cells"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_colab_badge_opens_this_notebook_at_this_version(path: Path):
    title = "".join(_cells(path)[0]["source"])
    assert f"]({COLAB}{path.relative_to(ROOT).as_posix()})" in title


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_first_code_cell_installs_this_version(path: Path):
    first_code = next(c for c in _cells(path) if c["cell_type"] == "code")
    assert "".join(first_code["source"]) == INSTALL


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_cell_passes_whatever_it_raises(path: Path):
    tagged = [
        i
        for i, cell in enumerate(_cells(path))
        if "raises-exception" in cell["metadata"].get("tags", [])
    ]
    assert not tagged, (
        f"cells {tagged} are tagged raises-exception, so nbmake passes them whatever "
        "they raise; catch the exception the cell demonstrates instead"
    )
