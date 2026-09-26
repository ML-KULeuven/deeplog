"""Notebook-related Sphinx build helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import jupytext
import nbformat


#  Copyright (c) 2024-2026. KU Leuven

_PATH_STEPS = {
    "ml": ["shape", "deeplogmodule", "formula_to_module", "semantic_loss"],
    "nesy": [
        "symbol",
        "shape",
        "predicates",
        "01_aggregation_basics",
        "03_free_variables_and_batching",
        "formula_to_module",
        "problog",
        "deepproblog",
    ],
}

_COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/ML-KULeuven/deeplog/blob/v{version}/{path})"
)


def write_example_notebooks(app) -> None:
    """Write the example notebooks and curated notebook paths into the doc tree.

    Each is written as ``.ipynb`` and carries an Open in Colab badge under its
    title, which opens the notebook at the release the docs are built for.
    """
    source_root = Path(app.srcdir)
    repo_root = (source_root / ".." / "..").resolve()
    examples_root = repo_root / "examples"
    target_root = source_root / "examples"
    paths_root = source_root / "paths"

    if not examples_root.exists():
        print("WARNING: Examples not found")
        return

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    notebook_by_name: dict[str, Path] = {}
    for notebook_path in examples_root.glob("*.md"):
        notebook_by_name[notebook_path.stem] = notebook_path
        destination = target_root / f"{notebook_path.stem}.ipynb"
        _write_with_colab_badge(
            notebook_path, destination, repo_root, app.config.release
        )

    if paths_root.exists():
        shutil.rmtree(paths_root)
    paths_root.mkdir(parents=True, exist_ok=True)

    for path_name, notebooks in _PATH_STEPS.items():
        dest_dir = paths_root / path_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for notebook in notebooks:
            source_path = notebook_by_name.get(notebook)
            if source_path is None:
                print(f"WARNING: Missing notebook '{notebook}.md' for path {path_name}")
                continue
            destination = dest_dir / f"{notebook}.ipynb"
            _write_with_colab_badge(
                source_path, destination, repo_root, app.config.release
            )


def _write_with_colab_badge(
    source: Path, destination: Path, repo_root: Path, version: str
) -> None:
    """Write a MyST Markdown notebook as ``.ipynb``, with a Colab badge under its title.

    Raises:
        ValueError: If the notebook does not open with a markdown cell whose first
            line is its ``# `` title.
    """
    notebook = jupytext.read(source)
    first = notebook.cells[0] if notebook.cells else None
    if (
        first is None
        or first.cell_type != "markdown"
        or not first.source.startswith("# ")
    ):
        raise ValueError(
            f"{source} does not open with a markdown cell holding its title"
        )
    title, _, rest = first.source.partition("\n")
    badge = _COLAB_BADGE.format(
        version=version,
        path=source.with_suffix(".ipynb").relative_to(repo_root).as_posix(),
    )
    first.source = f"{title}\n\n{badge}\n{rest}"
    nbformat.write(notebook, destination)
    # Keep the source's mtime: Sphinx rereads a document only when its file changed.
    shutil.copystat(source, destination)
