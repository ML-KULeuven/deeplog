"""Notebook-related Sphinx build helpers."""

from __future__ import annotations

import shutil
from pathlib import Path


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
        "mnist_addition",
    ],
}


def copy_example_notebooks(app) -> None:
    """Copy example notebooks and curated notebook paths into the doc tree."""
    source_root = Path(app.srcdir)
    examples_root = (source_root / ".." / ".." / "examples").resolve()
    target_root = source_root / "examples"
    paths_root = source_root / "paths"

    if not examples_root.exists():
        print("WARNING: Examples not found")
        return

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    notebook_by_name: dict[str, Path] = {}
    for notebook_path in examples_root.rglob("*.ipynb"):
        notebook_by_name[notebook_path.stem] = notebook_path
        destination = target_root / notebook_path.name
        shutil.copy2(notebook_path, destination)

    if paths_root.exists():
        shutil.rmtree(paths_root)
    paths_root.mkdir(parents=True, exist_ok=True)

    for path_name, notebooks in _PATH_STEPS.items():
        dest_dir = paths_root / path_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for notebook in notebooks:
            source_path = notebook_by_name.get(notebook)
            if source_path is None:
                print(
                    f"WARNING: Missing notebook '{notebook}.ipynb' for path {path_name}"
                )
                continue
            destination = dest_dir / f"{notebook}.ipynb"
            shutil.copy2(source_path, destination)
