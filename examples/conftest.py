#  Copyright (c) 2026. KU Leuven
"""Collects every example notebook as a test that executes it."""

import os
from contextlib import AbstractContextManager
from contextlib import nullcontext
from pathlib import Path

import jupytext
import pytest
from filelock import FileLock
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


EXAMPLES = Path(__file__).parent

#: The variable naming the directory the notebooks keep downloaded datasets in,
#: which defaults to ``data/`` beside them.
DATA_DIR = "DEEPLOG_DATA_DIR"


def pytest_collect_file(
    parent: pytest.Collector, file_path: Path
) -> pytest.File | None:
    """Collect each Markdown file in ``examples/`` as a notebook."""
    if file_path.suffix == ".md" and file_path.parent == EXAMPLES:
        return Notebook.from_parent(parent, path=file_path)
    return None


class Notebook(pytest.File):
    """An example notebook, written in MyST Markdown."""

    def collect(self):
        """The notebook's one test: running it."""
        yield NotebookRun.from_parent(self, name="run")


class NotebookRun(pytest.Item):
    """A run of an example notebook."""

    def runtest(self) -> None:
        """Execute the cells in order, in the notebook's directory.

        A notebook that keeps datasets in the data directory runs while no other
        such notebook does, since a dataset's download is not safe for two
        processes at once and ``pytest -n`` runs notebooks in parallel.

        Raises:
            CellExecutionError: If a cell raises.
            CellTimeoutError: If a cell runs longer than 120 seconds.
        """
        notebook = jupytext.read(self.path)
        resources = {"metadata": {"path": str(self.path.parent)}}
        with _data_lock(self.path):
            NotebookClient(notebook, timeout=120, resources=resources).execute()

    def repr_failure(self, excinfo, style=None):
        """The failing cell's source and traceback, for a cell that raised."""
        if isinstance(excinfo.value, CellExecutionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        """The notebook's path, relative to ``examples/``."""
        return self.path, None, self.path.relative_to(EXAMPLES).as_posix()


def _data_lock(path: Path) -> AbstractContextManager[object]:
    """A lock on the data directory if the notebook at ``path`` reads :data:`DATA_DIR`.

    The directory is resolved as the notebook resolves it, from its own
    directory, so every process locks the same file.
    """
    if DATA_DIR not in path.read_text():
        return nullcontext()
    data = EXAMPLES / os.environ.get(DATA_DIR, "data")
    data.mkdir(parents=True, exist_ok=True)
    return FileLock(data / ".lock")
