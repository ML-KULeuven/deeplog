#  Copyright (c) 2026. KU Leuven
"""Collects every example notebook as a test that executes it."""

from pathlib import Path

import jupytext
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


EXAMPLES = Path(__file__).parent


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

        Raises:
            CellExecutionError: If a cell raises.
            CellTimeoutError: If a cell runs longer than 120 seconds.
        """
        notebook = jupytext.read(self.path)
        resources = {"metadata": {"path": str(self.path.parent)}}
        NotebookClient(notebook, timeout=120, resources=resources).execute()

    def repr_failure(self, excinfo, style=None):
        """The failing cell's source and traceback, for a cell that raised."""
        if isinstance(excinfo.value, CellExecutionError):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)

    def reportinfo(self):
        """The notebook's path, relative to ``examples/``."""
        return self.path, None, self.path.relative_to(EXAMPLES).as_posix()
