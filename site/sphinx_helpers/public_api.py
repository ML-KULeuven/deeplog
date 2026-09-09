"""One-page public API reference, derived from what the packages export."""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import sys
from functools import cache
from pathlib import Path
from types import ModuleType

from sphinx.errors import PycodeError
from sphinx.pycode import ModuleAnalyzer


#  Copyright (c) 2024-2026. KU Leuven

_INTRO = """\
Public API
==========

Every name exported by a DeepLog package, with its signature and docstring, on
one page. Names appear under the package they are first exported from, so a name
re-exported for convenience elsewhere is documented once. For the full module
tree, including internals, see the :doc:`API reference <autoapi/index>`.
"""


def _public_packages(root: ModuleType) -> list[ModuleType]:
    """Return ``root`` and every public subpackage declaring ``__all__``."""
    modules = [root]
    for info in pkgutil.walk_packages(root.__path__, f"{root.__name__}."):
        if info.ispkg and not any(
            part.startswith("_") for part in info.name.split(".")
        ):
            modules.append(importlib.import_module(info.name))
    return sorted(
        (module for module in modules if hasattr(module, "__all__")),
        key=lambda module: (module.__name__.count("."), module.__name__),
    )


def _directive(obj: object) -> str:
    """Return the autodoc directive documenting ``obj``."""
    if inspect.isclass(obj):
        return "autoclass"
    if inspect.isroutine(obj):
        return "autofunction"
    return "autodata"


@cache
def _alias_comments(module_name: str) -> dict[str, tuple[str, ...]]:
    """The ``#:`` comments a module's ``type`` statements carry.

    Sphinx reads an attribute comment off an assignment; a type alias is a
    statement of its own, so the comment above one reaches no page unless it is
    handed to the directive as content.
    """
    module = sys.modules[module_name]
    try:
        lines = inspect.getsource(module).splitlines()
    except (OSError, TypeError):
        return {}
    documented = {}
    for node in ast.parse("\n".join(lines)).body:
        if not isinstance(node, ast.TypeAlias):
            continue
        comment = []
        for line in reversed(lines[: node.lineno - 1]):
            if not line.strip().startswith("#:"):
                break
            comment.append(line.strip().removeprefix("#:").strip())
        if comment:
            documented[node.name.id] = tuple(reversed(comment))
    return documented


def _documents(module: ModuleType, name: str) -> bool:
    """Whether ``module`` gives its own ``name`` a docstring."""
    if name in _alias_comments(module.__name__):
        return True
    try:
        analyzer = ModuleAnalyzer.for_module(module.__name__)
        analyzer.analyze()
    except PycodeError:
        return False
    return ("", name) in analyzer.attr_docs


def _home(module: ModuleType, name: str, obj: object) -> ModuleType:
    """Return the module ``name`` is defined in, else the one exporting it.

    Docstrings reference their neighbours relatively, and both those references
    and a variable's docstring resolve against the module the name is assigned in.
    """
    defined_in = getattr(obj, "__module__", None)
    if (inspect.isclass(obj) or inspect.isroutine(obj)) and defined_in in sys.modules:
        return sys.modules[defined_in]
    package = module.__name__.partition(".")[0]
    return next(
        (
            loaded
            for loaded_name, loaded in sorted(sys.modules.items())
            if loaded_name.startswith(f"{package}.")
            and getattr(loaded, name, None) is obj
            and _documents(loaded, name)
        ),
        module,
    )


def _target(home: ModuleType, name: str, obj: object) -> str:
    """Return the path ``obj`` is documented under, within its home module."""
    qualname = getattr(obj, "__qualname__", None)
    if (inspect.isclass(obj) or inspect.isroutine(obj)) and qualname:
        return f"{home.__name__}.{qualname}"
    return f"{home.__name__}.{name}"


def _summary(module: ModuleType) -> str:
    """Return the first paragraph of ``module``'s docstring."""
    docstring = inspect.cleandoc(module.__doc__ or "").strip()
    return docstring.split("\n\n")[0]


def drop_borrowed_docstring(app, what, name, obj, options, lines) -> None:
    """Drop a variable's docstring when it is the docstring of the value's type.

    A variable documents itself with a docstring under its assignment. Without
    one, ``__doc__`` resolves to the type of the value it happens to hold.
    """
    if what != "data":
        return
    borrowed = inspect.cleandoc(getattr(type(obj), "__doc__", "") or "").strip()
    if borrowed and "\n".join(lines).strip() == borrowed:
        lines.clear()


def write_public_api(app) -> None:
    """Write the single-page public API reference into the doc tree."""
    root = importlib.import_module("deeplog")
    lines = [_INTRO]
    documented: set[int] = set()
    for module in _public_packages(root):
        exported = []
        for name in sorted(module.__all__, key=str.lower):
            obj = getattr(module, name)
            if id(obj) in documented:
                continue
            documented.add(id(obj))
            exported.append((name, obj))
        if not exported:
            continue
        heading = f"``{module.__name__}``"
        lines += [heading, "-" * len(heading), "", _summary(module), ""]
        for name, obj in exported:
            home = _home(module, name, obj)
            lines.append(f".. {_directive(obj)}:: {_target(home, name, obj)}")
            if inspect.isclass(obj):
                lines.append("   :members:")
            # The same objects are documented by autoapi; index them there only.
            lines.append("   :no-index:")
            lines.append("")
            lines += [
                f"   {line}" for line in _alias_comments(home.__name__).get(name, ())
            ]
            lines.append("")
    (Path(app.srcdir) / "public_api.rst").write_text("\n".join(lines))
