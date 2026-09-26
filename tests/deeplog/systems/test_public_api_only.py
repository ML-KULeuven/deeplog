#  Copyright (c) 2024-2026. KU Leuven
"""The systems use only DeepLog's public API.

A name is public when a public DeepLog package lists it in ``__all__``. Importing
it from any other module, or overriding or calling a single-underscore member of a
DeepLog class, depends on an internal that a minor release may change. On the
Prolog side, the public API is what the ``deeplog(grounding)`` library exports.
"""

import ast
import importlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

import deeplog.systems

from ..test_public_api import public_packages


SYSTEMS = Path(deeplog.systems.__file__).parent
OWN = "deeplog.systems"


def _is_private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _is_own(module: str) -> bool:
    return module == OWN or module.startswith(OWN + ".")


def _is_deeplog(module: str) -> bool:
    return module == "deeplog" or module.startswith("deeplog.")


#: Every public DeepLog package outside the systems, with the names it exports.
PUBLIC = {
    package.__name__: frozenset(package.__all__)
    for package in public_packages()
    if not _is_own(package.__name__)
}


def _homes(name: str) -> str:
    """Where ``name`` can be imported from publicly, for the failure message."""
    homes = sorted(
        (module for module, names in PUBLIC.items() if name in names), key=len
    )
    return f"import it from {homes[0]}" if homes else "it is not public anywhere"


class _Checker(ast.NodeVisitor):
    """Collect every use of a non-public DeepLog name in one module."""

    def __init__(self, package: str):
        self.package = package
        self.violations: list[tuple[int, str]] = []
        self.aliases: set[str] = set()
        self.classes: dict[str, type] = {}

    def _flag(self, node: ast.AST, message: str) -> None:
        self.violations.append((getattr(node, "lineno", 0), message))

    def _resolve(self, node: ast.ImportFrom) -> str:
        if not node.level:
            return node.module or ""
        base = self.package.split(".")[: len(self.package.split(".")) - node.level + 1]
        return ".".join(base + ([node.module] if node.module else []))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = self._resolve(node)
        if not _is_deeplog(module) or _is_own(module):
            return
        for alias in node.names:
            if alias.name == "*":
                self._flag(node, f"`from {module} import *` imports unlisted names")
            elif module not in PUBLIC:
                self._flag(
                    node,
                    f"`{module}` is not a public package; {_homes(alias.name)}",
                )
            elif alias.name not in PUBLIC[module]:
                self._flag(
                    node,
                    f"`{module}` does not export `{alias.name}`; {_homes(alias.name)}",
                )
            else:
                value = getattr(importlib.import_module(module), alias.name)
                if isinstance(value, type):
                    self.classes[alias.asname or alias.name] = value

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "deeplog":
                self.aliases.add(alias.asname or "deeplog")
            elif _is_deeplog(alias.name) and not _is_own(alias.name):
                self._flag(
                    node, f"`import {alias.name}`; import the names it exports instead"
                )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.aliases
            and node.attr not in PUBLIC["deeplog"]
        ):
            self._flag(node, f"`deeplog` does not export `{node.attr}`")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dynamic = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
        ) or (isinstance(node.func, ast.Name) and node.func.id == "__import__")
        if dynamic and node.args and isinstance(node.args[0], ast.Constant):
            module = node.args[0].value
            if (
                isinstance(module, str)
                and _is_deeplog(module)
                and not _is_own(module)
                and module not in PUBLIC
            ):
                self._flag(node, f"`{module}` is not a public package")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        bases = [
            self.classes[base.id]
            for base in node.bases
            if isinstance(base, ast.Name) and base.id in self.classes
        ]
        for base in bases:
            for member in _members(node):
                if _is_private(member) and hasattr(base, member):
                    self._flag(node, f"overrides private `{base.__name__}.{member}`")
            for access in _own_accesses(node):
                if _is_private(access.attr) and hasattr(base, access.attr):
                    self._flag(access, f"uses private `{base.__name__}.{access.attr}`")
        self.generic_visit(node)


def _members(node: ast.ClassDef) -> Iterator[str]:
    """The names a class body defines."""
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
            yield statement.name
        elif isinstance(statement, ast.Assign):
            yield from (t.id for t in statement.targets if isinstance(t, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            yield statement.target.id


def _own_accesses(node: ast.ClassDef) -> Iterator[ast.Attribute]:
    """Attribute accesses on ``self``, ``cls`` or ``super()`` inside a class body."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        receiver = child.value
        if (isinstance(receiver, ast.Name) and receiver.id in {"self", "cls"}) or (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "super"
        ):
            yield child


def violations(source: str, package: str) -> list[tuple[int, str]]:
    """Every non-public DeepLog use in ``source``, a module of ``package``."""
    checker = _Checker(package)
    checker.visit(ast.parse(source))
    return checker.violations


def _package(path: Path) -> str:
    """The package a module file belongs to; for ``__init__.py``, its own."""
    return ".".join(path.relative_to(SYSTEMS.parent.parent).parent.parts)


def prolog_violations(source: str) -> list[tuple[int, str]]:
    """Every non-public DeepLog use in the Prolog ``source``."""
    found = []
    for number, line in enumerate(source.splitlines(), start=1):
        code = line.split("%", 1)[0]
        for library in re.findall(r"\bdeeplog\((\w+)\)", code):
            if library != "grounding":
                found.append((number, f"`deeplog({library})` is not a public library"))
        for module in re.findall(r"\b(deeplog_\w+):", code):
            found.append((number, f"calls into `{module}` past its exports"))
        for method in re.findall(r":\s*'(_\w+)'\s*\(", code):
            found.append((number, f"calls the private Python method `{method}`"))
    return found


def _files(pattern: str):
    for path in sorted(SYSTEMS.rglob(pattern)):
        yield pytest.param(path, id=path.relative_to(SYSTEMS).as_posix())


@pytest.mark.parametrize("path", _files("*.py"))
def test_module_uses_only_the_public_api(path):
    found = violations(path.read_text(), _package(path))
    assert not found, "\n".join(f"{path.name}:{line}: {msg}" for line, msg in found)


@pytest.mark.parametrize("path", _files("*.pl"))
def test_prolog_uses_only_the_public_library(path):
    found = prolog_violations(path.read_text())
    assert not found, "\n".join(f"{path.name}:{line}: {msg}" for line, msg in found)


ACCEPTED = {
    "a top-level name": "from deeplog import Symbol",
    "a subpackage name": "from deeplog.grounding import SimpleGrounder",
    "the package as a namespace": "import deeplog\ndeeplog.Symbol",
    "the system's own module": "from .ad import instantiate",
    "a public method of a subclassed class": (
        "from deeplog.grounding import ProofBuilder\n"
        "class B(ProofBuilder):\n"
        "    def leaf(self, atom):\n"
        "        return super().leaf(atom)\n"
    ),
}

REJECTED = {
    "a module path": "from deeplog.symbol import Symbol",
    "an unlisted name": "from deeplog import util",
    "a star import": "from deeplog import *",
    "a relative path into core": "from ...symbol import Symbol",
    "a submodule import": "import deeplog.symbol",
    "an unlisted attribute": "import deeplog as d\nd.util",
    "a dynamic import": "import importlib\nimportlib.import_module('deeplog.symbol')",
    "a private override": (
        "from deeplog.grounding import ProofBuilder\n"
        "class B(ProofBuilder):\n"
        "    def _combine(self, operator, lhs, rhs, identity):\n"
        "        return lhs\n"
    ),
    "a private call": (
        "from deeplog.grounding import ProofBuilder\n"
        "class B(ProofBuilder):\n"
        "    def f(self, lhs, rhs):\n"
        "        return self._combine('and', lhs, rhs, lhs)\n"
    ),
}


@pytest.mark.parametrize("source", ACCEPTED.values(), ids=ACCEPTED)
def test_the_checker_accepts(source):
    assert violations(source, "deeplog.systems.example") == []


@pytest.mark.parametrize("source", REJECTED.values(), ids=REJECTED)
def test_the_checker_rejects(source):
    assert violations(source, "deeplog.systems.example")


PROLOG_ACCEPTED = {
    "the library": ":- use_module(deeplog(grounding)).",
    "an exported predicate": "p(P, G) :- is_builtin(P, G), call_builtin(P, G).",
    "a public Python method": "p(F, L) :- py_call(F:get_true(), L).",
}

PROLOG_REJECTED = {
    "another library file": ":- use_module(deeplog(engine)).",
    "a module prefix": "p(G) :- deeplog_grounding:allowed_builtin(G).",
    "a private Python method": "p(P, S, R) :- py_call(P:'_call_builtin'(S), R).",
}


@pytest.mark.parametrize("source", PROLOG_ACCEPTED.values(), ids=PROLOG_ACCEPTED)
def test_the_prolog_checker_accepts(source):
    assert prolog_violations(source) == []


@pytest.mark.parametrize("source", PROLOG_REJECTED.values(), ids=PROLOG_REJECTED)
def test_the_prolog_checker_rejects(source):
    assert prolog_violations(source)
