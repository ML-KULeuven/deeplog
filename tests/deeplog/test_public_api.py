#  Copyright (c) 2024-2026. KU Leuven
"""What the public surface promises: every name it mentions can be imported.

A signature that names a type the caller cannot import leaves them guessing at
an import path or reaching into a module. The rule: a type named in a public
signature is exported by the package that exports the signature, or by
``deeplog`` itself.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import pkgutil
import re
import typing

import pytest

import deeplog


def public_packages():
    """Every package that declares a public surface."""
    modules = [deeplog] + [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(deeplog.__path__, "deeplog.")
        if info.ispkg and not any(part.startswith("_") for part in info.name.split("."))
    ]
    return [module for module in modules if hasattr(module, "__all__")]


def defined_names():
    """Every type DeepLog defines, mapped to the module that defines it.

    Type parameters are left out: they stand for a caller's own type rather
    than for one the caller imports.
    """
    defined: dict[str, str] = {}
    for info in pkgutil.walk_packages(deeplog.__path__, "deeplog."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:
            continue
        for name, value in vars(module).items():
            home = getattr(value, "__module__", None)
            if name.startswith("_") or isinstance(value, typing.TypeVar):
                continue
            if str(home).startswith("deeplog"):
                defined.setdefault(name, home)
    return defined


def reachable_from(package, exported):
    """The names a caller holding ``package.exported`` can already import.

    Its own package and ``deeplog``, plus every package above the one that
    defines it: a re-export carries no new contract, so a type stays reachable
    where the object it belongs to lives.
    """
    surfaces = {deeplog.__name__, package.__name__}
    home = getattr(getattr(package, exported), "__module__", "")
    parts = str(home).split(".")
    surfaces |= {".".join(parts[:i]) for i in range(1, len(parts) + 1)}
    return {
        name
        for module in public_packages()
        if module.__name__ in surfaces
        for name in module.__all__
    }


def annotations_of(obj):
    """The annotations of ``obj``, and of its public members when it is a class."""
    members = [obj]
    if inspect.isclass(obj):
        members += [
            value.__func__ if isinstance(value, staticmethod | classmethod) else value
            for name, value in vars(obj).items()
            if not name.startswith("_") or name == "__init__"
        ]
    for member in members:
        yield from getattr(member, "__annotations__", {}).values()


def names_in(annotation):
    """The identifiers an annotation mentions, as written."""
    if not isinstance(annotation, str):
        found = {getattr(annotation, "__name__", None)}
        for argument in typing.get_args(annotation):
            found |= names_in(argument)
        return {name for name in found if name}
    try:
        tree = ast.parse(annotation, mode="eval")
    except SyntaxError:
        return set()
    return {
        node.id if isinstance(node, ast.Name) else node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        or (isinstance(node, ast.Constant) and isinstance(node.value, str))
    }


@pytest.mark.parametrize(
    "package", public_packages(), ids=lambda module: module.__name__
)
def test_a_type_named_in_a_public_signature_is_importable(package):
    """Public signatures name only types the package or ``deeplog`` exports."""
    defined = defined_names()
    unreachable = {}
    for exported in package.__all__:
        reachable = reachable_from(package, exported)
        for annotation in annotations_of(getattr(package, exported)):
            for name in names_in(annotation) - reachable:
                if name in defined:
                    unreachable.setdefault(name, set()).add(exported)
    assert not unreachable, "\n".join(
        f"{package.__name__}.{', '.join(sorted(users))} names {name}, "
        f"defined in {defined[name]}, which no package a caller already "
        f"holds exports"
        for name, users in sorted(unreachable.items())
    )


def test_every_exported_name_resolves():
    """A package exports nothing it cannot hand out."""
    missing = {
        f"{package.__name__}.{name}"
        for package in public_packages()
        for name in package.__all__
        if not hasattr(package, name)
    }
    assert not missing


#: A role or an inline literal carries its own colons; napoleon skips those.
_PROTECTED = re.compile(r":\w+:`[^`]*`|``[^`]*``|`[^`]*`")


def attribute_comments():
    """Every ``#:`` comment block in the source, as (file, line, first line)."""
    for path in sorted(pathlib.Path("src/deeplog").rglob("*.py")):
        previous_was_comment = False
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#:"):
                if not previous_was_comment:
                    yield path, number, stripped.removeprefix("#:").strip()
                previous_was_comment = True
            else:
                previous_was_comment = False


def test_no_attribute_comment_opens_with_a_colon():
    """A ``#:`` comment's first line reads as prose, not as ``type: description``."""
    offenders = [
        f"{path}:{number}: {text}"
        for path, number, text in attribute_comments()
        if ":" in _PROTECTED.sub("", text)
    ]
    assert not offenders, "\n".join(offenders)
