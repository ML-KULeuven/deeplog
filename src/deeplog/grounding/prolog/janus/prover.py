#  Copyright (c) 2024-2026. KU Leuven
"""Run a prover written as an SWI-Prolog module through Janus."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import get_predicate
from deeplog.symbol import parse_symbol

from ..grounder import Builtin
from ..grounder import UnknownPredicateException


try:
    from janus_swi import PrologError
    from janus_swi import janus

    #: Whether ``janus_swi`` imported, so a :class:`JanusProver` can be built.
    JANUS_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    JANUS_AVAILABLE = False


#: The directory the ``deeplog`` file search path names.
LIBRARY = Path(__file__).parent


class JanusNotAvailableException(Exception):
    """Raised when a Janus prover is built but janus_swi is unavailable."""


class JanusProver:
    """A prover written as an SWI-Prolog module, run through Janus."""

    def __init__(self, module_file: Path):
        """Load the prover in ``module_file``, unless it is loaded already.

        The file is an SWI-Prolog module. It reaches DeepLog's Prolog library
        with ``:- use_module(deeplog(grounding)).``

        Raises:
            JanusNotAvailableException: ``janus_swi`` cannot be imported.
        """
        if not JANUS_AVAILABLE:
            raise JanusNotAvailableException()
        janus.query_once(
            "user:file_search_path(deeplog, Library) -> true"
            " ; assertz(user:file_search_path(deeplog, Library))",
            {"Library": str(LIBRARY)},
        )
        loaded = janus.query_once(
            "use_module(File, []),"
            " absolute_file_name(File, Path, [file_type(prolog), access(read)]),"
            " module_property(Module, file(Path))",
            {"File": str(module_file)},
        )
        if not loaded["truth"]:
            raise ValueError(f"{module_file} defines no Prolog module.")
        self._module: str = loaded["Module"]
        self._builtins: dict[tuple[str, int], Builtin] = {}

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Make ``functor/arity`` a builtin of the goals this prover runs."""
        self._builtins[(functor, arity)] = builtin_function

    def consult(self, clauses: Iterable[str]) -> str:
        """Load ``clauses``, in order, into a module of their own, and return its name.

        The module is named after the clauses, so the same clauses load once.
        Loading them reports no singleton variables.
        """
        text = "\n".join(clauses)
        module = f"program_{hashlib.sha256(text.encode()).hexdigest()}"
        if not janus.query_once(
            "atom_string(Module, Name), current_module(Module)", {"Name": module}
        )["truth"]:
            # On the first line, so the clauses keep their line numbers.
            janus.consult(
                module, data=f":- style_check(-singleton). {text}", module=module
            )
        return module

    def query(
        self, goal: str, bindings: Mapping[str, object]
    ) -> Iterator[dict[str, Any]]:
        """The solutions of ``goal``, run in the prover's module.

        A goal that reaches ``unknown_predicate/1`` raises
        :class:`~deeplog.grounding.prolog.UnknownPredicateException`, and one that
        reaches ``non_ground_open_predicate/1`` raises :class:`ValueError`.
        """
        try:
            yield from janus.query(f"{self._module}:({goal})", dict(bindings))
        except PrologError as err:
            translated = _translate(err)
            if translated is None:
                raise
            raise translated from err

    def _is_builtin(self, functor: str, arity: int) -> bool:
        return (functor, arity) in self._builtins

    def _call_builtin(self, goal: Symbol) -> list[Symbol]:
        return [
            apply_substitution(goal, answer)
            for answer in self._builtins[get_predicate(goal)](*goal[1:])
        ]


def _translate(err: Exception) -> Exception | None:
    """The Python exception for an error DeepLog's Prolog library throws, if any."""
    thrown = parse_symbol(repr(err))
    if len(thrown) != 3 or thrown[0] != "error" or len(thrown[1]) != 3:
        return None
    kind, name, arity = thrown[1]
    if kind == "unknown_procedure":
        return UnknownPredicateException(
            f"No clauses known for {(name[0], int(arity[0]))}."
        )
    if kind == "open_predicate_not_ground":
        return ValueError(
            f"Open predicate instance of {(name[0], int(arity[0]))} is not ground "
            "after proving its rule body."
        )
    return None
