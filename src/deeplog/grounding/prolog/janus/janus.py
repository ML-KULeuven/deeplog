#  Copyright (c) 2024-2026. KU Leuven
"""A Prolog grounder based on a meta-prover implemented with SWI-Prolog Janus."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeVar

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import get_predicate
from deeplog.symbol import parse_symbol

from ...builder import ProofBuilder
from ..grounder import Builtin
from ..grounder import OpenPredicates
from ..grounder import PrologGrounder
from ..grounder import UnknownPredicateException
from ..program import Program
from ..program import RuleType
from ..program import is_constraint
from ..program import is_fact
from ..program import is_query


try:
    from janus_swi import PrologError
    from janus_swi import janus

    #: Whether ``janus_swi`` imported, so a :class:`JanusGrounder` can be built.
    JANUS_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    JANUS_AVAILABLE = False


if TYPE_CHECKING:
    from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory


T = TypeVar("T")

ROOT = Path(__file__).parent


def _is_prolog_bare_token(s: str) -> bool:
    """Return true if ``s`` can be emitted bare (unquoted) into Prolog source.

    Bare-safe: numbers (parsed as int/float), lowercase identifiers
    ``[a-z][a-zA-Z0-9_]*``, and variable-form names (uppercase- or
    underscore-prefixed identifiers) which Prolog should read as
    variables -- quoting them would convert them into atoms.
    """
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    if not all(c.isalnum() or c == "_" for c in s):
        return False
    return s[0].islower() or s[0].isupper() or s[0] == "_"


def _quote_prolog_atom(s: str) -> str:
    if _is_prolog_bare_token(s):
        return s
    escaped = s.replace("'", "\\'")
    return f"'{escaped}'"


def symbol_to_prolog_str(symbol: Symbol) -> str:
    """Convert a Symbol to a valid Prolog term string using strict prefix notation.

    Translates ``cons``/``nil`` chains (deeplog's internal list encoding) to
    Prolog list syntax ``[h1, h2, ... | tail]`` so SWI list builtins like
    ``nth0/3`` and ``member/2`` operate on them. The reverse direction is
    handled in ``janus_translation.pl`` so round-trips stay symmetric.
    """
    list_str = _try_render_prolog_list(symbol)
    if list_str is not None:
        return list_str
    if symbol == ("nil",):
        return "[]"
    functor = symbol[0]
    if len(symbol) == 1:
        return _quote_prolog_atom(functor)
    args = ",".join(symbol_to_prolog_str(s) for s in symbol[1:])
    return f"{_quote_prolog_atom(functor)}({args})"


def _try_render_prolog_list(symbol: Symbol) -> str | None:
    """Render a ``cons``/``nil`` chain as ``[h1, h2, ... | tail]``; else None."""
    elements: list[str] = []
    current: Symbol = symbol
    while isinstance(current, tuple) and len(current) == 3 and current[0] == "cons":
        elements.append(symbol_to_prolog_str(current[1]))
        current = current[2]
    if not elements:
        return None
    if isinstance(current, tuple) and current == ("nil",):
        return f"[{','.join(elements)}]"
    tail = symbol_to_prolog_str(current)
    return f"[{','.join(elements)}|{tail}]"


class JanusNotAvailableException(Exception):
    """Raised when a Janus grounder is instantiated but janus_swi is unavailable."""


class JanusGrounder(PrologGrounder):
    """A Prolog grounder based on a meta-prover implemented with SWI-Prolog Janus."""

    program_identifiers: dict = {}
    #: Shared across the whole hierarchy, because a Prolog module name is
    #: global — a subclass with its own ``program_identifiers`` must not
    #: reuse a name.
    program_counter = 0
    code_loaded = False
    _shared_prolog_dir_registered = False
    engine_counter = 0
    _id_prefix = "engine"
    _engine_code_path = ROOT / "engine.pl"

    def __init_subclass__(cls, **kwargs):
        """Give each subclass its own program cache and code-loaded flag."""
        super().__init_subclass__(**kwargs)
        cls.program_identifiers = {}
        cls.code_loaded = False

    def __init__(self):
        """Initialize a Janus-backed grounder and ensure the Prolog code is loaded."""
        super().__init__()
        if not JANUS_AVAILABLE:
            raise JanusNotAvailableException()
        self._ensure_code_loaded()
        self._id = f"{type(self)._id_prefix}_{JanusGrounder.engine_counter}"
        JanusGrounder.engine_counter += 1
        self._builtins: dict[tuple[str, int], Builtin] = {}

    @classmethod
    def _ensure_code_loaded(cls):
        if not JanusGrounder._shared_prolog_dir_registered:
            # Point the `deeplog_prolog` file-search alias at this package's
            # plain-Prolog helper modules (builtins.pl, janus_translation.pl).
            # Every engine.pl — including subclasses that live in other packages,
            # such as k-best — loads that single shared copy via
            # `use_module(deeplog_prolog(...))`, so the helper modules are never
            # duplicated across prover directories.
            janus.query_once(
                "assertz(user:file_search_path(deeplog_prolog, Dir))",
                {"Dir": str(ROOT.absolute())},
            )
            JanusGrounder._shared_prolog_dir_registered = True
        if not cls.__dict__.get("code_loaded", False):
            janus.consult(str(cls._engine_code_path.absolute()))
            cls.code_loaded = True

    def ground(
        self,
        program: Program,
        goal: Symbol,
        factory: DeepLogFormulaFactory[T] | ProofBuilder[T],
        open_predicates: OpenPredicates = frozenset(),
    ) -> dict[Symbol, T]:
        """Prove ``goal`` in ``program`` to one proof formula per ground answer."""
        builder = ProofBuilder.wrapping(factory)
        variables = {
            "ProgramID": self._assert_program(program, open_predicates),
            "Query": goal,
            "Factory": builder,
        }
        try:
            results = janus.query(
                "prove_query(ProgramID,Query,Factory,GroundQuery,Formula)", variables
            )
            return {result["GroundQuery"]: result["Formula"] for result in results}
        except PrologError as err:
            raise self._translate_prolog_error(err) from err

    @staticmethod
    def _translate_prolog_error(err: Exception) -> Exception:
        """Map a SWI ``PrologError`` to a domain exception when recognised."""
        error_symbol = parse_symbol(repr(err))
        if len(error_symbol) == 3 and error_symbol[0] == "error":
            error_code = error_symbol[1]
            if len(error_code) == 3 and error_code[0] == "unknown_procedure":
                predicate = error_code[1][0], int(error_code[2][0])
                return UnknownPredicateException(f"No clauses known for {predicate}.")
            if len(error_code) == 3 and error_code[0] == "open_predicate_not_ground":
                predicate = error_code[1][0], int(error_code[2][0])
                return ValueError(
                    f"Open predicate instance of {predicate} is not ground "
                    "after proving its rule body."
                )
        return err

    def _assert_program(
        self, program: Program, open_predicates: OpenPredicates = frozenset()
    ):
        """Consult ``program`` into its own Prolog module once, and return its id.

        Subclasses that render a different dialect of the program override
        :meth:`_rules_to_janus_code` rather than this method, so the caching and
        ``engine_id`` bookkeeping lives in one place.
        """
        key = (program, frozenset(open_predicates))
        try:
            return self.program_identifiers[key]
        except KeyError:
            identifier = f"program_{JanusGrounder.program_counter}"
            JanusGrounder.program_counter += 1
            program_text = "\n".join(
                sorted(self._rules_to_janus_code(program, open_predicates))
            )
            janus.consult(identifier, data=program_text, module=identifier)
            self.program_identifiers[key] = identifier
            janus.query_once(
                "assertz(ProgramID:engine_id(Engine,ID))",
                {"ProgramID": identifier, "Engine": self, "ID": self._id},
            )

        return identifier

    def _rules_to_janus_code(
        self, program: Iterable[RuleType], open_predicates: OpenPredicates
    ) -> Iterable[str]:
        """Render ``program`` as the Prolog source this prover's engine expects."""
        # The directive sorts before any fact (":" < letters), so it survives
        # the sorted() in _assert_program; it keeps open_predicate/2 callable
        # even when no declarations follow.
        yield ":- dynamic open_predicate/2."
        for name, arity in open_predicates:
            yield f"open_predicate({name},{arity})."
        for rule in program:
            if is_query(rule) or is_constraint(rule):
                continue
            if is_fact(rule) and get_predicate(rule[1]) in open_predicates:
                yield f"leaf({symbol_to_prolog_str(rule[1])})."
            else:
                yield (
                    f"rule({symbol_to_prolog_str(rule[1])},"
                    f"{symbol_to_prolog_str(rule[2])})."
                )

    @staticmethod
    def is_available() -> bool:
        """Return true if janus_swi is available and the class can be instantiated."""
        return JANUS_AVAILABLE

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register a builtin in both the Python registry and the Janus engine."""
        self._builtins[(functor, arity)] = builtin_function
        janus.query_once(
            "assertz(EngineID:extern_builtin(Functor,Arity))",
            {"EngineID": self._id, "Functor": functor, "Arity": arity},
        )

    def _call_builtin(self, goal: Symbol):
        results = []
        for answer in self._builtins[get_predicate(goal)](*goal[1:]):
            results.append(apply_substitution(goal, answer))
        return results
