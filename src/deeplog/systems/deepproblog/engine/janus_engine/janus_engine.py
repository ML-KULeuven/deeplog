#  Copyright (c) 2024-2026. KU Leuven
"""
A module providing the Janus-based Prolog engine class.
"""

from collections.abc import Iterable
from pathlib import Path

from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import get_predicate
from deeplog.symbol import parse_symbol
from deeplog.symbol import symbol_to_pretty_string


try:
    from janus_swi import PrologError
    from janus_swi import janus

    JANUS_AVAILABLE = True
except (ModuleNotFoundError, RuntimeError):
    JANUS_AVAILABLE = False

from ...program import Program
from ...program import RuleType
from ...program import expand_annotated_disjunctions
from ...program import remove_labeled_rules
from ...program.program import get_fact_atom
from ...program.program import get_fact_label
from ...program.program import is_constraint
from ...program.program import is_fact
from ...program.program import is_query
from ..engine import Builtin
from ..engine import Engine
from ..engine import EngineFactory
from ..engine import F
from ..engine import Result
from ..engine import UnknownPredicateException


ROOT = Path(__file__).parent


def _is_prolog_bare_token(s: str) -> bool:
    """Return true if ``s`` can be emitted bare (unquoted) into Prolog source.

    Bare-safe: numbers (parsed as int/float), lowercase identifiers
    ``[a-z][a-zA-Z0-9_]*``, and variable-form names (uppercase- or
    underscore-prefixed identifiers) which Prolog should read as
    variables — quoting them would convert them into atoms.

    Operator-prefixed names like ``@cat_id_0`` are *not* bare-safe: the
    SWI tokeniser would otherwise read them as compounds (``@(cat_id_0)``).
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


def _symbol_to_prolog_str(symbol: Symbol) -> str:
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
    args = ",".join(_symbol_to_prolog_str(s) for s in symbol[1:])
    return f"{_quote_prolog_atom(functor)}({args})"


def _try_render_prolog_list(symbol: Symbol) -> str | None:
    """Render a ``cons``/``nil`` chain as ``[h1, h2, ... | tail]``; else None."""
    elements: list[str] = []
    current: Symbol = symbol
    while isinstance(current, tuple) and len(current) == 3 and current[0] == "cons":
        elements.append(_symbol_to_prolog_str(current[1]))
        current = current[2]
    if not elements:
        return None
    if isinstance(current, tuple) and current == ("nil",):
        return f"[{','.join(elements)}]"
    tail = _symbol_to_prolog_str(current)
    return f"[{','.join(elements)}|{tail}]"


class JanusNotAvailableException(Exception):
    """
    This exception is raised when JanusEngine is instantiated, but janus_swi is not available.
    """


class JanusEngine(Engine):
    """
    A Prolog Engine based on a meta-prover implemented with the SWI Prolog Janus.
    """

    program_identifiers: dict = {}
    code_loaded = False
    engine_counter = 0
    _id_prefix = "engine"
    _engine_code_path = ROOT / "engine.pl"

    def __init_subclass__(cls, **kwargs):
        """Give each subclass its own program cache and code-loaded flag.

        Without this, subclasses would inherit (and silently mutate) the
        parent's ``program_identifiers`` dict and ``code_loaded`` flag.
        """
        super().__init_subclass__(**kwargs)
        cls.program_identifiers = {}
        cls.code_loaded = False

    def __init__(self):
        """Initialize a Janus-backed engine and ensure the Prolog code is loaded."""
        super().__init__()
        if not JANUS_AVAILABLE:
            raise JanusNotAvailableException()
        self._ensure_code_loaded()
        self._id = f"{type(self)._id_prefix}_{JanusEngine.engine_counter}"
        JanusEngine.engine_counter += 1
        self._builtins: dict[tuple[str, int], Builtin] = {}

    @classmethod
    def _ensure_code_loaded(cls):
        if not cls.__dict__.get("code_loaded", False):
            janus.consult(str(cls._engine_code_path.absolute()))
            cls.code_loaded = True

    def _get_result(
        self, program: Program, goal: Symbol, factory: EngineFactory
    ) -> Result[F]:
        variables = {
            "ProgramID": self._assert_program(program),
            "Query": goal,
            "Factory": factory,
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
        """Map a SWI ``PrologError`` to a domain exception when recognised.

        ``unknown_procedure`` errors become :class:`UnknownPredicateException`;
        anything else is returned unchanged so the caller's ``raise`` re-raises
        the original Prolog error.
        """
        error_symbol = parse_symbol(repr(err))
        if len(error_symbol) == 3 and error_symbol[0] == "error":
            error_code = error_symbol[1]
            if len(error_code) == 3 and error_code[0] == "unknown_procedure":
                predicate = error_code[1][0], int(error_code[2][0])
                return UnknownPredicateException(f"No clauses known for {predicate}.")
        return err

    def _assert_program(self, program: Program):
        try:
            return self.program_identifiers[program]
        except KeyError:
            identifier = f"program_{len(self.program_identifiers)}"
            program_text = "\n".join(
                sorted(
                    self._rules_to_janus_code(
                        remove_labeled_rules(expand_annotated_disjunctions(program))
                    )
                )
            )
            janus.consult(identifier, data=program_text, module=identifier)
            self.program_identifiers[program] = identifier
            janus.query_once(
                "assertz(ProgramID:engine_id(Engine,ID))",
                {"ProgramID": identifier, "Engine": self, "ID": self._id},
            )

        return identifier

    @staticmethod
    def _rules_to_janus_code(program: Iterable[RuleType]) -> Iterable[str]:
        for rule in program:
            if is_query(rule) or is_constraint(rule):
                continue
            label = get_fact_label(rule)
            if is_fact(rule) and get_fact_label(rule) and label is not None:
                yield f"fact({_symbol_to_prolog_str(get_fact_atom(rule))},{_symbol_to_prolog_str(label)})."
            elif label is None:
                yield f"rule({_symbol_to_prolog_str(rule[1])},{_symbol_to_prolog_str(rule[2])})."
            else:
                raise ValueError(
                    f"{symbol_to_pretty_string(rule)} is not handled by the JanusEngine."
                )

    @staticmethod
    def is_available() -> bool:
        """
        Returns true if janus_swi is available and the class can be instantiated.
        """
        return JANUS_AVAILABLE

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register a builtin in both the Python registry and the Janus Prolog engine."""
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
