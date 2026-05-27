#  Copyright (c) 2024-2026. KU Leuven
"""K-best variant of :class:`JanusEngine` using a heap-based best-first prover."""

from collections import defaultdict
from functools import reduce
from pathlib import Path


try:
    from janus_swi import PrologError
    from janus_swi import janus
except (ModuleNotFoundError, RuntimeError):
    # JanusEngine.__init__ raises JanusNotAvailableException long before any
    # code path that needs ``janus``/``PrologError`` runs, so the names being
    # unbound here is safe.
    pass

from ...program import Program
from ..engine import EngineFactory
from ..engine import F
from ..engine import Result
from ..janus_engine import JANUS_AVAILABLE
from ..janus_engine import JanusEngine
from ..janus_engine import JanusNotAvailableException


ROOT = Path(__file__).parent


class KBestJanusEngine(JanusEngine):
    """Heap-based k-best variant of :class:`JanusEngine`.

    Returns up to ``k`` highest-probability proof formulas per ground goal,
    folded into a single disjunction. Builds proofs as boolean formulas
    via the wrapped factory just like :class:`JanusEngine` does — ranking
    only consumes scalar probabilities (via
    :meth:`~deeplog.systems.deepproblog.engine.engine.EngineFactory.get_scalar_probability`).
    For neural-network-backed
    labels, callers pass an ``evaluator`` to
    :meth:`get_query_result`/:meth:`get_result` that maps the label symbol
    to its probability scalar.

    Reuses ``JanusEngine``'s program assertion, rule translation, and
    builtin registration; only the prover entry point and error-mapping
    differ.
    """

    _id_prefix = "kbest_engine"
    _engine_code_path = ROOT / "engine.pl"

    _HEURISTICS = {"pp", "gm"}

    def __init__(self, k: int, heuristic: str = "pp"):
        """Build a k-best engine.

        Args:
            k: Maximum number of proofs to enumerate per goal (>= 1).
            heuristic: ``"pp"`` (partial probability) or ``"gm"``
                (geometric mean of -log probabilities).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if heuristic not in self._HEURISTICS:
            raise ValueError(f"heuristic must be one of {sorted(self._HEURISTICS)}")
        if not JANUS_AVAILABLE:
            raise JanusNotAvailableException()
        super().__init__()
        self._k = k
        self._heuristic = heuristic

    def _get_result(self, program: Program, goal, factory: EngineFactory) -> Result[F]:
        variables = {
            "ProgramID": self._assert_program(program),
            "Query": goal,
            "Factory": factory,
            "K": self._k,
            "Heuristic": self._heuristic,
        }
        try:
            rows = janus.query(
                "kbest_prove_query(ProgramID,Query,Factory,K,Heuristic,"
                "GroundQuery,Formula)",
                variables,
            )
            per_goal: dict = defaultdict(list)
            for row in rows:
                per_goal[row["GroundQuery"]].append(row["Formula"])
            return {
                ground: reduce(factory.disjoin, formulas)
                for ground, formulas in per_goal.items()
            }
        except PrologError as err:
            raise self._translate_prolog_error(err) from err
