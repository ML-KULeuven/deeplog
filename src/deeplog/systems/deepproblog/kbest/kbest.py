#  Copyright (c) 2024-2026. KU Leuven
"""K-best DeepProbLog prover: a probability-guided Janus grounder.

Unlike the plain grounders, k-best must know probabilities *while* it searches
(to rank and prune proofs), so it keeps DeepProbLog's labeled program
representation and builds labeled leaves in-line via
:class:`~deeplog.systems.deepproblog.kbest.probabilistic_factory.ProbabilisticFactory`.
It reuses :class:`~deeplog.grounding.prolog.JanusGrounder` for the Janus session
plumbing (code loading, program caching, builtins, error translation), overriding
only ``_rules_to_janus_code`` to render the labeled program its engine needs.
"""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from functools import reduce
from pathlib import Path

from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_query
from deeplog.grounding.prolog.janus import JANUS_AVAILABLE
from deeplog.grounding.prolog.janus import JanusGrounder
from deeplog.grounding.prolog.janus import JanusNotAvailableException
from deeplog.grounding.prolog.janus.janus import symbol_to_prolog_str
from deeplog.grounding.prolog.program import is_fact
from deeplog.symbol import Symbol
from deeplog.symbol import to_symbol

from ..ad import declare_neural
from ..ad import get_ad_branches
from ..ad import is_annotated_disjunction
from ..parser import create_labeled_fact
from ..parser import get_atom
from ..parser import get_fact_atom
from ..parser import get_fact_label
from ..parser import get_label
from ..solver import EngineResult
from ..transformation import remove_labeled_rules
from .probabilistic_factory import ProbabilisticFactory


try:
    from janus_swi import PrologError
    from janus_swi import janus
except (ModuleNotFoundError, RuntimeError):
    # JanusGrounder.__init__ raises JanusNotAvailableException long before any
    # code path that needs ``janus``/``PrologError`` runs, so the names being
    # unbound here is safe.
    pass


ROOT = Path(__file__).parent

#: Functor ``engine.pl`` matches on to recognize an annotated-disjunction branch
#: *during* search, which is what lets it hoist negation over a variable it has
#: not yet decided. The base grounder needs no such tag -- it is semantics-free,
#: so the disjunction is recognized after grounding
#: (:func:`~deeplog.systems.deepproblog.ad.instantiate`) rather than
#: carried through it.
CATEGORICAL_LABEL_FUNCTOR = "@cat"


def create_categorical_label(
    inner_label: Symbol | str, cat_id: int, value_idx: int
) -> Symbol:
    """Wrap ``inner_label`` so ``engine.pl`` recognizes a branch of an AD.

    The resulting label has the form
    ``("@cat", inner_label, ("@cat_id_<n>",), ("<value_idx>",))``, which
    round-trips into the engine's Prolog as
    ``'@cat'(InnerLabel, '@cat_id_<n>', <value_idx>)``. The ``@cat_id_`` prefix
    on the cat-id avoids collisions with user-defined predicates there.
    """
    return (
        CATEGORICAL_LABEL_FUNCTOR,
        to_symbol(inner_label),
        (f"@cat_id_{cat_id}",),
        (str(value_idx),),
    )


def expand_annotated_disjunctions(program: Iterable[RuleType]) -> Iterable[RuleType]:
    """Split each annotated disjunction into n facts, tagging each for the engine.

    This is the k-best rendering of an annotated disjunction, in either spelling
    -- the enumerated ``p1::a1; ...; pn::an.`` and the neural
    ``nn(net,[inputs],Var,[domain]) :: atom.`` both become the same n tagged
    facts. The base grounder takes
    :func:`~deeplog.systems.deepproblog.ad.split_annotated_disjunctions`
    instead, which splits without tagging and keeps what the disjunction
    declares, so the variable is recovered after grounding rather than during it.

    Cat-ids are assigned per-call, monotonically. Every other rule passes through
    unchanged.
    """
    cat_id = 0
    for rule in program:
        branches = _tagged_branches(rule)
        if branches is None:
            yield rule
            continue
        for value_idx, (atom, inner_label) in enumerate(branches):
            yield create_labeled_fact(
                atom, create_categorical_label(inner_label, cat_id, value_idx)
            )
        cat_id += 1


def _tagged_branches(rule: RuleType) -> list[tuple[Symbol, Symbol]] | None:
    """The ``(atom, label)`` branches of ``rule``, or ``None`` if it declares none."""
    if is_annotated_disjunction(rule):
        branches = []
        for branch in get_ad_branches(rule):
            label = get_label(branch)
            assert label is not None
            branches.append((get_atom(branch), label))
        return branches
    if not is_fact(rule):
        return None
    label = get_label(rule[1])
    if label is None:
        return None
    neural = declare_neural(get_atom(rule[1]), label)
    return None if neural is None else list(neural[0])


class KBestJanusGrounder(JanusGrounder):
    """Heap-based k-best DeepProbLog prover.

    Returns up to ``k`` highest-probability proof formulas per ground goal, folded
    into a single disjunction. Ranking consumes scalar probabilities via
    :meth:`~deeplog.systems.deepproblog.kbest.probabilistic_factory.ProbabilisticFactory.get_scalar_probability`;
    for neural-network-backed labels, callers pass an ``evaluator`` that maps a
    label symbol to its probability scalar.
    """

    _id_prefix = "kbest_engine"
    _engine_code_path = ROOT / "engine.pl"

    _HEURISTICS = {"pp", "gm"}

    def __init__(self, k: int, heuristic: str = "pp"):
        """Build a k-best grounder.

        Args:
            k: Maximum number of proofs to enumerate per goal (>= 1).
            heuristic: ``"pp"`` (partial probability) or ``"gm"`` (geometric mean
                of -log probabilities).
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

    def ground(self, program, goal, factory, open_predicates=frozenset()):
        """Not supported: use :meth:`get_query_result` / :meth:`get_result`.

        K-best builds labeled leaves during search rather than emitting a
        semantics-free proof structure, so it does not implement the plain
        grounder contract.
        """
        raise NotImplementedError(
            "KBestJanusGrounder is probability-guided; use get_query_result / "
            "get_result rather than the plain ground() contract."
        )

    def get_result(
        self,
        program,
        goal: Symbol,
        factory,
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult:
        """Prove a single ``goal`` with k-best search."""
        program_id = self._assert_program(program)
        prob_factory = ProbabilisticFactory(factory, evaluator)
        formulas = self._kbest_ground(program_id, goal, prob_factory)
        return EngineResult(formulas, prob_factory.labels, prob_factory.variables)

    def get_query_result(
        self,
        program,
        factory,
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult:
        """Evaluate every query with k-best search, conditioned on constraints."""
        program_id = self._assert_program(program)
        prob_factory = ProbabilisticFactory(factory, evaluator)
        evidence = self._build_evidence(program, program_id, prob_factory)
        all_formulas: dict[Symbol, object] = {}
        for query in filter(is_query, program):
            for answer, formula in self._kbest_ground(
                program_id, query[2], prob_factory
            ).items():
                all_formulas[answer] = (
                    formula
                    if evidence is None
                    else prob_factory.conjoin(formula, evidence)
                )
        return EngineResult(
            all_formulas, prob_factory.labels, prob_factory.variables, evidence
        )

    def _build_evidence(self, program, program_id, factory: ProbabilisticFactory):
        constraints = list(filter(is_constraint, program))
        if not constraints:
            return None
        evidence = factory.get_true()
        for constraint in constraints:
            body = get_constraint_body(constraint)
            proofs = self._kbest_ground(program_id, body, factory)
            body_formula = reduce(factory.disjoin, proofs.values(), factory.get_false())
            evidence = factory.conjoin(evidence, factory.negate(body_formula))
        return evidence

    def _kbest_ground(self, program_id, goal, factory: ProbabilisticFactory):
        self._bind_engine(program_id)
        variables = {
            "ProgramID": program_id,
            "Query": goal,
            "Factory": factory,
            "K": self._k,
            "Heuristic": self._heuristic,
        }
        try:
            rows = janus.query(
                "Engine:kbest_prove_query(ProgramID,Query,Factory,K,Heuristic,"
                "GroundQuery,Formula)",
                {**variables, "Engine": self._engine_module},
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

    def _rules_to_janus_code(self, program, open_predicates=frozenset()):
        """Render the *labeled* program k-best's engine.pl expects.

        The base grounder emits ``leaf/1`` for the facts of its open predicates
        and knows nothing of labels; k-best instead keeps the ``::`` labels, so
        its engine can rank partial proofs by probability, and has no use for
        ``open_predicates``. Only the rendering differs — the caching is
        inherited from
        :meth:`~deeplog.grounding.prolog.JanusGrounder._assert_program`.
        """
        program = remove_labeled_rules(expand_annotated_disjunctions(program))
        yield ":- dynamic fact/2, rule/2, engine_id/2."
        for rule in program:
            if is_query(rule) or is_constraint(rule):
                continue
            label = get_fact_label(rule)
            if is_fact(rule) and label is not None:
                yield (
                    f"fact({symbol_to_prolog_str(get_fact_atom(rule))},"
                    f"{symbol_to_prolog_str(label)})."
                )
            elif get_label(rule[1]) is None:
                yield (
                    f"rule({symbol_to_prolog_str(rule[1])},"
                    f"{symbol_to_prolog_str(rule[2])})."
                )
            else:
                raise ValueError(f"{rule} is not handled by the KBestJanusGrounder.")
