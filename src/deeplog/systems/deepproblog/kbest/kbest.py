#  Copyright (c) 2024-2026. KU Leuven
"""K-best DeepProbLog prover: a probability-guided Janus grounder.

Unlike the plain grounders, k-best must know probabilities *while* it searches
(to rank and prune proofs), so it keeps DeepProbLog's labeled program
representation and builds labeled leaves in-line via
:class:`~deeplog.systems.deepproblog.kbest.probabilistic_factory.ProbabilisticFactory`.
It runs its own engine on a :class:`~deeplog.grounding.prolog.JanusProver`,
which loads the engine and the programs, keeps the builtins, and runs the queries.
"""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterable
from functools import lru_cache
from functools import reduce
from pathlib import Path

from deeplog import FormulaNode
from deeplog import Symbol
from deeplog import get_term_variables
from deeplog import symbol_to_pretty_string
from deeplog import to_symbol
from deeplog.grounding.prolog import Builtin
from deeplog.grounding.prolog import JanusGrounder
from deeplog.grounding.prolog import JanusProver
from deeplog.grounding.prolog import Program
from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog import get_constraint_body
from deeplog.grounding.prolog import is_constraint
from deeplog.grounding.prolog import is_fact
from deeplog.grounding.prolog import is_query
from deeplog.grounding.prolog import symbol_to_prolog_str

from ..ad import declare_neural
from ..ad import get_ad_branches
from ..ad import is_annotated_disjunction
from ..parser import create_labeled_fact
from ..parser import get_atom
from ..parser import get_fact_atom
from ..parser import get_fact_label
from ..parser import get_label
from ..solver import EngineResult
from ..solver import query_answers
from ..transformation import remove_labeled_rules
from .probabilistic_factory import ProbabilisticFactory


ROOT = Path(__file__).parent

#: Functor ``engine.pl`` matches on to recognize an annotated-disjunction branch
#: *during* search, which is what lets it hoist negation over a variable it has
#: not yet decided. The base grounder needs no such tag -- it is semantics-free,
#: so the disjunction is recognized after grounding
#: (:func:`~deeplog.systems.deepproblog.ad.instantiate`) rather than
#: carried through it.
CATEGORICAL_LABEL_FUNCTOR = "@cat"


def create_categorical_label(
    inner_label: Symbol | str, cat_id: Symbol, value_idx: int
) -> Symbol:
    """Wrap ``inner_label`` so ``engine.pl`` recognizes a branch of an AD.

    The resulting label has the form
    ``("@cat", inner_label, cat_id, ("<value_idx>",))``, which round-trips into
    the engine's Prolog as ``'@cat'(InnerLabel, CatId, <value_idx>)``.
    """
    return (
        CATEGORICAL_LABEL_FUNCTOR,
        to_symbol(inner_label),
        cat_id,
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

    A disjunction chooses one branch for each ground instance of the variables
    its branches share, so the cat-id of the n-th disjunction is
    ``@cat_id_<n>`` applied to those variables: ``nn(m, [X], Y, [0..9]) ::
    digit(X,Y).`` tags its branches with ``'@cat_id_0'(X)``, and
    ``digit(i1,_)`` and ``digit(i2,_)`` are two choices. The ``@cat_id_`` prefix
    avoids collisions with user-defined predicates. Every other rule passes
    through unchanged.
    """
    cat_id = 0
    for rule in program:
        branches = _tagged_branches(rule)
        if branches is None:
            yield rule
            continue
        shared = _shared_variables([atom for atom, _ in branches])
        instance: Symbol = (f"@cat_id_{cat_id}", *shared)
        for value_idx, (atom, inner_label) in enumerate(branches):
            yield create_labeled_fact(
                atom, create_categorical_label(inner_label, instance, value_idx)
            )
        cat_id += 1


def _shared_variables(atoms: list[Symbol]) -> list[Symbol]:
    """The variables every one of ``atoms`` holds, in the order they first occur."""
    variables = dict.fromkeys(get_term_variables(atoms[0]))
    for atom in atoms[1:]:
        held = set(get_term_variables(atom))
        variables = {variable: None for variable in variables if variable in held}
    return list(variables)


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

    Returns up to ``k`` highest-probability proof formulas for each answer to a
    goal, folded into a single disjunction. The search drops a partial proof once
    its answer has ``k`` proofs and ends when none is left, so for a goal with
    variables it expands every partial proof whose answer is still open. Ranking
    consumes scalar probabilities via
    :meth:`~deeplog.systems.deepproblog.kbest.probabilistic_factory.ProbabilisticFactory.get_scalar_probability`;
    for neural-network-backed labels, callers pass an ``evaluator`` that maps a
    label symbol to its probability scalar.

    The search keeps every partial proof it has yet to expand, and raises
    :class:`MemoryError` when they outgrow Prolog's stack. Partial proofs of about
    the same probability, as an untrained network gives them, leave it little to
    choose by, and it keeps most of them. It does not table its calls, so a goal
    that calls itself before its proof holds a new fact, as
    ``path(X,Y) :- path(X,Z), edge(Z,Y).`` does, is expanded without end;
    :class:`~deeplog.grounding.prolog.JanusGrounder` tables its calls and proves
    such a program.
    """

    _HEURISTICS = {"pp", "gm"}

    def __init__(self, k: int, heuristic: str = "pp"):
        """Build a k-best grounder.

        Args:
            k: Maximum number of proofs to enumerate per answer (>= 1).
            heuristic: ``"pp"`` (partial probability) or ``"gm"`` (geometric mean
                of -log probabilities).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if heuristic not in self._HEURISTICS:
            raise ValueError(f"heuristic must be one of {sorted(self._HEURISTICS)}")
        super().__init__()
        self._kbest = JanusProver(ROOT / "engine.pl")
        self._k = k
        self._heuristic = heuristic

    def ground(self, program, goal, open_predicates=frozenset()):
        """Not supported: use :meth:`get_query_result` / :meth:`get_result`.

        K-best builds labeled leaves during search rather than emitting a
        semantics-free proof structure, so it does not implement the plain
        grounder contract.
        """
        raise NotImplementedError(
            "KBestJanusGrounder is probability-guided; use get_query_result / "
            "get_result rather than the plain ground() contract."
        )

    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """Register a builtin for the goals this grounder proves."""
        self._kbest.add_builtin(functor, arity, builtin_function)

    def get_result(
        self,
        program: Program,
        goal: Symbol,
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult:
        """Prove a single ``goal`` with k-best search."""
        program_id = self._kbest.consult(_clauses(program))
        prob_factory = ProbabilisticFactory(evaluator)
        formulas = query_answers(
            goal, self._kbest_ground(program_id, goal, prob_factory)
        )
        return EngineResult(formulas, prob_factory.labels, prob_factory.variables)

    def get_query_result(
        self,
        program: Program,
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult:
        """Evaluate every query with k-best search, conditioned on constraints."""
        program_id = self._kbest.consult(_clauses(program))
        prob_factory = ProbabilisticFactory(evaluator)
        evidence = self._build_evidence(program, program_id, prob_factory)
        all_formulas: dict[Symbol, FormulaNode] = {}
        for query in filter(is_query, program):
            proofs = self._kbest_ground(program_id, query[2], prob_factory)
            for answer, formula in query_answers(query[2], proofs).items():
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
        rows = self._kbest.query(
            "kbest_prove_query(Program, Prover, Query, Factory, K, Heuristic,"
            " GroundQuery, Formula)",
            {
                "Program": program_id,
                "Prover": self._kbest,
                "Query": goal,
                "Factory": factory,
                "K": self._k,
                "Heuristic": self._heuristic,
            },
        )
        per_goal: dict = defaultdict(list)
        try:
            for row in rows:
                per_goal[row["GroundQuery"]].append(row["Formula"])
        except MemoryError as err:
            raise MemoryError(
                f"{err}, in the k-best search for {symbol_to_pretty_string(goal)}. "
                "It keeps every partial proof it has yet to "
                "expand, so partial proofs of about the same probability, as an "
                "untrained network gives them, can outgrow Prolog's stack. It "
                "does not table its calls, so a goal that calls itself before its "
                "proof holds a new fact, as the left-recursive "
                "path(X,Y) :- path(X,Z), edge(Z,Y). does, is expanded without "
                "end. JanusGrounder tables its calls and proves such a program."
            ) from err
        return {
            ground: reduce(factory.disjoin, formulas)
            for ground, formulas in per_goal.items()
        }


@lru_cache(maxsize=64)
def _clauses(program: Program) -> tuple[str, ...]:
    """``program`` as the *labeled* Prolog clauses k-best's engine reads.

    Unlike the base grounder, which turns the facts of open predicates into
    unlabeled leaves, k-best keeps the ``::`` labels as ``fact/2``, so its engine
    can rank partial proofs by probability. The renderings of the 64 most recent
    programs are cached.
    """
    clauses = []
    for rule in remove_labeled_rules(expand_annotated_disjunctions(program)):
        if is_query(rule) or is_constraint(rule):
            continue
        label = get_fact_label(rule)
        if is_fact(rule) and label is not None:
            clauses.append(
                f"fact({symbol_to_prolog_str(get_fact_atom(rule))},"
                f"{symbol_to_prolog_str(label)})."
            )
        elif get_label(rule[1]) is None:
            clauses.append(
                f"rule({symbol_to_prolog_str(rule[1])},"
                f"{symbol_to_prolog_str(rule[2])})."
            )
        else:
            raise ValueError(f"{rule} is not handled by the KBestJanusGrounder.")
    return (":- dynamic fact/2, rule/2.", *sorted(clauses))
