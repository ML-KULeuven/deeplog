"""
A module providing the abstract Engine class.
"""

#  Copyright (c) 2024-2026. KU Leuven

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import field
from typing import TypeVar

from deeplog.algebraic import BOOLEAN
from deeplog.algebraic import get_algebraic_structure
from deeplog.formula.deeplogformulafactory import DeepLogFormulaFactory
from deeplog.symbol import Symbol

from ..program import Program
from ..program.program import is_query


F = TypeVar("F")
L = TypeVar("L")
type Result[F] = "dict[Symbol, F]"
type Builtin = Callable[..., Iterable[dict[Symbol, Symbol]]]


@dataclass
class EngineResult[F]:
    """Result of proving a goal: boolean formulas and atom labels."""

    #: Maps ground goals to boolean formulas (the proof structure).
    formulas: dict[Symbol, F]
    #: Maps ground atom symbols to their annotations (weights/labels).
    labels: dict[Symbol, Symbol] = field(default_factory=dict)
    #: For categorical (annotated-disjunction) leaves only, maps the leaf's
    #: identifying symbol to ``(cat_id, value)`` where ``value`` is either
    #: an int (the AD-branch index) or the string ``"none"`` (the residual
    #: outcome). Downstream MV-SDD compilation uses this to fan a single
    #: multi-valued literal out at each cat_id.
    categoricals: dict[Symbol, tuple[str, int | str]] = field(default_factory=dict)


class Engine(ABC):
    """Abstract DeepProbLog engine that evaluates programs to boolean proofs."""

    @abstractmethod
    def _get_result(
        self, program: Program, goal: Symbol, factory: "EngineFactory"
    ) -> Result[F]:
        """Per engine implementation of get_result"""

    def get_result(
        self,
        program: Program,
        goal: Symbol,
        factory: DeepLogFormulaFactory[F],
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult[F]:
        """Return the result of proving ``goal`` within ``program`` using ``factory``.

        ``evaluator`` is consulted by
        :meth:`~deeplog.systems.deepproblog.engine.engine.EngineFactory.get_scalar_probability`
        for non-numeric labels (e.g. neural-network-backed labels in kbest);
        formula-construction engines that never need a scalar can leave it
        unset.
        """
        engine_factory = EngineFactory(factory, evaluator=evaluator)
        result = self._get_result(program, goal, engine_factory)
        return EngineResult(
            formulas=result,
            labels=engine_factory.labels,
            categoricals=engine_factory.categoricals,
        )

    def get_query_result(
        self,
        program: Program,
        factory: DeepLogFormulaFactory,
        evaluator: Callable[[Symbol], float] | None = None,
    ) -> EngineResult:
        """Evaluate all queries in ``program`` and merge their results."""
        all_formulas: dict = {}
        all_labels: dict[Symbol, Symbol] = {}
        all_categoricals: dict[Symbol, tuple[str, int | str]] = {}
        for query in filter(is_query, program):
            result = self.get_result(program, query[2], factory, evaluator)
            all_formulas.update(result.formulas)
            all_labels.update(result.labels)
            all_categoricals.update(result.categoricals)
        return EngineResult(
            formulas=all_formulas,
            labels=all_labels,
            categoricals=all_categoricals,
        )

    @abstractmethod
    def add_builtin(self, functor: str, arity: int, builtin_function: Builtin) -> None:
        """
        Add a builtin of the given functor and arity to the Engine, implemented by the given function.
        """


class EngineFactory[F]:
    """Helper that builds boolean proof formulas and resolves label probabilities."""

    def __init__(
        self,
        factory: DeepLogFormulaFactory,
        evaluator: Callable[[Symbol], float] | None = None,
    ):
        """Wrap a DeepLogFormulaFactory.

        ``evaluator`` (optional) supplies a probability scalar for labels
        that aren't numeric constants — e.g. neural-network-backed labels.
        Engines that only need formula construction can omit it.
        """
        self._factory = factory
        self._evaluator = evaluator
        self._labels: dict[Symbol, Symbol] = {}
        self._categoricals: dict[Symbol, tuple[str, int | str]] = {}

    @property
    def labels(self) -> dict[Symbol, Symbol]:
        """Return the collected atom-to-label mapping."""
        return self._labels

    @property
    def categoricals(self) -> dict[Symbol, tuple[str, int | str]]:
        """Return the collected leaf-to-(cat_id, value) mapping for AD literals."""
        return self._categoricals

    def get_false(self) -> F:
        """Return the boolean false literal."""
        return self._factory.create_atom(("_", BOOLEAN.zero, ("boolean",)))

    def get_true(self) -> F:
        """Return the boolean true literal."""
        return self._factory.create_atom(("_", BOOLEAN.one, ("boolean",)))

    def get_boolean(self, goal: Symbol, label: Symbol) -> F:
        """Create a free boolean leaf for ``goal`` and record its ``label``."""
        self._labels[goal] = label
        return self._factory.create_atom(("_", goal, ("boolean",)))

    def get_categorical_value(
        self, goal: Symbol, label: Symbol, cat_id: str, value_idx: int
    ) -> F:
        """Create a leaf for AD branch ``(cat_id, value_idx)`` reaching ``goal``.

        Same shape as :meth:`get_boolean` (a free indicator leaf tagged
        ``("boolean",)``); the AD-specific information — which goals
        belong to the same multi-valued variable, and at which value
        each one sits — is carried separately in
        :attr:`categoricals`. Downstream MV-SDD compilation reads the
        tagging off the engine result to fan one multi-valued literal
        at the cat_id's vtree leaf rather than treating each branch as
        an independent boolean.
        """
        self._labels[goal] = label
        self._categoricals[goal] = (cat_id, value_idx)
        return self._factory.create_atom(("_", goal, ("boolean",)))

    def get_categorical_none(self, cat_id: str) -> F:
        """Create a leaf for the residual ("no branch chosen") outcome of an AD.

        Used by NAF over a categorical: the world where the categorical
        RV took none of its declared values. Probability scalar is
        ``1 - Σ P_i`` (computed downstream from the cat_id's branch
        labels). The leaf is a regular boolean indicator; the
        residual-outcome tagging is recorded in :attr:`categoricals`.
        """
        none_symbol = ("@cat_none", (cat_id,))
        self._categoricals[none_symbol] = (cat_id, "none")
        return self._factory.create_atom(("_", none_symbol, ("boolean",)))

    def disjoin(self, lhs: F, rhs: F) -> F:
        """Combine two formulas with boolean disjunction (OR)."""
        return self._factory.create_binary_node("or", lhs, rhs)

    def conjoin(self, lhs: F, rhs: F) -> F:
        """Combine two formulas with boolean conjunction (AND)."""
        return self._factory.create_binary_node("and", lhs, rhs)

    def negate(self, operand: F) -> F:
        """Negate a formula with boolean NOT."""
        return self._factory.create_unary_node("not", operand)

    def get_scalar_probability(self, label: Symbol) -> float:
        """Return a scalar probability for ``label`` (used by kbest's heuristic).

        Numeric-constant labels resolve structurally. Other labels delegate
        to the optional evaluator passed at construction; if no evaluator
        was provided, raises ``ValueError``.
        """
        constant = get_algebraic_structure("probability").get_constant_value(label)
        if constant is not None:
            return float(constant)
        if self._evaluator is None:
            raise ValueError(
                f"Cannot resolve scalar probability for label {label}: not a "
                f"numeric constant and no evaluator was supplied."
            )
        return float(self._evaluator(label))


class UnknownPredicateException(Exception):
    """
    An exception that is raised by an engine when no clauses are known for the given predicate.
    """
