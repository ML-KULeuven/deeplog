#  Copyright (c) 2024-2026. KU Leuven
"""The builders that compute an ``expectation``.

``expectation(X; P): φ`` is the expectation of the boolean body ``φ`` over the
binders ``X``, distributed as ``P``. ``P`` is a product, in its algebra, of
factors. The binders one factor mentions are one variable, and so are binders
factors join; the product of a variable's factors weighs each combination of
its binders' values. Where the algebra has a complement, a variable has one more
outcome, *missing*: no named value, weighing the complement of the named
values' total, where every leaf over its binders is false. The named values'
weights therefore need not sum to one; the mass they leave is the missing
outcome's, as ProbLog reads an annotated disjunction whose probabilities sum
below one. They must not sum above one. A factor that mentions no binder scales
the expectation, and every binder needs a factor. Without ``P``, the binders are
independent variables in probability, each naming the values ``φ`` tests it for:
the value ``v`` of ``V`` weighs as the leaf ``=(V, v)`` in probability, an input
of the compiled module, and the mass the named values leave is ``V``'s missing
outcome. Every leaf over a binder must be such a test.

:func:`weighted_model_count` computes it by knowledge compilation in probability
and log-probability, and :func:`enumeration` by evaluating every assignment.
They give the same number wherever both apply: counting enumerates what it
cannot count, and every expectation in another algebra. :func:`sampling`
returns a builder that estimates it from drawn assignments instead.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from functools import reduce
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ...algebraic import BOOLEAN
from ...algebraic import LOGPROBABILITY
from ...algebraic import PROBABILITY
from ...algebraic import Algebra
from ...algebraic import AlgebraicStructure
from ...algebraic import Semiring
from ...circuit.circuit import Circuit
from ...circuit.knowledge_compilation.dispatch import compiler_installed
from ...circuit.knowledge_compilation.dispatch import knowledge_compile
from ...module.aggregation_modules import ExpectationModule
from ...module.aggregation_modules import WeightedVariable
from ...module.deeplog_module import DeepLogModule
from ...symbol import Symbol
from ...symbol import apply_substitution
from ...symbol import get_args
from ...symbol import get_term_variables
from ...symbol import is_variable
from ...symbol import symbol_to_pretty_string
from ...symbol import unwrap_structure
from ...symbol import with_structure
from ...symbol import without_structure
from ...variable import OPEN
from ...variable import Domain
from ...variable import SymbolicDomain
from ...variable import Variable
from ...variable import VariableAtoms
from ..ast import Aggregation
from ..ast import Atom
from ..ast import BinaryOp
from ..ast import CircuitNode
from ..ast import FormulaNode
from ..ast import Transformation
from ..ast import UnaryOp
from ..ast import fold
from ..ast import map_children
from ..ast import operands
from ..circuit_factory import CircuitFactory
from ..circuit_node import transform_nodes
from .builder_protocols import AggregationBuilder


if TYPE_CHECKING:
    from .lowering import Lowering


def weighted_model_count(
    nodes: Sequence[Aggregation], lowering: Lowering
) -> list[tuple[DeepLogModule, int]]:
    """Exact: knowledge-compile each group's bodies into their distribution's algebra.

    An expectation is counted when its body is a boolean circuit lump, every leaf
    of which tests one of its binders for a named value of the binder's domain,
    and it either has no distribution or one, in probability or log-probability,
    that is a product of atoms the compiler has builders for, one for each binder
    and mentioning only that binder, as an argument. The leaf testing ``V`` for
    ``v`` then weighs as ``V``'s factor with ``v`` substituted, which its
    predicate reads as ``V`` holding ``v``, since the argument's sort is ``V``'s
    domain; or, without a distribution, as itself. Expectations over one circuit,
    counted into one algebra, whose factors agree on the binders they share, are
    one group, and each group takes one knowledge compilation. A group reaching
    two values of one binder compiles with mv-sdd, which ``pydeeplog[mvsdd]``
    installs. Every other expectation, and every one in a group whose knowledge
    compiler is not installed, is enumerated (:func:`enumeration`).
    """
    columns: dict[int, tuple[DeepLogModule, int]] = {}
    for group in _groups(nodes, lowering):
        if compiler_installed(group.circuit, group.roots, group.variables):
            columns.update(_count(group, lowering))
    rest = [node for node in nodes if id(node) not in columns]
    if rest:
        columns.update(zip(map(id, rest), enumeration(rest, lowering), strict=True))
    return [columns[id(node)] for node in nodes]


def enumeration(
    nodes: Sequence[Aggregation], lowering: Lowering
) -> list[tuple[DeepLogModule, int]]:
    """Exact, by evaluating the body at every combination of its variables' outcomes.

    Each variable's factor is evaluated at every value of its binders, and the
    body at every combination, missing outcomes included.

    Raises:
        ValueError: If an expectation's body is not boolean, it has more than one
            param, its distribution's algebra has no sum and product, one of its
            binders has no factor, a leaf that is not boolean reads a binder
            that may have no value, or, without a distribution, a leaf reads a
            binder other than by testing it for a value.
    """
    columns = []
    for node in nodes:
        body, algebra, variables = _distribution(node, lowering)
        if not variables:
            columns.append((lowering.lower(Transformation(algebra.name, body)), 0))
        else:
            columns += _expect([(node, body, algebra, variables)], lowering)
    return columns


def score_function(values: Tensor, log_probability: Tensor) -> Tensor:
    """The score-function estimator, with the other draws' mean as its baseline.

    ``values`` are what each expectation's body takes at each drawn assignment
    and ``log_probability`` the log-probability of drawing it, both of shape
    ``(batch, samples, expectations)``. The result equals ``values``, and its
    gradient adds each draw's log-probability gradient, weighted by how far its
    value lies from the mean of the other draws. That mean does not depend on the
    draw, so the estimate stays unbiased; a single draw has no baseline.
    """
    samples = values.shape[1]
    if samples > 1:
        baseline = (values.sum(dim=1, keepdim=True) - values) / (samples - 1)
    else:
        baseline = torch.zeros_like(values)
    score = log_probability - log_probability.detach()
    return values + (values - baseline).detach() * score


def sampling(
    samples: int,
    *,
    gradient: Callable[[Tensor, Tensor], Tensor] = score_function,
) -> AggregationBuilder:
    """The builder estimating each expectation from ``samples`` drawn assignments.

    An expectation whose distribution is in probability, as one without a
    distribution is, is sampled. Each variable's outcome, missing included, is
    drawn with probability its weight divided by the total; the estimate is the
    mean of the body over the draws, times those totals. Expectations whose
    variables agree where they share binders read the same draws, so the ratio of
    two is estimated on one set of assignments. Every other expectation is
    enumerated.

    ``gradient`` receives the values of the bodies at the draws and the
    log-probability of drawing them, both of shape ``(batch, samples,
    expectations)``, and returns values equal to the first whose gradient
    estimates the expectations'.

    Raises:
        ValueError: If ``samples`` is not positive.
    """
    if samples < 1:
        raise ValueError(f"Sampling draws at least one assignment, not {samples}.")

    def build(
        nodes: Sequence[Aggregation], lowering: Lowering
    ) -> list[tuple[DeepLogModule, int]]:
        columns: dict[int, tuple[DeepLogModule, int]] = {}
        groups: list[list[_Member]] = []
        for node in nodes:
            member = (node, *_distribution(node, lowering))
            if member[2] is not PROBABILITY or not member[3]:
                continue
            group = next((g for g in groups if _agree(g, member)), None)
            if group is None:
                groups.append([member])
            else:
                group.append(member)
        for group in groups:
            answered = _expect(group, lowering, samples, gradient)
            columns.update(zip((id(m[0]) for m in group), answered, strict=True))
        rest = [node for node in nodes if id(node) not in columns]
        if rest:
            columns.update(zip(map(id, rest), enumeration(rest, lowering), strict=True))
        return [columns[id(node)] for node in nodes]

    return build


# -- Counting -----------------------------------------------------------------


# The algebras in which an expectation is counted, by identity.
_COUNTED: tuple[Algebra, ...] = (PROBABILITY, LOGPROBABILITY)


@dataclass
class _Group:
    """Expectations counted together: one circuit, one algebra, agreeing factors.

    ``factors`` maps each binder to its factor atom, bare; it is ``None`` for
    expectations without a distribution. ``domains`` maps each binder to its
    named values, and ``members`` pairs each expectation with its body.
    """

    circuit: Circuit
    algebra: Semiring
    factors: dict[Symbol, Symbol] | None
    domains: dict[Symbol, Domain] = field(default_factory=dict)
    members: list[tuple[Aggregation, CircuitNode]] = field(default_factory=list)

    def admits(self, reading: _Reading) -> bool:
        """Whether ``reading`` counts with this group's circuit, algebra and factors."""
        if reading.body.circuit is not self.circuit:
            return False
        if reading.algebra is not self.algebra:
            return False
        if self.factors is None or reading.factors is None:
            if self.factors is not None or reading.factors is not None:
                return False
        elif any(
            self.factors.get(binder, factor) != factor
            for binder, factor in reading.factors.items()
        ):
            return False
        return all(
            self.domains.get(binder, domain) == domain
            for binder, domain in reading.domains.items()
        )

    @property
    def roots(self) -> list[int]:
        """The members' bodies, each node once."""
        return list(dict.fromkeys(body.node for _, body in self.members))

    @property
    def variables(self) -> VariableAtoms:
        """Each binder over its named values and the missing one, asserted by ``=``."""
        return {
            Variable(binder, Domain.of([*domain.values, _MISSING])): (
                ("=", binder, OPEN),
            )
            for binder, domain in self.domains.items()
        }


@dataclass(frozen=True)
class _Reading:
    """What counting reads off one expectation."""

    body: CircuitNode
    algebra: Semiring
    factors: dict[Symbol, Symbol] | None
    domains: dict[Symbol, Domain]


def _groups(nodes: Sequence[Aggregation], lowering: Lowering) -> list[_Group]:
    """The countable ``nodes``, in groups that each take one knowledge compilation."""
    groups: list[_Group] = []
    tests = _Tests(lowering, [node.child for node in nodes])
    for node in nodes:
        reading = _read(node, lowering, tests)
        if reading is None:
            continue
        group = next((group for group in groups if group.admits(reading)), None)
        if group is None:
            group = _Group(reading.body.circuit, reading.algebra, reading.factors)
            groups.append(group)
        if group.factors is not None and reading.factors is not None:
            group.factors.update(reading.factors)
        group.domains.update(reading.domains)
        group.members.append((node, reading.body))
    return groups


def _read(node: Aggregation, lowering: Lowering, tests: _Tests) -> _Reading | None:
    """How to count ``node``, or ``None`` if it is not countable."""
    body = node.child
    if not isinstance(body, CircuitNode) or body.feeders:
        return None
    if body.structure != BOOLEAN.name or len(node.params) > 1:
        return None
    if not node.params:
        tested = tests.of(body)
        if tested is None or not tested <= set(node.binders):
            return None
        return _Reading(body, PROBABILITY, None, _named(body, lowering))
    (distribution,) = node.params
    if not isinstance(distribution, CircuitNode) or distribution.feeders:
        return None
    algebra = distribution.circuit.structure
    if not isinstance(algebra, Algebra) or algebra not in _COUNTED:
        return None
    factors = _factors(distribution, node.binders, lowering)
    if factors is None or set(factors) != set(node.binders):
        return None
    domains = {binder: lowering.compiler.domain(binder) for binder in node.binders}
    if not all(isinstance(domain, SymbolicDomain) for domain in domains.values()):
        return None
    tested = tests.of(body)
    if tested is None or not tested <= domains.keys():
        return None
    return _Reading(body, algebra, dict(factors), domains)


def _direct(atom: Symbol, binder: Symbol) -> bool:
    """Whether every occurrence of ``binder`` in ``atom`` is an argument of it.

    Only such an occurrence has a sort, so only there may a value be written.
    """
    arguments = get_args(atom)
    occurrences = sum(
        variable == binder
        for argument in arguments
        for variable in get_term_variables(argument)
    )
    return 0 < sum(argument == binder for argument in arguments) == occurrences


def _named(body: CircuitNode, lowering: Lowering) -> dict[Symbol, Domain]:
    """Each variable ``body`` tests, with the values it tests it for, in domain order.

    Every leaf of ``body`` must be a test ``=(V, v)``.
    """
    tested: dict[Symbol, set[Symbol]] = {}
    for leaf in body.circuit.reachable_leaf_names([body.node]):
        _, binder, value = without_structure(leaf)
        tested.setdefault(binder, set()).add(value)
    return {
        binder: Domain.of(
            [v for v in lowering.compiler.domain(binder).values if v in values]
        )
        for binder, values in tested.items()
    }


def _factors(
    distribution: CircuitNode, binders: Sequence[Symbol], lowering: Lowering
) -> dict[Symbol, Symbol] | None:
    """Each binder's factor, if ``distribution`` is a product of one per binder.

    A factor is an atom the compiler has a builder for, mentioning exactly one
    of ``binders``. Returns ``None`` if ``distribution`` is not such a product.
    """
    circuit = distribution.circuit
    algebra = circuit.structure
    if not isinstance(algebra, Semiring):
        return None
    factors: dict[Symbol, Symbol] = {}
    stack = [distribution.node]
    while stack:
        node = stack.pop()
        operation, operands = circuit.operation(node)
        if operation == algebra.product:
            stack.extend(operands)
            continue
        atom = circuit.get_leaf_name(node)
        if atom is None:
            return None
        mentioned = _over(atom, binders)
        builder = lowering.compiler.atom_builder(atom[0], len(atom) - 1, algebra.name)
        if len(mentioned) != 1 or mentioned[0] in factors or builder is None:
            return None
        if not _direct(atom, mentioned[0]):
            return None
        factors[mentioned[0]] = atom
    return factors


def _over(symbol: Symbol, binders: Sequence[Symbol]) -> list[Symbol]:
    """The ``binders`` that occur in ``symbol``, in their order.

    Walks ``symbol`` once, whatever the number of binders.
    """
    found: set[Symbol] = set()
    stack = [symbol]
    while stack:
        term = stack.pop()
        found.add(term)
        stack.extend(part for part in term[1:] if isinstance(part, tuple))
    if len(found) < len(binders):
        return [binder for binder in binders if binder in found]
    order = {binder: i for i, binder in enumerate(binders)}
    return sorted((term for term in found if term in order), key=order.__getitem__)


class _Tests:
    """The variables each node of the bodies' circuits tests.

    A node tests the variables its leaves do; it is ``None`` when one of its
    leaves is not ``=(V, v)`` for a variable ``V`` and a named value ``v`` of its
    domain. Each circuit is walked once, however many bodies it holds.
    """

    def __init__(self, lowering: Lowering, bodies: Iterable[FormulaNode]) -> None:
        """Walk every circuit the lumps among ``bodies`` belong to."""
        self._lowering = lowering
        self._domains: dict[Symbol, Domain | None] = {}
        roots: dict[int, tuple[Circuit, list[int]]] = {}
        for body in bodies:
            if isinstance(body, CircuitNode):
                roots.setdefault(id(body.circuit), (body.circuit, []))[1].append(
                    body.node
                )
        self._tested = {
            key: self._walk(circuit, nodes) for key, (circuit, nodes) in roots.items()
        }

    def of(self, body: CircuitNode) -> frozenset[Symbol] | None:
        """The variables ``body`` tests, or ``None`` if a leaf of it is not a test."""
        return self._tested[id(body.circuit)][body.node]

    def _walk(
        self, circuit: Circuit, roots: list[int]
    ) -> dict[int, frozenset[Symbol] | None]:
        """What each node under ``roots`` tests, children before parents."""
        tested: dict[int, frozenset[Symbol] | None] = {}
        for node in circuit._iter_topological(roots):
            leaf = circuit.get_leaf_name(node)
            if leaf is not None:
                tested[node] = self._leaf(leaf)
                continue
            found: frozenset[Symbol] | None = frozenset()
            for operand in circuit.operation(node)[1]:
                below = tested[operand]
                if below is None:
                    found = None
                    break
                if not below <= found:
                    found = found | below
            tested[node] = found
        return tested

    def _leaf(self, leaf: Symbol) -> frozenset[Symbol] | None:
        """The variable ``leaf`` tests, or ``None`` if it is no test of a named value."""
        if len(leaf) != 3 or leaf[0] != "=" or not is_variable(leaf[1]):
            return None
        if leaf[1] not in self._domains:
            try:
                self._domains[leaf[1]] = self._lowering.compiler.domain(leaf[1])
            except ValueError:
                self._domains[leaf[1]] = None
        domain = self._domains[leaf[1]]
        if not isinstance(domain, SymbolicDomain) or leaf[2] not in domain.names:
            return None
        return frozenset((leaf[1],))


def _count(group: _Group, lowering: Lowering) -> dict[int, tuple[DeepLogModule, int]]:
    """The module counting every member of ``group``, and each member's column."""
    roots = group.roots
    algebra = group.algebra
    assert isinstance(algebra, Algebra)
    weight = partial(_weight, group.factors, algebra.name)
    leaf = partial(_weight_leaf, group.factors, algebra.name)
    counted, node_map = knowledge_compile(
        group.circuit,
        roots,
        variables=group.variables,
        structure=algebra,
        leaf_mapping=leaf,
    )
    # The totals are lumps of one circuit; their complements cannot be, since a
    # compiled circuit only complements a leaf.
    totals = _constructed(
        [
            reduce(
                partial(BinaryOp, algebra.sum),
                (Atom(weight(("=", binder, v))) for v in domain.values),
            )
            for binder, domain in group.domains.items()
        ],
        algebra,
    )
    feeders: dict[Symbol, FormulaNode] = {}
    for (binder, domain), total in zip(group.domains.items(), totals, strict=True):
        for value in domain.values:
            feeders[leaf(("=", binder, value))] = Atom(weight(("=", binder, value)))
        feeders[leaf(("=", binder, _MISSING))] = UnaryOp(algebra.negation, total)
    # Knowledge compilation is canonical, so equivalent bodies count to one node.
    counts = list(dict.fromkeys(node_map[root] for root in roots))
    reached = set(counted.reachable_symbol_names(counts))
    fed = tuple((name, feeder) for name, feeder in feeders.items() if name in reached)
    module = lowering.lower(*(CircuitNode(counted, count, fed) for count in counts))
    return {
        id(node): (module, counts.index(node_map[body.node]))
        for node, body in group.members
    }


def _weight(
    factors: dict[Symbol, Symbol] | None, structure: str, leaf: Symbol
) -> Symbol:
    """The weight of ``leaf``, ``=(V, v)``: ``V``'s factor with ``v`` substituted.

    Without ``factors``, the weight is ``leaf`` itself.
    """
    if factors is None:
        return with_structure(leaf, structure)
    _, binder, value = leaf
    return with_structure(
        apply_substitution(factors[binder], {binder: value}), structure
    )


def _weight_leaf(
    factors: dict[Symbol, Symbol] | None, structure: str, leaf: Symbol
) -> Symbol:
    """The counted circuit's leaf for ``leaf``, ``=(V, v)``, named by it and ``V``'s factor.

    It is fed its weight (:func:`_count`): :func:`_weight` for a named value, and
    the complement of ``V``'s named values' total for the missing one.
    """
    if factors is None:
        return with_structure(("@weight", leaf), structure)
    _, binder, _ = leaf
    return with_structure(("@weight", leaf, factors[binder]), structure)


# -- Enumeration and sampling -------------------------------------------------


#: The value a variable takes where it has no named one.
_MISSING: Symbol = ("@missing",)


@dataclass(frozen=True)
class _Variable:
    """A variable of a distribution, as its weight formula reads it.

    ``inputs`` are the symbols its binders' values are read under, ``domains``
    their values, and ``weight`` the product of its factors. With ``missing``,
    it has a missing outcome, where its presences read zero.
    """

    inputs: tuple[Symbol, ...]
    domains: tuple[Domain, ...]
    weight: FormulaNode
    missing: bool


#: An expectation, its body as evaluated, its distribution's algebra and variables.
type _Member = tuple[Aggregation, FormulaNode, Semiring, tuple[_Variable, ...]]


def _distribution(
    node: Aggregation, lowering: Lowering
) -> tuple[FormulaNode, Semiring, tuple[_Variable, ...]]:
    """``node``'s body, its distribution's algebra, and its variables.

    An expectation without a distribution has a variable for each binder its
    body tests, and none if it tests none.

    Raises:
        ValueError: If the body is not boolean, ``node`` has more than one param,
            the distribution's algebra has no sum and product, a binder has no
            factor, a leaf that is not boolean reads a binder that may be
            missing, or, without a distribution, a leaf reads a binder other
            than by testing it for a value.
    """
    if node.child.structure != BOOLEAN.name:
        raise ValueError(
            f"An expectation's body is a boolean formula, got {node.child}."
        )
    if len(node.params) > 1:
        raise ValueError(
            f"An expectation takes one param, its distribution, got {len(node.params)}."
        )
    if not node.params:
        return _default(node, lowering)
    (distribution,) = node.params
    algebra = lowering.compiler.algebra(str(distribution.structure))
    if not isinstance(algebra, Semiring):
        raise ValueError(
            f"An expectation sums products in its distribution's algebra, which "
            f"'{algebra.name}' does not declare."
        )
    joined: list[tuple[frozenset[Symbol], list[FormulaNode]]] = []
    scale: list[FormulaNode] = []
    for operand in _product_operands(distribution, algebra):
        mentioned = frozenset(_mentioned(operand, node.binders))
        if not mentioned:
            scale.append(operand)
            continue
        meets = [part for part in joined if part[0] & mentioned]
        joined = [part for part in joined if not part[0] & mentioned]
        binders = mentioned.union(*(part[0] for part in meets))
        joined.append((binders, [o for part in meets for o in part[1]] + [operand]))
    unweighted = [b for b in node.binders if not any(b in part[0] for part in joined)]
    if unweighted:
        raise ValueError(
            f"{node} binds {symbol_to_pretty_string(unweighted[0])}, which its "
            f"distribution gives no factor; an unweighted sum over it is a sum."
        )
    if complemented := isinstance(algebra, Algebra):
        _check_leaves(node.child, node.binders)
    variables = [
        _Variable(
            binders,
            tuple(lowering.compiler.domain(binder) for binder in binders),
            reduce(partial(BinaryOp, algebra.product), parts),
            complemented,
        )
        for binders, parts in (
            (tuple(b for b in node.binders if b in part[0]), part[1]) for part in joined
        )
    ]
    if scale:
        variables.append(
            _Variable((), (), reduce(partial(BinaryOp, algebra.product), scale), False)
        )
    return node.child, algebra, tuple(variables)


def _product_operands(
    distribution: FormulaNode, algebra: Semiring
) -> list[FormulaNode]:
    """The operands of ``distribution`` as a product in ``algebra``, itself if none."""
    found: list[FormulaNode] = []
    stack = [distribution]
    while stack:
        node = stack.pop()
        if isinstance(node, BinaryOp) and node.operator == algebra.product:
            stack.extend((node.rhs, node.lhs))
            continue
        if isinstance(node, CircuitNode) and node.structure == algebra.name:
            operation, below = node.circuit.operation(node.node)
            if operation == algebra.product:
                stack.extend(_part(node, part) for part in reversed(below))
                continue
        found.append(node)
    return found


def _part(lump: CircuitNode, node: int) -> CircuitNode:
    """The lump over ``node`` of ``lump``'s circuit, fed as ``lump`` feeds it."""
    reached = set(lump.circuit.reachable_symbol_names([node]))
    return CircuitNode(
        lump.circuit,
        node,
        tuple((name, fed) for name, fed in lump.feeders if name in reached),
    )


def _mentioned(formula: FormulaNode, binders: Sequence[Symbol]) -> set[Symbol]:
    """The ``binders`` that occur in an atom of ``formula``."""
    found: set[Symbol] = set()
    seen: set[int] = set()
    stack = [formula]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Atom):
            atom = unwrap_structure(node.atom)
            found.update(_over(atom, binders))
        elif isinstance(node, CircuitNode):
            stack.extend(node.children)
        else:
            stack.extend(operands(node))
    return found


def _presence(binder: Symbol) -> Symbol:
    """The input reading one where ``binder`` has a named value, zero where it has none."""
    return with_structure(("@present", binder), BOOLEAN.name)


def _check_leaves(body: FormulaNode, binders: Sequence[Symbol]) -> None:
    """Refuse a leaf of ``body`` over ``binders`` that has no truth value.

    Raises:
        ValueError: If a leaf over one of ``binders`` is not boolean.
    """
    seen: set[int] = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Atom):
            over = _over(unwrap_structure(node.atom), binders)
            if over and node.structure != BOOLEAN.name:
                raise ValueError(
                    f"{node} reads {symbol_to_pretty_string(over[0])}, which may "
                    f"have no value; only a boolean leaf is false there."
                )
        elif isinstance(node, CircuitNode):
            stack.extend(node.children)
        else:
            stack.extend(operands(node))


def _present(
    bodies: Sequence[FormulaNode], binders: Sequence[Symbol]
) -> list[FormulaNode]:
    """``bodies`` with every leaf over ``binders`` conjoined with their presences.

    The lumps of one circuit are rewritten together, so they stay one circuit.
    """
    if not binders:
        return list(bodies)
    by_circuit: dict[int, list[CircuitNode]] = {}
    for body in bodies:
        if isinstance(body, CircuitNode):
            by_circuit.setdefault(id(body.circuit), []).append(body)
    present: dict[int, FormulaNode] = {}
    for lumps in by_circuit.values():
        present.update(zip(map(id, lumps), _present_lumps(lumps, binders), strict=True))
    return [
        present[id(body)] if id(body) in present else _present_formula(body, binders)
        for body in bodies
    ]


def _present_lumps(
    lumps: Sequence[CircuitNode], binders: Sequence[Symbol]
) -> list[CircuitNode]:
    """``lumps``, one circuit's, each leaf over ``binders`` fed its conjunction.

    Such a leaf is renamed, and the conjunction of its atom and the presences
    feeds the new name: fed under its own name, a leaf would read its atom's
    column.
    """
    circuit = lumps[0].circuit
    structure = circuit.structure.name
    feeders = {name: fed for lump in lumps for name, fed in lump.feeders}
    over: dict[Symbol, list[Symbol]] = {}
    for name in circuit.reachable_symbol_names([lump.node for lump in lumps]):
        bare = without_structure(name)
        mentioned = _over(bare, binders)
        if mentioned and name not in feeders:
            over[bare] = mentioned
    if not over and not feeders:
        return list(lumps)
    renamed = {bare: ("@present", bare) for bare in over}
    moved = transform_nodes(
        *lumps,
        target_structure=structure,
        leaf_mapping=lambda bare: renamed.get(bare, bare),
    )
    conjunctions = _constructed(
        [
            reduce(
                partial(BinaryOp, BOOLEAN.product),
                [Atom(with_structure(bare, structure))]
                + [Atom(_presence(b)) for b in mentioned],
            )
            for bare, mentioned in over.items()
        ],
        BOOLEAN,
    )
    masks = tuple(
        zip(
            (with_structure(renamed[bare], structure) for bare in over),
            conjunctions,
            strict=True,
        )
    )
    return [
        CircuitNode(
            new.circuit,
            new.node,
            tuple(
                (name, _present_formula(formula, binders))
                for name, formula in lump.feeders
            )
            + masks,
        )
        for lump, new in zip(lumps, moved, strict=True)
    ]


def _constructed(
    formulas: Sequence[FormulaNode], algebra: AlgebraicStructure
) -> list[FormulaNode]:
    """``formulas``, made during lowering, constructed as compiling them would.

    Their regions a circuit can hold become lumps of one circuit, which lower to
    one core rather than a vertex per operator.
    """
    factory = CircuitFactory({algebra.name: algebra})
    built: dict[int, tuple[FormulaNode, FormulaNode]] = {}
    return [fold(formula, factory, memo=built) for formula in formulas]


def _present_formula(formula: FormulaNode, binders: Sequence[Symbol]) -> FormulaNode:
    """``formula`` with every boolean leaf over ``binders`` conjoined with their presences."""
    rewritten: dict[int, FormulaNode] = {}

    def rewrite(node: FormulaNode) -> FormulaNode:
        if id(node) in rewritten:
            return rewritten[id(node)]
        if isinstance(node, Atom):
            over = _over(unwrap_structure(node.atom), binders)
            result: FormulaNode = reduce(
                partial(BinaryOp, BOOLEAN.product),
                [node, *(Atom(_presence(b)) for b in over)],
            )
        elif isinstance(node, CircuitNode):
            (result,) = _present_lumps([node], binders)
        else:
            result = map_children(node, rewrite)
        rewritten[id(node)] = result
        return result

    return rewrite(formula)


def _agree(group: list[_Member], member: _Member) -> bool:
    """Whether ``member``'s variables agree with ``group``'s where they share inputs."""
    ours = [v for m in group for v in m[3] if v.inputs]
    return all(v == w for v in member[3] for w in ours if set(v.inputs) & set(w.inputs))


def _expect(
    members: Sequence[_Member],
    lowering: Lowering,
    samples: int | None = None,
    gradient: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> list[tuple[DeepLogModule, int]]:
    """One module computing every member, and each member's column.

    Members share variables that are equal, a scale excepted, and bodies that
    are equal.
    """
    algebra = members[0][2]
    variables: list[_Variable] = []
    bodies: list[FormulaNode] = []
    missing = list(
        dict.fromkeys(b for m in members for v in m[3] if v.missing for b in v.inputs)
    )
    expectations: list[tuple[int, frozenset[int]]] = []
    for _, body, _, own in members:
        if body not in bodies:
            bodies.append(body)
        over = set()
        for variable in own:
            if variable.inputs and variable in variables:
                over.add(variables.index(variable))
            else:
                variables.append(variable)
                over.add(len(variables) - 1)
        expectations.append((bodies.index(body), frozenset(over)))
    distinct = list(dict.fromkeys(expectations))
    module = ExpectationModule(
        lowering.lower(
            *(Transformation(algebra.name, body) for body in _present(bodies, missing))
        ),
        [
            WeightedVariable(
                v.inputs,
                tuple(domain.as_tensor() for domain in v.domains),
                lowering.lower(v.weight),
                tuple(_presence(i) for i in v.inputs) if v.missing else (),
            )
            for v in variables
        ],
        [(column, sorted(over)) for column, over in distinct],
        algebra,
        samples,
        gradient,
    )
    return [(module, distinct.index(e)) for e in expectations]


def _default(
    node: Aggregation, lowering: Lowering
) -> tuple[FormulaNode, Semiring, tuple[_Variable, ...]]:
    """``node`` without a distribution: its body, and a variable per binder it tests.

    A binder's weight is, at each value its body tests it for, that test in
    probability, and zero at every other.

    Raises:
        ValueError: If a leaf reads a binder other than as a test ``=(V, v)``.
    """
    variables = []
    for binder, tests in _tests(node.child, node.binders).items():
        weight = reduce(
            partial(BinaryOp, PROBABILITY.sum),
            (
                BinaryOp(
                    PROBABILITY.product,
                    Transformation(
                        PROBABILITY.name, Atom(with_structure(test, BOOLEAN.name))
                    ),
                    Atom(with_structure(test, PROBABILITY.name)),
                )
                for test in tests
            ),
        )
        variables.append(
            _Variable((binder,), (lowering.compiler.domain(binder),), weight, True)
        )
    return node.child, PROBABILITY, tuple(variables)


def _tests(body: FormulaNode, binders: Sequence[Symbol]) -> dict[Symbol, list[Symbol]]:
    """The leaves of ``body`` testing each of ``binders`` for a value, ``=(V, v)``.

    Raises:
        ValueError: If a leaf reads one of ``binders`` other than as such a test.
    """
    found: dict[Symbol, dict[Symbol, None]] = {}
    seen: set[int] = set()
    stack = [body]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Atom):
            atom = unwrap_structure(node.atom)
            over = _over(atom, binders)
            if not over:
                continue
            if (
                node.structure != BOOLEAN.name
                or len(atom) != 3
                or atom[0] != "="
                or atom[1] not in binders
                or _over(atom[2], binders)
            ):
                raise ValueError(
                    f"{node} reads {symbol_to_pretty_string(over[0])} other than as "
                    f"a test =(V, v), the one way an expectation without a "
                    f"distribution weighs a binder; give it a distribution."
                )
            found.setdefault(atom[1], {})[atom] = None
        elif isinstance(node, CircuitNode):
            stack.extend(node.children)
        else:
            stack.extend(operands(node))
    return {binder: list(found[binder]) for binder in binders if binder in found}
