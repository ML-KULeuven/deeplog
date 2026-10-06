#  Copyright (c) 2024-2026. KU Leuven
"""Compile DeepProbLog engine results into DeepLogModules.

A proof is the expectation of its formula, modelled as the DeepLog paper models
ProbLog: each random atom is reified by a variable of its own. A probabilistic
fact is a two-valued binder, and an annotated disjunction one binder over its
values. The formula's atoms become tests of their binders' values, and its
distribution is the product of the atoms themselves in probability, each
extended with its binder, such as
``burglary(B)`` or ``digit(i1, D)``. What the program's labels say is those
atoms' labelling function, which :func:`compile_to_module` declares to the
compiler for their predicates.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from functools import reduce
from typing import TYPE_CHECKING
from typing import cast

import torch

from deeplog import BOOLEAN
from deeplog import OPEN
from deeplog import PROBABILITY
from deeplog import Aggregation
from deeplog import Atom
from deeplog import BinaryOp
from deeplog import Domain
from deeplog import Symbol
from deeplog import SymTensor
from deeplog import UnaryOp
from deeplog import WrappedModule
from deeplog import apply_substitution
from deeplog import compose_modules
from deeplog import symbol_to_pretty_string
from deeplog import with_structure
from deeplog import without_structure

from .solver import EngineResult


if TYPE_CHECKING:
    from deeplog import AtomBuilder
    from deeplog import Compiler
    from deeplog import DeepLogModule
    from deeplog import FormulaNode


def compile_to_module(result: EngineResult, compiler: Compiler) -> DeepLogModule:
    """Compile an engine result to a DeepLogModule with a column per answer.

    Each of ``result.formulas`` is compiled as the expectation of it under the
    program's labels. With evidence (``result.evidence`` set) each column is the
    posterior ``P(q | e) = E[q ∧ e] / E[e]`` instead, every answer dividing by
    the one expectation of the evidence. ``compiler`` compiles them together,
    declaring the program's variables and their labelling function, and a label
    naming an atom it has a builder for, such as a network's output, is that
    builder's value. Columns are named by the answer atoms; two answers with
    equal proofs still get one column each.

    Raises:
        ValueError: If ``result`` has no formula, a random atom's predicate
            already has a builder in ``compiler``, or a label is neither a
            number nor an atom in probability.
    """
    lifting = _Lifting(result)
    queries = [lifting.expectation(formula) for formula in result.formulas.values()]
    if result.evidence is not None:
        evidence = lifting.expectation(result.evidence)
        queries = [lifting.posterior(query, evidence) for query in queries]
    if not queries:
        raise ValueError("The engine result holds no formula to compile.")
    compiler = compiler.declaring(
        {variable.binder: variable.domain for variable in lifting.variables},
        lifting.labelling(compiler),
    )
    return compiler.compile(dict(zip(result.formulas, queries, strict=True)))


@dataclass(frozen=True)
class _Label:
    """A value's probability: a number or an atom's value, or the complement of one."""

    source: float | Symbol
    complement: bool = False


@dataclass(frozen=True, eq=False)
class _Variable:
    """A random variable of the program, and how it is reified.

    ``atoms`` maps each program atom asserting a value to that value, and
    ``labels`` gives each value of ``domain`` its probability, in value order.
    ``factor`` is the atom standing for the variable in the distribution, with
    ``binder`` in the position its value takes.
    """

    binder: Symbol
    domain: Domain
    factor: Symbol
    atoms: dict[Symbol, Symbol]
    labels: tuple[_Label, ...]

    def at(self, value: Symbol) -> Symbol:
        """The factor with ``value`` in the binder's position."""
        return apply_substitution(self.factor, {self.binder: value})


#: The truth values, which a fact's binder ranges over.
_TRUTH = Domain.of_structure(BOOLEAN)
_TRUE, _FALSE = BOOLEAN.one, BOOLEAN.zero

#: The values, the domain of a factor's arguments other than its binder.
_VALUES = Domain.of_values()


class _Lifting:
    """Lifts a result's proofs into expectations over the program's variables.

    A proof's atoms become tests of their variables' values, and every node it
    makes is interned, so equal proofs become one formula and are compiled once.
    """

    def __init__(self, result: EngineResult) -> None:
        """Read the disjunctions' variables off ``result``; facts are read as reached.

        Raises:
            ValueError: If a disjunction's values have no names, so no atom
                asserts them.
        """
        self._labels = result.labels
        self._variables: dict[Symbol, _Variable] = {}
        for variable, occurrences in result.variables.items():
            (occurrence,) = occurrences
            self._add(
                _disjunction(variable.name, variable.domain, occurrence, self._labels)
            )
        #: What each proof node became, by id, beside the node, which keeps its id.
        self._lifted: dict[int, tuple[FormulaNode, FormulaNode]] = {}
        #: Every node made, by what it is made of, which interns it.
        self._made: dict[tuple, FormulaNode] = {}
        #: The binders each node made tests, by node id.
        self._tested: dict[int, frozenset[Symbol]] = {}

    @property
    def variables(self) -> list[_Variable]:
        """Every variable a lifted proof may test."""
        return list({id(v): v for v in self._variables.values()}.values())

    def expectation(self, formula: FormulaNode) -> FormulaNode:
        """The expectation of ``formula`` under the program's labels."""
        body = self._lift(formula)
        key = ("expectation", id(body))
        if key not in self._made:
            self._made[key] = self._expect(body)
        return self._made[key]

    def posterior(self, joint: FormulaNode, evidence: FormulaNode) -> FormulaNode:
        """``joint`` divided by ``evidence``."""
        key = ("posterior", id(joint), id(evidence))
        if key not in self._made:
            self._made[key] = BinaryOp(PROBABILITY.division, joint, evidence)
        return self._made[key]

    def labelling(self, compiler: Compiler) -> dict[tuple[str, int, str], AtomBuilder]:
        """The builder of each factor's predicate: the variables' labelling function.

        A label naming an atom ``compiler`` has a builder for is that builder's
        value, and one it has none for is an input named after the atom.
        """
        keys: dict[tuple[str, int, str], list[_Variable]] = {}
        for variable in self.variables:
            key = (variable.factor[0], len(variable.factor) - 1, PROBABILITY.name)
            keys.setdefault(key, []).append(variable)
        return {
            key: partial(_labelling, variables, compiler)
            for key, variables in keys.items()
        }

    def _lift(self, formula: FormulaNode) -> FormulaNode:
        """``formula`` with each atom a test of its variable's value.

        Operands before operators, with an explicit stack: a proof can be a
        chain of disjunctions thousands deep.
        """
        stack: list[tuple[FormulaNode, bool]] = [(formula, False)]
        while stack:
            node, ready = stack.pop()
            if id(node) in self._lifted:
                continue
            if ready:
                self._lifted[id(node)] = (node, self._lifted_node(node))
                continue
            stack.append((node, True))
            match node:
                case UnaryOp(_, operand):
                    stack.append((operand, False))
                case BinaryOp(_, lhs, rhs):
                    stack.extend(((rhs, False), (lhs, False)))
        return self._lifted[id(formula)][1]

    def _lifted_node(self, node: FormulaNode) -> FormulaNode:
        """``node`` lifted, its operands already lifted and interned.

        Raises:
            TypeError: If ``node`` is not an atom or an operator.
        """
        match node:
            case Atom(atom):
                return self._test(atom)
            case UnaryOp(operator, operand):
                lifted = self._lifted[id(operand)][1]
                key = ("unary", operator, id(lifted))
                made = self._made.get(key)
                if made is None:
                    made = self._made[key] = UnaryOp(operator, lifted)
                    self._tested[id(made)] = self._tested[id(lifted)]
                return made
            case BinaryOp(operator, lhs, rhs):
                left = self._lifted[id(lhs)][1]
                right = self._lifted[id(rhs)][1]
                key = ("binary", operator, id(left), id(right))
                made = self._made.get(key)
                if made is None:
                    made = self._made[key] = BinaryOp(operator, left, right)
                    below, above = self._tested[id(left)], self._tested[id(right)]
                    self._tested[id(made)] = below if above <= below else below | above
                return made
        raise TypeError(f"A proof is built of atoms and operators, got {node!r}.")

    def _test(self, atom: Symbol) -> FormulaNode:
        """A random atom's test of its variable's value; a constant as it is."""
        ground = cast("Symbol", atom[1])
        if ground in (_TRUE, _FALSE):
            return self._intern(("atom", atom), frozenset(), lambda: Atom(atom))
        variable = self._variables.get(ground) or self._add(
            _fact(ground, self._labels.get(ground, ground))
        )
        test = with_structure(
            ("=", variable.binder, variable.atoms[ground]), BOOLEAN.name
        )
        return self._intern(
            ("atom", test), frozenset((variable.binder,)), lambda: Atom(test)
        )

    def _intern(
        self, key: tuple, tested: frozenset[Symbol], make: Callable[[], FormulaNode]
    ) -> FormulaNode:
        """The one node for ``key``, testing ``tested``, made the first time."""
        node = self._made.get(key)
        if node is None:
            node = self._made[key] = make()
            self._tested[id(node)] = tested
        return node

    def _expect(self, body: FormulaNode) -> Aggregation:
        """The expectation of the lifted ``body`` over the variables it tests."""
        binders = tuple(sorted(self._tested[id(body)], key=symbol_to_pretty_string))
        if not binders:
            return Aggregation("expectation", (), (), body)
        distribution = reduce(
            partial(BinaryOp, PROBABILITY.product),
            (
                Atom(with_structure(self._variables[binder].factor, PROBABILITY.name))
                for binder in binders
            ),
        )
        return Aggregation("expectation", binders, (distribution,), body)

    def _add(self, variable: _Variable) -> _Variable:
        """Record ``variable`` under its binder and every atom asserting its values."""
        self._variables[variable.binder] = variable
        for atom in variable.atoms:
            self._variables[atom] = variable
        return variable


def _fact(atom: Symbol, label: Symbol) -> _Variable:
    """The variable of the probabilistic fact ``atom``: whether it holds."""
    binder = ("_" + symbol_to_pretty_string(atom),)
    probability = _label(label)
    return _Variable(
        binder,
        _TRUTH,
        (*atom, binder),
        {atom: _TRUE},
        (_Label(probability.source, complement=True), probability),
    )


def _disjunction(
    name: Symbol,
    domain: Domain,
    occurrence: Symbol,
    labels: dict[Symbol, Symbol],
) -> _Variable:
    """The variable of an annotated disjunction: which of its values holds.

    Its factor is the atom asserting a value, with the binder in the value's
    position, or, where the values are unrelated atoms, the variable's name
    extended with the binder.
    """
    atoms = {apply_substitution(occurrence, {OPEN: v}): v for v in domain.values}
    if occurrence == OPEN:
        binder = ("_" + symbol_to_pretty_string(name),)
        factor = (*name, binder)
    else:
        binder = ("_" + symbol_to_pretty_string(occurrence),)
        factor = apply_substitution(occurrence, {OPEN: binder})
    return _Variable(
        binder,
        domain,
        factor,
        atoms,
        tuple(_label(labels.get(atom, atom)) for atom in atoms),
    )


def _label(label: Symbol) -> _Label:
    """What the program's ``label`` says a probability is.

    Raises:
        ValueError: If ``label`` is an atom in another algebra than probability.
    """
    if len(label) == 3 and label[0] == "_":
        if label[2] != (PROBABILITY.name,):
            raise ValueError(
                f"The label {symbol_to_pretty_string(label)} is not a probability."
            )
        label = label[1]
    if len(label) == 1:
        try:
            return _Label(float(label[0]))
        except (TypeError, ValueError):
            pass
    return _Label(label)


def _labelling(
    variables: Sequence[_Variable],
    compiler: Compiler,
    atoms: Sequence[Symbol],
) -> DeepLogModule:
    """The module labelling the factor atoms ``atoms``, a column named by each.

    A factor atom with a value in its binder's position is labelled by that
    value's label, and one with its binder there by the label of the value the
    binder holds. The module declares that position's domain, its variable's.

    Raises:
        ValueError: If an atom asked for is no variable's factor.
    """
    known: dict[Symbol, tuple[_Variable, int | None]] = {}
    for variable in variables:
        known[variable.factor] = (variable, None)
        for position, value in enumerate(variable.domain.values):
            known[variable.at(value)] = (variable, position)
    factors = [without_structure(atom) for atom in atoms]
    missing = [factor for factor in factors if factor not in known]
    if missing:
        raise ValueError(
            f"{symbol_to_pretty_string(missing[0])} is not the factor of a variable "
            "of this program."
        )
    table = _Table([known[factor] for factor in factors])
    output = SymTensor(list(atoms))
    gather = WrappedModule(table, SymTensor(table.inputs), output, name="labelling")
    built = _label_modules(table.label_atoms, compiler)
    return _Labels(
        compose_modules([*built, gather], output) if built else gather,
        {
            factor[1:]: (variable.factor.index(variable.binder) - 1, variable.domain)
            for factor, (variable, _) in known.items()
        },
    )


class _Labels(WrappedModule):
    """Factor atoms' labels, declaring the domain of each binder's position."""

    def __init__(
        self,
        module: DeepLogModule,
        domains: dict[tuple[Symbol, ...], tuple[int, Domain]],
    ) -> None:
        """Wrap ``module``; ``domains`` gives each factor's binder position and domain."""
        super().__init__(
            module,
            module.get_input_shape(),
            module.get_output_shape(),
            name="labelling",
        )
        self._domains = domains

    def domains_of(self, arguments: tuple[Symbol, ...], /) -> tuple[Domain, ...]:
        """The variable's domain at the binder's position, the values elsewhere."""
        position, domain = self._domains[arguments]
        return tuple(
            domain if index == position else _VALUES for index in range(len(arguments))
        )


def _label_modules(atoms: Sequence[Symbol], compiler: Compiler) -> list[DeepLogModule]:
    """A module computing the label ``atoms`` ``compiler`` has builders for."""
    grouped: dict[tuple[str, int, str], list[Symbol]] = {}
    for atom in atoms:
        grouped.setdefault((atom[0], len(atom) - 1, PROBABILITY.name), []).append(
            with_structure(atom, PROBABILITY.name)
        )
    modules = []
    for key, labelled in grouped.items():
        builder = compiler.atom_builder(*key)
        if builder is not None:
            modules.append(builder(labelled))
    return modules


class _Table(torch.nn.Module):
    """Reads each factor atom's label off a bank of constants and label columns.

    Its inputs are the label atoms' columns and the binders' positions, in that
    order (:attr:`inputs`).
    """

    _rows: torch.Tensor
    _complements: torch.Tensor
    _fixed: torch.Tensor
    _bound: torch.Tensor

    def __init__(self, evaluations: Sequence[tuple[_Variable, int | None]]) -> None:
        """Lay out each evaluation's labels, one row of the bank per value."""
        super().__init__()
        constants: dict[float, None] = {}
        atoms: dict[Symbol, None] = {}
        binders: dict[Symbol, None] = {}
        for variable, position in evaluations:
            for label in _read(variable, position):
                if isinstance(label.source, tuple):
                    atoms.setdefault(label.source)
                else:
                    constants.setdefault(label.source)
            if position is None:
                binders.setdefault(variable.binder)
        bank = {source: i for i, source in enumerate([*constants, *atoms])}
        width = max(len(variable.labels) for variable, _ in evaluations)
        rows = torch.zeros(len(evaluations), width, dtype=torch.long)
        complements = torch.zeros(len(evaluations), width, dtype=torch.bool)
        fixed = torch.zeros(len(evaluations), dtype=torch.long)
        bound = torch.full((len(evaluations),), -1, dtype=torch.long)
        binder_column = {binder: i for i, binder in enumerate(binders)}
        for e, (variable, position) in enumerate(evaluations):
            for value, label in enumerate(variable.labels):
                if position is None or value == position:
                    rows[e, value] = bank[label.source]
                    complements[e, value] = label.complement
            if position is None:
                bound[e] = binder_column[variable.binder]
            else:
                fixed[e] = position
        #: The label atoms read as columns, and then the binders read as positions.
        self.label_atoms = list(atoms)
        self.inputs = [
            *(with_structure(atom, PROBABILITY.name) for atom in atoms),
            *binders,
        ]
        #: Held exactly, and made a tensor in the dtype of the values they meet,
        #: or the default dtype where those are positions alone.
        self.constants = list(constants)
        self.register_buffer("_rows", rows)
        self.register_buffer("_complements", complements)
        self.register_buffer("_fixed", fixed)
        self.register_buffer("_bound", bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Each evaluation's label, one column per evaluation."""
        batch = values.shape[0]
        labels = len(self.label_atoms)
        dtype = (
            values.dtype if values.is_floating_point() else torch.get_default_dtype()
        )
        bank = torch.cat(
            [
                torch.tensor(self.constants, dtype=dtype, device=values.device).expand(
                    batch, -1
                ),
                values[:, :labels],
            ],
            dim=1,
        )
        positions = self._fixed.expand(batch, -1).clone()
        bound = self._bound >= 0
        if bool(bound.any()):
            binders = values[:, labels:][:, self._bound[bound]]
            positions[:, bound] = binders.to(torch.long)
        evaluation = torch.arange(len(self._fixed), device=values.device)
        chosen = bank.gather(1, self._rows[evaluation, positions])
        return torch.where(self._complements[evaluation, positions], 1 - chosen, chosen)


def _read(variable: _Variable, position: int | None) -> Iterable[_Label]:
    """The labels an evaluation at ``position`` reads: all of them for a binder."""
    return variable.labels if position is None else (variable.labels[position],)
