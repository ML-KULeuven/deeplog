#  Copyright (c) 2024-2026. KU Leuven
"""Annotated disjunctions: the variable they declare, ground or not.

An annotated disjunction says its branches hold mutually exclusively -- which is
a *variable* over n values, not n independent facts. DeepProbLog spells it two
ways, and both declare the same thing: an occurrence, the atom with the
variable's term position left open, and the domain that position ranges over.

``nn(m_digit, [X], Y, [0..9]) :: digit(X,Y).`` says it directly -- ``Y`` is the
variable, ``[0..9]`` its domain -- and means ``m_digit(X,Y) :: digit(X,Y)`` plus
that declaration. The ground ``p1::a1; ...; pn::an.`` form says it by
enumeration, so the declaration is read back off the branches: the one argument
position they differ in is the variable's.

Grounding then substitutes the variable away. The grounder is semantics-free and
can only prove ordinary facts, so a disjunction reaches it as n of them and
:func:`instantiate` recovers one variable per *image* of the declaration --
``digit(i1,_)`` and ``digit(i2,_)`` are two variables. Which atom asserts which
value never needs recording: it is what substituting the value into the
occurrence yields.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import cast

from deeplog.grounding.prolog import RuleType
from deeplog.grounding.prolog.unify import calculate_mgu
from deeplog.symbol import Symbol
from deeplog.symbol import apply_substitution
from deeplog.symbol import flatten_symbol
from deeplog.symbol import get_term_variables
from deeplog.symbol import is_variable
from deeplog.symbol import symbol_to_str
from deeplog.variable import OPEN
from deeplog.variable import Domain
from deeplog.variable import SymbolicDomain
from deeplog.variable import Variable
from deeplog.variable import VariableAtoms

from .parser import create_labeled_fact
from .parser import get_atom
from .parser import get_label


#: What a disjunction declares — an atom with the variable's term position left
#: :data:`~deeplog.variable.OPEN`, and the domain that position ranges over.
type Declaration = tuple[Symbol, SymbolicDomain]

#: Functor of the neural annotation that declares a disjunction non-ground —
#: ``nn(network, [inputs], OutputVariable, [domain]) :: atom.``
NEURAL_FUNCTOR = "nn"


def is_annotated_disjunction(rule: Symbol) -> bool:
    """Return true if ``rule`` is a probabilistic AD-fact ``p1::a1; ...; pn::an.``.

    The parser already splits these on ``;`` into a fact whose head is a
    ``;``-tree of labeled atoms with body ``true``. An AD must have at least
    two labeled branches; a regular labeled fact ``p::a.`` is not an AD.
    """
    from deeplog.grounding.prolog.program import is_fact

    if not is_fact(rule):
        return False
    head = rule[1]
    if not (isinstance(head, tuple) and len(head) == 3 and head[0] == ";"):
        return False
    branches = list(flatten_symbol(head, ";"))
    if len(branches) < 2:
        return False
    return all(get_label(b) is not None for b in branches)


def get_ad_branches(rule: RuleType) -> list[Symbol]:
    """Return the labeled-atom branches of an AD-fact (in source order)."""
    return list(flatten_symbol(rule[1], ";"))


def split_annotated_disjunctions(
    program: Iterable[RuleType],
) -> tuple[list[RuleType], tuple[Declaration, ...]]:
    """Split each AD into n ordinary facts, keeping what it declares.

    Returns ``(rules, declarations)``: ``rules`` is the program the grounder
    sees, in which every AD has become n plain labeled facts; ``declarations``
    is what each AD said, which :func:`instantiate` turns back into variables
    once the atoms are ground. Non-AD rules pass through unchanged.
    """
    rules: list[RuleType] = []
    declarations: list[Declaration] = []
    for rule in program:
        if not is_annotated_disjunction(rule):
            rules.append(rule)
            continue
        branches = get_ad_branches(rule)
        declarations.append(declare_branches([get_atom(b) for b in branches]))
        rules.extend(create_labeled_fact(get_atom(b), get_label(b)) for b in branches)
    return rules, tuple(declarations)


def declare_branches(branches: Sequence[Symbol]) -> Declaration:
    """What an enumerated disjunction declares.

    The one argument position the branches differ in is the variable's, so
    ``digit(i1,0); ...; digit(i1,9)`` declares ``digit(i1,_)`` over ``0 ... 9``.
    Branches sharing no such shape are unrelated atoms, which the theory has no
    variable *term* for: the whole atom is the position, so the occurrence is
    bare and the values are the branch atoms themselves.
    """
    position = _value_position(branches)
    if position is None:
        return OPEN, Domain.of(branches)
    occurrence = cast(
        "Symbol", (*branches[0][:position], OPEN, *branches[0][position + 1 :])
    )
    return occurrence, Domain.of(branch[position] for branch in branches)


def declare_neural(
    atom: Symbol, label: Symbol
) -> tuple[tuple[tuple[Symbol, Symbol], ...], Declaration] | None:
    """Read a non-ground disjunction off ``nn(net, [inputs], Var, [domain])``.

    ``nn(m_digit, [X], Y, [0..9]) :: digit(X,Y).`` is the enumerated disjunction
    ``m_digit(X,0)::digit(X,0); ...; m_digit(X,9)::digit(X,9).`` written once:
    the returned branches are those labeled atoms, one per declared value, and
    the declaration is what they say. Returns ``None`` if ``label`` is not that
    annotation, so a label of any other shape passes through untouched.

    Raises:
        ValueError: If the annotation is well-formed but its output variable
            does not occur exactly once in ``atom``, which leaves the
            declaration with no term position to open.
    """
    if len(label) != 5 or label[0] != NEURAL_FUNCTOR:
        return None
    network, inputs, output, domain = label[1], label[2], label[3], label[4]
    arguments = _list_elements(inputs)
    values = _list_elements(domain)
    if arguments is None or values is None or not is_variable(output):
        return None
    positions = [index for index in range(1, len(atom)) if atom[index] == output]
    if len(positions) != 1:
        raise ValueError(
            f"Neural annotation {symbol_to_str(label)} declares "
            f"{symbol_to_str(output)} over a domain, but it occurs "
            f"{len(positions)} times in {symbol_to_str(atom)}; it must occur "
            f"exactly once, as the argument the domain ranges over."
        )
    position = positions[0]
    declared = Domain.of(_expand_values(values))
    template = cast("Symbol", (cast("str", network[0]), *arguments, output))
    branches = tuple(
        (
            apply_substitution(atom, {output: value}),
            apply_substitution(template, {output: value}),
        )
        for value in declared.values
    )
    occurrence = cast("Symbol", (*atom[:position], OPEN, *atom[position + 1 :]))
    return branches, (occurrence, declared)


def instantiate(
    declarations: Iterable[Declaration], atoms: Iterable[Symbol]
) -> VariableAtoms:
    """The variables ``declarations`` have once ``atoms`` fix their images.

    A declaration over free arguments stands for one variable per image of them:
    ``digit(X,_)`` is ``digit(i1)`` and ``digit(i2)``, told apart by the ground
    atoms that match it. A declaration with no free argument already names its
    variable and needs no atom to do so -- a disjunction the program never
    grounds still declares one, which simply has no reachable value.
    """
    ground = list(atoms)
    variables: dict[Variable, tuple[Symbol, ...]] = {}
    for occurrence, domain in declarations:
        for image in _images(occurrence, domain, ground):
            instance = apply_substitution(occurrence, image)
            variables[Variable(_variable_name(instance, domain), domain)] = (instance,)
    return variables


def _images(
    occurrence: Symbol, domain: SymbolicDomain, atoms: Sequence[Symbol]
) -> Iterable[Mapping[Symbol, Symbol]]:
    """The bindings of ``occurrence``'s free arguments that ``atoms`` witness."""
    free = {variable for variable in get_term_variables(occurrence) if variable != OPEN}
    if not free:
        yield {}
        return
    seen: set[tuple[tuple[Symbol, Symbol], ...]] = set()
    for atom in atoms:
        mgu = calculate_mgu(occurrence, atom)
        if mgu is None or mgu.get(OPEN) not in domain.values:
            continue
        image = {name: value for name, value in mgu.items() if name in free}
        key = tuple(sorted(image.items(), key=lambda item: symbol_to_str(item[0])))
        if key not in seen:
            seen.add(key)
            yield image


def _variable_name(occurrence: Symbol, domain: SymbolicDomain) -> Symbol:
    """The variable occupying ``occurrence``'s open position.

    ``digit(i1,3)`` asserts a value of ``digit(i1)``, so the name is the
    occurrence without the position it leaves open. A bare occurrence has none
    to drop: its name is minted, because unrelated branch atoms give the
    variable itself no user-side referent, only its values.
    """
    positions = [
        index for index in range(1, len(occurrence)) if occurrence[index] == OPEN
    ]
    if len(positions) != 1:
        return "@variable", domain.values[0]
    position = positions[0]
    return cast("Symbol", (*occurrence[:position], *occurrence[position + 1 :]))


def _value_position(branches: Sequence[Symbol]) -> int | None:
    """The one argument position the branches differ in, or ``None``.

    When there is one, the disjunction is a variable occurring in that argument
    -- ``digit(i1,0); ...; digit(i1,9)`` is ``digit(i1,N)`` with ``N`` over the
    digits -- and its domain is the values found there. Branches that share no
    such shape (unrelated predicates) have no argument to read a value from.
    """
    first = branches[0]
    if not all(len(b) == len(first) and b[0] == first[0] for b in branches):
        return None
    differing = [i for i in range(1, len(first)) if len({b[i] for b in branches}) > 1]
    if len(differing) != 1:
        return None
    if len({b[differing[0]] for b in branches}) != len(branches):
        return None
    return differing[0]


def _list_elements(term: Symbol) -> list[Symbol] | None:
    """The elements of a ``cons``/``nil`` list term, or ``None`` if not one."""
    elements: list[Symbol] = []
    while term != ("nil",):
        if len(term) != 3 or term[0] != "cons":
            return None
        elements.append(cast("Symbol", term[1]))
        term = cast("Symbol", term[2])
    return elements


def _expand_values(elements: Iterable[Symbol]) -> Iterable[Symbol]:
    """The declared values, expanding each ``low..high`` range element."""
    for element in elements:
        name = element[0]
        low, separator, high = (
            name.partition("..") if isinstance(name, str) else ("", "", "")
        )
        if separator and low.isdigit() and high.isdigit():
            yield from ((str(value),) for value in range(int(low), int(high) + 1))
        else:
            yield element
