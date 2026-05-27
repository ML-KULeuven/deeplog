#  Copyright (c) 2024-2026. KU Leuven

from deeplog.symbol import apply_substitution
from deeplog.systems.deepproblog.engine.unify import calculate_mgu
from deeplog.systems.deepproblog.engine.unify import replace_with_fresh_variables
from deeplog.systems.deepproblog.engine.unify import unify


a, b, c, X, Y = ((sym,) for sym in "abcXY")


def test_substitution():
    term1 = ("p", X, ("a", X, Y))
    substitution = {X: ("b", c), Y: c}
    term2 = ("p", ("b", c), ("a", ("b", c), c))
    term3 = apply_substitution(term1, substitution)
    assert term2 == term3


def test_apply_substitution_variable():
    substitution = {X: Y}
    assert Y == apply_substitution(X, substitution)
    assert Y == apply_substitution(Y, substitution)


def test_unify():
    term1 = ("p", X, ("b", c))
    term2 = ("p", a, ("b", Y))
    term3 = ("p", a, ("b", c))

    term4, substitution = unify(term1, term2)
    assert term3 == term4
    assert a == substitution[X]
    assert c == substitution[Y]


def test_unify2():
    term1 = ("p", X, X)
    term2 = ("p", a, Y)
    term3 = ("p", a, a)

    term4, substitution1 = unify(term1, term2)
    term5, substitution2 = unify(term2, term1)
    assert term4 == term3
    assert term5 == term3


def test_unify3():
    term1 = ("p", ("a", X), X)
    term2 = ("p", ("a", X), b)

    term = ("p", ("a", b), b)

    term4, substitution1 = unify(term1, term2)
    term5, substitution2 = unify(term2, term1)

    assert substitution1 == substitution2

    assert term4 == term
    assert term5 == term


def test_unify4():
    term1 = ("p", X, X)
    term2 = ("p", a, a)
    assert {X: a} == calculate_mgu(term1, term2)


def test_unify5():
    term1 = ("fact", ("t", a, b, X), ("t", b, a, X))
    term2 = ("fact", ("t", a, b, c), Y)

    term = ("fact", ("t", a, b, c), ("t", b, a, c))
    substitution = {X: c, Y: ("t", b, a, c)}

    term4, substitution1 = unify(term1, term2)
    term5, substitution2 = unify(term2, term1)

    assert substitution1 == substitution2 == substitution
    assert term4 == term
    assert term5 == term


def test_unify_fail():
    term1 = ("p", X, X)
    term2 = ("p", a, b)

    unification = unify(term1, term2)
    assert unification is None


def test_fresh_variables():
    counter = 0

    def get_fresh_variable():
        nonlocal counter
        counter += 1
        return (f"VAR_{counter}",)

    var_1, var_2 = ("VAR_1",), ("VAR_2",)

    term1 = ("p", X, ("a", X, Y))
    term2 = ("p", var_1, ("a", var_1, var_2))
    term3, substitution = replace_with_fresh_variables(term1, get_fresh_variable)
    assert term2 == term3
    assert {X: var_1, Y: var_2} == substitution
