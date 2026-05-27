#  Copyright (c) 2024-2026. KU Leuven

import numpy as np
import pytest

from deeplog import SymTensor
from deeplog import get_all_symbols
from deeplog import map_shape


a, b, c, d = ("a",), ("b",), ("c",), ("d",)


def test_symtensor_to_set():
    symtensor = SymTensor([a, b])
    assert {a, b} == set(symtensor)


def test_symtensor_parse():
    assert SymTensor([a, b]) == SymTensor(["a", "b"])
    assert SymTensor(a) == SymTensor("a")


def test_symtensor_hash_matches_equality():
    # Equal SymTensors must hash equal (dict/set invariant).
    assert hash(SymTensor([a, b])) == hash(SymTensor(["a", "b"]))
    # Different shapes with same flattened symbols must not collide.
    assert hash(SymTensor([a, b])) != hash(SymTensor([[a, b]]))
    # Usable as dict keys.
    d = {SymTensor([a, b]): 1}
    assert d[SymTensor(["a", "b"])] == 1


def test_symtensor_len():
    assert len(SymTensor([a, b, c])) == 3
    assert len(SymTensor([[a, b], [c, d]])) == 2
    assert len(SymTensor(a)) == 0  # scalar SymTensor


def test_symtensor_array_is_immutable():
    sym = SymTensor([a, b])
    with pytest.raises(ValueError, match="read-only|writeable"):
        sym.array[0] = c
    # Going through __array__ exposes the same frozen view.
    arr = np.asarray(sym)
    with pytest.raises(ValueError, match="read-only|writeable"):
        arr[0] = c


def test_get_all_symbols():
    assert set(get_all_symbols(SymTensor([a, c]))) == {a, c}
    assert set(get_all_symbols((SymTensor([a, c]), SymTensor([b])))) == {a, b, c}
    assert set(
        get_all_symbols((SymTensor([d]), (SymTensor([a, c]), SymTensor([b]))))
    ) == {a, b, c, d}


def test_map_shape():
    def func(x):
        return "a", x

    assert SymTensor([("a", a), ("a", c)]) == map_shape(func, SymTensor([a, c]))
    assert SymTensor([[("a", a), ("a", b)], [("a", c), ("a", d)]]) == map_shape(
        func, SymTensor([[a, b], [c, d]])
    )
    assert (
        SymTensor([("a", a), ("a", b)]),
        SymTensor([("a", c), ("a", d)]),
    ) == map_shape(func, (SymTensor([a, b]), SymTensor([c, d])))
