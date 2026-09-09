#  Copyright (c) 2024-2026. KU Leuven

import numpy as np
import pytest
import torch

from deeplog import ShapeMismatchException
from deeplog import SymTensor
from deeplog import get_all_symbols
from deeplog import map_shape
from deeplog import sole_structure
from deeplog import structures
from deeplog import to_dict
from deeplog import with_structure


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


def test_to_dict_single_symtensor():
    out = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    result = to_dict(out, SymTensor([a, b]))
    assert set(result) == {a, b}
    assert torch.equal(result[a], out[:, 0])
    assert torch.equal(result[b], out[:, 1])


def test_to_dict_tuple_shape():
    first = torch.tensor([[0.1], [0.2]])
    second = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    result = to_dict((first, second), (SymTensor([c]), SymTensor([a, b])))
    assert torch.equal(result[c], first[:, 0])
    assert torch.equal(result[a], second[:, 0])
    assert torch.equal(result[b], second[:, 1])


def test_to_dict_multidimensional():
    shape = SymTensor([[a, b], [c, d]])
    out = torch.arange(8.0).reshape(2, 2, 2)
    result = to_dict(out, shape)
    assert torch.equal(result[a], out[:, 0, 0])
    assert torch.equal(result[d], out[:, 1, 1])


def test_to_dict_mismatch_raises():
    with pytest.raises(ShapeMismatchException):
        to_dict(torch.zeros(2, 3), SymTensor([[a, b], [c, d]]))
    with pytest.raises(ShapeMismatchException):
        to_dict((torch.zeros(2, 1),), (SymTensor([a]), SymTensor([b])))


def test_to_dict_duplicate_symbol_raises():
    with pytest.raises(ValueError, match="twice"):
        to_dict(torch.zeros(2, 2), SymTensor([a, a]))


def test_to_dict_prints_readably():
    result = to_dict(torch.tensor([[0.5, 0.0]]), SymTensor([a, b]))
    assert str(result) == "{a: 0.5, b: 0}"


# --- Algebraic structure of a shape ---


def test_structures_reports_one_entry_per_symbol():
    """A shape naming values in different algebras is described honestly."""
    shape = SymTensor([with_structure(a, "probability"), b])

    assert structures(shape) == ("probability", None)


def test_sole_structure_of_a_uniformly_labelled_shape():
    shape = SymTensor([with_structure(sym, "boolean") for sym in (a, b)])

    assert sole_structure(shape) == "boolean"


def test_sole_structure_of_an_unlabelled_shape_is_none():
    """No label is a state, not an error — it is what a raw-tensor module has."""
    assert sole_structure(SymTensor([a, b])) is None
    assert sole_structure(SymTensor([])) is None


def test_sole_structure_raises_on_a_mixed_shape():
    """A mixed shape has no single structure, and saying ``None`` would make
    that indistinguishable from having no label at all."""
    mixed = SymTensor([with_structure(a, "probability"), with_structure(b, "boolean")])

    with pytest.raises(ValueError, match="mixes algebraic structures"):
        sole_structure(mixed)

    partly = SymTensor([with_structure(a, "probability"), b])
    with pytest.raises(ValueError, match="mixes algebraic structures"):
        sole_structure(partly)


def test_sole_structure_spans_a_tuple_shape():
    """A multi-channel shape is one value space, so all its channels must agree."""
    labelled = SymTensor([with_structure(a, "probability")])
    other = SymTensor([with_structure(b, "boolean")])

    assert sole_structure((labelled, labelled)) == "probability"
    with pytest.raises(ValueError, match="mixes algebraic structures"):
        sole_structure((labelled, other))
