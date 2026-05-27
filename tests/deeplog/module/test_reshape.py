#  Copyright (c) 2024-2026. KU Leuven

import operator
from functools import reduce
from typing import Any

import pytest
import torch

from deeplog import SymTensor
from deeplog import TransformationNotPossible
from deeplog import construct_transformation
from deeplog.module.reshape import _IndexingTransform


a, b = ("a",), ("b",)
t_a = torch.tensor([0, 1, 2, 3, 4]).unsqueeze(0)
t_b = torch.tensor([5, 6, 7, 8, 9]).unsqueeze(0)
empty = SymTensor([])

shapes = [
    (SymTensor(a), t_a),
    (SymTensor([a]), t_a.unsqueeze(1)),
    (SymTensor([[a]]), t_a.unsqueeze(1).unsqueeze(1)),
    (SymTensor([a, b]), torch.stack([t_a, t_b], dim=1)),
    (SymTensor([b, a]), torch.stack([t_b, t_a], dim=1)),
    ((SymTensor(a), SymTensor([b])), (t_a, t_b.unsqueeze(1))),
    ((SymTensor(a), SymTensor(b)), (t_a, t_b)),
    ((SymTensor([b]), SymTensor([a])), (t_b.unsqueeze(1), t_a.unsqueeze(1))),
]


def test_construct_transformation_empty_output_from_symtensor_input():
    batch = 3
    input_shape = SymTensor([a])
    transform = construct_transformation(input_shape, empty)
    output = transform(torch.ones(batch, 1))
    assert output.shape == torch.Size([batch, 0])


def test_construct_transformation_empty_output_from_tuple_input():
    batch = 2
    input_shapes = (empty, SymTensor([a]))
    transform = construct_transformation(input_shapes, empty)
    output = transform(torch.zeros(batch, 0), torch.ones(batch, 1))
    assert output.shape == torch.Size([batch, 0])


def test_construct_transformation_empty_output_from_empty_tuple_input():
    batch = 1
    input_shapes = tuple()
    transform = construct_transformation(input_shapes, empty)
    output = transform()
    assert output.shape == torch.Size([batch, 0])


def _get_symbols(shape: SymTensor | tuple[SymTensor]):
    if isinstance(shape, tuple):
        return reduce(operator.or_, (set(s) for s in shape))
    return set(shape)


@pytest.mark.parametrize("output_shape", shapes)
@pytest.mark.parametrize("input_shape", shapes)
def test_construct_transformation(
    input_shape: tuple[SymTensor | tuple[SymTensor, ...], Any],
    output_shape: tuple[SymTensor | tuple[SymTensor, ...], Any],
):
    input_symbols = _get_symbols(input_shape[0])
    output_symbols = _get_symbols(output_shape[0])

    if len(output_symbols - input_symbols) > 0:
        with pytest.raises(TransformationNotPossible):
            construct_transformation(input_shape[0], output_shape[0])
    else:
        reshape_module = construct_transformation(input_shape[0], output_shape[0])

        assert reshape_module.get_input_shape() == input_shape[0]
        assert reshape_module.get_output_shape() == output_shape[0]

        if isinstance(input_shape[1], tuple):
            result = reshape_module(*input_shape[1])
        else:
            result = reshape_module(input_shape[1])

        if isinstance(output_shape[1], tuple):
            assert all(
                expected.shape == result[i].shape
                for i, expected in enumerate(output_shape[1])
            )
            assert all(
                (expected == result[i]).all()
                for i, expected in enumerate(output_shape[1])
            )
        else:
            assert output_shape[1].shape == result.shape
            assert (output_shape[1] == result).all()


def test_indexing_transform_0():
    shape1 = SymTensor(a)
    shape2 = SymTensor([a])
    transform = _IndexingTransform(shape1, shape2)
    expected = t_a.unsqueeze(0)
    result = transform(t_a)
    assert result.shape == expected.shape
    assert (result == expected).all()


def test_indexing_transform_1():
    shape1 = SymTensor([a, b])
    shape2 = SymTensor([a])
    transform = _IndexingTransform(shape1, shape2)
    expected = t_a.unsqueeze(1)
    result = transform(torch.stack([t_a, t_b], dim=1))
    assert result.shape == expected.shape
    assert (result == expected).all()


def test_indexing_transform_2():
    shape1 = SymTensor([a, b])
    shape2 = SymTensor(a)
    transform = _IndexingTransform(shape1, shape2)
    expected = t_a
    result = transform(torch.stack([t_a, t_b], dim=1))
    assert result.shape == expected.shape
    assert (result == expected).all()


def test_indexing_transform_3():
    shape1 = SymTensor([a])
    shape2 = SymTensor([a, a])
    transform = _IndexingTransform(shape1, shape2)
    expected = torch.stack([t_a, t_a], dim=1)
    result = transform(t_a.unsqueeze(1))
    assert result.shape == expected.shape
    assert (result == expected).all()
