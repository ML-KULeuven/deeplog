#  Copyright (c) 2024-2026. KU Leuven
import torch

from deeplog import SymTensor
from deeplog import with_structure
from deeplog.formula.lowering.transform_cast import build_transform
from deeplog.shape import sole_structure


a, b = ("a",), ("b",)


def test_boolean_to_real_cast():
    """A cast reports its algebra on its outputs, not on itself.

    Its inputs and outputs live in *different* algebras, which is exactly the
    thing a single per-module annotation could not express.
    """
    input_shape = SymTensor([with_structure(sym, "boolean") for sym in (a, b)])
    module = build_transform(input_shape, "boolean", "real")

    assert sole_structure(module.get_input_shape()) == "boolean"

    expected_output_shape = SymTensor(
        [with_structure(("transform", ("real",), sym), "real") for sym in input_shape]
    )
    assert module.get_output_shape() == expected_output_shape
    assert sole_structure(module.get_output_shape()) == "real"

    input_tensor = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    torch.testing.assert_close(module(input_tensor), input_tensor)


def test_probability_and_log_probability_cast_into_each_other():
    """The two spellings of a probability convert both ways.

    A formula written in one space can be read in the other, which is what
    makes log space a target of the language rather than a place a circuit can
    only be built in by hand.
    """
    probabilities = torch.tensor([[0.8, 0.25]])
    input_shape = SymTensor([with_structure(sym, "probability") for sym in (a, b)])

    into_log = build_transform(input_shape, "probability", "logprobability")
    logs = into_log(probabilities)
    torch.testing.assert_close(logs, probabilities.log())
    assert sole_structure(into_log.get_output_shape()) == "logprobability"

    back = build_transform(into_log.get_output_shape(), "logprobability", "probability")
    torch.testing.assert_close(back(logs), probabilities)
    assert sole_structure(back.get_output_shape()) == "probability"
