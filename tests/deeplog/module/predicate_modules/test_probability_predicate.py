#  Copyright (c) 2024-2026. KU Leuven
from math import log

import torch

from deeplog import LogProbabilityPredicate
from deeplog import ProbabilityPredicate
from deeplog import SymTensor
from deeplog import reshape


def test_probability_predicate():
    module = ProbabilityPredicate(
        [
            (("a",), ("_", ("p_a",), ("probability",))),
            (("b",), ("_", ("0.4",), ("probability",))),
            (("c",), ("_", ("p_c",), ("probability",))),
        ]
    )

    inputs = (
        [("a",), ("b",), ("c",)],
        [("_", ("p_a",), ("probability",)), ("_", ("p_c",), ("probability",))],
    )
    assert set(inputs[0]) == set(module.get_input_shape()[0])
    assert set(inputs[1]) == set(module.get_input_shape()[1])
    assert module.get_output_shape() == SymTensor(
        [
            ("_", ("p", ("a",), ("_", ("p_a",), ("probability",))), ("probability",)),
            ("_", ("p", ("b",), ("_", ("0.4",), ("probability",))), ("probability",)),
            ("_", ("p", ("c",), ("_", ("p_c",), ("probability",))), ("probability",)),
        ]
    )
    module = reshape(module, input=(SymTensor(inputs[0]), SymTensor(inputs[1])))

    atom_values = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    probability_values = torch.tensor([[0.2, 0.8], [0.1, 0.7]])

    result = module(atom_values, probability_values)
    expected = torch.tensor([[0.2, 0.6, 0.2], [0.9, 0.4, 0.7]])
    torch.testing.assert_close(result, expected)


def test_logprobability_predicate_log_constants():
    module = LogProbabilityPredicate(
        [
            (("a",), ("_", ("0.4",), ("probability",))),
            (("b",), ("_", ("0",), ("probability",))),
        ],
    )
    inputs = [("a",), ("b",)]

    assert set(module.get_input_shape()[0]) == set(inputs)
    assert module.get_output_shape() == SymTensor(
        [
            (
                "_",
                ("logp", ("a",), ("_", ("0.4",), ("probability",))),
                ("logprobability",),
            ),
            (
                "_",
                ("logp", ("b",), ("_", ("0",), ("probability",))),
                ("logprobability",),
            ),
        ]
    )

    # TODO allow to reshape away empty input tensor
    # print(module)
    # module = reshape_input(module, SymTensor(inputs))
    # print(module)

    result = module(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.empty(2, 0))
    expected = torch.tensor([[log(0.6), 0.0], [log(0.4), float("-inf")]])
    torch.testing.assert_close(result, expected)


def test_probability_predicate_plain_constants_match_structured():
    structured = ProbabilityPredicate(
        [
            (("a",), ("_", ("0.25",), ("probability",))),
            (("b",), ("_", ("0.75",), ("probability",))),
        ]
    )
    unlabeled = ProbabilityPredicate(
        [
            (("a",), ("0.25",)),
            (("b",), ("0.75",)),
        ]
    )

    input_atoms = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    empty_probabilities = torch.empty(2, 0)

    torch.testing.assert_close(
        structured(input_atoms, empty_probabilities),
        unlabeled(input_atoms, empty_probabilities),
    )


def test_logprobability_predicate_plain_constants_convert():
    structured = LogProbabilityPredicate(
        [(("atom",), ("_", ("0.2",), ("probability",)))],
    )
    unlabeled = LogProbabilityPredicate(
        [(("atom",), ("0.2",))],
    )

    input_atoms = torch.tensor([[1.0], [0.0]])
    empty_probabilities = torch.empty(2, 0)

    torch.testing.assert_close(
        structured(input_atoms, empty_probabilities),
        unlabeled(input_atoms, empty_probabilities),
    )


def test_probability_predicate_converts_log_labels_to_probability_outputs():
    value = 0.8
    module = ProbabilityPredicate(
        [
            (("atom",), ("_", (str(log(value)),), ("logprobability",))),
        ],
    )

    input_atoms = torch.tensor([[1.0], [0.0]])
    output = module(input_atoms, torch.empty(2, 0))

    expected = torch.tensor([[value], [1 - value]])
    torch.testing.assert_close(output, expected)


def test_logprobability_predicate_converts_probability_labels_for_log_outputs():
    value = 0.65
    module = LogProbabilityPredicate(
        [
            (("atom",), ("_", (str(value),), ("probability",))),
        ],
    )

    input_atoms = torch.tensor([[1.0], [0.0]])
    output = module(input_atoms, torch.empty(2, 0))

    expected = torch.tensor([[log(value)], [log(1 - value)]])
    torch.testing.assert_close(output, expected)
