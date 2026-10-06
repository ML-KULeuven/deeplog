#  Copyright (c) 2024-2026. KU Leuven
from math import log

import pytest
import torch

from deeplog import LogProbabilityPredicate
from deeplog import ProbabilityPredicate
from deeplog import SymTensor
from deeplog import parse_formula_to_module
from deeplog import reshape
from deeplog import with_structure


def _p(*arguments):
    """The atom ``p(*arguments)`` in probability."""
    return with_structure(("p", *arguments), "probability")


def _logp(*arguments):
    """The atom ``logp(*arguments)`` in logprobability."""
    return with_structure(("logp", *arguments), "logprobability")


def test_probability_predicate():
    module = ProbabilityPredicate(
        [
            _p(("A",), ("_", ("p_a",), ("probability",))),
            _p(("B",), ("_", ("0.4",), ("probability",))),
            _p(("C",), ("_", ("p_c",), ("probability",))),
        ]
    )

    inputs = (
        [("A",), ("B",), ("C",)],
        [("_", ("p_a",), ("probability",)), ("_", ("p_c",), ("probability",))],
    )
    assert set(inputs[0]) == set(module.get_input_shape()[0])
    assert set(inputs[1]) == set(module.get_input_shape()[1])
    assert module.get_output_shape() == SymTensor(
        [
            ("_", ("p", ("A",), ("_", ("p_a",), ("probability",))), ("probability",)),
            ("_", ("p", ("B",), ("_", ("0.4",), ("probability",))), ("probability",)),
            ("_", ("p", ("C",), ("_", ("p_c",), ("probability",))), ("probability",)),
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
            _logp(("A",), ("_", ("0.4",), ("probability",))),
            _logp(("B",), ("_", ("0",), ("probability",))),
        ],
    )
    inputs = [("A",), ("B",)]

    assert set(module.get_input_shape()[0]) == set(inputs)
    assert module.get_output_shape() == SymTensor(
        [
            (
                "_",
                ("logp", ("A",), ("_", ("0.4",), ("probability",))),
                ("logprobability",),
            ),
            (
                "_",
                ("logp", ("B",), ("_", ("0",), ("probability",))),
                ("logprobability",),
            ),
        ]
    )

    result = module(torch.tensor([[0.0, 0.0], [1.0, 1.0]]), torch.empty(2, 0))
    expected = torch.tensor([[log(0.6), 0.0], [log(0.4), float("-inf")]])
    torch.testing.assert_close(result, expected)


def test_probability_predicate_plain_constants_match_structured():
    structured = ProbabilityPredicate(
        [
            _p(("A",), ("_", ("0.25",), ("probability",))),
            _p(("B",), ("_", ("0.75",), ("probability",))),
        ]
    )
    unlabeled = ProbabilityPredicate(
        [
            _p(("A",), ("0.25",)),
            _p(("B",), ("0.75",)),
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
        [_logp(("Atom",), ("_", ("0.2",), ("probability",)))],
    )
    unlabeled = LogProbabilityPredicate(
        [_logp(("Atom",), ("0.2",))],
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
            _p(("Atom",), ("_", (str(log(value)),), ("logprobability",))),
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
            _logp(("Atom",), ("_", (str(value),), ("probability",))),
        ],
    )

    input_atoms = torch.tensor([[1.0], [0.0]])
    output = module(input_atoms, torch.empty(2, 0))

    expected = torch.tensor([[log(value)], [log(1 - value)]])
    torch.testing.assert_close(output, expected)


def test_the_language_reaches_both_label_predicates():
    """``p`` and ``logp`` are builders of the language, each in its own space.

    A labelled fact weighs an assignment: the label where the atom is true, its
    complement where the atom is false. The two predicates say the same thing in
    probability and in log-probability space.
    """
    atoms = SymTensor([("A",), ("B",)])
    probability = reshape(
        parse_formula_to_module("p(A,0.8)_probability times p(B,0.3)_probability"),
        atoms,
    )
    logprobability = reshape(
        parse_formula_to_module(
            "logp(A,0.8)_logprobability times logp(B,0.3)_logprobability"
        ),
        atoms,
    )

    assignments = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    weights = torch.tensor([0.8 * 0.3, 0.8 * 0.7, 0.2 * 0.7])

    torch.testing.assert_close(probability(assignments).flatten(), weights)
    torch.testing.assert_close(logprobability(assignments).flatten(), weights.log())


def test_the_atom_p_weighs_is_a_truth_value():
    """``p``'s first argument ranges over the truth values, so a word is refused."""
    with pytest.raises(ValueError, match="burglary is not a value of the domain"):
        ProbabilityPredicate([_p(("burglary",), ("0.8",))])
