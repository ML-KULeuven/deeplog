"""Label → probability evaluator used by :class:`KBestJanusGrounder` for ranking.

KBest builds boolean proof formulas just like the non-kbest engines; the only
thing it needs from the user is a way to resolve non-numeric labels (e.g.
neural-network-backed labels) to a probability scalar for its heuristic ranking. Numeric-constant labels
already resolve structurally inside
:meth:`~deeplog.systems.deepproblog.kbest.probabilistic_factory.ProbabilisticFactory.get_scalar_probability`,
so the evaluator only needs to cover the network-backed case.
"""

#  Copyright (c) 2024-2026. KU Leuven

from collections.abc import Mapping

import torch

from deeplog import PROBABILITY
from deeplog import AtomBuilder
from deeplog import Symbol
from deeplog import SymTensor
from deeplog import get_all_symbols
from deeplog import reshape
from deeplog import symbol_to_pretty_string
from deeplog import with_structure


class NeuralPredicateEvaluator:
    """Evaluator that runs a registered network predicate per non-numeric label.

    Numeric-constant labels resolve via the standard probability-structure
    constant path (:meth:`~deeplog.algebraic.AlgebraicStructure.get_constant_value`).
    Compound labels like ``("classifier", ("img1",), ("1",))`` are dispatched
    to the atom builder keyed by ``(functor, arity, "probability")``, and the
    label's atom is read, on the supplied tensor mapping, wherever the module
    that builder builds holds it.
    """

    def __init__(
        self,
        tensors: Mapping[Symbol, torch.Tensor],
        atom_builders: Mapping[tuple[str, int, str], AtomBuilder],
    ):
        """Bind a tensor source and atom-builder registry for label dispatch."""
        self._tensors = dict(tensors)
        self._atom_builders = dict(atom_builders)

    def __call__(self, label: Symbol) -> float:
        """Evaluate ``label`` to a probability scalar.

        The module built for a label's atom may hold it anywhere among its
        outputs, beside other atoms. It runs on one example, each symbol it
        reads taken from the tensors.

        Raises:
            ValueError: If no atom builder is registered for ``label``, or the
                module it builds holds no atom ``label``.
            KeyError: If the module reads a symbol the tensors do not hold.
        """
        constant = PROBABILITY.get_constant_value(label)
        if constant is not None:
            return float(constant)
        functor = label[0]
        arity = len(label) - 1
        key = (functor, arity, "probability")
        builder = self._atom_builders.get(key)
        if builder is None:
            raise ValueError(f"No atom builder registered for {key}")
        atom = with_structure(label, PROBABILITY.name)
        module = builder([atom])
        if atom not in set(get_all_symbols(module.get_output_shape())):
            raise ValueError(
                f"The atom builder for {key} built a module that holds no "
                f"{symbol_to_pretty_string(label)}."
            )
        read = list(get_all_symbols(module.get_input_shape()))
        reader = reshape(
            module,
            input=tuple(SymTensor([symbol]) for symbol in read),
            output=SymTensor([atom]),
        )
        output = reader(*(self._tensors[symbol][None, None] for symbol in read))
        return float(output.reshape(()).item())
