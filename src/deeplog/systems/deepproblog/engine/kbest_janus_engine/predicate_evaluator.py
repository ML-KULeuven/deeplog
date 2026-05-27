"""Label → probability evaluator used by :class:`KBestJanusEngine` for ranking.

KBest builds boolean proof formulas via the wrapped factory just like the
non-kbest engines; the only thing it needs from the user *beyond* a factory
is a way to resolve non-numeric labels (e.g. neural-network-backed labels)
to a probability scalar for its heuristic ranking. Numeric-constant labels
already resolve structurally inside
:meth:`~deeplog.systems.deepproblog.engine.engine.EngineFactory.get_scalar_probability`,
so the evaluator only needs to cover the network-backed case.
"""

#  Copyright (c) 2024-2026. KU Leuven

from collections.abc import Mapping

import torch

from deeplog.algebraic import get_algebraic_structure
from deeplog.formula.deeplogmodulefactory.builder_protocols import AtomBuilder
from deeplog.formula.predicates.predicate import Predicate
from deeplog.symbol import Symbol


class NeuralPredicateEvaluator:
    """Evaluator that runs a registered network predicate per non-numeric label.

    Numeric-constant labels resolve via the standard probability-structure
    constant path (delegated to :func:`~deeplog.algebraic.get_algebraic_structure`).
    Compound labels like ``("classifier", ("img1",), ("1",))`` are dispatched
    to the atom builder keyed by ``(functor, arity, "probability")``; the
    resulting :class:`~deeplog.formula.predicates.predicate.Predicate`
    is evaluated eagerly with the supplied tensor mapping.
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
        """Evaluate ``label`` to a probability scalar."""
        constant = get_algebraic_structure("probability").get_constant_value(label)
        if constant is not None:
            return float(constant)
        functor = label[0]
        arity = len(label) - 1
        key = (functor, arity, "probability")
        builder = self._atom_builders.get(key)
        if builder is None:
            raise ValueError(f"No atom builder registered for {key}")
        node = builder([label[1:]])
        if not isinstance(node, Predicate):
            raise ValueError(
                f"Atom builder for {key} produced a {type(node).__name__}, "
                f"expected a Predicate to evaluate eagerly."
            )
        output = node.eager_eval(self._tensors)
        return float(output.reshape(()).item())
