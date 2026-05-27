#  Copyright (c) 2024-2026. KU Leuven
"""Abstract base class for aggregation modules."""

from abc import ABC

import torch

from ..shape import Shape
from ..shape import SymTensor
from ..shape import map_shape
from ..symbol import Symbol
from ..util import broadcast_tensors
from .deeplog_module import DeepLogModule


class AbstractAggregationModule(DeepLogModule, ABC):
    """Base class for modules that aggregate over bound variables and domains."""

    structure: str

    @staticmethod
    def _validate_binders(variables: list[Symbol], domains: list[torch.Tensor]) -> None:
        """Check that binder variables and domains are valid."""
        if len(variables) == 0:
            raise ValueError("Aggregation requires at least one binder.")
        if len(variables) != len(domains):
            raise ValueError("Each variable needs a matching domain.")

    @staticmethod
    def _binder_output_shape(
        name: str, variables: list[Symbol], child_output_shape: Shape
    ) -> Shape:
        """Wrap child output shape with the aggregation binder symbol."""
        binder_symbol = ("binders", *variables)
        return map_shape(lambda x: (name, binder_symbol, x), child_output_shape)

    @staticmethod
    def _remaining_input_shape(
        all_symbols: list[Symbol], variable_set: set[Symbol]
    ) -> Shape:
        """Compute input shape by filtering out binder variables."""
        remaining = [s for s in all_symbols if s not in variable_set]
        return SymTensor(remaining) if remaining else ()

    def _init_binders(
        self, variables: list[Symbol], domains: list[torch.Tensor]
    ) -> None:
        """Store variables and domains. Call after ``super().__init__``."""
        self.variables = list(variables)
        self._variable_set = set(variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}
        self.domains = [domains[var_to_idx[v]] for v in variables]

    def _enumerate_domain(
        self, *x: torch.Tensor
    ) -> tuple[int, tuple[torch.Tensor, ...]]:
        """Expand inputs across domain values and flatten for child evaluation."""
        batch_size = x[0].shape[0] if x else 1
        expanded_x = [t.unsqueeze(1) for t in x]
        expanded_domains = [
            d.unsqueeze(0).repeat([batch_size] + [1] * len(d.shape))
            for d in self.domains
        ]
        all_tensors = broadcast_tensors(*expanded_x, *expanded_domains)
        flat = tuple(t.view(-1, *t.shape[2:]) for t in all_tensors)
        return batch_size, flat

    def get_structure(self) -> str:
        """Return the structure assigned to this module."""
        if not hasattr(self, "structure"):
            raise AttributeError("Structure not set on aggregation module.")
        return self.structure
