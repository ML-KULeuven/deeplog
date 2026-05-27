#  Copyright (c) 2024-2026. KU Leuven
"""
Utility DeepLog modules that support test scenarios.
"""

from __future__ import annotations

import torch
from torch import nn


class IndexClassifier(nn.Module):
    """
    Deterministic classifier that concentrates probability mass on a provided class index.

    The module expects inputs shaped ``(batch_size, 1)`` containing (possibly floating point)
    encodings of the target class index. During the forward pass it returns a probability
    distribution where ``peak_probability`` is assigned to the indicated index and the remaining
    mass is distributed evenly across all other classes.
    """

    def __init__(self, num_classes: int, peak_probability: float = 0.9):
        """
        Parameters
        ----------
        num_classes:
                Number of classes in the simulated classifier. Must be positive.
        peak_probability:
                Probability mass allocated to the provided class index. The remainder is distributed
                uniformly over the other classes. Must lie in ``(0, 1]`` and be strictly greater than
                ``1 / num_classes`` whenever ``num_classes > 1`` to ensure a dominant class.
        """
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if not (0.0 < peak_probability <= 1.0):
            raise ValueError("peak_probability must lie in (0, 1].")
        if num_classes > 1 and peak_probability <= 1.0 / num_classes:
            raise ValueError(
                "peak_probability must be greater than 1/num_classes for a dominant class."
            )

        self.num_classes = num_classes
        self.peak_probability = peak_probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Produce per-class probabilities aligned with the provided class indices.

        Parameters
        ----------
        inputs:
                Tensor of shape ``(batch_size,)`` containing class indices (floats or integers).

        Returns
        -------
        torch.Tensor
                Tensor of shape ``(batch_size, num_classes)`` holding the simulated probability
                distributions.
        """

        batch_size = inputs.shape[0]
        device = inputs.device

        dtype = inputs.dtype if torch.is_floating_point(inputs) else torch.float32
        indices = inputs.to(torch.long)

        if self.num_classes == 1:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)

        rest_mass = (1.0 - self.peak_probability) / (self.num_classes - 1)
        probs = torch.full(
            (batch_size, self.num_classes), rest_mass, device=device, dtype=dtype
        )
        probs.scatter_(1, indices.unsqueeze(-1), self.peak_probability)
        return probs
