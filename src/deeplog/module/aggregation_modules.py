#  Copyright (c) 2024-2026. KU Leuven
"""Concrete aggregation module implementations."""

from collections.abc import Callable

import torch

from ..algebraic import HasStructure
from ..shape import Shape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..util import as_tuple
from .abstract_aggregation_module import AbstractAggregationModule
from .deeplog_module import SupportsToModule
from .reshape import construct_transformation


class AggregationModule(AbstractAggregationModule):
    """Aggregate an inner module's output over the joint domain of one or more variables.

    For each combination of values from ``domains`` (one tensor per variable
    in ``variables``), the inner module is evaluated with those values bound
    to the corresponding input symbols. The resulting outputs are stacked
    along a new dimension and reduced with ``op`` (default: ``sum``).

    The aggregated variables are removed from this module's input shape —
    they become internal "loop" variables, not external inputs. Any other
    input symbols of the inner module remain as inputs of the aggregation.

    Example (sum aggregation over a single variable ``X`` with a 3-element
    domain): ``output = sum(inner(other_inputs, X=v) for v in domain)``.
    """

    #: Built-in reductions keyed by ``name``. Used as the fallback when ``op``
    #: is not provided to :meth:`__init__`. Each value is a callable that
    #: reduces along dim 1 (the stacked-assignments dim).
    aggregation_operations = {"sum": lambda x: x.sum(dim=1)}

    def __init__(
        self,
        aggregated_module: SupportsToModule,
        variables: list[Symbol],
        domains: list[torch.Tensor],
        name: str,
        op: Callable | None = None,
    ):
        """Configure a joint aggregation over ``variables`` of ``aggregated_module``.

        Args:
            aggregated_module: The inner module whose output is aggregated.
                Must declare each ``variables[i]`` symbol among its inputs.
            variables: Symbols to aggregate over. Must all appear in
                ``aggregated_module.get_input_shape()``.
            domains: Tensor of values per variable, in the same order as
                ``variables``. The joint domain is the Cartesian product.
            name: Identifies the reduction; also used to look up the default
                ``op`` from :attr:`aggregation_operations` if ``op`` is None.
            op: Reduction function applied to the stacked outputs along
                dim 1. Defaults to ``aggregation_operations[name]``.

        Raises:
            ValueError: If any of ``variables`` isn't an input symbol of
                ``aggregated_module``, or if ``len(variables) != len(domains)``.
        """
        child_structure = (
            aggregated_module.get_structure()
            if isinstance(aggregated_module, HasStructure)
            else None
        )
        aggregated_module = aggregated_module.to_module()
        self._validate_binders(variables, domains)
        input_symbols = list(
            dict.fromkeys(get_all_symbols(aggregated_module.get_input_shape()))
        )
        missing = [var for var in variables if var not in input_symbols]
        if missing:
            raise ValueError(f"Missing variables {missing} from {aggregated_module}.")
        self.name = name
        self.op = op or self.aggregation_operations[self.name]

        agg_input_shape = aggregated_module.get_input_shape()
        if isinstance(agg_input_shape, SymTensor):
            input_shape: Shape = self._remaining_input_shape(
                input_symbols, set(variables)
            )
        else:
            input_shape = tuple(
                st
                for st in (
                    SymTensor([sym for sym in st if sym not in set(variables)])
                    for st in agg_input_shape
                )
                if not st.is_empty()
            )

        output_shape = self._binder_output_shape(
            name, variables, aggregated_module.get_output_shape()
        )
        super().__init__(input_shape, output_shape)
        self._init_binders(variables, domains)

        self._pre_transform = construct_transformation(
            as_tuple(input_shape) + tuple(SymTensor(v) for v in variables),
            aggregated_module.get_input_shape(),
        )
        self._aggregated_module = aggregated_module
        if child_structure is not None:
            self.structure = child_structure

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Evaluate the inner module across the joint variable domain and reduce.

        Enumerates every combination of variable assignments, feeds each
        (along with the caller's inputs) through the inner module, stacks
        the results along dim 1, and applies :attr:`op` to that dimension.
        """
        batch_size, flat = self._enumerate_domain(*x)
        x = self._pre_transform(*flat)
        y = self._aggregated_module(*as_tuple(x))
        y = y.view(batch_size, -1, *y.shape[1:])
        return self.op(y)

    def to_graph(self, graph=None):
        """Render the aggregation as a graph node connected to the inner module."""
        graph, in_node, out_node = self._aggregated_module.to_graph(graph)
        _, new_in_node, new_out_node = super().to_graph(graph)
        graph.add_edge(
            new_out_node, in_node, label=str(self._aggregated_module.get_input_shape())
        )
        return graph, new_in_node, out_node
