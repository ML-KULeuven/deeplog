#  Copyright (c) 2024-2026. KU Leuven
"""Concrete aggregation module implementations."""

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
from dataclasses import dataclass
from functools import reduce

import torch

from ..algebraic import Algebra
from ..algebraic import Semiring
from ..shape import Shape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..symbol import retag
from ..symbol import without_structure
from ..util import as_tuple
from .abstract_aggregation_module import AbstractAggregationModule
from .deeplog_module import DeepLogModule
from .reshape import construct_transformation


class AggregationModule(AbstractAggregationModule):
    """Reduce an inner module's output over the joint domain of one or more variables.

    For each combination of values from ``domains`` (one tensor per variable
    in ``variables``), the inner module is evaluated with those values bound
    to the corresponding input symbols. Each of its output columns is stacked
    over the combinations along a new dimension 1, and ``op`` reduces the
    stacked columns to one.

    The aggregated variables are removed from this module's input shape —
    they become internal "loop" variables, not external inputs. Any other
    input symbols of the inner module remain as inputs of the aggregation.

    Example (sum aggregation over a single variable ``X`` with a 3-element
    domain): ``output = sum(inner(other_inputs, X=v) for v in domain)``.
    """

    def __init__(
        self,
        aggregated_module: DeepLogModule,
        variables: list[Symbol],
        domains: list[torch.Tensor],
        name: str,
        op: Callable[..., torch.Tensor],
    ):
        """Configure a joint aggregation over ``variables`` of ``aggregated_module``.

        The output is one column, named ``(name, ("binders", *variables),
        *columns)`` for the inner module's columns, and labelled with the
        algebra of the first.

        Args:
            aggregated_module: The inner module whose output is aggregated.
                Must output one symbolic tensor. A variable it does not read
                leaves its output the same at each of that variable's values.
            variables: Symbols to aggregate over.
            domains: Tensor of values per variable, in the same order as
                ``variables``. The joint domain is the Cartesian product.
            name: Identifies the reduction in the output's name.
            op: Receives each output column of the inner module, stacked over
                the joint domain, as a tensor of shape ``(batch, combinations,
                1)``, in column order, and returns their reduction, of shape
                ``(batch, 1)``.

        Raises:
            ValueError: If ``len(variables) != len(domains)``, or if the inner
                module outputs no column or more than one tensor.
        """
        self._validate_binders(variables, domains)
        input_symbols = list(
            dict.fromkeys(get_all_symbols(aggregated_module.get_input_shape()))
        )
        output = aggregated_module.get_output_shape()
        if not isinstance(output, SymTensor) or output.is_empty():
            raise ValueError(
                f"An aggregation reduces the columns of one symbolic tensor, got "
                f"{output!r} from {aggregated_module}."
            )
        self.name = name
        self.op = op

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

        columns = list(get_all_symbols(output))
        reduced = (name, ("binders", *variables), *map(without_structure, columns))
        super().__init__(input_shape, SymTensor([retag(reduced, columns[0])]))
        self._init_binders(variables, domains)

        self._pre_transform = construct_transformation(
            as_tuple(input_shape) + tuple(SymTensor(v) for v in variables),
            aggregated_module.get_input_shape(),
        )
        self._aggregated_module = aggregated_module

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Evaluate the inner module across the joint variable domain and reduce.

        Enumerates every combination of variable assignments, feeds each
        (along with the caller's inputs) through the inner module, stacks
        the results along dim 1, and applies :attr:`op` to its columns.
        """
        batch_size, flat = self._enumerate_domain(*x)
        x = self._pre_transform(*flat)
        y = self._aggregated_module(*as_tuple(x))
        y = y.view(batch_size, -1, *y.shape[1:])
        return self.op(*y.split(1, dim=2))

    def to_graph(self, graph=None):
        """Render the aggregation as a graph node connected to the inner module."""
        graph, in_node, out_node = self._aggregated_module.to_graph(graph)
        _, new_in_node, new_out_node = super().to_graph(graph)
        graph.add_edge(
            new_out_node, in_node, label=str(self._aggregated_module.get_input_shape())
        )
        return graph, new_in_node, out_node


@dataclass(frozen=True, eq=False)
class WeightedVariable:
    """A variable of an expectation's distribution: inputs whose values are drawn together.

    ``inputs`` are the symbols the variable's values are fed under, one per
    binder, and ``domains`` their values, one row per value. The variable's
    named outcomes are the combinations of those rows, in row-major order, and
    ``factor`` weighs each: a module with one column that reads the inputs, or
    ``None`` for a weight of one. A variable with no input has one outcome. With
    ``presences``, one symbol per input, the variable has one more outcome,
    *missing*, weighing the complement of the named outcomes' total; there each
    presence reads zero and each input its first row.
    """

    inputs: tuple[Symbol, ...]
    domains: tuple[torch.Tensor, ...]
    factor: DeepLogModule | None
    presences: tuple[Symbol, ...] = ()


class ExpectationModule(DeepLogModule):
    """Compute expectations of bodies under a distribution over independent variables.

    Each expectation reads one column of ``bodies`` and ranges over some of
    ``variables``: its value is the sum, over their outcomes, of the column at
    those outcomes times the product of their weights, in ``algebra``. Without
    ``samples``, every combination of outcomes is evaluated. With ``samples``,
    that many combinations are drawn per batch row instead, each variable's
    outcome with probability its weight divided by their total; an expectation's
    value is then the mean of its column over the draws, times the totals of its
    variables, and every expectation reads the same draws. A drawn missing
    outcome weighs its complement at no less than zero.

    The variables' inputs and presences are internal, as an
    :class:`AggregationModule`'s binders are: the inputs of this module are those
    of ``bodies`` and the factors, less those.
    """

    def __init__(
        self,
        bodies: DeepLogModule,
        variables: Sequence[WeightedVariable],
        expectations: Sequence[tuple[int, Collection[int]]],
        algebra: Semiring,
        samples: int | None = None,
        gradient: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Configure the expectations of ``bodies``' columns.

        The output is a column per expectation, labelled with the algebra of the
        column of ``bodies`` it reads.

        Args:
            bodies: Outputs one symbolic tensor, a column per body, in
                ``algebra``, and reads the variables' inputs and presences.
            variables: The variables the expectations range over.
            expectations: For each expectation, the column of ``bodies`` it
                reads and the positions in ``variables`` of the variables it
                ranges over.
            algebra: Sums and multiplies the weights, and complements a
                variable's total when it has a missing outcome. Drawing needs
                probabilities.
            samples: The number of combinations drawn per batch row, or ``None``
                to evaluate every combination.
            gradient: When drawing, receives the values each expectation's column
                takes at the draws and the log-probability of drawing them, both
                of shape ``(batch, samples, expectations)``, and returns values
                equal to the first whose gradient estimates the expectations'.

        Raises:
            ValueError: If an expectation ranges over a position not in
                ``variables``, a variable's inputs and domains differ in length or
                it has presences without a complement to weigh them, or
                ``samples`` is given without ``gradient``.
        """
        if any(not 0 <= i < len(variables) for _, over in expectations for i in over):
            raise ValueError("An expectation ranges over a variable it was not given.")
        for variable in variables:
            if len(variable.inputs) != len(variable.domains):
                raise ValueError(f"Each input of {variable} needs a domain.")
            if variable.presences and not isinstance(algebra, Algebra):
                raise ValueError(
                    f"A missing outcome weighs the complement of a total, which "
                    f"'{algebra.name}' does not define."
                )
        if samples is not None and gradient is None:
            raise ValueError("Drawing needs a gradient estimator.")
        internal = {s for v in variables for s in (*v.inputs, *v.presences)}
        inner = [bodies, *(v.factor for v in variables if v.factor is not None)]
        remaining = (
            SymTensor([symbol for symbol in tensor if symbol not in internal])
            for module in inner
            for tensor in as_tuple(module.get_input_shape())
        )
        given = tuple(dict.fromkeys(t for t in remaining if not t.is_empty()))
        input_shape: Shape = given[0] if len(given) == 1 else given
        columns = list(get_all_symbols(bodies.get_output_shape()))
        names = [
            retag(
                (
                    "expectation",
                    ("binders", *(s for i in over for s in variables[i].inputs)),
                    without_structure(columns[column]),
                ),
                columns[column],
            )
            for column, over in expectations
        ]
        super().__init__(input_shape, SymTensor(names))
        self.algebra = algebra
        self.samples = samples
        self.gradient = gradient
        self.variables = list(variables)
        self._bodies = bodies
        self._factors = torch.nn.ModuleList(
            [torch.nn.Identity() if v.factor is None else v.factor for v in variables]
        )
        self._columns = [column for column, _ in expectations]
        self._ranges = [frozenset(over) for _, over in expectations]
        self._rows = [_combinations(v.domains) for v in variables]
        self._routes = torch.nn.ModuleList(
            [
                construct_transformation(
                    given + tuple(SymTensor(s) for s in v.inputs),
                    v.factor.get_input_shape(),
                )
                if v.factor is not None
                else torch.nn.Identity()
                for v in variables
            ]
        )
        self._body_route = construct_transformation(
            given
            + tuple(SymTensor(s) for v in variables for s in (*v.inputs, *v.presences)),
            bodies.get_input_shape(),
        )

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Each expectation's value, a column per expectation."""
        batch = x[0].shape[0] if x else 1
        weights = [self._weigh(i, x, batch) for i in range(len(self.variables))]
        if self.samples is None:
            return self._enumerate(x, weights, batch)
        return self._draw(x, weights, batch)

    def _weigh(self, i: int, x: tuple[torch.Tensor, ...], batch: int) -> torch.Tensor:
        """Variable ``i``'s weight at each of its outcomes, ``(batch, outcomes)``."""
        variable, rows = self.variables[i], self._rows[i]
        named = rows[0].shape[0] if rows else 1
        device = x[0].device if x else None
        if variable.factor is None:
            one = _constant(self.algebra, self.algebra.one)
            weights = torch.full((batch, named), one, device=device)
        else:
            values = [
                d.to(device)[r.to(device)]
                for d, r in zip(variable.domains, rows, strict=True)
            ]
            inputs = self._routes[i](
                *_repeat(x, batch, named), *(_tile(v, batch) for v in values)
            )
            weights = self._factors[i](*as_tuple(inputs))
            weights = weights.expand(batch * named, -1).reshape(batch, named)
        if not variable.presences:
            return weights
        assert isinstance(self.algebra, Algebra)
        total = reduce(self.algebra.sum_fn, weights.unbind(dim=1))
        missing = self.algebra.negation_fn(total)
        return torch.cat([weights, missing.unsqueeze(1)], dim=1)

    def _feed(self, i: int, outcomes: torch.Tensor) -> list[torch.Tensor]:
        """Variable ``i``'s inputs and presences at ``outcomes``, of any shape."""
        variable, rows = self.variables[i], self._rows[i]
        named = rows[0].shape[0] if rows else 1
        present = outcomes < named
        index = torch.where(present, outcomes, 0)
        fed = [
            d.to(outcomes.device)[r.to(outcomes.device)[index]]
            for d, r in zip(variable.domains, rows, strict=True)
        ]
        presence = present.to(torch.get_default_dtype())
        return fed + [presence] * len(variable.presences)

    def _bodies_at(
        self, x: tuple[torch.Tensor, ...], fed: list[torch.Tensor], batch: int, per: int
    ) -> torch.Tensor:
        """The bodies' columns at ``per`` fed assignments per batch row."""
        flat = [f.reshape(batch * per, *f.shape[2:]) for f in fed]
        inputs = self._body_route(*_repeat(x, batch, per), *flat)
        values = self._bodies(*as_tuple(inputs))
        values = values.expand(batch * per, -1).reshape(batch, per, -1)
        return values[..., self._columns]

    def _enumerate(
        self, x: tuple[torch.Tensor, ...], weights: list[torch.Tensor], batch: int
    ) -> torch.Tensor:
        """Every combination of outcomes, each weighed by its variables."""
        counts = [w.shape[1] for w in weights]
        grid = torch.cartesian_prod(
            *(torch.arange(c, device=weights[0].device) for c in counts)
        ).reshape(-1, len(counts))
        per = grid.shape[0]
        fed = [
            f.unsqueeze(0).expand(batch, *f.shape)
            for i in range(len(counts))
            for f in self._feed(i, grid[:, i])
        ]
        values = self._bodies_at(x, fed, batch, per)
        algebra = self.algebra
        one = torch.full_like(weights[0][:, :1], _constant(algebra, algebra.one))
        zero = torch.full_like(one, _constant(algebra, algebra.zero))
        estimates = []
        for j, over in enumerate(self._ranges):
            factors = [
                weights[i][:, grid[:, i]]
                if i in over
                else torch.where(grid[:, i] == 0, one, zero)
                for i in range(len(counts))
            ]
            weighted = algebra.product_fn(
                reduce(algebra.product_fn, factors), values[..., j]
            )
            estimates.append(reduce(algebra.sum_fn, weighted.unbind(dim=1)))
        return torch.stack(estimates, dim=1)

    def _draw(
        self, x: tuple[torch.Tensor, ...], weights: list[torch.Tensor], batch: int
    ) -> torch.Tensor:
        """``samples`` drawn combinations per batch row, averaged."""
        assert self.samples is not None and self.gradient is not None
        fed, logs, totals = [], [], []
        for i, w in enumerate(weights):
            total = w.sum(dim=1, keepdim=True)
            tiny = torch.finfo(w.dtype).tiny
            chances = torch.where(total > 0, w / total.clamp_min(tiny), 1 / w.shape[1])
            index = torch.multinomial(chances.detach(), self.samples, replacement=True)
            logs.append(chances.gather(1, index).log())
            totals.append(total)
            fed += self._feed(i, index)
        values = self._bodies_at(x, fed, batch, self.samples)
        ranges = torch.tensor(
            [[i in over for i in range(len(weights))] for over in self._ranges],
            dtype=values.dtype,
            device=values.device,
        )
        log_probability = torch.stack(logs, dim=-1) @ ranges.T
        scale = torch.where(ranges.bool(), torch.cat(totals, dim=1)[:, None, :], 1)
        estimate = self.gradient(values, log_probability).mean(dim=1)
        return scale.prod(dim=-1) * estimate


def _constant(algebra: Semiring, symbol: Symbol) -> float:
    """The value ``algebra`` gives the constant ``symbol``.

    Raises:
        ValueError: If ``algebra`` gives it none.
    """
    value = algebra.get_constant_value(symbol)
    if value is None:
        raise ValueError(f"'{algebra.name}' gives {symbol} no value.")
    return value


def _combinations(domains: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Each domain's row at every combination of rows, in row-major order."""
    if not domains:
        return []
    grids = torch.meshgrid(*(torch.arange(d.shape[0]) for d in domains), indexing="ij")
    return [g.reshape(-1) for g in grids]


def _repeat(x: tuple[torch.Tensor, ...], batch: int, times: int) -> list[torch.Tensor]:
    """Each of ``x`` with every batch row repeated ``times`` times in place."""
    return [
        t.unsqueeze(1).expand(batch, times, *t.shape[1:]).reshape(-1, *t.shape[1:])
        for t in x
    ]


def _tile(values: torch.Tensor, batch: int) -> torch.Tensor:
    """``values`` once for every batch row, rows of one batch row together."""
    return (
        values.unsqueeze(0).expand(batch, *values.shape).reshape(-1, *values.shape[1:])
    )
