#  Copyright (c) 2024-2026. KU Leuven
"""ModuleCircuit implementation and helpers."""

import math
import operator
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from graphlib import TopologicalSorter
from typing import cast

import torch
import torch.nn

from ..shape import Shape
from ..shape import SymTensor
from ..shape import get_all_symbols
from ..symbol import Symbol
from ..symbol import free_variables
from ..symbol import symbol_to_pretty_string
from ..symbol import symbol_to_str
from ..util import as_tuple
from .deeplog_module import DeepLogModule
from .reshape import construct_transformation
from .reshape import reshape


def _get_missing_transformations(
    input_tensors: tuple[SymTensor, ...],
    modules: Iterable[DeepLogModule],
    required_shapes: Iterable[SymTensor] = (),
) -> list[DeepLogModule]:
    """Determine extra transformation modules needed so every symbolic input has a producer.

    A transformation takes each output it reads from once, however many of
    that output's symbols it reads.
    """
    all_inputs: set[SymTensor] = reduce(
        operator.or_, (set(as_tuple(m.get_input_shape())) for m in modules)
    ) | set(required_shapes)
    all_outputs: set[SymTensor] = reduce(
        operator.or_, (set(as_tuple(m.get_output_shape())) for m in modules)
    ) | set(input_tensors)
    missing_symtensors = all_inputs - all_outputs

    #: The first output, in ``all_outputs``' order, holding each symbol.
    producer: dict[Symbol, SymTensor] = {}
    for output in all_outputs:
        for symbol in output:
            producer.setdefault(symbol, output)

    all_transformation_inputs: list[tuple[list[SymTensor], SymTensor]] = []
    transformations = []
    for symtensor in missing_symtensors:
        transformation_inputs: dict[SymTensor, None] = {}
        for symbol in symtensor:
            if symbol not in producer:
                raise ValueError(
                    f"Symbol {symbol} required by {symtensor} is not produced by "
                    f"any module or provided as an external input."
                )
            transformation_inputs[producer[symbol]] = None
        all_transformation_inputs.append((list(transformation_inputs), symtensor))

    for inputs, tensor in all_transformation_inputs:
        if len(inputs) == 1:
            transformations.append(construct_transformation(inputs[0], tensor))
        if len(inputs) > 1:
            transformations.append(construct_transformation(tuple(inputs), tensor))

    return transformations


def _external_shapes(
    symbols: Iterable[Symbol], modules: Iterable[DeepLogModule]
) -> tuple[SymTensor, ...]:
    """The shape to declare for each of ``symbols`` as an external input of a circuit.

    A one-symbol tensor has two spellings — ``SymTensor([symbol])`` and the
    scalar ``SymTensor(symbol)``, which differ because a symbol is itself a
    tuple (the ambiguity :class:`~deeplog.shape.SymTensor` documents) — and both
    occur among modules. Declaring the one its consumer declared is what lets the
    circuit hand the caller's tensor straight to it; declaring the other earns a
    transformation that exists only to reconcile the two spellings, and one of
    them changes the tensor's rank. Where nothing consumes the symbol alone
    there is no consumer to agree with, and the one-element spelling stands.
    """
    alone: dict[Symbol, SymTensor] = {}
    for module in modules:
        for shape in as_tuple(module.get_input_shape()):
            held = tuple(get_all_symbols(shape))
            if len(held) == 1:
                alone.setdefault(held[0], shape)
    return tuple(alone.get(symbol, SymTensor([symbol])) for symbol in symbols)


def build_module_graph(modules: list[DeepLogModule]) -> dict[int, set[int]]:
    """Build a predecessor map from module index to the indices producing its inputs."""
    inputs_by_shape: dict[Shape, list[int]] = defaultdict(list)
    outputs_by_shape: dict[Shape, list[int]] = defaultdict(list)

    for i, module in enumerate(modules):
        for s in as_tuple(module.get_input_shape()):
            inputs_by_shape[s].append(i)
        for s in as_tuple(module.get_output_shape()):
            outputs_by_shape[s].append(i)

    predecessors: dict[int, set[int]] = {i: set() for i in range(len(modules))}
    for s, producers in outputs_by_shape.items():
        for j in inputs_by_shape.get(s, []):
            predecessors[j].update(p for p in producers if p != j)

    return predecessors


class ModuleCircuit(DeepLogModule):
    """Module that evaluates a circuit of DeepLogModules."""

    _sub_modules: torch.nn.ModuleList

    def __init__(
        self,
        modules: Iterable[DeepLogModule],
        output_shape: Shape,
        input_shape: Shape | None = None,
    ):
        """Assemble a circuit of modules and compute ordering/transformations.

        The circuit's inputs are ``input_shape``, which must hold no symbol a
        module produces; without it, they are the symbols no module produces,
        one tensor each, in the order ``modules`` first read them.

        The circuit ranges over the variables its modules bind: those a module's
        output holds and none of its inputs do. A value whose symbols hold one has
        an axis for it after its batch, and gives it, as an output, behind its
        symbols.
        """
        all_modules = list(modules)

        if input_shape is None:
            produced = set(get_all_symbols(m.get_output_shape() for m in all_modules))
            input_symbols = [
                symbol
                for symbol in dict.fromkeys(
                    get_all_symbols(m.get_input_shape() for m in all_modules)
                )
                if symbol not in produced
            ]
            input_shape = _external_shapes(input_symbols, all_modules)
        input_tensors = as_tuple(input_shape)
        transform_modules = _get_missing_transformations(
            input_tensors, all_modules, as_tuple(output_shape)
        )
        super().__init__(input_shape, output_shape)
        all_modules += transform_modules

        sorted_indices = list(
            TopologicalSorter(build_module_graph(all_modules)).static_order()
        )
        all_modules = [all_modules[i] for i in sorted_indices]
        self._sub_modules = torch.nn.ModuleList(all_modules)
        #: The variables this circuit ranges over. A value holding one has its axis.
        self._ranges = _bound(all_modules)
        self._plans = [self._plan(module) for module in all_modules]
        self._given = [self._held(shape) for shape in input_tensors]
        self._returned = [
            (
                held := self._held(shape),
                tuple(held.index(v) for v in _variables((shape,)) if v in held),
                len(shape.sym_shape),
            )
            for shape in as_tuple(output_shape)
        ]

    def _held(self, shape: SymTensor) -> tuple[Symbol, ...]:
        """The variables this circuit ranges over that ``shape`` holds, in its order."""
        if not self._ranges:
            return ()
        held = set(_variables((shape,)))
        return tuple(variable for variable in self._ranges if variable in held)

    def _plan(self, module: DeepLogModule) -> tuple:
        """How ``module`` is called: the variables folded into its batch, and per tensor the axes.

        The variables both its inputs and its outputs hold are folded into the
        batch. An input holds behind its symbols an axis for each variable it
        holds that no output does, which the module removes, and an output one
        for each it holds that no input does, which the module adds, in the
        order they occur in that tensor's symbols.
        """
        inputs = as_tuple(module.get_input_shape())
        outputs = as_tuple(module.get_output_shape())
        if not self._ranges:
            return inputs, outputs, True, (), [], []
        ranged = set(self._ranges)
        read = {v for s in inputs for v in _variables((s,)) if v in ranged}
        made = {v for s in outputs for v in _variables((s,)) if v in ranged}
        folded = tuple(v for v in self._ranges if v in read and v in made)
        feeds = [
            (
                self._held(shape),
                tuple(v for v in _variables((shape,)) if v in read and v not in made),
                len(shape.sym_shape),
            )
            for shape in inputs
        ]
        gives = [
            (
                self._held(shape),
                tuple(v for v in _variables((shape,)) if v in made and v not in read),
                len(shape.sym_shape),
            )
            for shape in outputs
        ]
        plain = (
            not folded and not any(f[1] for f in feeds) and not any(g[1] for g in gives)
        )
        return inputs, outputs, plain, folded, feeds, gives

    def forward(self, *inputs: torch.Tensor):
        """Evaluate the circuit by propagating cacheable tensors through each submodule.

        A value holding variables the circuit ranges over has, after its batch,
        an axis for each, in the circuit's order of them.
        """
        cache = {}
        # The empty input stands in for the batch, on the inputs' device. With no
        # runtime inputs every leaf was baked to a constant, so there is no tensor
        # to read the batch from; evaluate a single constant row.
        batch, device = (inputs[0].shape[0], inputs[0].device) if inputs else (1, None)
        cache[SymTensor([])] = torch.zeros(batch, 0, device=device)
        for shape, tensor, held in zip(
            as_tuple(self.get_input_shape()), inputs, self._given, strict=True
        ):
            cache[shape] = (
                tensor.reshape(batch, *(1 for _ in held), *tensor.shape[1:])
                if held
                else tensor
            )

        for module, (reads, writes, plain, folded, feeds, gives) in zip(
            cast(Iterable[DeepLogModule], self._sub_modules), self._plans, strict=True
        ):
            if plain:
                output = module(*(cache[shape] for shape in reads))
                for shape, o in zip(writes, as_tuple(output), strict=True):
                    cache[shape] = o
                continue
            arranged = [
                (_arrange(cache[shape], held, folded + removed), removed, symbolic)
                for shape, (held, removed, symbolic) in zip(reads, feeds, strict=True)
            ]
            # A value the same along a variable has an axis of one there.
            rows = [1] * len(folded)
            for value, _, _ in arranged:
                for k in range(len(folded)):
                    if value.shape[1 + k] != 1:
                        rows[k] = value.shape[1 + k]
            fed = []
            for value, removed, symbolic in arranged:
                if folded:
                    value = value.expand(
                        value.shape[0], *rows, *value.shape[1 + len(folded) :]
                    ).reshape(batch * math.prod(rows), *value.shape[1 + len(folded) :])
                fed.append(_behind(value, len(removed), symbolic))
            output = module(*fed)
            for shape, o, (held, added, symbolic) in zip(
                writes, as_tuple(output), gives, strict=True
            ):
                if o.dim() < 1 + symbolic + len(added):
                    raise ValueError(_no_axis(module, shape, added))
                o = _ahead(o, len(added), symbolic)
                if folded:
                    o = o.reshape(batch, *rows, *o.shape[1:])
                axes = folded + added
                for k in reversed(range(len(folded))):
                    if folded[k] not in held:
                        o = o.select(1 + k, 0)
                axes = tuple(v for v in axes if v in held)
                if axes != held:
                    o = o.permute(
                        0,
                        *(1 + axes.index(v) for v in held),
                        *range(1 + len(held), o.dim()),
                    )
                cache[shape] = o

        returned = [
            _behind(
                cache[shape].permute(
                    0,
                    *(1 + k for k in order),
                    *range(1 + len(held), cache[shape].dim()),
                )
                if held
                else cache[shape],
                len(order),
                symbolic,
            )
            for shape, (held, order, symbolic) in zip(
                as_tuple(self.get_output_shape()), self._returned, strict=True
            )
        ]
        if isinstance(self.get_output_shape(), tuple):
            return tuple(returned)
        return returned[0]


def _bound(modules: Iterable[DeepLogModule]) -> tuple[Symbol, ...]:
    """The variables some module's output holds and none of its inputs do."""
    bound = {}
    for module in modules:
        read = set(_variables(as_tuple(module.get_input_shape())))
        for variable in _variables(as_tuple(module.get_output_shape())):
            if variable not in read:
                bound[variable] = None
    return tuple(sorted(bound, key=symbol_to_str))


def _no_axis(
    module: DeepLogModule, shape: SymTensor, variables: tuple[Symbol, ...]
) -> str:
    """Why ``module`` cannot give ``shape``: it binds ``variables`` but has no axis for them."""
    names = ", ".join(symbol_to_pretty_string(v) for v in variables)
    columns = ", ".join(
        name[:1].lower() + name[1:]
        for name in (symbol_to_str(v).lstrip("_") for v in variables)
    )
    return (
        f"{type(module).__name__} giving {shape} binds {names}, which its output "
        "holds and its inputs do not, but gives no axis behind its symbols for "
        "each. A capitalised name is a variable; if one names a column, write it in "
        f"lowercase ({columns})."
    )


def _variables(shapes: Iterable[SymTensor]) -> tuple[Symbol, ...]:
    """The variables free in the symbols of ``shapes``, in the order they first occur."""
    return tuple(
        dict.fromkeys(
            variable
            for symbol in get_all_symbols(shapes)
            for variable in free_variables(symbol)
        )
    )


def _arrange(
    value: torch.Tensor, held: tuple[Symbol, ...], wanted: tuple[Symbol, ...]
) -> torch.Tensor:
    """``value``, with an axis per ``held`` variable, given one per ``wanted`` variable.

    Its axes are put in ``wanted``'s order, and one it does not hold is added
    with a size of one.
    """
    if not held and not wanted:
        return value
    present = [v for v in wanted if v in held]
    value = value.permute(
        0, *(1 + held.index(v) for v in present), *range(1 + len(held), value.dim())
    )
    for k, variable in enumerate(wanted):
        if variable not in held:
            value = value.unsqueeze(1 + k)
    return value


def _behind(value: torch.Tensor, count: int, symbolic: int) -> torch.Tensor:
    """``value``, its ``count`` axes after the batch moved behind its ``symbolic`` dimensions."""
    if not count:
        return value
    return value.movedim(
        tuple(range(1, 1 + count)), tuple(range(1 + symbolic, 1 + symbolic + count))
    )


def _ahead(value: torch.Tensor, count: int, symbolic: int) -> torch.Tensor:
    """``value``, the ``count`` axes behind its ``symbolic`` dimensions moved after the batch."""
    if not count:
        return value
    return value.movedim(
        tuple(range(1 + symbolic, 1 + symbolic + count)), tuple(range(1, 1 + count))
    )


def compose_modules(
    modules: Iterable[DeepLogModule],
    output_shape: Shape,
    input_shape: Shape | None = None,
) -> DeepLogModule:
    """Compose modules into one module taking ``input_shape`` and producing ``output_shape``.

    Each symbol is produced by at most one module. The composed module's inputs
    are the symbols no module produces. ``input_shape`` lays them out, and must
    hold each of them and none a module produces; without it, each is a tensor of
    its own, in the order the modules first read them. A single module is
    reshaped to the requested shapes, and returned as it is when it already has
    them; several become a :class:`ModuleCircuit`, which orders them by what each
    one's inputs need and inserts the transformations between them.

    A capitalised name is a variable. A module whose output holds a variable
    none of its inputs hold binds it, and gives its values along an axis behind
    its symbols. A value holding the variable then has an axis for it after its
    batch, and a module reading such a value runs on a row per value of it.
    Calling the composed module raises :class:`ValueError` where a module
    binding a variable gives no axis for it.

    Raises:
        ValueError: If no module is given, two modules produce one symbol, or
            ``input_shape`` holds a symbol a module produces.
    """
    modules = list(modules)
    if len(modules) == 0:
        raise ValueError("At least one module is required to compose_modules.")
    produced: set[Symbol] = set()
    for module in modules:
        made = list(get_all_symbols(module.get_output_shape()))
        if twice := [symbol for symbol in made if symbol in produced]:
            raise ValueError(
                f"Two modules produce {symbol_to_pretty_string(twice[0])}; a "
                "composition reads each symbol from one module, so name their "
                "outputs apart."
            )
        produced.update(made)
    if input_shape is not None:
        if clash := [s for s in get_all_symbols(input_shape) if s in produced]:
            raise ValueError(
                f"input_shape holds {clash}, which the modules produce; a symbol "
                "is either an input of the composed module or produced in it."
            )
    if len(modules) == 1:
        return reshape(modules[0], input=input_shape, output=output_shape)
    # TODO: potential optimization. If dependency graph is linear, return a sequential.
    return ModuleCircuit(modules, output_shape, input_shape)
