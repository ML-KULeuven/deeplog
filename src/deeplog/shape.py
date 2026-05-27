#  Copyright (c) 2024-2026. KU Leuven
"""Symbolic tensor shapes used to validate DeepLog module inputs and outputs."""

import itertools
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from .symbol import Symbol
from .symbol import is_symbol
from .symbol import parse_symbol
from .symbol import symbol_to_pretty_string
from .symbol import symbol_to_str


type SymTensorLike = (
    "SymTensor" | Symbol | str | Sequence["SymTensorLike"] | NDArray[np.object_]
)


class SymTensor:
    """Numpy-backed container that represents symbolic tensor indices."""

    def __init__(
        self,
        symtensor_like: SymTensorLike,
        subtensor_shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Creates a symbolic tensor from a SymTensor-like input.
        If an array-like input of strings is given, then parse_symbol will be used to construct symbols.
        This flexibility can lead to an ambiguous cases, e.g. with symtensor_like = ('a',) which can be
        interpreted as scalar SymTensor of the constant a, or parsed into a 1-dimensionsal symtensor [a].
        In these cases, it will first consider the input to be an array-like of symbols.
        To avoid this ambiguity, use nested lists instead of tuples as input.

        :param symtensor_like: An object that can be cast to a SymTensor.
        :param subtensor_shape: The shape of the sub-tensors.
        """
        self._array: NDArray[np.object_] = SymTensor._symtensor_like_as_array(
            symtensor_like
        )
        # Freeze the backing array so callers obtaining it via `.array` cannot
        # mutate the symbols underneath the cached hash.
        self._array.flags.writeable = False
        self.size = self._array.size
        self.sym_shape = self._array.shape
        self.shape = None
        if subtensor_shape is not None:
            self.shape = self._array.shape + subtensor_shape
        self._hash = hash((self.sym_shape, tuple(self._array.ravel())))

    @property
    def array(self) -> NDArray[np.object_]:
        """The underlying numpy array of symbols."""
        return self._array

    def __eq__(self, other: object) -> bool:
        """Return True when another SymTensor has identical shape and symbols."""
        if isinstance(other, SymTensor):
            return bool(np.array_equal(self._array, other._array))
        return False

    def __array__(self) -> NDArray[np.object_]:
        """Expose the underlying numpy array for interoperability."""
        return self._array

    def __getitem__(self, item: Any) -> Any:
        """Delegate indexing to the underlying numpy array."""
        return self._array[item]

    def __iter__(self) -> Iterator[Symbol]:
        """Iterate over the flattened symbolic entries."""
        yield from cast(Iterator[Symbol], self._array.flat)

    def __len__(self) -> int:
        """Return the size of the leading symbolic dimension (0 for scalars)."""
        return self._array.shape[0] if self._array.ndim else 0

    def __str__(self):
        """Pretty-print the symbolic tensor with shortened arrays and formatted symbols."""
        return np.array2string(
            self._array,
            edgeitems=1,
            threshold=3,
            formatter={"object": lambda x: symbol_to_pretty_string(cast(Symbol, x))},
        )

    def __repr__(self):
        """Represent the symbolic tensor using parsable symbol strings."""
        return np.array2string(
            self._array, formatter={"object": lambda x: symbol_to_str(cast(Symbol, x))}
        )

    def __hash__(self):
        """Return a stable hash derived from shape and symbol contents."""
        return self._hash

    def is_empty(self) -> bool:
        """Return True if this SymTensor contains no symbolic entries."""
        return self.size == 0

    @staticmethod
    def _symtensor_like_as_array(symtensor_like: SymTensorLike) -> NDArray[np.object_]:
        # Trivial case
        if isinstance(symtensor_like, SymTensor):
            return symtensor_like._array  # pylint: disable=protected-access

        # Scalar SymTensors
        if is_symbol(symtensor_like):
            scalar_array = np.empty([], dtype=object)
            scalar_array[...] = symtensor_like
            return scalar_array
        if isinstance(symtensor_like, str):
            return SymTensor._symtensor_like_as_array(parse_symbol(symtensor_like))

        # Runtime guard for type-contract violations (e.g. SymTensor(3.14)),
        # which static type checkers reject but Python lets through.
        # Pyright sees this as redundant because is_symbol is a TypeGuard
        # that doesn't narrow the negative branch (Symbol is a tuple, hence
        # always a Sequence at the static level).
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            symtensor_like, (Sequence, np.ndarray)
        ):
            raise ValueError(
                f"Cannot convert {symtensor_like} of type "
                f"{type(symtensor_like)} to a SymTensor"
            )

        seq = cast(Sequence[Any], symtensor_like)
        # 1-D
        if len(seq) == 0 or is_symbol(seq[0]):
            return np.fromiter(seq, object)
        if isinstance(seq[0], str):
            str_seq = cast(Sequence[str], seq)
            return np.fromiter([parse_symbol(s) for s in str_seq], object)

        # Higher dimensions
        return np.array([SymTensor._symtensor_like_as_array(x) for x in seq])


type Shape = SymTensor | tuple[SymTensor, ...]


def get_all_symbols(shape: Shape | Iterable[Shape]) -> Iterator[Symbol]:
    """Yield every symbol contained in ``shape``."""
    if isinstance(shape, SymTensor):
        yield from iter(shape)
    else:
        yield from itertools.chain.from_iterable(get_all_symbols(x) for x in shape)


def _shape_symbol_count(shape: Shape | Iterable[Shape]) -> int:
    if isinstance(shape, SymTensor):
        return shape.size
    return sum(_shape_symbol_count(x) for x in shape)


def get_only_symbol(shape: Shape | Iterable[Shape]) -> Symbol:
    """Yield the only symbol contained in ``shape``. Raises an error if there is not exactly one."""
    count = _shape_symbol_count(shape)
    if count != 1:
        raise ValueError(f"Shape contains {count} elements.")
    return next(get_all_symbols(shape))


def map_shape(func: Callable[[Symbol], Symbol], shape: Shape) -> Shape:
    """Apply ``func`` to all symbols present in ``shape``."""
    if isinstance(shape, tuple):
        return tuple(cast(SymTensor, map_shape(func, x)) for x in shape)
    ufunc = np.frompyfunc(func, 1, 1)
    return SymTensor(ufunc(shape))


class ShapeMismatchException(Exception):
    """
    This exception is raised when a tensor does not match a Shape.
    """

    def __init__(
        self,
        shape: Shape,
        args: tuple[torch.Tensor],
        location: object = None,
    ):
        """Describe the expected shape vs. received tensor shapes and optional location."""
        expected = shape if isinstance(shape, tuple) else (shape,)
        lines = ["Shape mismatch:"]
        if location is not None:
            lines.append(f"  location: {location}")
        if len(expected) != len(args):
            lines.append(f"  expected {len(expected)} tensor(s), got {len(args)}")
        for i in range(max(len(expected), len(args))):
            sym = expected[i] if i < len(expected) else None
            tensor = args[i] if i < len(args) else None
            lines.extend(self._describe_tensor(i, sym, tensor))
        message = "\n".join(lines)
        super().__init__(message)
        self.shape = shape
        self._args = args
        self.location = location
        self.message = message

    @staticmethod
    def _describe_tensor(
        index: int, sym: "SymTensor | None", tensor: "torch.Tensor | None"
    ) -> list[str]:
        prefix = f"  tensor[{index}]"
        if sym is None:
            assert tensor is not None
            return [f"{prefix}: unexpected, got shape={tuple(tensor.shape)}"]
        if tensor is None:
            return [
                f"{prefix}: missing, expected sym_shape={sym.sym_shape} symbols={sym}"
            ]
        expected_dims = sym.sym_shape
        # _validate_symtensor compares tensor.shape[1:1+N] to expected_dims[:T-1]
        cmp_len = min(len(expected_dims), max(len(tensor.shape) - 1, 0))
        mismatch_dim = None
        for k in range(cmp_len):
            if tensor.shape[1 + k] != expected_dims[k]:
                mismatch_dim = k
                break
        too_few_dims = cmp_len < len(expected_dims)
        if mismatch_dim is None and not too_few_dims:
            return []  # this tensor is fine
        header = (
            f"{prefix}: expected sym_shape={expected_dims} (symbols={sym}),"
            f" got tensor.shape={tuple(tensor.shape)} (dim 0 = batch, unchecked)"
        )
        details: list[str] = [header]
        if mismatch_dim is not None:
            details.append(
                f"    dim {1 + mismatch_dim}: expected size "
                f"{expected_dims[mismatch_dim]}, got {tensor.shape[1 + mismatch_dim]}"
            )
        elif too_few_dims:
            details.append(
                f"    too few dimensions: needs at least "
                f"{1 + len(expected_dims)}, got {len(tensor.shape)}"
            )
        return details


def _validate_symtensor(tensor: torch.Tensor, symtensor: SymTensor) -> bool:
    """Check whether ``tensor`` matches the leading dimensions of ``symtensor``."""
    return (
        tensor.shape[1 : len(symtensor.sym_shape) + 1]
        == symtensor.sym_shape[: len(tensor.shape)]
    )


def correct_shape(args: tuple[torch.Tensor], shape: Shape) -> bool:
    """Return True if the given tensors conform to the symbolic shape."""
    if isinstance(shape, tuple):
        if len(shape) != len(args):
            return False
        return all(
            _validate_symtensor(tensor, symtensor)
            for tensor, symtensor in zip(args, shape, strict=True)
        )
    return (len(args) == 1) and _validate_symtensor(args[0], shape)
