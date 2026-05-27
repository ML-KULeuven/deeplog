"""
A module that contains various utility functions.
"""

import os
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Reversible

#  Copyright (c) 2024-2026. KU Leuven
from functools import reduce
from typing import cast

import torch


def bracket_aware_split(
    s: str, split_str: str, l_bracket: str = "([", r_bracket: str = ")]"
) -> Iterable[str]:
    """
    A simple function that performs similar to the string split function, but does not split when it is inside brackets.
    :param s: The string to split.
    :param split_str:  The string to split on.
    :param l_bracket: The left-handed brackets to take into account.
    :param r_bracket: right-handed brackets to take into account.
    :return: An iterable over all string splits.
    """
    depth = 0
    current_string = ""
    for sym in s:
        if depth == 0 and sym == split_str:
            yield current_string
            current_string = ""
        else:
            current_string += sym
            if sym in l_bracket:
                depth += 1
            elif sym in r_bracket:
                depth -= 1
    yield current_string


def foldr[T](func: Callable[[T, T], T], iterable: Reversible[T]) -> T:
    """
    A foldr implementation.
    """
    return reduce(lambda x, y: func(y, x), reversed(iterable))


def as_tuple[T](item: T | tuple[T, ...]) -> tuple[T, ...]:
    """
    Returns the input as a tuple. If it is already a tuple, it returns it unchanged. If it is not a tuple, it returns
    a tuple with the item as its only element.
    """
    if isinstance(item, tuple):
        return cast(tuple[T, ...], item)
    return (item,)


def broadcast_tensors(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Broadcasts the tensors across each other."""
    result: list[torch.Tensor] = []
    total_length = 1
    for tensor in tensors:
        result = [
            t.repeat(1, tensor.shape[1], *[1] * (len(t.shape) - 2)) for t in result
        ]
        result.append(tensor.repeat_interleave(total_length, dim=1))
        total_length *= tensor.shape[1]
    return tuple(result)


def fast_dev_run_enabled() -> bool:
    """Return whether notebooks should use Lightning's fast_dev_run mode."""
    value = os.getenv("DEELOG_FAST_DEV_RUN", "")
    return value.lower() in {"1", "true", "yes", "on"}
