#  Copyright (c) 2024-2026. KU Leuven
"""Benchmark for generic vs klay circuit evaluators."""

import time

import torch

from deeplog.algebraic import AlgebraicStructure
from deeplog.circuit import Circuit


def _make_klay_circuit(depth: int, num_leaves: int = 4):
    """Build a klay-compatible circuit (and/or only)."""
    circuit = Circuit("boolean")
    leaves = [circuit.get_leaf_node((f"x{i}",)) for i in range(num_leaves)]
    and_op = circuit.get_operator("and")
    or_op = circuit.get_operator("or")

    node = and_op(leaves[0], leaves[1])
    for i in range(2, num_leaves):
        node = or_op(node, and_op(leaves[i - 1], leaves[i]))
    for _ in range(depth):
        node = and_op(leaves[0], node)

    return circuit, node, num_leaves


def _make_generic_circuit(depth: int, num_leaves: int = 4):
    """Build a generic circuit (has 'implies', so not klay-compatible)."""
    fuzzy = AlgebraicStructure(
        name="fuzzy",
        operator_fns={
            "and": lambda a, b: a * b,
            "or": lambda a, b: a + b - a * b,
            "not": lambda x: 1.0 - x,
            "implies": lambda a, b: 1.0 - a + a * b,
        },
    )
    circuit = Circuit(fuzzy)
    leaves = [circuit.get_leaf_node((f"x{i}",)) for i in range(num_leaves)]
    and_op = circuit.get_operator("and")
    implies = circuit.get_operator("implies")

    node = and_op(leaves[0], leaves[1])
    for i in range(2, num_leaves):
        node = implies(node, and_op(leaves[i - 1], leaves[i]))
    for _ in range(depth):
        node = and_op(leaves[0], node)

    return circuit, node, num_leaves


def bench_build(label: str, make_fn, depth: int, n_iters: int = 100):
    # Warmup
    for _ in range(3):
        c, r, n = make_fn(depth)
        c.to_module({r: ("out",)})

    t0 = time.perf_counter()
    for _ in range(n_iters):
        c, r, n = make_fn(depth)
        c.to_module({r: ("out",)})
    elapsed = time.perf_counter() - t0

    us_per_call = elapsed / n_iters * 1e6
    print(f"  {label:20s}  depth={depth:4d}  {us_per_call:10.1f} µs/build")


def bench_eval(label: str, module, num_leaves: int, batch: int, n_iters: int = 500):
    x = torch.rand(batch, num_leaves)

    # Warmup
    for _ in range(20):
        module(x)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        module(x)
    elapsed = time.perf_counter() - t0

    us_per_call = elapsed / n_iters * 1e6
    print(
        f"  {label:20s}  depth={depth:4d}  batch={batch:5d}  {us_per_call:8.1f} µs/eval"
    )


if __name__ == "__main__":
    print("BUILD TIME")
    print("=" * 70)
    for depth in [10, 50, 200]:
        bench_build("klay", _make_klay_circuit, depth)
        bench_build("generic", _make_generic_circuit, depth)
    print()

    print("EVAL TIME")
    print("=" * 70)
    for depth in [10, 50, 200]:
        for batch in [1, 64, 1024]:
            kc, kr, kn = _make_klay_circuit(depth)
            gc, gr, gn = _make_generic_circuit(depth)

            k_mod = kc.to_module({kr: ("out",)})
            g_mod = gc.to_module({gr: ("out",)})
            k_compiled = torch.compile(k_mod)
            g_compiled = torch.compile(g_mod)

            bench_eval("klay", k_mod, kn, batch)
            bench_eval("klay+compile", k_compiled, kn, batch)
            bench_eval("generic", g_mod, gn, batch)
            bench_eval("generic+compile", g_compiled, gn, batch)
        print()
