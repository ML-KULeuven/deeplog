---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Circuit Transformation

A DeepLog circuit is a graph over *one* algebraic structure. **Transforming** it re-reads the same graph in another one — the mechanism behind the `expectation` aggregation, which reads a boolean proof circuit in the probability semiring.

Two entry points, both free functions: `transform_circuit()` works on a raw circuit and node ids, and `transform_nodes()` on `CircuitNode` handles. We start from a boolean circuit for `(a OR b) AND (NOT a OR c)`.

```{code-cell}
import torch
from deeplog import BOOLEAN, PROBABILITY
from deeplog.circuit import Circuit, transform_circuit
from deeplog.formula import CircuitNode, to_module, transform_nodes

bool_circuit = Circuit("boolean")
a, b, c = (bool_circuit.get_leaf_node((name,)) for name in "abc")
or_op, and_op, not_op = (bool_circuit.get_operator(op) for op in ("or", "and", "not"))

clause1 = or_op(a, b)  # (a OR b)
clause2 = or_op(not_op(a), c)  # (NOT a OR c)
root = and_op(clause1, clause2)

bool_module = bool_circuit.to_module({root: ("result",)})
# a=1, b=0, c=1 -> (1 OR 0) AND (NOT 1 OR 1) = 1 AND 1 = 1
print("Boolean result:", bool_module(torch.tensor([[1.0, 0.0, 1.0]])))
```

## Transforming to another structure

Operators map by the **roles** the two structures declare. A structure spells each role it has with an operator name — `AlgebraicStructure.roles` is that table — so the mapping between two of them is simply the intersection: `and` → `times` (product), `or` → `plus` (sum), `not` → `negate` (negation), and `divide` → `divide` (division) between two semifields.

A role the target does not declare is not refused up front. Whether it matters depends on which nodes you are transforming, so it raises only at a node that actually uses that operator. Named constants cross the same way — by the role they play (`identities`), never by the value they happen to have: the additive identity is `("0",)` in `probability` but `("-inf",)` in `logprobability`, where `("0",)` is the *multiplicative* one.

```{code-cell}
prob_circuit, node_map = transform_circuit(bool_circuit, "probability", roots=[root])

print("Source structure:", bool_circuit.structure)
print("Target structure:", prob_circuit.structure)
print()
print("BOOLEAN roles:    ", BOOLEAN.roles)
print("PROBABILITY roles:", PROBABILITY.roles)
for role, source_operator in BOOLEAN.roles.items():
    print(f"  {source_operator:4s} -> {PROBABILITY.roles[role]:6s} ({role})")

prob_module = prob_circuit.to_module({node_map[root]: ("result",)})
# Read as written, in the probability semiring: a=0.8, b=0.3, c=0.6 gives
# (0.8 + 0.3) * ((1 - 0.8) + 0.6) = 1.1 * 0.8 = 0.88. That is a sum, not a
# probability -- reading `or` as a union of overlapping events needs knowledge
# compilation first, which is the weighted model count in `circuits.md`.
probs = torch.tensor([[0.8, 0.3, 0.6]])
print("\nProbability result:", prob_module(probs))
```

## Renaming leaves

`leaf_mapping` renames symbols during the transformation, which is how you keep boolean atoms distinct from their probability counterparts.

```{code-cell}
def bool_to_prob_symbol(symbol):
    """Tag boolean leaf symbols with the probability structure."""
    return ("_", symbol, ("probability",))


mapped_circuit, mapped_nodes = transform_circuit(
    bool_circuit,
    "probability",
    roots=[root],
    leaf_mapping=bool_to_prob_symbol,
)

mapped_module = mapped_circuit.to_module({mapped_nodes[root]: ("result",)})

# Input symbols are now tagged: ('_', ('a',), ('probability',)), etc.
print("Input shape:", mapped_module.get_input_shape())
print("Result:", mapped_module(probs))
```

## Transforming `CircuitNode` handles

A `CircuitNode` is pure data — a `(circuit, node)` handle with no methods of its own — so transforming and compiling one goes through the same free functions, `transform_nodes()` and `to_module()`. Passing several from the same circuit transforms them in a single pass, which traverses a shared subgraph once; a single node is just the one-argument case.

```{code-cell}
transformed = transform_nodes(
    CircuitNode(bool_circuit, clause1),
    CircuitNode(bool_circuit, clause2),
    target_structure="probability",
)

print("Both share the same circuit:", transformed[0].circuit is transformed[1].circuit)

multi_module = to_module(*transformed, names=(("clause1",), ("clause2",)))
print("Output shape:", multi_module.get_output_shape())
print("Results:", multi_module(probs))
```

## A structure that declares no roles

A bare `AlgebraicStructure` has `operator_fns` but no axiom names them, so its `roles` are empty and there is nothing to intersect. Give it an explicit `operator_mapping` instead.

```{code-cell}
from deeplog import AlgebraicStructure

fuzzy = AlgebraicStructure(
    name="fuzzy",
    operator_fns={
        "t_norm": lambda a, b: a * b,
        "t_conorm": lambda x, y: x + y - x * y,
        "complement": lambda x: 1.0 - x,
    },
)
print("fuzzy roles:", fuzzy.roles)

fuzzy_circuit, fuzzy_map = transform_circuit(
    bool_circuit,
    fuzzy,
    roots=[root],
    operator_mapping={"and": "t_norm", "or": "t_conorm", "not": "complement"},
)

fuzzy_module = fuzzy_circuit.to_module({fuzzy_map[root]: ("result",)})
print("Fuzzy result:", fuzzy_module(probs))
```
