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

# Circuits in DeepLog

A [Circuit](deeplog.circuit.circuit.Circuit) is a DAG over a single algebraic structure. `to_module()` turns it into a [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) whose inputs are the circuit's leaves.

```{code-cell} ipython3
import torch

from deeplog.circuit import Circuit


circuit = Circuit("boolean")
a, b, c = (circuit.get_leaf_node((name,)) for name in "abc")
and_node = circuit.get_operator("and")(a, b)
or_node = circuit.get_operator("or")(c, and_node)

module = circuit.to_module({or_node: ("or_root",), and_node: ("and_root",)})
print(module)
print(module(torch.tensor([[1.0, 0.0, 1.0]])))  # a = 1, b = 0, c = 1
```

Each leaf is named by a symbol, `circuit.get_leaf_node(symbol)`, and an operator is looked up by name, `circuit.get_operator(name)`, among those the circuit's algebra defines: `and`, `or` and `not` for `boolean`, and `times`, `plus`, `negate` and `divide` for `probability`, which is a semifield.

```{code-cell} ipython3
from deeplog import BOOLEAN, LOGPROBABILITY, PROBABILITY


print("Boolean operators:", BOOLEAN.operators)
print("Probability operators:", PROBABILITY.operators)
print("LogProbability operators:", LOGPROBABILITY.operators)
```

Combining nodes builds the graph. `to_module` takes the roots to compute, each with the name of its output column.

The same in the `probability` structure:

```{code-cell} ipython3
circuit = Circuit("probability")
a, b = (circuit.get_leaf_node((name,)) for name in "ab")
total = circuit.get_operator("plus")(a, b)

module = circuit.to_module({total: ("total",)})
print(module(torch.tensor([[0.4, 0.25]])))  # a = 0.4, b = 0.25
```

## Weighted model counts

The `probability` structure uses `plus` and `times` as semiring operators, so the circuit above is read exactly as written: `plus(a, b)` is the sum `0.4 + 0.25`, which is the probability of "a or b" only if the two events are disjoint.

The other reading, where `plus` is a union of events that may overlap, is a *weighted model count*, and it is two ordinary rewrites rather than a mode you switch on. `knowledge_compile` rewrites the formula into an equivalent one that is deterministic (a disjunction's branches are mutually exclusive) and decomposable (a conjunction's operands share no variables); `transform_circuit` then maps its operators onto the probability semiring, which is exact *because* of those two properties. The result is an ordinary arithmetic circuit, evaluated as written like any other.

The two are spelled separately below to keep them visible. Passing `structure=` to `knowledge_compile` fuses them: it emits the compiled diagram straight into the target algebra, so the boolean circuit in between is never built.

```{code-cell} ipython3
from deeplog.circuit import knowledge_compile, transform_circuit


# Write the formula in boolean...
circuit = Circuit("boolean")
a, b = (circuit.get_leaf_node((name,)) for name in "ab")
either = circuit.get_operator("or")(a, b)

# ...knowledge-compile it, so that reading `or` as `plus` is exact...
compiled, compiled_map = knowledge_compile(circuit, [either])

# ...and only then map it into probability.
counted, counted_map = transform_circuit(compiled, "probability", [compiled_map[either]])

module = counted.to_module({counted_map[compiled_map[either]]: ("either",)})
print(module(torch.tensor([[0.4, 0.25]])))  # 0.25 + (1 - 0.25) * 0.4 = 0.55
```

`transform_circuit` maps operators by the roles both algebras declare, so the product is the same whether it is spelled `and` or `times`. A bare [`AlgebraicStructure`](deeplog.algebraic.AlgebraicStructure) declares no roles, so a transform into one takes an explicit `operator_mapping`.
