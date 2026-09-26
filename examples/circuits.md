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

A [Circuit](deeplog.circuit.circuit.Circuit) is a DAG over a single algebraic structure. After construction, `to_module()` bundles the circuit into a [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) whose inputs are the circuit's own leaves.

```{code-cell} ipython3
import torch

from deeplog.circuit import Circuit


# Construct circuit
circuit = Circuit("boolean")
symbol_a = ("a",)
symbol_b = ("b",)
symbol_c = ("c",)
leaf_a = circuit.get_leaf_node(symbol_a)
leaf_b = circuit.get_leaf_node(symbol_b)
leaf_c = circuit.get_leaf_node(symbol_c)

and_operator = circuit.get_operator("and")
or_operator = circuit.get_operator("or")
and_node = and_operator(leaf_a, leaf_b)
or_node = or_operator(leaf_c, and_node)

# Use circuit
module = circuit.to_module({or_node: ("or_root",), and_node: ("and_root",)})
print("Module created:", module)

inputs = torch.tensor([[1, 0, 1]], dtype=torch.float32)  # Example input: a=1, b=0, c=1
output = module(inputs)
print(output)
```

Observe in the example: each leaf node has an associated unique, symbolic name (cf. `circuit.get_leaf_node(_)`). To combine these leaf nodes, we first obtain from the circuit a reference to the desired operator (cf. `circuit.get_operator(_)`). The available operators are defined by the algebraic structure. For `boolean` this is `and`, `or`, and `not`, while `probability` is a semifield and so also has a division: `times`, `plus`, `negate`, and `divide`.

```{code-cell} ipython3
from deeplog import BOOLEAN, PROBABILITY, LOGPROBABILITY


print("Boolean operators:", BOOLEAN.operators)
print("Probability operators:", PROBABILITY.operators)
print("LogProbability operators:", LOGPROBABILITY.operators)
```

As we combine nodes we construct a computational graph. We then set the root nodes to indicate the nodes we are interested in, before converting the circuit into an evaluatable [DeepLogModule](deeplog.module.deeplog_module.DeepLogModule).

Next we show a small example for the `"probability"` structure.

```{code-cell} ipython3
import torch

from deeplog.circuit import Circuit


# Construct circuit
circuit = Circuit("probability")  # <--- probability
symbol_a = ("a",)
symbol_b = ("b",)
leaf_a = circuit.get_leaf_node(symbol_a)
leaf_b = circuit.get_leaf_node(symbol_b)
or_operator = circuit.get_operator("plus")  # <--- plus
or_node = or_operator(leaf_a, leaf_b)

# Use circuit
module = circuit.to_module({or_node: ("or_root",)})
print("Module created:", module)

inputs = torch.tensor(
    [[0.4, 0.25]], dtype=torch.float32
)  # Example input: a=0.4, b=0.25
output = module(inputs)
print(output)
```

## Weighted model counts
The `probability` structure uses `plus`/`times` as semiring operators, so the circuit above is read exactly as written: `plus(a, b)` is the sum `0.4 + 0.25`, which is only the probability of "a or b" if the two events are disjoint.

The other reading — where `plus` is a union of events that may overlap — is a *weighted model count*, and it is two ordinary rewrites rather than a mode you switch on. `knowledge_compile` rewrites the formula into an equivalent one that is deterministic (a disjunction's branches are mutually exclusive) and decomposable (a conjunction's operands share no variables); `transform` then maps its operators onto the probability semiring, which is exact *because* of those two properties. The result is an ordinary arithmetic circuit, evaluated as written like any other.

The two are spelled separately below to keep them visible. Passing `structure=` to `knowledge_compile` fuses them — it emits the compiled diagram straight into the target algebra, so the boolean circuit in between is never built.

```{code-cell} ipython3
import torch

from deeplog.circuit import Circuit
from deeplog.circuit.knowledge_compile import knowledge_compile


# Write the formula in boolean...
circuit = Circuit("boolean")  # <--- boolean
symbol_a = ("a",)
symbol_b = ("b",)
leaf_a = circuit.get_leaf_node(symbol_a)
leaf_b = circuit.get_leaf_node(symbol_b)
or_node = circuit.get_operator("or")(leaf_a, leaf_b)

# ...knowledge-compile it, so that reading `or` as `plus` is exact...
compiled, compiled_map = knowledge_compile(circuit, [or_node])

# ...and only then map it into probability.
counted, node_map = compiled.transform([compiled_map[or_node]], "probability")

module = counted.to_module({node_map[compiled_map[or_node]]: ("or_root",)})
print("Module created:", module)

inputs = torch.tensor(
    [[0.4, 0.25]], dtype=torch.float32
)  # Example input: a=0.4, b=0.25
output = module(inputs)
print(output)  # 0.25 + (1 - 0.25) * 0.4 = 0.55
```
