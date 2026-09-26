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

# Mastering Composition in DeepLog

If you're coming from standard PyTorch, you know that building complex architectures involves manually wiring tensors through your `forward` methods. DeepLog simplifies this process! By leveraging **symbolic shapes** (`SymTensor`), DeepLog can often figure out how to wire, reshape, and combine modules automatically.

We'll cover:
1. **Composition Strategies**: How to combine modules using `Sequential` and `ModuleCircuit`.
2. **Automatic Transformations**: How to use `reshape` and `simplify_module` to gracefully handle shape mismatches.
3. **Playing with Inputs**: How `ModuleCircuit` can intelligently handle missing producers and synthesize inputs for you.

Let's get started by importing our tools!

```{code-cell} ipython3
import torch
import torch.nn as nn

from deeplog import SymTensor, reshape
from deeplog.module import WrappedModule
from deeplog.module.sequential import Sequential
from deeplog.module.reshape import simplify_module
from deeplog.module.module_circuit import ModuleCircuit, compose_modules
```

## 1. Composition Strategies

DeepLog offers several containers that orchestrate multiple modules. Because each `DeepLogModule` declares its input and output layouts symbolically, the containers can validate and route data automatically.

+++

### 1.1 Sequential: Chaining Modules

`Sequential` behaves similarly to `nn.Sequential` in PyTorch, but with an added feature: it validates that the symbolic output of one module exactly matches the expected symbolic input of the next. It pipes data forward seamlessly.

```{code-cell} ipython3
# Let's define some symbolic features
raw_features = SymTensor([f"Raw{i}" for i in range(4)])
hidden_features = SymTensor([f"Hidden{i}" for i in range(8)])
output_classes = SymTensor([f"Class{i}" for i in range(2)])

# Wrap standard PyTorch modules into DeepLogModules
feature_extractor = WrappedModule(
    module=nn.Linear(4, 8),
    input_shape=raw_features,
    output_shape=hidden_features,
    name="FeatureExtractor"
)

classifier_head = WrappedModule(
    module=nn.Linear(8, 2),
    input_shape=hidden_features,
    output_shape=output_classes,
    name="ClassifierHead"
)

# Chain them together
sequential_model = Sequential(feature_extractor, classifier_head)

print("Sequential Model Input Shape:", sequential_model.get_input_shape())
print("Sequential Model Output Shape:", sequential_model.get_output_shape())

# Run a forward pass
sample_input = torch.randn(16, 4) # Batch of 16, 4 features
predictions = sequential_model(sample_input)
print("Predictions shape:", predictions.shape)
```

### 1.2 ModuleCircuit

`ModuleCircuit` is arguably the most powerful container in DeepLog. When you have a complex graph of modules with branched dependencies, manually routing outputs to inputs becomes tedious.

Instead, you just give `ModuleCircuit` an unordered list of modules and specify what outputs you want. It builds a Directed Acyclic Graph (DAG) based on the symbolic shapes, topologically sorts the operations, and manages an internal cache to route data!

Furthermore, if multiple modules share the exact same input dependencies, `ModuleCircuit` seamlessly handles parallel evaluation and caches the intermediate results nicely so they are only computed once.

```{code-cell} ipython3
task1_out = SymTensor(["Task1_Score"])
task2_out = SymTensor(["Task2_Score"])

head_1 = WrappedModule(
    module=nn.Linear(8, 1),
    input_shape=hidden_features,
    output_shape=task1_out,
    name="Head1"
)

head_2 = WrappedModule(
    module=nn.Linear(8, 1),
    input_shape=hidden_features,
    output_shape=task2_out,
    name="Head2"
)

# Using the feature_extractor and the two heads.
# Notice how head_1 and head_2 evaluate in parallel on the cached hidden_features!
circuit_model = ModuleCircuit(
    modules=[head_2, feature_extractor, head_1], # Order doesn't matter!
    output_shape=(task1_out, task2_out) # We want both task scores
)

# Fix the input shape so it matches our single tensor containing all features
circuit_model = reshape(circuit_model, input=raw_features)

out1, out2 = circuit_model(sample_input)
print("Circuit Input Shape:", circuit_model.get_input_shape())
print("Circuit Output Shape:", circuit_model.get_output_shape())
print("Circuit Outputs:", out1.shape, out2.shape)
```

### 1.3 compose_modules: The High-Level Helper

If you don't want to think about whether to use `ModuleCircuit` or return a single module, you can use the `compose_modules` helper. It analyzes the provided modules and automatically builds the necessary container.

```{code-cell} ipython3
composed = compose_modules(
    modules=[feature_extractor, head_1, head_2],
    output_shape=(task1_out, task2_out),
)
print("Composed Module Type:", type(composed).__name__)
```

## 2. Automatic Transformations

Sometimes, your modules don't perfectly align. DeepLog provides utilities to safely mutate input and output signatures without changing the underlying mathematical logic.

+++

### 2.1 Reshaping Signatures

The `reshape()` function allows you to change the input or output signature of a module. DeepLog will automatically construct and prepend/append indexing, flattening, or broadcasting transformations so that the internal module still gets what it expects!

```{code-cell} ipython3
# Suppose our feature extractor outputs [Hidden0...Hidden7]
# What if we want it to artificially output ONLY [Hidden2, Hidden5]?

narrow_output = SymTensor(["Hidden2", "Hidden5"])
reshaped_extractor = reshape(feature_extractor, output=narrow_output)

print("Original Output Shape:", feature_extractor.get_output_shape())
print("Reshaped Output Shape:", reshaped_extractor.get_output_shape())

out = reshaped_extractor(sample_input)
print("Reshaped Execution Output Shape:", out.shape) # Notice it's [16, 2] now!
```

### 2.2 Simplifying Module Signatures

If a module has a messy signature (e.g., duplicate inputs, empty tensors), `simplify_module()` cleans it up into a minimal, deduplicated form while maintaining correctness.

```{code-cell} ipython3
messy_input = (
    SymTensor(["X1", "X2"]),
    SymTensor([]),             # Empty tensor!
    SymTensor(["X1", "X2"])    # Duplicate!
)
messy_output = SymTensor(["Out"])

messy_module = WrappedModule(
    module=lambda a, b, c: a.sum(dim=1, keepdim=True) + c.sum(dim=1, keepdim=True),
    input_shape=messy_input,
    output_shape=messy_output,
    name="MessyModule"
)

simplified = simplify_module(messy_module)

print("Messy Input Shape:", messy_module.get_input_shape())
print("Simplified Input Shape:", simplified.get_input_shape()) # Cleaned up!

clean_input = torch.tensor([[1.0, 2.0]])
print("Execution result:", simplified(clean_input))
```

## 3. Playing with Inputs: Missing Producers in ModuleCircuit
One of the most magical features of `ModuleCircuit` is how it handles missing producers. Imagine module **A** produces `[X, Y, Z]`, and module **B** requires `[Y, Z]`. Normally, you would have to write a custom lambda or forward method to slice the tensor. DeepLog's `ModuleCircuit` realizes that **B** needs `[Y, Z]` and that the symbols exist in the universe of available tensors (produced by **A** or provided as circuit inputs). It will **automatically synthesize the required transformation modules** on the fly!

```{code-cell} ipython3
# A module that takes a single input and produces a wide tensor [F0, F1, F2, F3]
producer_in = SymTensor(["Input"])
wide_out = SymTensor([f"F{i}" for i in range(4)])
producer = WrappedModule(
    module=lambda x: torch.cat([x, x, x, x], dim=1), # duplicate the input 4 times
    input_shape=producer_in,
    output_shape=wide_out,
    name="Producer"
)

# A module that only needs the middle two features [F1, F2]
narrow_in = SymTensor(["F1", "F2"])
narrow_out = SymTensor(["Result"])
consumer = WrappedModule(
    module=nn.Linear(2, 1),
    input_shape=narrow_in,
    output_shape=narrow_out,
    name="Consumer"
)

# We ask ModuleCircuit to wire them together.
# Note that no one explicitly produces "narrow_in" exactly.
smart_circuit = ModuleCircuit(
    modules=[producer, consumer],
    output_shape=narrow_out
)

# Fix input shape to match the single tensor we will pass
smart_circuit = reshape(smart_circuit, input=producer_in)

print("Smart Circuit Input Shape:", smart_circuit.get_input_shape())
print("Smart Circuit Output Shape:", smart_circuit.get_output_shape())

# Let's run it
sample_in = torch.randn(2, 1) # Batch 2, 1 feature
result = smart_circuit(sample_in)
print("Result Shape:", result.shape)
print("\nThe circuit synthesized the slicing operation [F1, F2] behind the scenes!")
```

### What if a symbol is completely missing?

If `ModuleCircuit` cannot find a symbol anywhere in the provided modules or the declared inputs, it will raise a clear `ValueError` before you even start training, saving you from nasty runtime surprises.

```{code-cell} ipython3
impossible_in = SymTensor(["F1", "MAGIC_FEATURE"])
impossible_consumer = WrappedModule(
    module=nn.Linear(2, 1),
    input_shape=impossible_in,
    output_shape=narrow_out,
    name="ImpossibleConsumer"
)

try:
    ModuleCircuit([producer, impossible_consumer], output_shape=narrow_out)
except ValueError as e:
    print("DeepLog properly caught the error!")
    print("Error message:", e)
```
