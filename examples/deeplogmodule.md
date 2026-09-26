---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# DeepLogModule

A [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) extends `torch.nn.Module` with symbolic shape metadata. Each module declares its input and output layout via [`SymTensor`](deeplog.shape.SymTensor) objects so DeepLog can validate compositions, reshape tensors automatically, and render graphs that show how modules connect. You implement the actual computation as usual (either by subclassing `DeepLogModule` or by wrapping a callable with `deeplog.module.deeplog_module.WrappedModule`).

The example below shows how to instantiate two placeholder modules—a classifier and a regressor—by only specifying their symbolic signatures.

```{code-cell} ipython3
import torch.nn as nn

from deeplog import DeepLogModule
from deeplog import SymTensor


classifier_module = DeepLogModule(
    input_shape=SymTensor("X"),
    output_shape=SymTensor([f"classifier(X,{i})" for i in range(4)]),
)

regressor_module = DeepLogModule(
    input_shape=SymTensor("X"), output_shape=("regressor(X)")
)

print("Classifier module", classifier_module)
print("Regressor module", regressor_module)
```

## Containers
For convenience, DeepLog provides higher-level containers that understand symbolic shapes while orchestrating multiple modules.

- [`Sequential`](deeplog.module.sequential.Sequential): Chains together multiple modules, piping the symbolic output of one module into the next (similar to function composition).
- [`ModuleCircuit`](deeplog.module.module_circuit.ModuleCircuit): Evaluates a circuit/DAG of modules, auto-wiring intermediate tensors while sharing cached results.

+++

### Sequential: chain modules
`Sequential` feeds the output of one module into the next while ensuring their symbolic shapes line up. Below, a vector is normalized before a classifier head; the container keeps the declared layout visible and would fail fast if one of the modules expected a different shape.

```{code-cell} ipython3
import torch

from deeplog import SymTensor
from deeplog.module import WrappedModule
from deeplog.module.sequential import Sequential


features = SymTensor([f"X{i}" for i in range(8)])
normalized_features = SymTensor([f"Norm_X{i}" for i in range(8)])
classes = SymTensor([f"Class{i}" for i in range(3)])

normalize = WrappedModule(
    module=nn.LayerNorm(8),
    input_shape=features,
    output_shape=normalized_features,
    name="Normalize",
)
classifier_head = WrappedModule(
    module=nn.Linear(8, 3),
    input_shape=normalized_features,
    output_shape=classes,
    name="ClassifierHead",
)

sequential_model = Sequential(normalize, classifier_head)

logits = sequential_model(torch.randn(2, 8))
print("Sequential input shape:", sequential_model.get_input_shape())
print("Sequential output shape:", sequential_model.get_output_shape())
print("Logits shape:", logits.shape)
```

### ModuleCircuit: branched DAG
[`ModuleCircuit`](deeplog.module.module_circuit.ModuleCircuit) solves the "wire everything up" problem when multiple modules read and write different symbolic tensors. Instead of manually passing every intermediate, it builds a DAG from the declared [`SymTensor`](deeplog.shape.SymTensor) inputs/outputs of each [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule), caches results, and routes them to any downstream consumer. It will even synthesize reshaping/concatenation transformations when an input symbol is produced elsewhere but with a compatible layout.

Initialization highlights:
- `modules`: the set of `DeepLogModule` instances (order does not matter; a topological sort is derived).
- `output_shape`: which symbolic tensors you want returned; everything else stays in the cache.
- optional `structure` label: used only for graph visualization metadata.

In the example below, a splitter fans out features to a classifier and a regressor, and the circuit returns both outputs in a single call.

```{code-cell} ipython3
import torch
import torch.nn as nn

from deeplog import SymTensor, reshape
from deeplog.module import WrappedModule
from deeplog.module.module_circuit import ModuleCircuit


input_features = SymTensor([f"X{i}" for i in range(8)])
left_features = SymTensor([f"X_left{i}" for i in range(4)])
right_features = SymTensor([f"X_right{i}" for i in range(4)])
class_logits = SymTensor([f"Class{i}" for i in range(3)])
regression_score = SymTensor(["Score"])

split = WrappedModule(
    module=lambda x: (x[:, :4], x[:, 4:]),
    input_shape=input_features,
    output_shape=(left_features, right_features),
    name="SplitFeatures",
)
classifier_branch = WrappedModule(
    module=nn.Linear(4, 3),
    input_shape=left_features,
    output_shape=class_logits,
    name="ClassifierHead",
)
regressor_branch = WrappedModule(
    module=nn.Linear(4, 1),
    input_shape=right_features,
    output_shape=regression_score,
    name="RegressorHead",
)

circuit = ModuleCircuit(
    modules=[split, classifier_branch, regressor_branch],
    output_shape=(class_logits, regression_score),
)
# Fix input ordering using reshape
circuit = reshape(circuit, input=input_features)

logits, score = circuit(torch.randn(2, 8))
print("Circuit input shape:", circuit.get_input_shape())
print("Circuit output shape:", circuit.get_output_shape())
print("Circuit outputs:", logits.shape, score.shape)
```

## Validation during inference
The class attribute `DeepLogModule.validate` defaults to `True`, so every forward pass checks that provided tensors match the declared [`SymTensor`](deeplog.shape.SymTensor) shapes **and** that the callable returns tensors with the expected shape. This catches wiring bugs early, before training loops keep running with bad inputs/outputs.

```{code-cell} ipython3
import torch

from deeplog import ShapeMismatchException
from deeplog import SymTensor
from deeplog.module import WrappedModule


features = SymTensor([f"X{i}" for i in range(4)])
preds = SymTensor([f"Y{i}" for i in range(2)])

# Wrapped module that obeys the declared input/output shapes
good_head = WrappedModule(
    lambda x: x[:, :2], input_shape=features, output_shape=preds, name="GoodHead"
)

# Wrapped module that returns the wrong output shape (only 1 dim instead of 2)
buggy_head = WrappedModule(
    lambda x: x[:, :1], input_shape=features, output_shape=preds, name="BuggyHead"
)

valid_input = torch.randn(2, 4)
invalid_input = torch.randn(2, 3)  # wrong feature length

print("--- Valid input, correct output ---")
try:
    out = good_head(valid_input)
    print("Accepted; output shape", out.shape)
except ShapeMismatchException as e:
    print("Unexpected validation error:", e)

print("--- Invalid input shape ---")
try:
    good_head(invalid_input)
except ShapeMismatchException as e:
    print("Input validation caught mismatch:")
    print(e)

print("--- Valid input, bad output from callable ---")
try:
    buggy_head(valid_input)
except ShapeMismatchException as e:
    print("Output validation caught mismatch:")
    print(e)
```
