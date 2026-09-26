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

# Shapes

DeepLog models carry symbolic information about their inputs and outputs.
That structure lets us compose modules and build transforms without running data through them first.
In this notebook, we’ll start with a tiny example and then explain the pieces it uses.

+++

## Quick demo

We’ll build a tiny reshape module that splits classifier outputs into even and odd digits, then run it on a toy tensor.

```{code-cell} ipython3
from deeplog import SymTensor
from deeplog.module.reshape import construct_transformation
import torch

symbolic_input = SymTensor("X")
symbolic_output = SymTensor([f"classifier(X,{i})" for i in range(10)])

even = SymTensor([f"classifier(X,{i})" for i in range(0, 10, 2)])
odd = SymTensor([f"classifier(X,{i})" for i in range(1, 10, 2)])

transformation = construct_transformation(symbolic_output, (even, odd))

input_tensor = torch.tensor([0.1 * i for i in range(10)])
output_tensor = transformation(input_tensor.unsqueeze(0))

print("Symbolic output:", symbolic_output)
print("Input:", input_tensor)
print("Output:", output_tensor)
```

## SymTensor and Shape

DeepLog modules (instances of [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule)) carry symbolic input/output structure: number of inputs/outputs, dimensionality, and what each input represents. That symbolic structure is what lets us compose modules automatically.

The symbolic information for a module input or output is stored in a [`SymTensor`](deeplog.shape.SymTensor), which is a tensor of symbols. A [`Shape`](deeplog.shape.Shape) is a [`SymTensor`](deeplog.shape.SymTensor) or a tuple of [`SymTensors`](deeplog.shape.SymTensor).

+++

## Transformations

`construct_transformation` analyzes how symbols appear in the input shape(s) and builds a [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) that reorders, splits, or groups those symbols to match a desired output shape.

Think of it as a symbolic `reshape` that works on named pieces rather than raw dimensions. The function matches output symbols to input symbols, determines the necessary indexing/slicing, and returns a module you can call like any other DeepLog module.

In the demo, we start from a flat list of class symbols and request a tuple `(even, odd)`. The transformation groups every other symbol into two outputs, preserving order within each group.

+++

## When a transformation fails

If an output symbol doesn’t exist in the input, there’s no valid reshape. In that case `construct_transformation` raises [`TransformationNotPossible`](deeplog.module.reshape.TransformationNotPossible).

```{code-cell} ipython3
from deeplog.module.reshape import TransformationNotPossible

try:
    construct_transformation(symbolic_output, SymTensor(["a"]))
except TransformationNotPossible as err:
    print("Transformation is not possible:", err)
```
