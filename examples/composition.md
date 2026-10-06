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

# Composing Modules

In PyTorch, a `forward` method passes tensors from one module to the next by hand. A DeepLog module declares what its inputs and outputs are, as [`SymTensor`](deeplog.shape.SymTensor) shapes, so DeepLog can do that wiring by name: `Sequential` chains modules and `compose_modules` wires a graph of them.

```{code-cell} ipython3
import torch
from torch import nn

from deeplog import Sequential, SymTensor, WrappedModule, compose_modules
```

## Sequential: a chain

`Sequential` works like `nn.Sequential`, and checks that each module's output shape is the next module's input shape.

```{code-cell} ipython3
raw = SymTensor([f"Raw{i}" for i in range(4)])
hidden = SymTensor([f"Hidden{i}" for i in range(8)])
classes = SymTensor([f"Class{i}" for i in range(2)])

extractor = WrappedModule(nn.Linear(4, 8), raw, hidden)
classifier = WrappedModule(nn.Linear(8, 2), hidden, classes)

model = Sequential(extractor, classifier)
batch = torch.randn(16, 4)

print("input :", model.get_input_shape())
print("output:", model.get_output_shape())
print("result:", model(batch).shape)
```

## compose_modules: a graph

`compose_modules` looks symbols up the way `reshape` does in the `deeplogmodule` notebook, across a graph. It takes modules in any order and the outputs you want, and feeds each module from the module that produces what it reads, or the part of it the module reads. A result that several modules read is computed once: below, both heads read the extractor's output, the second only its first four columns.

The composed module's inputs are the symbols no module produces, here `Raw0` to `Raw3`. Without `input_shape`, each is an input of its own; `input_shape` lays them out, here as the one tensor a batch comes in.

```{code-cell} ipython3
first_half = SymTensor([f"Hidden{i}" for i in range(4)])
score1 = SymTensor(["Score1"])
score2 = SymTensor(["Score2"])
head1 = WrappedModule(nn.Linear(8, 1), hidden, score1)
head2 = WrappedModule(nn.Linear(4, 1), first_half, score2)

composed = compose_modules(
    [head2, extractor, head1], output_shape=(score1, score2), input_shape=raw
)

out1, out2 = composed(batch)
print("outputs:", out1.shape, out2.shape)
```

## Inputs no module produces

A module may read a symbol that no module produces. This head reads the extractor's output and a `Prior` beside it, in one tensor, so `Prior` becomes an input of the composed module:

```{code-cell} ipython3
score3 = SymTensor(["Score3"])
head3 = WrappedModule(
    nn.Linear(9, 1), SymTensor([f"Hidden{i}" for i in range(8)] + ["Prior"]), score3
)

composed = compose_modules([extractor, head3], output_shape=score3)
print("inputs:", composed.get_input_shape())
```

`input_shape` must hold every such symbol, and a layout that leaves one out is refused. Here the prior comes as a second tensor beside the batch:

```{code-cell} ipython3
prior = SymTensor(["Prior"])
composed = compose_modules(
    [extractor, head3], output_shape=score3, input_shape=(raw, prior)
)

print("inputs:", composed.get_input_shape())
print("result:", composed(batch, torch.ones(16, 1)).shape)
```
