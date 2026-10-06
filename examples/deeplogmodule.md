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

A [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) is a `torch.nn.Module` that declares its input and output layout as [`SymTensor`](deeplog.shape.SymTensor) shapes, so DeepLog can validate compositions, reshape tensors automatically, and render graphs that show how modules connect. You implement the computation as usual, by subclassing `DeepLogModule` or by wrapping a callable with `WrappedModule`.

A signature alone already makes a module: here a classifier and a regressor, declared by what they read and what they produce.

```{code-cell} ipython3
from deeplog import DeepLogModule, SymTensor


classifier = DeepLogModule(
    input_shape=SymTensor("X"),
    output_shape=SymTensor([f"classifier(X,{i})" for i in range(4)]),
)
regressor = DeepLogModule(
    input_shape=SymTensor("X"),
    output_shape=SymTensor("regressor(X)"),
)

print(classifier)
print(regressor)
```

Wiring modules together by these shapes, with `Sequential` and `compose_modules`, is the `composition` notebook.

+++

## Validation

Every forward pass checks that the tensors a module receives match its input shape, and that what it returns matches its output shape, so a wiring mistake raises at the module where it happens. Validation is on by default; setting the environment variable `DEEPLOG_VALIDATE=0` turns it off for modules constructed afterwards.

```{code-cell} ipython3
import torch

from deeplog import ShapeMismatchException, WrappedModule


features = SymTensor([f"X{i}" for i in range(4)])
predictions = SymTensor([f"Y{i}" for i in range(2)])

head = WrappedModule(lambda x: x[:, :2], features, predictions)
buggy_head = WrappedModule(lambda x: x[:, :1], features, predictions)  # one column, not two

print("accepted:", head(torch.randn(2, 4)).shape)

try:
    head(torch.randn(2, 3))
except ShapeMismatchException as error:
    print("wrong input:", error)

try:
    buggy_head(torch.randn(2, 4))
except ShapeMismatchException as error:
    print("wrong output:", error)
```

## Reshaping

`reshape` returns a new module with the input or output layout you ask for, adding the indexing that maps one onto the other by symbol. Asking for fewer outputs selects them:

```{code-cell} ipython3
from torch import nn

from deeplog import reshape


hidden = SymTensor([f"Hidden{i}" for i in range(8)])
extractor = WrappedModule(nn.Linear(4, 8), features, hidden)
batch = torch.randn(16, 4)

narrow = reshape(extractor, output=SymTensor(["Hidden2", "Hidden5"]))

print("output:", narrow.get_output_shape())
print("result:", narrow(batch).shape)
```

On the input side, each symbol the module reads is looked up in the layout you give. A batch whose columns come in reverse order is read column by column under their names, so the result is the extractor's:

```{code-cell} ipython3
flipped = reshape(extractor, input=SymTensor(["X3", "X2", "X1", "X0"]))

print("input:", flipped.get_input_shape())
print("same result:", torch.allclose(flipped(batch.flip(1)), extractor(batch)))
```

A layout that names a symbol the module does not have cannot be fitted, and `reshape` raises [`TransformationNotPossible`](deeplog.module.reshape.TransformationNotPossible):

```{code-cell} ipython3
from deeplog import TransformationNotPossible


try:
    reshape(extractor, output=SymTensor(["Hidden9"]))
except TransformationNotPossible as error:
    print("not possible:", error)
```
