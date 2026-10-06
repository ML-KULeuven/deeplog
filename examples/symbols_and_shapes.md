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

# Symbols and shapes

DeepLog names the values it computes. A symbol is a name, such as `digit(img,3)` for the chance that an image shows a 3, and a [`SymTensor`](deeplog.shape.SymTensor) is a tensor of names, one for each entry of a tensor of values. Every module declares what it reads and returns with these names, so this notebook introduces them first.

+++

## Symbols

A [`Symbol`](deeplog.symbol.Symbol) is a term: a functor applied to arguments, which are symbols themselves. `parse_symbol` reads one from text. It is a plain tuple, the functor followed by the arguments, and a constant such as `bob` is a functor without arguments:

```{code-cell} ipython3
from deeplog import parse_symbol, symbol_to_pretty_string


symbol = parse_symbol("parentOf(bob,charlie)")
print(symbol)
print(symbol_to_pretty_string(symbol))
```

A symbol with functor `f` and `n` arguments is written `f/n`, so this one is `parentOf/2`. An argument that starts with an upper-case letter or `_` is a variable, which stands for any value:

```{code-cell} ipython3
from deeplog import get_term_variables


atom = parse_symbol("digit(Image,3)")
print(*map(symbol_to_pretty_string, get_term_variables(atom)))
```

## A symbol's algebra

A symbol can also say which algebra its value is in: `burglary _ probability` names a probability. A module names its outputs this way, so each output says what it computes and how to read the number. In a formula, an atom's algebra follows it after an underscore, as in `p(Rain,0.2)_probability`. Code that builds such a symbol uses `with_structure`:

```{code-cell} ipython3
from deeplog import structure_of, with_structure


weight = with_structure(parse_symbol("burglary"), "probability")
print(symbol_to_pretty_string(weight))
print(structure_of(weight))
print(weight == parse_symbol("burglary _ probability"))
```

## SymTensors

A [`SymTensor`](deeplog.shape.SymTensor) names the entries of a tensor. A digit classifier's ten outputs for an image are `digit(img,0)` to `digit(img,9)`, and the strings are parsed as symbols:

```{code-cell} ipython3
from deeplog import SymTensor


digits = SymTensor([f"digit(img,{d})" for d in range(10)])
print(digits)
print(digits.sym_shape)
```

Nested lists name more dimensions, here two images by ten digits:

```{code-cell} ipython3
pairs = SymTensor([[f"digit(img{i},{d})" for d in range(10)] for i in (1, 2)])
print(pairs.sym_shape)
```

The names cover every dimension but the first, which runs over the batch: a tensor laid out by `digits` has shape `(batch, 10)`. An entry can itself be a tensor, such as an image: `SymTensor(["img1", "img2"])` names the two images of a `(batch, 2, 784)` tensor.

## Shapes

A module's inputs and outputs are each laid out by a [`Shape`](deeplog.shape.Shape): a `SymTensor`, or a tuple of them when there are several tensors. `to_dict` reads a tensor back by the names its shape gives it:

```{code-cell} ipython3
import torch

from deeplog import to_dict


scores = torch.softmax(torch.randn(1, 10), dim=1)
print(to_dict(scores, digits))
```

The `deeplogmodule` notebook declares modules by their shapes, which lets DeepLog check every tensor a module receives and returns, and fit a module to another layout by name.
