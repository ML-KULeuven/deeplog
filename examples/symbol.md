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

# Symbols

One of the core components of DeepLog is the [`Symbol`](deeplog.symbol.Symbol), which is used to identify anything of symbolic nature.


The [`Symbol`](deeplog.symbol.Symbol) itself is a tuple of length $n+1$ where the first element is the functor, and the subsequent n elements are the arguments, which are symbols themselves. A symbol with functor $f$ and arity $n$ is denoted $f/n$.
Here are some example symbolic definitions. Note the notation for tuples with a single element.

```{code-cell} ipython3
TrueSymbol = ("true",)  # true
FalseSymbol = ("false",)  # false
an = ("an",)  # an
term1 = ("parentOf", an, ("bob",))  # parentOf(an, bob)
```

As constructing symbols manually is cumbersome, we have provided a helper function for constructing terms from strings.

```{code-cell} ipython3
from deeplog.symbol import parse_symbol


print(parse_symbol("parentOf(bob,charlie)"))
```
