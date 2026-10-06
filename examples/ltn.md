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

# Logic Tensor Networks

This notebook follows the LTN tutorial [Grounding connectives and quantifiers](https://github.com/logictensornetworks/logictensornetworks/blob/master/tutorials/2-grounding_connectives.ipynb): the same fuzzy semantics and the same connectives, named as LTN names them, with each formula compiled into a PyTorch module.

```{code-cell} ipython3
import torch

from deeplog import AlgebraicStructure, Compiler, Domain, Predicate, with_structure
from deeplog import Aggregation, Atom, BinaryOp, UnaryOp
```

## The fuzzy algebra

LTN's semantics is an algebra over truth values in `[0, 1]`: the product t-norm for `and`, the probabilistic sum for `or`, the standard negation, and Reichenbach's implication. Its quantifiers are the algebra's aggregators, which receive a formula's truth values over a variable's domain stacked along dimension 1. As in the tutorial, `exists` is the p-mean with `p = 5`, and `forall` the p-mean error with `p = 2`.

```{code-cell} ipython3
def p_mean(values, p):
    return values.pow(p).mean(dim=1).pow(1 / p)


fuzzy = AlgebraicStructure(
    name="fuzzy",
    operator_fns={
        "and": lambda a, b: a * b,
        "or": lambda a, b: a + b - a * b,
        "implies": lambda a, b: 1 - a + a * b,
        "not": lambda a: 1 - a,
    },
    aggregation_fns={
        "exists": lambda values: p_mean(values, 5),
        "forall": lambda values: 1 - p_mean(1 - values, 2),
    },
)
```

## The predicate

`eq` measures how close two points are, `exp(-‖x - y‖)`. It is a predicate in the fuzzy algebra, evaluated in one batch over every pair of points it is asked about.

```{code-cell} ipython3
class Closeness(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of_values(), Domain.of_values()))

    def forward_predicate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.norm(x - y, dim=1))
```

The compiler knows the algebra, the predicate and the domains of the two variables: `X` ranges over ten points in the plane and `Y` over five, drawn as in the tutorial.

```{code-cell} ipython3
torch.manual_seed(0)
x, y = ("X",), ("Y",)

compiler = Compiler(
    structures={"fuzzy": fuzzy},
    variables={
        x: Domain.of_tensor(torch.randn(10, 2)),
        y: Domain.of_tensor(torch.randn(5, 2) * 2),
    },
    atom_builders={("eq", 2, "fuzzy"): Closeness},
)
```

## Connectives

As in LTN, each connective and quantifier is a function that builds a formula. The formula is data, and prints as the text it stands for.

```{code-cell} ipython3
Not = lambda a: UnaryOp("not", a)
And = lambda a, b: BinaryOp("and", a, b)
Or = lambda a, b: BinaryOp("or", a, b)
Implies = lambda a, b: BinaryOp("implies", a, b)
Equiv = lambda a, b: And(Implies(a, b), Implies(b, a))
Forall = lambda variables, a: Aggregation("forall", tuple(variables), (), a)
Exists = lambda variables, a: Aggregation("exists", tuple(variables), (), a)
Eq = lambda terms: Atom(with_structure(("eq", *terms), "fuzzy"))

print(Implies(Eq([x, y]), Eq([x, y])))
```

`compiler.compile` turns a formula into a module whose inputs are its free variables, here a point for `X` and one for `Y`, each of shape `(batch, 1, 2)`.

```{code-cell} ipython3
point_x = torch.tensor([[[1.0, 0.0]]])
point_y = torch.tensor([[[0.5, 0.5]]])

close = Eq([x, y])
for formula in [
    close,
    Not(close),
    And(close, close),
    Or(close, close),
    Implies(close, close),
    Equiv(close, close),
]:
    print(f"{compiler.compile(formula)(point_x, point_y).item():.4f}  {formula}")
```

Reichenbach's implication gives `p → p` the value `1 - p + p²`, which is below one unless `p` is 0 or 1: neither `Implies(p, p)` nor `Equiv(p, p)` is a tautology.

+++

## Quantifiers

A quantifier binds its variables: they range over their domains inside the module and are no longer inputs. `Forall([x], Eq([x, y]))` leaves `Y` free, so its module takes a point for `Y`.

```{code-cell} ipython3
compiler.compile(Forall([x], Eq([x, y])))(point_y).item()
```

Binding both variables leaves no inputs at all.

```{code-cell} ipython3
for formula in [
    Forall([x, y], Eq([x, y])),
    Exists([x, y], Eq([x, y])),
    Forall([x], Exists([y], Eq([x, y]))),
]:
    print(f"{compiler.compile(formula)().item():.4f}  {formula}")
```
