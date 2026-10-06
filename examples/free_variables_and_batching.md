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

# Free Variables and Batching

A variable that no aggregation binds is **free**: it becomes an input of the compiled module. A formula can aggregate over some variables and take the others as inputs, and one call evaluates a whole batch of them.

+++

## Bound and free

```
sum(X): (lt(X, Y)_boolean)_real
```

`sum` binds `X`, which ranges over its domain inside the module. `Y` is free, so the module takes it as an input. Binding decides this, not declaring: below, the compiler declares a domain for both variables, and each formula takes the variables it leaves free.

+++

## Counting values below a threshold

For a given `Y`, count the values of `X` in `1..10` below it.

```{code-cell} ipython3
import torch

from deeplog import Compiler, Domain, Predicate, SymTensor, parse_formula_to_module, reshape


class Less(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of_values(), Domain.of_values()))

    def forward_predicate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a < b).to(a.dtype)


one_to_ten = Domain.of_tensor(torch.arange(1, 11))
compiler = Compiler(
    variables={("X",): one_to_ten, ("Y",): one_to_ten},
    atom_builders={("lt", 2, "boolean"): Less},
)

count_below = parse_formula_to_module("sum(X): (lt(X, Y)_boolean)_real", compiler=compiler)

print("input:", count_below.get_input_shape())
print("Y = 5:", count_below(torch.tensor([[5.0]])).item())  # 1, 2, 3 and 4
```

## Batching

A batch of `Y` values is one call, a row each:

```{code-cell} ipython3
thresholds = torch.tensor([[2.0], [5.0], [100.0]])
print(count_below(thresholds).flatten().tolist())  # 1; 1 to 4; all ten
```

## Several free variables

Every free variable is an input. Counting the values of `Y` strictly between `X` and `Z` binds `Y` and leaves `X` and `Z` free. The module takes them in the order it first reads them; `reshape` states the order instead.

```{code-cell} ipython3
count_between = reshape(
    parse_formula_to_module(
        "sum(Y): (lt(X, Y)_boolean and lt(Y, Z)_boolean)_real", compiler=compiler
    ),
    input=(SymTensor(["X"]), SymTensor(["Z"])),
)

x = torch.tensor([[2.0], [7.0]])
z = torch.tensor([[6.0], [10.0]])
print(count_between(x, z).flatten().tolist())  # 3, 4, 5; 8, 9
```

## Key points

- A variable an aggregation binds ranges over its domain inside the module.
- A free variable is an input of the module.
- A batch of values for the free variables is evaluated in one call.

+++

## Next steps

- The `formula_to_module` notebook: a formula as a module, fed by a network.
- The `problog` notebook: the same aggregations, driven from a ProbLog program.
- The `semantic_loss` notebook: an aggregation inside a training loop, with the gradient flowing through it.
