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

So far, every variable in our formulas has been **aggregated over** (bound).

But what if a variable is **not** in an aggregation operator? It becomes a **free variable**—an input to the compiled module. This enables:
- Partial computations (aggregate over some variables, keep others as inputs)
- Batching (evaluate multiple instances in parallel)
- Conditional aggregations

+++

## Bound vs. Free Variables

**Bound variable**: Appears in an aggregation operator → becomes internal (not an input)

**Free variable**: Does NOT appear in any aggregation → becomes an input to the module

Example:
```
sum(X):
    lt(X, Y)_boolean
```
- `X` is **bound** by aggregation → internal
- `Y` is **free** → external input

+++

## Example: Count Values Below a Threshold

**Problem**: For a given value Y, count how many values in {1..10} are strictly less than Y.

**Formula**: Aggregate over X (domain {1..10}), but leave Y as an input.

```{code-cell} ipython3
import torch
from deeplog import DeepLogModuleFactory, Domain, Predicate, Symbol, parse_formula_to_module


class LessPredicate(Predicate):
    """Binary comparison: lt(A, B) returns 1 if A < B, else 0."""

    functor = "lt"
    arity = 2
    structure = "boolean"

    def resolve_argument(self, symbol: Symbol, index: int):
        try:
            return int(symbol[0])
        except (ValueError, TypeError, IndexError):
            return symbol

    def forward_predicate(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a < b).to(a.dtype)


# Domain for X
x_symbol = ("X",)
x_domain = Domain.of_tensor(torch.arange(1, 11))  # {1, 2, ..., 10}

# Y is not in the factory's variables—it will be a free input
factory = DeepLogModuleFactory(
    variables={x_symbol: x_domain},
    atom_builders={("lt", 2, "boolean"): LessPredicate},
)

formula = """
sum(X):
    lt(X, Y)_boolean
"""

count_less_than_module = parse_formula_to_module(formula, factory=factory)
print("Module:")
print(count_less_than_module)
```

## Calling with Free Variables

The module now expects Y as an input. Let's call it with different thresholds:

```{code-cell} ipython3
# Test with Y=5
y_value = torch.tensor([[5.0]])  # Batch size 1, Y=5
result = count_less_than_module(y_value)
print(f"Y=5: Count X < Y = {result.item():.0f}")
print("  Expected: 4 (1, 2, 3, 4)\n")

# Test with Y=3
y_value = torch.tensor([[3.0]])
result = count_less_than_module(y_value)
print(f"Y=3: Count X < Y = {result.item():.0f}")
print("  Expected: 2 (1, 2)")
```

## Batching with Free Variables

Since Y is a free variable, we can evaluate multiple thresholds **in a single batch call**:

```{code-cell} ipython3
# Test with multiple Y values in one batch
y_values = torch.tensor([
    [2.0],   # Y=2 -> only 1 is less than 2
    [5.0],   # Y=5 -> 1, 2, 3, 4
    [100.0], # Y=100 -> every value in the domain matches
])  # Shape: [3, 1]

results = count_less_than_module(y_values)
print("Batch results:")
print(f"  Y=2:   {results[0].item():.0f}")
print(f"  Y=5:   {results[1].item():.0f}")
print(f"  Y=100: {results[2].item():.0f}")
```

## Multiple Free Variables

A formula can have multiple free variables. They all become inputs:

```{code-cell} ipython3
# Aggregate over Y, but leave X and Z as free inputs
# Count how many Y satisfy: X < Y < Z

y_symbol = ("Y",)

factory_multi = DeepLogModuleFactory(
    variables={y_symbol: x_domain},
    atom_builders={
        ("lt", 2, "boolean"): LessPredicate,
    },
)

formula_multi = """
sum(Y):
    (lt(X, Y)_boolean) and (lt(Y, Z)_boolean)
"""

multi_free_module = parse_formula_to_module(formula_multi, factory=factory_multi)
print("Module with two free variables Y and Z:")
print(multi_free_module)
```

```{code-cell} ipython3
# Call with X and Z as inputs (batch size 2)
# multi_free_module expects two input tensors: X first, then Z.
x_inputs = torch.tensor([[2.0], [7.0]])
z_inputs = torch.tensor([[6.0], [10.0]])

results = multi_free_module(x_inputs, z_inputs)
print("Batch results:")
print(f"  X=2, Z=6:   {results[0].item():.0f} (expected: 3)")
print(f"  X=7, Z=10:  {results[1].item():.0f} (expected: 2)")
```

## Key Points

- **Bound variables** (in aggregations) -> covered internally, not inputs
- **Free variables** (not in aggregations) -> become module inputs
- **Multiple free variables** -> let you express e.g. interval constraints such as `X < Y < Z`
- **Batching**: Batching over free variables let you evaluate multiple scenarios in one call

+++

## Next Steps

Now you understand aggregation mechanics:

- `examples/formula_to_module.md` — weighted model counting and the expectation operator.
- `examples/problog.md` — the same machinery driven from a ProbLog program.
- `examples/semantic_loss.md` — aggregation inside a training loop, with the gradient flowing through it.
