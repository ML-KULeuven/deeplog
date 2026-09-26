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

# Aggregation Basics

Aggregation is the fundamental mechanism in DeepLog that allows formulas to range over multiple variable assignments and combine their contributions into a single value.

This notebook starts with **Boolean model counting**, then shows that the exact same pattern works over any finite domain.

+++

## Aggregation Syntax

An aggregation specifies:
- **how to aggregate** (sum, max, product, …)
- **what to aggregate over** (variables and their domains)
- **what formula to evaluate** for each assignment

Syntactically:
```
operator(Var1, Var2, ...): <subformula>
```

**Example:** `sum(X): <subformula>` sums the subformula's result over all values in X's domain.

+++

## Example: Simple Model Counting

Consider two Boolean variables: `Burglary` and `Earthquake`.

How many Boolean assignments satisfy: `Burglary = true OR Earthquake = true`?

Intuitively:
- (false, false) → false ✗
- (false, true) → true ✓
- (true, false) → true ✓
- (true, true) → true ✓

**Answer: 3 models**

+++

## Writing the Formula

We aggregate over both variables, evaluating the constraint for each assignment, and sum the results:

```{code-cell} ipython3
model_count_formula = """
sum(Burglary, Earthquake):
    =(Burglary, true)_boolean or =(Earthquake, true)_boolean
"""

print("Formula:")
print(model_count_formula)
```

## Compiling and Evaluating

Parse the formula, compile it into a PyTorch module, and evaluate it:

```{code-cell} ipython3
import torch
from deeplog import parse_formula_to_module

model_count_module = parse_formula_to_module(model_count_formula)
print("Compiled module:")
print(model_count_module)

result = model_count_module()
print(f"Model count: {result.item()}")
```

## What Happens Internally

When aggregation evaluates `sum(Burglary, Earthquake)`, DeepLog:

1. **Determines domains**: Each Boolean variable has domain `{0, 1}` (false, true)
2. **Enumerates all combinations**: Creates the Cartesian product: `[(0,0), (0,1), (1,0), (1,1)]`
3. **Broadcasts**: Repeats the input batch and fills variable columns with each assignment
4. **Evaluates**: Runs the inner formula on all assignments in parallel via broadcasting
5. **Reduces**: Sums the results along the assignment dimension

This is all done via pure tensor operations—no loops, fully differentiable.

+++

## Another Example: Three Variables

Count assignments satisfying: `A OR (B AND C)` where A, B, C are Boolean.

```{code-cell} ipython3
formula_abc = """
sum(A, B, C):
    (=(A, true)_boolean) or ((=(B, true)_boolean) and (=(C, true)_boolean))
"""

module_abc = parse_formula_to_module(formula_abc)
result_abc = module_abc()
print(f"Models satisfying A OR (B AND C): {result_abc.item()}")

# Verify: A=true, B=true, C=true (3 combos with B or C true) = 3
print("Expected: 5 (only when A is true, or both B and C are true)")
```

## Same Pattern, Larger Domains

So far we've summed over Boolean assignments. The same aggregation pattern works over any finite domain: you register the domain in a `DeepLogModuleFactory`, then provide a predicate that can evaluate each assignment.

We'll keep the predicate implementation minimal here and focus on the aggregation setup. For predicate design details, see `examples/predicates.md`.

```{code-cell} ipython3
from deeplog import DeepLogModuleFactory, Domain, Predicate, Symbol


class EvenPredicate(Predicate):
    functor = "even"
    arity = 1
    structure = "boolean"

    def resolve_argument(self, symbol: Symbol, _: int):
        try:
            return int(symbol[0])
        except (ValueError, TypeError, IndexError):
            return symbol

    def forward_predicate(self, digits: torch.Tensor) -> torch.Tensor:
        return (digits % 2 == 0).to(digits.dtype)


digit_symbol = ("Digit",)
digit_domain = Domain.of_tensor(torch.arange(10))
factory = DeepLogModuleFactory(
    variables={digit_symbol: digit_domain},
    atom_builders={("even", 1, "boolean"): EvenPredicate},
)
```

```{code-cell} ipython3
even_formula = """
sum(Digit):
    even(Digit)_boolean
"""

even_module = parse_formula_to_module(even_formula, factory=factory)
print("Compiled module:")
print(even_module)

even_result = even_module()
print(f"Even digits in {{0..9}}: {even_result.item()}")
print("Expected: 5 (0, 2, 4, 6, 8)")
```

## Aggregation Operators

`sum` is the only aggregation operator DeepLog registers by default. The operator name in the formula is looked up in the factory's aggregation builders, so a new operator means a new builder: a callable `(child, variables, params, domains) -> SupportsToModule`, where `domains` holds the domain tensor of each bound variable. Pass one to a factory as `aggregators={...}`, or make it a default with `register_aggregation_builder`.

Reducing with `max` instead of `sum` turns the counting formula above into existential quantification — "is there an even digit?" rather than "how many?".

```{code-cell} ipython3
from deeplog import AggregationModule


def build_max(child, variables, _params, domains):
    return AggregationModule(
        child, variables, domains, name="max", op=lambda x: x.amax(dim=1)
    )


max_factory = DeepLogModuleFactory(
    variables={digit_symbol: digit_domain},
    atom_builders={("even", 1, "boolean"): EvenPredicate},
    aggregators={"max": build_max},
)

any_even_module = parse_formula_to_module(
    "max(Digit): even(Digit)_boolean", factory=max_factory
)
print(f"Any even digit in {{0..9}}? {any_even_module().item()}")
print("Expected: 1.0 (yes)")
```

`expectation(...)` looks like a second operator, but it is not an aggregation builder: the construction fold absorbs it into a knowledge-compiled probability circuit before lowering, so it never reaches the enumerating factory. See `examples/ast_and_rewrites.md`.

+++

## Next Steps

Now that you've seen aggregation over both Boolean and non-Boolean domains:

- **Next notebook**: Free variables, batching, and partial aggregations
- **Related notebook**: `examples/predicates.md` for custom predicate design
- **Beyond**: `examples/formula_to_module.md` for weighted model counting and the expectation operator
