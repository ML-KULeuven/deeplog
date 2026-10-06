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

An aggregation evaluates a formula at every assignment of some variables and combines the values into one. This notebook starts with boolean model counting, aggregates over a larger domain, and ends with expectations, which weigh each assignment by a distribution.

+++

## Syntax

An aggregation names how to combine, the variables it binds, and the formula to evaluate at each of their assignments:

```
operator(Var1, Var2, ...): formula
```

`sum(X): φ` adds up the values of `φ` over every value of `X`. How to combine is part of the formula's algebra, as its operators are: each [algebra](deeplog.algebraic.AlgebraicStructure) declares its aggregators. `real` and `probability` aggregate with `sum`, and `boolean` with `exists` and `forall`.

+++

## Model counting

How many assignments of two boolean variables, `Burglary` and `Earthquake`, satisfy `Burglary = true or Earthquake = true`? Three: all but the one where both are false.

The formula sums the constraint over both variables. The constraint is boolean, and `boolean` has no `sum`, so it is cast into the reals first: `(…)_real` reads `true` as 1 and `false` as 0, and the model count is the sum of those.

```{code-cell} ipython3
import torch

from deeplog import parse_formula_to_module


model_count = parse_formula_to_module("""
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_real
""")

print(model_count)
print("model count:", model_count().item())
```

`sum` evaluates its body at every assignment of its variables, here the four pairs of `false` and `true`, in one batched pass rather than a loop, and adds the values with `real`'s `sum`. The result is differentiable in everything the body reads.

+++

## Larger domains

The same aggregation ranges over any finite domain. A predicate evaluates the atoms, and `even` reads a digit as the number it is, so nothing in the formula says which numbers `Digit` ranges over: a `Compiler` declares it. The `predicates` notebook covers writing predicates.

```{code-cell} ipython3
from deeplog import Compiler, Domain, Predicate


class Even(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of_values(),))

    def forward_predicate(self, digits: torch.Tensor) -> torch.Tensor:
        return (digits % 2 == 0).to(digits.dtype)


compiler = Compiler(
    variables={("Digit",): Domain.of_tensor(torch.arange(10))},
    atom_builders={("even", 1, "boolean"): Even},
)

evens = parse_formula_to_module("sum(Digit): (even(Digit)_boolean)_real", compiler=compiler)
print("even digits in 0..9:", evens().item())
```

## Aggregation operators

The operator an aggregation names is one of its body's algebra's aggregators. `boolean` has `exists` and `forall`, so asking whether there is an even digit, or whether every digit is even, rather than how many, aggregates the boolean body itself, with no cast.

```{code-cell} ipython3
any_even = parse_formula_to_module("exists(Digit): even(Digit)_boolean", compiler=compiler)
all_even = parse_formula_to_module("forall(Digit): even(Digit)_boolean", compiler=compiler)
print("an even digit in 0..9:", bool(any_even().item()))
print("every digit in 0..9 even:", bool(all_even().item()))
```

A new aggregator is declared on an algebra, beside its operators, as `AlgebraicStructure(aggregation_fns={...})`; the `ltn` notebook declares fuzzy quantifiers this way. How the compiler computes an aggregation can be replaced too, with a builder in its `aggregation_builders`; the `formula_to_module` notebook swaps the one for `expectation`.

+++

## Expectations

`expectation(X; P): φ` is the expectation of the boolean formula `φ` over the variables `X`, distributed as `P`, the param after the `;`. Where each variable's weights sum to one, as `p`'s do, it is the sum over the assignments of `φ`, cast into probability, times its weight. Weighted, the model count above is a probability:

```{code-cell} ipython3
weighted_sum = parse_formula_to_module("""
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
    times p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability
""")
expectation = parse_formula_to_module("""
expectation(Burglary, Earthquake;
            p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability):
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
""")

print("sum:", weighted_sum().item())
print("expectation:", expectation().item())
```

The sum evaluates every assignment, as written. The expectation is computed by knowledge compilation: the compiler compiles `φ` into a circuit where reading `or` as a sum is exact, and weighs each value a test reads by its weight, so the sum over assignments happens inside the circuit. A variable's weights may also sum below one. The expectation then gives the rest of the mass to a missing value, where every test of the variable is false, as ProbLog reads an annotated disjunction whose probabilities sum below one; the sum has no such value.

Without a distribution, the variables are independent, and the probability of each value `φ` tests is an input. The `formula_to_module` notebook uses such an expectation as a constraint on a classifier.

+++

## Dependent variables

Burglaries and earthquakes happen independently, so their distribution is one `p` per variable. Not every pair of variables is like that. Rain and a sprinkler both wet the grass, and the sprinkler's timer skips rainy days: it runs on 40% of dry days and on 1% of rainy ones, and it rains on 20% of days. Whether the sprinkler runs depends on whether it rains, so the distribution is `P(Rain) × P(Sprinkler | Rain)`, and the second factor mentions both variables.

`P(Sprinkler | Rain)` is a table, written as a predicate over two truth values. It declares that both its arguments are truth values, so the compiler knows `Rain` and `Sprinkler` range over them without being told.

```{code-cell} ipython3
truth = Domain.of(["false", "true"])


class Timer(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (truth, truth))

    def forward_predicate(
        self, rain: torch.Tensor, sprinkler: torch.Tensor
    ) -> torch.Tensor:
        on = torch.where(rain == 1, 0.01, 0.4)
        return torch.where(sprinkler == 1, on, 1 - on)


garden = Compiler(atom_builders={("timer", 2, "probability"): Timer})
wet = parse_formula_to_module(
    """
expectation(Rain, Sprinkler;
            p(Rain,0.2)_probability times timer(Rain,Sprinkler)_probability):
    =(Rain,true)_boolean or =(Sprinkler,true)_boolean
""",
    compiler=garden,
)
print("grass wet:", wet().item())
```

The grass is wet on rainy days, 0.2, and on the dry days the sprinkler runs, 0.8 × 0.4. Over all days, the sprinkler runs with probability 0.2 × 0.01 + 0.8 × 0.4 = 0.322. Two independent causes with those same chances wet the grass less often:

```{code-cell} ipython3
independent = parse_formula_to_module("""
expectation(Rain, Sprinkler;
            p(Rain,0.2)_probability times p(Sprinkler,0.322)_probability):
    =(Rain,true)_boolean or =(Sprinkler,true)_boolean
""")
print("grass wet:", independent().item())
```

Independent, the sprinkler runs on some rainy days, where the grass is wet already. The timer saves it for the dry days, so the same two chances cover more days. An expectation depends on how its variables vary together, not only on each one's own chance.

A factor that mentions two variables makes them one, whose values are their pairs, and the compiler evaluates the formula at each of the four pairs, without knowledge compilation. The same distribution can also be written with one factor per variable, as in the DeepLog paper's Appendix A: the sprinkler gets a coin for rainy days and one for dry days, and the formula says which coin it follows. Each variable then has a factor of its own, so the compiler counts the expectation by knowledge compilation, as it does for burglary and earthquake.

```{code-cell} ipython3
coins = parse_formula_to_module("""
expectation(Rain, OnIfRain, OnIfDry;
            p(Rain,0.2)_probability times p(OnIfRain,0.01)_probability
            times p(OnIfDry,0.4)_probability):
    =(Rain,true)_boolean
    or (=(Rain,true)_boolean and =(OnIfRain,true)_boolean)
    or (not =(Rain,true)_boolean and =(OnIfDry,true)_boolean)
""")
print("grass wet:", coins().item())
```

## Next steps

- The `free_variables_and_batching` notebook: variables an aggregation leaves free, and batching over them.
- The `predicates` notebook: writing predicates.
- The `formula_to_module` notebook: a formula as a module, fed by a network.
