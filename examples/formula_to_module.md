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

# From Formulas to Modules

+++

DeepLog parses formulas and, via the public `parse_formula_to_module` helper, lowers them with a {class}`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory`. The factory now ships defaults (equality/probability predicates and Klay circuits) so most notebooks don't need custom setup.

+++

## Model Counting Warm-up

Consider two Boolean variables (`Burglary` and `Earthquake`) and a rule that fires whenever either event holds. We define the model count as a DeepLog formula and send it straight to `parse_formula_to_module`, which instantiates a factory with the built-in equality predicate and Klay circuit backend to produce a {class}`~deeplog.module.deeplog_module.DeepLogModule`.

```{code-cell} ipython3
from deeplog import parse_formula_to_module


model_count_text = """
sum(Burglary): sum(Earthquake):
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
"""

model_count_module = parse_formula_to_module(model_count_text)
print("Model count: ", int(model_count_module()))
```

The module reports a model count of 3, matching the number of satisfying assignments (every valuation except the one where both burglary and earthquake are false).

+++

## Weighted Model Counting

A weighted model count scores each assignment before totalling it. `p(atom, label)` is the labelled fact that does the scoring: it reads the atom's truth value and returns `label` where the atom is true, `1 - label` where it is false. The default factory registers it as [`ProbabilityPredicate`](deeplog.formula.predicates.builtin_predicates.ProbabilityPredicate), so a formula can weigh its own atoms.

Below, the `sum` aggregations enumerate the assignments, `times` weighs each one, and the total is the weighted model count.

```{code-cell} ipython3
weighted_formula_text = """
sum(Burglary): sum(Earthquake):
    (
        (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
    )
    times
    (
        p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability
    )
"""

weighted_module = parse_formula_to_module(weighted_formula_text)

print("Weighted model count:", float(weighted_module()))
```

## Expectation Operator

The **expectation** operator computes the same number without spelling out the enumeration. Given a boolean formula it knowledge-compiles it and transforms the result into the probability semiring — the two rewrites that make reading `or` as a sum exact — so the atoms' probabilities go straight in, and the sum over assignments happens inside the compiled circuit instead of in a `sum` around it.

The compiled module is a count feeding the formula around it, so it declares one input channel per atom; `reshape` packs them into the single tensor we want to hand over.

```{code-cell} ipython3
import torch

from deeplog import reshape
from deeplog.shape import SymTensor
from deeplog.symbol import with_structure

expectation_text = """
expectation(Burglary, Earthquake):
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
"""

expectation_module = parse_formula_to_module(expectation_text)
expectation_module = reshape(
    expectation_module,
    input=SymTensor(
        [
            with_structure(("=", (name,), ("true",)), "probability")
            for name in ("Burglary", "Earthquake")
        ]
    ),
)

# Input: P(Burglary=true)=0.8, P(Earthquake=true)=0.3
result = expectation_module(torch.tensor([[0.8, 0.3]]))
print("Expectation (WMC):", float(result))
```
