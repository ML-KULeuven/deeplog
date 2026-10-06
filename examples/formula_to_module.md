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

`parse_formula_to_module` compiles a formula into a [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule): a PyTorch module you call on a batch, differentiate through and feed from a network like any other. This notebook does all three with one constraint.

+++

## A constraint as a module

A classifier tags images with `rain` and `wet`, each with a probability of its own. Whatever it predicts, rain implies wet. The probability that this holds, when `Rain` and `Wet` are independently true with the classifier's probabilities, is an expectation over the two variables:

```{code-cell} ipython3
import torch

from deeplog import SymTensor, parse_formula_to_module, reshape


constraint = parse_formula_to_module(
    "expectation(Rain, Wet): not =(Rain,true)_boolean or =(Wet,true)_boolean",
    name="rain_implies_wet",
)
print("inputs:", constraint.get_input_shape())
print("output:", constraint.get_output_shape())
```

Without a distribution, an expectation's variables are independent, and the probability of each value the formula tests is an input, named by the test: `=(Rain,true) _ probability` is the probability that `Rain` is true. The output is named by `name`, in the algebra of its value. `reshape` lays the two inputs out as the one tensor the classifier produces, rain first, as in the `deeplogmodule` notebook. Each row is one image:

```{code-cell} ipython3
labels = SymTensor(["=(Rain,true) _ probability", "=(Wet,true) _ probability"])
constraint = reshape(constraint, input=labels)

predictions = torch.tensor([[0.9, 0.1], [0.9, 0.9], [0.1, 0.5]], requires_grad=True)
satisfied = constraint(predictions)
print(satisfied)
```

Likely rain without wet satisfies the constraint with probability 0.19, and likely rain with wet with 0.91.

## Gradients

The module is differentiable in its inputs, and its gradient says how to change the predictions to satisfy the constraint more: lower `rain` and raise `wet`.

```{code-cell} ipython3
satisfied.sum().backward()
print(predictions.grad)
```

The probability is `1 - rain + rain * wet`, so its gradient is `wet - 1` for `rain` and `rain` for `wet`.

## A network feeding the constraint

A classifier whose outputs are named `=(Rain,true) _ probability` and `=(Wet,true) _ probability` feeds the constraint by name: `compose_modules` wires each module to the one that produces what it reads, as in the `composition` notebook. The classifier here is a stand-in that reads eight features per image.

```{code-cell} ipython3
from torch import nn

from deeplog import WrappedModule, compose_modules


torch.manual_seed(0)
features = SymTensor([f"Feature{i}" for i in range(8)])
classifier = WrappedModule(nn.Sequential(nn.Linear(8, 2), nn.Sigmoid()), features, labels)
model = compose_modules(
    [classifier, constraint],
    output_shape=SymTensor(["rain_implies_wet _ probability"]),
    input_shape=features,
)
```

The negative log of the probability that the constraint holds is a loss, the semantic loss. Training on it alone moves the classifier towards predictions that satisfy the constraint:

```{code-cell} ipython3
images = torch.randn(256, 8)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)

print("before:", model(images).mean().item())
for _ in range(100):
    loss = -model(images).log().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("after:", model(images).mean().item())
```

On its own, a constraint is easy to satisfy: predicting no rain, or wet everywhere, does it. The `semantic_loss` notebook adds one to a supervised loss on MNIST.

## How it is computed

`parse_formula_to_module(text, compiler)` is `compiler.compile(parse_formula(text))`. A [`Compiler`](deeplog.formula.lowering.compiler.Compiler) holds what a formula compiles with: the domains of its variables, the predicates that evaluate its atoms, and a builder for each aggregation. Its builder for `expectation` decides how an expectation is computed. By default that is [`weighted_model_count`](deeplog.formula.lowering.expectation.weighted_model_count), which compiles the formula into a circuit where reading `or` as a sum is exact, and enumerates what it cannot count. [`enumeration`](deeplog.formula.lowering.expectation.enumeration) evaluates every assignment instead: the same number, without compiling. [`sampling`](deeplog.formula.lowering.expectation.sampling) estimates it from assignments drawn from the distribution, which reaches formulas too large to compile; its gradient is the score-function estimator. An estimate is never chosen for you: sampling is a builder you register.

```{code-cell} ipython3
from deeplog import Compiler, enumeration, parse_formula, sampling


rain_implies_wet = parse_formula(
    "expectation(Rain, Wet): not =(Rain,true)_boolean or =(Wet,true)_boolean"
)

torch.manual_seed(0)
for name, compiler in [
    ("counting", Compiler()),
    ("enumeration", Compiler(aggregation_builders={"expectation": enumeration})),
    ("sampling", Compiler(aggregation_builders={"expectation": sampling(samples=10_000)})),
]:
    module = reshape(compiler.compile(rain_implies_wet), input=labels)
    print(f"{name:11s}:", module(torch.tensor([[0.9, 0.1]])).item())
```

## Next steps

- The `semantic_loss` notebook: a constraint as a loss on MNIST.
- The `aggregation_basics` notebook: aggregations, and an expectation under a distribution of your own.
- The `predicates` notebook: the atoms of a formula, and networks that evaluate them.
