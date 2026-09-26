---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: venv-deeplogdev
  language: python
  name: python3
---

# Predicates in DeepLog

Predicates are the mechanism through which symbolic atoms in DeepLog formulas become executable tensor operations.

An atom is the smallest symbolic unit in a logic formula (e.g.`=(X,true)`, `digit(I,N)`), representing a statement whose value we want to compute.

A predicate defines how such atoms should be evaluated: it maps symbolic arguments to tensors, and implements the function that determines whether the atom holds (Boolean predicates) or how strongly it holds (probability or neural predicates).

+++

When DeepLog compiles a formula such as:
```
or(=(X,true), p(X))
```

it needs to know how to evaluate:
- the Boolean equality atom `=(X,true)`
- the probability atom `p(X)`

A predicate is a {class}`~deeplog.module.deeplog_module.DeepLogModule` that defines how to evaluate a family of atoms.
This notebook shows how predicates are defined, what built-in predicates DeepLog provides, and how to write custom predicates.

+++


Every predicate class:
- declares a [`SymTensor`](deeplog.shape.SymTensor) input and output layout
- implements a batched forward computation
- can be symbolic, numeric, or neural

+++

## Boolean predicates

+++

### EqualityPredicate
The [`EqualityPredicate`](deeplog.formula.predicates.builtin_predicates.EqualityPredicate) evaluates atoms of the form:
```
=(X,true)
=(X,false)
```

It expects per-variable assignments (`false` or `true`) and checks equality.

```{code-cell} ipython3
import torch

from deeplog import parse_formula_to_module, reshape
from deeplog.shape import SymTensor


module = parse_formula_to_module(
    "=(Burglary,true)_boolean or =(Earthquake,true)_boolean"
)
module = reshape(module, input=SymTensor([("Burglary",), ("Earthquake",)]))

print("Module input shape:", module.get_input_shape())

inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = module(inputs)

print("-" * 80)
for i, row in enumerate(inputs[:]):
    print(
        f"Burglary: {bool(row[0])}\t|\tEarthquake: {bool(row[1])}\t|\tResult: {bool(outputs[i])}"
    )
```

## Probability predicate
[`ProbabilityPredicate`](deeplog.formula.predicates.builtin_predicates.ProbabilityPredicate) materializes literal weights by reading the numeric label encoded in the second argument of `p/2`. It mixes those probabilities with the Boolean value of each atom, returning `p` when the literal is true and `1-p` otherwise (or the log versions when configured).

```{code-cell} ipython3
from deeplog import ProbabilityPredicate
from deeplog import simplify_module


probability_module = simplify_module(
    ProbabilityPredicate(
        [
            (("burglary",), ("_", ("0.8",), ("probability",))),
            (("earthquake",), ("_", ("0.3",), ("probability",))),
        ]
    )
)

print("Input shape:", probability_module.get_input_shape())

assignments = torch.tensor(
    [
        [1.0, 0.0],  # burglary true, earthquake false
        [0.0, 1.0],  # burglary false, earthquake true
        [1.0, 1.0],  # both true
    ]
)

weights = probability_module(assignments)
print("Weights:")
print(weights)
```

Probabilities can also be symbolic variables rather than constants. In that case the predicate expects an extra probability tensor input whose symbols match those variable labels.

```{code-cell} ipython3
import torch

from deeplog import ProbabilityPredicate
from deeplog import SymTensor


# Two atoms, but their labels are variables p_a and p_b instead of numeric constants
variable_prob_predicate = ProbabilityPredicate(
    [
        (("a",), ("p_a",)),
        (("b",), ("p_b",)),
    ]
)

print("Input shape (atoms, probabilities):", variable_prob_predicate.get_input_shape())

atom_assignments = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
provided_probabilities = torch.tensor([[0.2, 0.8], [0.5, 0.6]])

weights = variable_prob_predicate(atom_assignments, provided_probabilities)
print("Weights with variable labels:")
print(weights)
```

Log-space support: use `LogProbabilityPredicate` to get log outputs with automatic conversions between probability and log labels. It takes the logarithm of labels provided as probabilities. Similarly, `ProbabilityPredicate` converts log-probability labels to probability outputs.

```{code-cell} ipython3
from math import log

from deeplog import LogProbabilityPredicate, ProbabilityPredicate


atoms = torch.tensor([[1.0], [0.0]])  # single atom true/false
# Request log-prob outputs from probability labels
logprob_module = simplify_module(
    LogProbabilityPredicate(
        [
            (("atom",), ("_", ("0.65",), ("probability",))),
        ],
    )
)
print("logprobability outputs from probability labels:")
print(logprob_module(atoms))

# Request probability outputs from log-probability labels
prob_module = simplify_module(
    ProbabilityPredicate(
        [
            (("atom",), ("_", (str(log(0.8)),), ("logprobability",))),
        ],
    )
)
print("probability outputs from logprobability labels:")
print(prob_module(atoms))
```

## Arithmetic Predicates

+++

### SumsPredicate

[`SumsPredicate`](deeplog.formula.predicates.builtin_predicates.SumsPredicate) evaluates digit-wise addition constraints:
```
sums(A, B, C)
```


It checks whether: `A + B == C`

This is used in tasks like MNIST addition, digit-by-digit arithmetic, temporal constraint reasoning, and logic with numeric structure.

Example:

```{code-cell} ipython3
from deeplog import SumsPredicate


triplets = [(("A",), ("B",), ("C",))]
predicate = SumsPredicate(triplets)

in_a = torch.tensor([[1.0], [1.0]])
in_b = torch.tensor([[2.0], [5.0]])
in_c = torch.tensor([[3.0], [9.0]])

result = predicate(in_a, in_b, in_c)
print("-" * 80)
for i in range(2):
    a, b, c, r = int(in_a[i]), int(in_b[i]), int(in_c[i]), bool(result[i])
    print(f"A: {a}	|	B: {b}t|	C: {c}	|	Result: {r}")
```

## NetworkPredicate

The `get_network_predicate` function wraps a torch module that predicts a distribution over a discrete output domain, returning a predicate class.

```{code-cell} ipython3
from deeplog import get_network_predicate


# Lets define a simple neural network for digit recognition


class MyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x)


net = MyMLP()  # must output logits/probabilities over the domain

Image1 = ("Image1",)
Image2 = ("Image2",)

# Create a predicate class using get_network_predicate
DigitPredicate = get_network_predicate("digit", 2, "probability", net)

# Instantiate with arguments (image, digit_index)
arguments = [
    (Image1, ("0",)),
    (Image1, ("1",)),
    (Image2, ("0",)),
]

predicate = DigitPredicate(arguments)
print(predicate)
```

## Custom Predicates
To define your own predicate, subclass `Predicate`:
- `resolve_argument(symbol, index)`, optional: return a numeric/boolean/tensor constant for literal arguments, or a `Symbol` (often the input itself) to keep it as a variable, which is what the default does for every argument. Constants are baked into the module buffers and injected automatically during `forward`.
- `forward_predicate(*x)`: the batched computation over fully materialized arguments (constants already filled in, variables coming from the input tensors). Return a tensor whose batch dimension is flattened over all evaluations; the base class reshapes it back to `(batch, num_evaluations, ...)`.

```{code-cell} ipython3
from collections.abc import Iterable

import torch

from deeplog import Predicate
from deeplog.symbol import Symbol


class EvenPredicate(Predicate):
    functor = "even"
    arity = 1
    structure = "boolean"

    def __init__(self, all_arguments: Iterable[tuple[Symbol, ...]]):
        super().__init__(all_arguments)

    def resolve_argument(self, symbol: Symbol, _: int):
        # Treat literal numbers as constants so they need no runtime input
        try:
            return float(symbol[0])
        except (ValueError, TypeError, IndexError):
            return symbol

    def forward_predicate(self, digits: torch.Tensor) -> torch.Tensor:
        # digits shape: (batch * num_evaluations, 1)
        return (digits % 2 == 0).to(digits.dtype)
```

```{code-cell} ipython3
even_predicate = EvenPredicate([(("Digit",),)])

print("Input shape:", even_predicate.get_input_shape())
print("Output shape:", even_predicate.get_output_shape())

digits = torch.tensor([[0.0], [1.0], [2.0], [7.0]])
outputs = even_predicate(digits)

print("Digits:", digits.view(-1).tolist())
print("Even flags:", outputs.view(-1).tolist())
```

The base `Predicate.forward` injects constants (when `resolve_argument` returns any non-symbol values) and duplicates the input tensors across evaluations. Your `forward_predicate` only needs to implement the pure logic; the example above shows how to supply batched numeric inputs directly without using factories or parsed formulas.

+++

## Predicate factory pattern

You can use `functools.partial` to pre-configure predicate parameters, creating a factory-like pattern for instantiation.

**What it does:** binds predicate-specific options (like thresholds) so you can instantiate consistently with just the per-atom arguments.

**Where it's used:** when registering predicates with {class}`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory` via `atom_builders`.

**Example:** set predicate-specific options once, then instantiate consistently.

```{code-cell} ipython3
from functools import partial


class MyPredicate(Predicate):
    functor = "my_functor"
    arity = 2
    structure = "boolean"

    def __init__(self, arguments, threshold: float = 0.5):
        super().__init__(arguments)
        self.threshold = threshold

    def resolve_argument(self, symbol, index):
        return symbol

    def forward_predicate(self, *x):
        score = (x[0] == x[1]).float().mean(dim=-1, keepdim=True)
        return (score >= self.threshold).float()


# Configure once using functools.partial
factory = partial(MyPredicate, threshold=0.8)

# Instantiate with atom arguments
pred = factory([(("A",), ("B",))])  # MyPredicate(arguments=[...], threshold=0.8)
print(pred)
```

This pattern lets you expose predicate-specific knobs (e.g., thresholds, domains) once while keeping a uniform interface. In {class}`~deeplog.formula.deeplogmodulefactory.deeplogmodulefactory.DeepLogModuleFactory`, you can register these factories via `atom_builders`, keyed by `(functor, arity, structure)` signature.
