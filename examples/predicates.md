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

# Predicates in DeepLog

A formula is built from atoms, such as `sky(photo1,rainy)` or `umbrella(rainy)`: statements whose values the formula combines. This notebook is about where those values come from. An atom's value is either given, as an input of the compiled module, or computed by a predicate, a [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) that computes every atom of one functor at once from the atoms' arguments.

The examples follow one small scenario: the weather is sunny, cloudy or rainy, people carry an umbrella more often the worse it is, and a network tells the weather from a photo.

+++

## Atoms as inputs

Without a predicate for its functor, an atom is an input of the compiled module, named by the atom. The formula below is the chance that `photo1` shows rain and someone carries an umbrella, and `_probability` says each atom's value is a probability. Nothing computes either atom, so both are columns of the module's input:

```{code-cell} ipython3
import torch

from deeplog import parse_formula_to_module


formula = "sky(photo1,rainy)_probability times umbrella(rainy)_probability"

given = parse_formula_to_module(formula)
print("input:", given.get_input_shape())
print(given(torch.tensor([[0.7, 0.9]])))
```

The rest of this notebook replaces each of these inputs with a predicate that computes it.

## Writing a predicate

A predicate subclasses [`Predicate`](deeplog.formula.predicates.predicate.Predicate). It is given its atoms and a domain for each argument, which the next section explains. Its `forward_predicate` receives one tensor per argument, with a row for each atom in each batch item, and returns a value per row: the predicate computes all of its atoms in one call.

`Umbrella` gives the chance that someone carries an umbrella in each weather, reading its argument over the weathers:

```{code-cell} ipython3
from deeplog import Domain, Predicate, parse_symbol


weathers = Domain.of(["sunny", "cloudy", "rainy"])


class Umbrella(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (weathers,))

    def forward_predicate(self, weather: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.1, 0.4, 0.9])[weather]


umbrella = Umbrella(map(parse_symbol, ["umbrella(rainy)", "umbrella(Weather)"]))
print("input :", *umbrella.get_input_shape())
print("output:", umbrella.get_output_shape())
print(umbrella(torch.tensor([[2], [0]])))  # Weather is rainy, then sunny
```

The module has a column for each atom. `umbrella(rainy)` reads no input: its argument is written, and the predicate keeps it.

A compiler looks up an atom's predicate in its `atom_builders`, by the atom's functor, arity and algebra: `umbrella(rainy)_probability` under `("umbrella", 1, "probability")`. An entry can be anything that takes the atoms and returns a module with a column for each, such as a predicate class. With `Umbrella` registered, the formula's input `umbrella(rainy)` is gone, and `umbrella(Weather)` reads `Weather` instead:

```{code-cell} ipython3
from deeplog import Compiler


compiler = Compiler(atom_builders={("umbrella", 1, "probability"): Umbrella})

partly_given = parse_formula_to_module(formula, compiler=compiler)
print("input:", *partly_given.get_input_shape())
print(partly_given(torch.tensor([[0.7]])))

in_weather = parse_formula_to_module("umbrella(Weather)_probability", compiler=compiler)
print("input:", *in_weather.get_input_shape())
print(in_weather(torch.tensor([[2], [1]])))  # Weather is rainy, then cloudy
```

## Reading arguments

An argument's domain, its sort, says how the predicate reads the values there, whether they are written in the atom or held by a variable:

- a named domain, `Domain.of([...])`: a finite set of names. A name reaches the predicate as its position in the domain, so `Umbrella` sees `rainy` as 2, whether the atom writes it or `Weather` holds it.
- the values, `Domain.of_values()`: numbers, points or images, read as they are. A number written as the argument is that number, and a word names an input.
- a tensor domain, `Domain.of_tensor(...)`: a finite set of unnamed values, such as the digits 0 to 9, read as they are.

`Warm` reads a temperature, over the values:

```{code-cell} ipython3
class Warm(Predicate):
    def __init__(self, atoms):
        super().__init__(atoms, (Domain.of_values(),))

    def forward_predicate(self, temperature: torch.Tensor) -> torch.Tensor:
        return (temperature >= 20).to(temperature.dtype)


warm = Warm(map(parse_symbol, ["warm(Temperature)", "warm(25)", "warm(12)"]))
print("output:", warm.get_output_shape())
print(warm(torch.tensor([[25.0], [12.0]])))  # Temperature is 25, then 12
```

An atom means the same either way: where `Temperature` is 25, `warm(Temperature)` agrees with `warm(25)`, and where it is 12, with `warm(12)`.

In a formula, a variable takes the domain of the arguments it fills: `Weather` ranges over the three weathers wherever `umbrella` reads it, without being declared. A variable read only over the values, as `warm` reads `Temperature`, has no finite domain to take, so summing over it needs one declared, as the `aggregation_basics` notebook does.

## Built-in predicates

Every compiler computes `p`, `logp` and `=`. `p(V,Label)` weighs a truth value: it is `Label` where `V` is true, and `1 - Label` where it is false. `=(V,Value)` holds where `V` holds `Value`, reading `Value` as its position in `V`'s domain, as a predicate over a named domain does. Here `Rain` is a truth value, which a module takes as `0` for false and `1` for true:

```{code-cell} ipython3
chance = parse_formula_to_module("p(Rain,0.2)_probability")
raining = parse_formula_to_module("=(Rain,true)_boolean")

rain = torch.tensor([[1], [0]])  # Rain is true, then false
print("p(Rain,0.2) :", chance(rain).flatten())
print("=(Rain,true):", raining(rain).flatten())
```

A label written as a word names an input, so `p(Rain,forecast)` takes the chance of rain from an input `forecast`. `logp` computes the same in log space, in the `logprobability` algebra, and converts a label written as a probability, as in `logp(Rain,0.2 _ probability)`.

## Networks

[`NetworkPredicate`](deeplog.formula.predicates.builtin_predicates.NetworkPredicate) runs a network that outputs a distribution. `sky(Photo,Weather)` is the probability a classifier gives the weather `Weather` for `Photo`; with the weathers as its domain, the classifier's outputs are the weathers in that order. The compiler gives a predicate only its atoms, so the network is bound at registration, here with `functools.partial`. With both predicates registered, the formula reads only the photo: `photo1` is a word where `sky` reads values, so it names an input:

```{code-cell} ipython3
from functools import partial

from torch import nn

from deeplog import NetworkPredicate


classifier = nn.Sequential(nn.Linear(64, 3), nn.Softmax(dim=1))
sky = partial(NetworkPredicate, module=classifier, domain=weathers)
compiler = Compiler(
    atom_builders={
        ("umbrella", 1, "probability"): Umbrella,
        ("sky", 2, "probability"): sky,
    }
)

from_photo = parse_formula_to_module(formula, compiler=compiler)
print("input:", *from_photo.get_input_shape())

photos = torch.rand(2, 1, 64)  # a batch of two photos, 64 features each
print(from_photo(photos).detach())
```

The formula covers rainy weather only. Summed over every weather, the same product is the chance of an umbrella in whatever weather the photo shows. Summing over a variable is an aggregation, the subject of the `aggregation_basics` notebook.

```{code-cell} ipython3
umbrella_in_photo = parse_formula_to_module(
    "sum(Weather): sky(photo1,Weather)_probability times umbrella(Weather)_probability",
    compiler=compiler,
)
print(umbrella_in_photo(photos).detach())
```
