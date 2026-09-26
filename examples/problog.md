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

# ProbLog: probabilistic logic programs

+++

DeepLog can run **ProbLog** programs — logic programs where facts carry probabilities. A program is a handful of probabilistic facts, some rules, and one or more queries; DeepLog proves the queries, compiles the proofs into a differentiable circuit, and gives you a module that returns each query's probability.

This notebook builds the textbook **alarm network** two ways: first asking for plain success probabilities `P(q)`, then *conditioning* on evidence to get `P(q | e)`. Backing a fact's probability with a neural network instead of a constant turns ProbLog into **DeepProbLog** — that's the `deepproblog` notebook.

+++

## Setup

A program is text. `Solver(SimpleGrounder()).get_query_result` proves every `?- ` query into a boolean proof, and `compile_to_module` lowers those proofs into one module whose outputs are the query probabilities.

A numeric fact like `0.6::burglary` is a *constant*: its probability is baked straight into the compiled circuit, so the module needs no probability inputs and returns `P(q)` directly — you write the probabilities once, in the program. (Probabilities backed by a neural network, `nn(...)::digit(...)`, are what stays a runtime input instead; that's the `deepproblog` notebook.) Since these programs are all constants, the helper below just compiles and reads the outputs back by name with `to_dict`: every query comes back under the symbol naming it, tagged with the algebra of the value — `burglary _ probability`.

```{code-cell} ipython3
from deeplog import to_dict
from deeplog.formula import DeepLogModuleFactory
from deeplog import CircuitFactory
from deeplog.systems.deepproblog import compile_to_module
from deeplog.grounding import SimpleGrounder
from deeplog.systems.deepproblog import Solver
from deeplog.grounding import str_to_rules


def run(program):
    """Compile a ProbLog program and read off its query probabilities."""
    result = Solver(SimpleGrounder()).get_query_result(tuple(str_to_rules(program)), CircuitFactory())
    module = compile_to_module(result, DeepLogModuleFactory())
    # The facts are baked into the circuit, so the module needs no inputs — call it bare.
    # `to_dict` reads the result through the shape that names it, so each query
    # comes back under its own symbol instead of at some column index.
    return to_dict(module(), module.get_output_shape())
```

## A first program: the alarm network

A burglary or an earthquake sets off the alarm. We declare each event's probability, two rules for the alarm, and ask for the probability of the alarm and of a burglary:

```{code-cell} ipython3
ALARM = """
0.6::burglary.
0.3::earthquake.
alarm :- burglary.
alarm :- earthquake.
?- alarm.
?- burglary.
"""

run(ALARM)
```

`P(alarm) = 1 - (1 - 0.6)(1 - 0.3) = 0.72` — the alarm fires unless *both* events stay quiet — and `P(burglary) = 0.6`, just the fact's probability. These are **prior** probabilities: we haven't observed anything yet.

+++

## Conditioning on evidence

Now suppose we *hear the alarm* and ask how likely a burglary is: the conditional probability `P(burglary | alarm)`.

DeepProbLog writes evidence as an **integrity constraint** `:- body`, which conditions the distribution on `body` being *false* — the operator dual of the `?- q` query. So you observe an atom is **true** with `:- not(atom)` and **false** with `:- atom`. Here, observing the alarm is `:- not(alarm)`:

```{code-cell} ipython3
OBSERVED_ALARM = """
0.6::burglary.
0.3::earthquake.
alarm :- burglary.
alarm :- earthquake.
:- not(alarm).
?- burglary.
"""

run(OBSERVED_ALARM)
```

Hearing the alarm raises the burglary probability from `0.6` to `P(burglary | alarm) = P(burglary) / P(alarm) = 0.6 / 0.72 = 0.8333…`.

Evidence can also be **negative**. If the alarm stayed silent (`:- alarm`), a burglary is impossible — the rule `alarm :- burglary` means a burglary would have triggered it:

```{code-cell} ipython3
SILENT_ALARM = """
0.6::burglary.
0.3::earthquake.
alarm :- burglary.
alarm :- earthquake.
:- alarm.
?- burglary.
"""

run(SILENT_ALARM)
```

## How it works

A conditional probability is a division, `P(q | e) = E[q ∧ e] / E[e]`. The engine proves the query *and* the evidence on one shared circuit; `compile_to_module` compiles every query and the shared evidence into one circuit and divides them as a final step (keeping the denominator just above zero, so impossible evidence gives a finite answer rather than a crash). Without any `:- ` constraints the division drops away and you get the plain `P(q)` from the first example.

The neural version — predicates backed by a network rather than by a constant — is in the `deepproblog` notebook.
