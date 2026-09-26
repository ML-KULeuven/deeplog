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

# The Formula AST and its Rewrites

+++

When you compile a DeepLog formula with `parse_formula_to_module`, it passes through a few steps:

1. **You write** a formula as text.
2. **DeepLog parses** it into a structured form — an *abstract syntax tree*, or AST.
3. **DeepLog rewrites** that structure, automatically recognizing useful patterns.
4. **DeepLog compiles** the result into a runnable module.

Usually you only deal with steps 1 and 4. This notebook opens up the middle: the tree your formula becomes, how to read it, and the rewrite that matters — how DeepLog spots a weighted model count and turns it into something that doesn't enumerate.

+++

## The tree your formula becomes

`parse_formula_to_ast` gives you the structure DeepLog builds from your text. Printing a node gives the formula back; `.tree()` draws the structure that formula stands for.

```{code-cell}
from deeplog.formula import parse_formula_to_ast

ast = parse_formula_to_ast("(=(A,true)_boolean or =(B,true)_boolean)_probability")

print(ast)         # a node prints as the formula it stands for ...
print()
print(ast.tree())  # ... and .tree() shows the nodes it is built from
```

There are six kinds of node, and the tree above uses three:

| node | what it holds |
| --- | --- |
| `Atom` | a leaf — one structure-tagged symbol |
| `UnaryOp` | an operator and one operand (`not`) |
| `BinaryOp` | an operator and two operands (`and`, `or`, `times`, `divide`, …) |
| `Transformation` | a structure cast — the `(…)_probability` above |
| `Aggregation` | an operation, its binders, its parameters, and a body |
| `CircuitNode` | a subformula already compiled into a circuit |

Each is a frozen dataclass, so you read a node off its fields, or match on it. `repr` gives the same structure on one line — what you see in a list, or in a failing test.

```{code-cell}
from deeplog.formula import children

print(repr(ast))
print("structure:", ast.structure)  # every node reports the algebra it lives in
print("operator :", ast.child.operator)  # read a field straight off the node
print("operands :", [str(c) for c in children(ast.child)])
```

## The same tree, whoever built it

The parser is not the only thing that builds formulas. A *formula factory* is the interface every producer goes through, and `AstFactory` is the one that hands back the tree — so a grounder can give you a proof as an AST, rather than as text or as an already-compiled circuit.

```{code-cell}
from deeplog.formula import AstFactory
from deeplog.grounding import SimpleGrounder
from deeplog.grounding.prolog import str_to_rules
from deeplog.symbol import parse_symbol

program = tuple(
    str_to_rules("""
    alarm :- burglary.
    alarm :- earthquake.
    burglary.
    earthquake.
    ?- alarm.
""")
)

proof = SimpleGrounder().ground(
    program,
    parse_symbol("alarm"),
    AstFactory(),
    open_predicates={("burglary", 0), ("earthquake", 0)},
)
print(proof[("alarm",)].tree())
```

## Recognizing a weighted model count

Between parsing and compiling, DeepLog rewrites the tree, recognizing shapes that compile better. This happens on formula *text* and nowhere else: `parse_formula_to_module` applies the passes, and nothing else does.

A **weighted model count** sums over every assignment of some variables and weights each by the probability of the atoms it makes true. The probability that *Burglary or Earthquake* holds is one. Written out, it is exactly what it sounds like: a `sum` over a boolean formula cast to probability, times the probabilities of the atoms that formula is built from.

```{code-cell}
from deeplog.formula import recognize_expectation

WMC = """
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
  times
    (=(Burglary,true)_probability times =(Earthquake,true)_probability)
"""

written = parse_formula_to_ast(WMC)

print("as written:")
print(written.tree())
print("\nrecognized as:")
print(recognize_expectation(written).tree())
```

`sum` means *enumerate*: four assignments here, `2ⁿ` in general, each one evaluated and added up. DeepLog recognizes that this particular shape — a boolean formula cast to probability, times the probabilities of the very atoms it is built from — is the **expectation** `E[Burglary or Earthquake]`, and rewrites it to the `expectation` form. That one knowledge-compiles the boolean formula and reads the result in the probability semiring instead, so it never enumerates anything.

The weights are gone from the rewritten tree because an `expectation` already implies them: they are what it takes the expectation *over*. You can write the count whichever way is clearer and get the same compiled circuit either way.

```{code-cell}
import torch
from deeplog import parse_formula_to_module, reshape
from deeplog.shape import SymTensor

B = ("_", ("=", ("Burglary",), ("true",)), ("probability",))
E = ("_", ("=", ("Earthquake",), ("true",)), ("probability",))

# P(Burglary)=0.8, P(Earthquake)=0.3  ->  P(Burglary or Earthquake) = 0.86
module = reshape(parse_formula_to_module(WMC), input=SymTensor([B, E]))
print("E[Burglary or Earthquake] =", float(module(torch.tensor([[0.8, 0.3]]))))
```

## The recognition is careful

It fires only when the weights really are the boolean atoms retagged to probability. Weight the same formula by something else and it is not a weighted model count any more, so DeepLog leaves your `sum` exactly as you wrote it — and still compiles it, enumeration and all:

```{code-cell}
NOT_A_WMC = """
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
  times
    (p(Burglary)_probability times p(Earthquake)_probability)
"""

print("still a:", recognize_expectation(parse_formula_to_ast(NOT_A_WMC)).operation)

# `p(Burglary)` and `p(Earthquake)` do not depend on the assignment, so what you
# wrote is the plain model count times two constants: 3 * 0.8 * 0.3 = 0.72.
pB = ("_", ("p", ("Burglary",)), ("probability",))
pE = ("_", ("p", ("Earthquake",)), ("probability",))
module = reshape(parse_formula_to_module(NOT_A_WMC), input=SymTensor([pB, pE]))
print("value  :", float(module(torch.tensor([[0.8, 0.3]]))))
```

## It all happens automatically

You don't call the rewrite yourself — `parse_formula_to_module` applies it for you between parsing and compiling, so writing a weighted model count the explicit way simply works and runs efficiently. If you ever want a formula compiled exactly as written, pass `passes=()` to skip the rewrites; you get the same number, just without the shortcut.
