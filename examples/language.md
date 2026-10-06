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

# DeepLog Language

A quick tour of the textual DeepLog language, how it maps to the parser and grammar, and how to compile formulas into modules.

+++

## Overview

DeepLog provides a compact textual language for writing formulas with aggregations, transformations, unary/binary operators, and leaves tagged with structures (e.g., `_boolean`, `_probability`, `_logprobability`).

Parser helpers:

- [`parse_formula`](deeplog.formula.text_parser_lark.parse_formula) to get the formula as a tree of nodes. Print a node to read it back as text, or call `.tree()` to see the structure; the `formula_ast` notebook takes that further.
- [`parse_formula_to_module`](deeplog.formula.text_parser_lark.parse_formula_to_module) to execute the same text as a module.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
import torch

from deeplog import parse_formula, parse_formula_to_module
```

## Leaves and structures

Leaves are written as `predicate(args)_structure`. Structures tag the algebra of the atom's value (e.g., `_boolean`, `_probability`, `_logprobability`).

```{code-cell} ipython3
text = "=(Burglary,true)_boolean"
parsed = parse_formula(text)
print(parsed)
```

## Aggregations, binary, and unary operators

- Aggregation: `op(Var1, Var2): expression` binds the variables in `expression`; extra parameter formulas go after a `;`, as in `op(Var; psi): expression`. `op` is one of the aggregators of `expression`'s algebra: `sum` in `real` and `probability`, `exists` and `forall` in `boolean`.
- Binary: `lhs op rhs` (all binary ops share the same precedence and associate left).
- Unary: `op expr` applies a prefix operator.
- Parentheses + `_structure` create transformations: `(expr)_probability`.

```{code-cell} ipython3
model_count_text = """
sum(Burglary): sum(Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_real
"""
parsed_model_count = parse_formula(model_count_text)
print(parsed_model_count)
```

You can also compile the same string directly into a module. The resulting [`DeepLogModule`](deeplog.module.deeplog_module.DeepLogModule) validates shapes and plugs into Torch code.

```{code-cell} ipython3
model_count_module = parse_formula_to_module(model_count_text)
model_count_module()  # no free variables, so no inputs
```

## Mixing structures: probabilities over Boolean formulas

Transformations let you change structures mid-formula (e.g., turn a Boolean result into a probability and weigh it).

```{code-cell} ipython3
weighted_text = """
sum(Burglary): sum(Earthquake):
    ((=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability)
    times
    (burglary(Burglary)_probability times earthquake(Earthquake)_probability)
"""
print(parse_formula(weighted_text))
```

No predicate computes `burglary` or `earthquake`, so their labels are inputs. The sums read `burglary(Burglary)` at each value of `Burglary`, so the module takes one input per ground atom: `burglary(true)` and `burglary(false)`, and likewise for `earthquake`.

```{code-cell} ipython3
from deeplog import SymTensor, reshape

weighted_module = parse_formula_to_module(weighted_text)
print("inputs:", weighted_module.get_input_shape())

# One tensor: burglary(true), burglary(false), earthquake(true), earthquake(false)
weighted_module = reshape(
    weighted_module,
    input=SymTensor(
        [
            "burglary(true) _ probability",
            "burglary(false) _ probability",
            "earthquake(true) _ probability",
            "earthquake(false) _ probability",
        ]
    ),
)

# Evaluate with P(burglary) = 0.2 and P(earthquake) = 0.6: 1 - 0.8 * 0.4
weighted_module(torch.tensor([[0.2, 0.8, 0.6, 0.4]]))
```

## Tips

- All binary operators share one precedence; use parentheses to group.
- Aggregations capture the following expression before binary/unary operators are applied.
- Comments starting with `#` and extra whitespace are ignored.
- Use `parse_formula` when you want the formula itself, to inspect or build on, and `parse_formula_to_module` when you want a runnable module.
- DIMACS CNF is a second front end onto formulas: `parse_dimacs_cnf(text)` reads clause lists instead of DeepLog syntax. Variable `i` becomes the variable `Vi`, and the literal `i` the test `=(Vi,true)`.
