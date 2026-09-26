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

# DeepLog Language

A quick tour of the textual DeepLog language, how it maps to the parser and grammar, and how to compile formulas into modules.

+++

## Overview

DeepLog provides a compact textual language for writing formulas with aggregations, transformations, unary/binary operators, and leaves tagged with structures (e.g., `_boolean`, `_probability`, `_logprobability`).

Parser helpers:

- {func}`~deeplog.formula.text_parser_lark.parse_formula_to_ast` to get the formula as a tree of nodes. Print a node to read it back as text, or call `.tree()` to see the structure — the `ast_and_rewrites` notebook takes that further.
- {func}`~deeplog.formula.text_parser_lark.parse_formula` to fold the text straight through a `DeepLogFormulaFactory` of your choice, without materializing the tree. Below we use `SymbolicFormulaFactory`, whose result is text again — which is how the surface syntax gets checked against itself.
- {func}`~deeplog.formula.text_parser_lark.parse_formula_to_module` to execute the same text as a module.

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
---
import torch

from deeplog import parse_formula_to_module
from deeplog.formula import parse_formula, SymbolicFormulaFactory
```

## Leaves and structures

Leaves are written as `predicate(args)_structure`. Structures tag the "domain" of the atom (e.g., `_boolean`, `_probability`, `_logprobability`).

```{code-cell} ipython3
text = "=(Burglary,true)_boolean"
parsed = parse_formula(text, SymbolicFormulaFactory())
parsed
```

## Aggregations, binary, and unary operators

- Aggregation: `op(Var1, Var2): expression` binds the variables in `expression`; extra parameter formulas go after a `;`, as in `op(Var; psi): expression`.
- Binary: `lhs op rhs` (all binary ops share the same precedence and associate left).
- Unary: `op expr` applies a prefix operator.
- Parentheses + `_structure` create transformations: `(expr)_probability`.

```{code-cell} ipython3
model_count_text = """
sum(Burglary): sum(Earthquake):
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
"""
parsed_model_count = parse_formula(model_count_text, SymbolicFormulaFactory())
parsed_model_count
```

You can also compile the same string directly into a module. The resulting {class}`~deeplog.module.deeplog_module.DeepLogModule` validates shapes and plugs into Torch code.

```{code-cell} ipython3
model_count_module = parse_formula_to_module(model_count_text)
# Evaluate on a dummy batch with no free variables (shape: batch x 0)
out = model_count_module()
out
```

## Mixing structures: probabilities over Boolean formulas

Transformations let you change structures mid-formula (e.g., turn a Boolean result into a probability and multiply by literal weights).

```{code-cell} ipython3
weighted_text = """
sum(Burglary): sum(Earthquake):
    ((=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability)
    times
    (p(Burglary)_probability times p(Earthquake)_probability)
"""
parse_formula(weighted_text, SymbolicFormulaFactory())
```

```{code-cell} ipython3
from deeplog import reshape
from deeplog.shape import SymTensor

weighted_module = parse_formula_to_module(weighted_text)
print("Original input shape:", weighted_module.get_input_shape())

# Fix input ordering: Burglary first, then Earthquake
expected_input = SymTensor([
    ("_", ("p", ("Burglary",)), ("probability",)),
    ("_", ("p", ("Earthquake",)), ("probability",)),
])
weighted_module = reshape(weighted_module, input=expected_input)

# Evaluate with P(Burglary) = 0.2 and P(Earthquake) = 0.6
weighted_module(torch.tensor([[0.2, 0.6]]))
```

## Tips

- All binary operators share the same precedence; use parentheses to force grouping.
- Aggregations capture the following expression before binary/unary operators are applied.
- Comments starting with `#` and extra whitespace are ignored.
- Leaf suffixes `_boolean`, `_probability`, `_logprobability` pick the structure of the atom.
- Use `parse_formula_to_ast` when you want to inspect the formula's structure, `parse_formula` to fold it through your own factory, and `parse_formula_to_module` when you want a runnable module.
- DIMACS CNF is a second front end onto the same factories: `parse_dimacs_cnf(text, SymbolicFormulaFactory())` reads clause lists instead of DeepLog syntax. Variables become `v1`, `v2`, … atoms — the `v` keeps them clear of the neutral constants `0`/`1` — tagged with the structure passed as `structure=`.
