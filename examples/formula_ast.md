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

# The Formula AST

+++

When you compile a DeepLog formula, the text first becomes an *abstract syntax tree* (AST), and the compiler works on that. Usually you never see it. This notebook opens it up: the tree your formula becomes, how to read it, and how the other ways of building a formula, from Python or from a grounder, build the same tree.

+++

## The tree your formula becomes

`parse_formula` gives you the structure DeepLog builds from your text. Printing a node gives the formula back; `.tree()` draws the structure that formula stands for.

```{code-cell} ipython3
from deeplog import parse_formula

ast = parse_formula("(=(A,true)_boolean or =(B,true)_boolean)_probability")

print(ast)         # a node prints as the formula it stands for ...
print()
print(ast.tree())  # ... and .tree() shows the nodes it is built from
```

There are five kinds of node, and the tree above uses three:

| node | what it holds |
| --- | --- |
| `Atom` | a leaf: one structure-tagged symbol |
| `UnaryOp` | an operator and one operand (`not`) |
| `BinaryOp` | an operator and two operands (`and`, `or`, `times`, `divide`, …) |
| `Transformation` | a structure cast, like the `(…)_probability` above |
| `Aggregation` | an operation, its binders, its parameters, and a body |

You read a node off its fields, or match on it:

```{code-cell} ipython3
from deeplog import Atom, BinaryOp

print("structure:", ast.structure)  # every node reports the algebra it lives in
print("operator :", ast.child.operator)  # read a field straight off the node

match ast.child:
    case BinaryOp("or", Atom() as lhs, Atom() as rhs):
        print("a disjunction of", lhs, "and", rhs)
```

## Building the tree yourself

The node types are also how you write a formula from Python, for when the formula is generated rather than typed. Nodes compare by value, so a tree built by hand is the same formula as the parsed one, and compiles the same way.

```{code-cell} ipython3
from deeplog import Transformation, with_structure


def is_true(variable):
    return Atom(with_structure(("=", (variable,), ("true",)), "boolean"))


built = Transformation("probability", BinaryOp("or", is_true("A"), is_true("B")))

print(built)
print("same formula:", built == ast)
```

## The same tree, whoever built it

The parser is not the only thing that builds formulas. A grounder proves a query and returns each proof as the same kind of tree.

```{code-cell} ipython3
from deeplog import parse_symbol
from deeplog.grounding.prolog import SimpleGrounder, str_to_rules

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
    open_predicates={("burglary", 0), ("earthquake", 0)},
)
print(proof[("alarm",)].tree())
```
