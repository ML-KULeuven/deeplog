# DeepLog textual formula grammar

This document specifies the textual format that will be parsed into DeepLog formulas and wired to a `DeepLogFormulaFactory`. The goal is to describe a surface syntax that can be read and written by users while still mapping one-to-one to the primitive factory operations (aggregation, transformation, binary/unary operator, and leaf creation).

## Lexical elements

* `Identifier` – ASCII strings produced by the regular expression `[A-Za-z_][A-Za-z0-9_]*`. Identifiers are used for operator names, aggregation names, structure names, and variables. They are case-sensitive.
* `SymbolLiteral` – any literal that can already be parsed by `deeplog.symbol.str_to_symbol`. This gives us predicate syntax such as `p(X,1)` or tuples like `('=',X,true)`.
* `Structure` – also an `Identifier`; examples today include `boolean` and `probability`.
* `Variable` – an `Identifier` that will be turned into a `Symbol` via `str_to_symbol`. Conventionally, DeepLog variables start with an uppercase letter.
* Whitespace – spaces, tabs, and newlines may appear between tokens and have no semantic meaning.
* Comments – lines that begin with `#` are ignored by the parser (matching the style already used in `str_to_rules`).

## Grammar

The following grammar uses EBNF-style notation where `{…}` is repetition, `[…]` is optional, and `|` denotes alternatives.

```
Formula        ::= Aggregation
                 | BinaryExpr
                 | UnaryExpr
                 | Transformation
                 | Leaf

Aggregation    ::= Identifier "(" BinderList [";" ParamList] ")" ":" Formula
BinderList     ::= Variable { "," Variable }
ParamList      ::= Formula { "," Formula }

# Aggregations bind more tightly than the other constructs because the binder captures the
# entire following expression before any binary/unary operator can intervene.

BinaryExpr     ::= UnaryExpr BinaryOp UnaryExpr { BinaryOp UnaryExpr }
BinaryOp       ::= Identifier                      # e.g. and, or, plus, times

# Binary operators associate to the left and all share the same precedence level for now.

UnaryExpr      ::= PrefixOp UnaryExpr
                 | "(" Formula ")"
                 | Transformation
PrefixOp       ::= Identifier                      # e.g. not, negate

Transformation ::= "(" Formula ")" "_" Structure

# Parentheses group sub-formulas; appending `_structure` turns the group into a transformation node.

Leaf           ::= SymbolLiteral "_" Structure

# Example: (=,Burglary,true)_boolean  or  p(goal,label)_probability
# The SymbolLiteral is parsed with str_to_symbol and paired with the structure for create_leaf_node.
```

### Notes on the grammar

1. **Aggregation scope** – the `:` separates the binders (and optional parameters) from the body. `sum(X, Y; q(X)_probability): f(X, Y)` binds `X` and `Y` inside both the parameter formulas and `f`. Nested aggregations are still written `sum(X): sum(Y): ...`.
2. **Aggregation operators** – `Op(Var1, Var2; ψ1, ψ2): expr` allows users to select any operation name that their `DeepLogFormulaFactory` supports (e.g., `sum`, `max`, `product`) and pass extra parameter formulas.
3. **Binary operator precedence** – **all binary operators share one precedence level** (no classical tiers such as `times` vs `plus`). They associate to the left (`a plus b plus c` parses as `(a plus b) plus c`). Parentheses **must** be used whenever you need to force a different evaluation order.
4. **Unary operators** – any identifier before an expression denotes a unary operation and feeds into `create_unary_node`. Parentheses without a suffix are also part of `UnaryExpr`, so `not (a or b)` parses as a unary operator whose operand is the grouped formula. Nested unary chains work because the grammar is right-recursive (`not negate leaf`). Transformations are listed as a third alternative, so expressions like `(φ)_probability times (ψ)_probability` remain valid without extra parentheses.
5. **Transformations** – appending `_structure` to a parenthesized formula turns it into a transformation node. Example: `(burglary_boolean)_probability`.
6. **Leaves** – every leaf provides both a symbol literal and its structure, which is exactly what `create_leaf_node` requires.
7. **Underscore usage** – because `_structure` is reserved for suffixes, predicate names that contain underscores outside parentheses should be quoted (e.g., `'=foo_bar'`) before applying the structure suffix.
8. **Structure aliases** – `_b` and `_p` act as shorthands for `_binary` and `_probabilistic`. The long forms continue to work alongside the aliases.

## Mapping to factory calls

| Grammar construct                | Factory call                                                     |
|----------------------------------|------------------------------------------------------------------|
| `Op(Var1, Var2; ψ1, ψ2): φ`      | `factory.create_aggregation(Op, [str_to_symbol(Var1), ...], [ψ1, ψ2], φ)` |
| `(ψ)_structure`                  | `factory.create_transformation(structure, ψ)`                    |
| `lhs Op rhs`                     | `factory.create_binary_node(Op, lhs, rhs)`                       |
| `Op ψ` (unary prefix)            | `factory.create_unary_node(Op, ψ)`                               |
| `symbol_structure`               | `factory.create_leaf_node(str_to_symbol(symbol), structure)`     |

The parser should be implemented as a recursive-descent parser that follows the above grammar. Each production directly invokes the matching factory method, which keeps the textual language intentionally close to the underlying semantic graph.

## Parsing

Use `deeplog.formula.text_parser.parse_formula(text, factory)` to turn a textual formula into the objects generated by your preferred `DeepLogFormulaFactory`. A shortcut `parse_symbolic_formula(text)` is available when you simply want the symbolic tuple representation described in this document.

## Examples

### Model counting

A Boolean model-counting query that matches the `test_model_count` unit test can be written as:

```
sum(Burglary, Earthquake):
    =(Burglary,true)_boolean or =(Earthquake,true)_boolean
```

* Each `sum(Var1, Var2):` introduces an aggregation node that quantifies jointly over those variables' domains.
* `=` is treated as an infix functor in the symbolic representation; the `_boolean` suffix makes the parser call `create_leaf_node`.
* The `or` binary operator is left-associative, so the body builds `factory.create_binary_node('or', burglary_leaf, earthquake_leaf)`.

### Weighted model counting

The weighted model-counting network from `test_weighted_model_count` becomes:

```
sum(Burglary, Earthquake):
      (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
    times
      (p(Burglary)_probability times p(Earthquake)_probability)
```

* The `( ... )_probability` wrapper creates a transformation node via `factory.create_transformation('probability', ...)`.
* Probability leaves re-use the `_structure` suffix to select the appropriate predicate factories.
* Multiplying the transformed Boolean result with the probability product and aggregating over both variables reproduces the exact execution in the module factory test.
