# DeepLog textual formula grammar

This document specifies the textual format that is parsed into DeepLog formulas. The goal is to describe a surface syntax that can be read and written by users while still mapping one-to-one to the formula's node types (aggregation, transformation, binary/unary operator, and leaf).

## Lexical elements

* `Identifier` – ASCII strings produced by the regular expression `[A-Za-z_][A-Za-z0-9_]*`. Identifiers are used for operator names, aggregation names, structure names, and variables. They are case-sensitive.
* `SymbolLiteral` – any literal that can already be parsed by `deeplog.symbol.parse_symbol`. This gives us predicate syntax such as `p(X,1)`, nested terms such as `p(V0,nn(0))`, or tuples like `('=',X,true)`.
* `Structure` – also an `Identifier`; examples today include `boolean` and `probability`.
* `Variable` – an `Identifier` that will be turned into a `Symbol` via `parse_symbol`. A variable starts with an uppercase letter or an underscore, and an aggregation binds only variables.
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
BinaryOp       ::= Identifier                      # e.g. and, or, plus, times, divide

# Binary operators associate to the left and all share the same precedence level for now.

UnaryExpr      ::= PrefixOp UnaryExpr
                 | "(" Formula ")"
                 | Transformation
PrefixOp       ::= Identifier                      # e.g. not, negate

Transformation ::= "(" Formula ")" "_" Structure

# Parentheses group sub-formulas; appending `_structure` turns the group into a transformation node.

Leaf           ::= SymbolLiteral "_" Structure

# Example: (=,Burglary,true)_boolean  or  p(goal,label)_probability
# The SymbolLiteral is parsed with parse_symbol and paired with the structure for create_atom.
```

### Notes on the grammar

1. **Aggregation scope** – the `:` separates the binders (and optional parameters) from the body. `sum(X, Y; q(X)_probability): f(X, Y)_probability` binds `X` and `Y` inside both the parameter formulas and `f`. Nested aggregations are still written `sum(X): sum(Y): ...`.
2. **Binder domains** – a binder ranges over a [`Domain`](../src/deeplog/variable.py), and the aggregation enumerates it. A *regular* variable's domain is declared per variable in the compiler's `variables` (`{("N1",): Domain.of_tensor(torch.arange(10))}`); an undeclared binder is a *reification* variable and inherits the values of the compiler's `reification` structure, `BOOLEAN` by default — so `sum(Burglary, Earthquake): ...` ranges over `{false, true}` without being told. A reification variable's structure is not derivable from the atoms it appears in (`burglary[B]_probability` reifies a *boolean* `B`), which is why it is one statement about the model rather than one per binder. A structure that declares no enumerable values hands out no domain at all, so a binder over `probability` is refused rather than silently enumerated as if it were boolean.
3. **Aggregation operators** – in `Op(Var1, Var2; ψ1, ψ2): expr`, `Op` is one of the aggregators of `expr`'s algebra (`sum` in `real` and `probability`, `exists` and `forall` in `boolean`), or `expectation`, and the extra parameter formulas follow the `;`. A compiler computes an aggregator by enumerating the binders, or with the builder it is given for the operation.
4. **Binary operator precedence** – **all binary operators share one precedence level** (no classical tiers such as `times` vs `plus`). They associate to the left (`a plus b plus c` parses as `(a plus b) plus c`). Parentheses **must** be used whenever you need to force a different evaluation order.
5. **Unary operators** – any identifier before an expression denotes a unary operation. Parentheses without a suffix are also part of `UnaryExpr`, so `not (a or b)` parses as a unary operator whose operand is the grouped formula. Nested unary chains work because the grammar is right-recursive (`not negate leaf`). Transformations are listed as a third alternative, so expressions like `(φ)_probability times (ψ)_probability` remain valid without extra parentheses. A functor written against its parentheses makes an atom, as in Prolog: `q(x_bar)_boolean` is the atom `q(x_bar)`, while `q (x_bar)_boolean` applies the operator `q` to a cast.
6. **Transformations** – appending `_structure` to a parenthesized formula turns it into a transformation node. Example: `(burglary_boolean)_probability`.
7. **Leaves** – every leaf provides both a symbol literal and its structure, which is exactly what an `Atom` holds. The same spelling names a lowered module's *outputs*: the algebra a value lives in is recorded on the symbol naming it (`addition(i1,i2,3) _ probability`), not as an annotation on the module, so one module can name values in different algebras or in none. Printing `module.get_output_shape()` shows each output's label. Inside a formula every module must label its outputs — an unlabelled child is refused rather than defaulted, so a forgotten label cannot silently select a conversion.
8. **Underscore usage** – a leaf's symbol reads up to its first underscore, with everything after read as the `_structure` suffix, so a bare leaf's functor name cannot itself contain an underscore (`foo_bar_boolean` reads as functor `foo` in structure `bar_boolean`, not functor `foo_bar` in structure `boolean`).
9. **Structure aliases** – `_b` and `_p` act as shorthands for `_boolean` and `_probability`. The long forms continue to work alongside the aliases.

## Mapping to formula nodes

| Grammar construct                | Formula node                                                     |
|----------------------------------|------------------------------------------------------------------|
| `Op(Var1, Var2; ψ1, ψ2): φ`      | `Aggregation(Op, (Var1, Var2), (ψ1, ψ2), φ)`                      |
| `(ψ)_structure`                  | `Transformation(structure, ψ)`                                   |
| `lhs Op rhs`                     | `BinaryOp(Op, lhs, rhs)`                                         |
| `Op ψ` (unary prefix)            | `UnaryOp(Op, ψ)`                                                 |
| `symbol_structure`               | `Atom(("_", symbol, (structure,)))`                              |

Each production builds the matching node, which keeps the textual language intentionally close to the underlying semantic graph.

## Parsing

Use `deeplog.parse_formula(text)` to turn a textual formula into its formula, and `str(formula)` to write one back as text.

## Examples

### Model counting

A Boolean model-counting query that matches the `test_model_count` unit test can be written as:

```
sum(Burglary, Earthquake):
    (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_real
```

* Each `sum(Var1, Var2):` introduces an aggregation node that quantifies jointly over those variables' domains — here the boolean values `{false, true}`, inherited because neither binder is declared (note 2).
* `=` is an ordinary two-argument predicate, written in functor form like any other atom (`=(Burglary,true)`); the `_boolean` suffix makes the leaf an `Atom` in the boolean structure. It is symmetric: `=(true,Burglary)` is the atom `=(Burglary,true)`, a variable compared with a written value always coming first.
* The `or` binary operator is left-associative, so the body is `BinaryOp('or', burglary_leaf, earthquake_leaf)`.
* `boolean` aggregates with `exists` and `forall`, not `sum`, so the body is cast into `real`, where `true` is 1 and `false` 0, and summed there.

### Weighted model counting

The weighted model-counting network from `test_weighted_model_count` becomes:

```
sum(Burglary, Earthquake):
      (=(Burglary,true)_boolean or =(Earthquake,true)_boolean)_probability
    times
      (p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability)
```

* The `( ... )_probability` wrapper is a transformation node, `Transformation('probability', ...)`.
* Probability leaves re-use the `_structure` suffix to select the appropriate atom builders.
* Multiplying the transformed Boolean result with the probability product and aggregating over both variables reproduces the exact execution in the compiler test.
* A sum is computed as written, by evaluating every assignment. Where the weights of each binder sum to one, it equals `expectation(X; P): φ`, which the compiler counts by knowledge compilation instead; where they sum below one, the expectation also gives the mass they leave to a missing value, and the two differ.

### Posterior (conditional probability)

A scalar posterior `P(q | e) = E[q∧e] / E[e]` is one weighted model count divided by another. The division is spelled by the name the algebra gives it — `divide` in the probability [`Semifield`](../src/deeplog/algebraic.py) — infix like every other binary operator. For `P(Burglary | Earthquake)` — joint `Burglary ∧ Earthquake`, evidence `Earthquake`:

```
(expectation(Burglary, Earthquake;
      p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability):
    =(Burglary,true)_boolean and =(Earthquake,true)_boolean)
divide
(expectation(Burglary, Earthquake;
      p(Burglary,0.8)_probability times p(Earthquake,0.3)_probability):
    =(Earthquake,true)_boolean)
```

* Each side is an expectation under the same distribution, in which `Burglary` is true with probability `0.8` and `Earthquake` with `0.3`, independently.
* `divide` is an ordinary binary operator here: it lowers exactly as `times` does, and how a quotient is compiled is the compiler's business ([`split`](../src/deeplog/circuit/split.py)).
* `E[B∧E] = 0.24` and `E[E] = 0.3`, so the posterior is `0.8`.
