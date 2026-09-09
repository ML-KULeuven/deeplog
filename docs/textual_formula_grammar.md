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
# The SymbolLiteral is parsed with str_to_symbol and paired with the structure for create_leaf_node.
```

### Notes on the grammar

1. **Aggregation scope** – the `:` separates the binders (and optional parameters) from the body. `sum(X, Y; q(X)_probability): f(X, Y)` binds `X` and `Y` inside both the parameter formulas and `f`. Nested aggregations are still written `sum(X): sum(Y): ...`.
2. **Binder domains** – a binder ranges over a [`Domain`](../src/deeplog/variable.py), and the aggregation enumerates it. A *regular* variable's domain is declared per variable in the factory's `variables` (`{("N1",): Domain.of_tensor(torch.arange(10))}`); an undeclared binder is a *reification* variable and inherits the values of the factory's `reification` structure, `BOOLEAN` by default — so `sum(Burglary, Earthquake): ...` ranges over `{false, true}` without being told. A reification variable's structure is not derivable from the atoms it appears in (`burglary[B]_probability` reifies a *boolean* `B`), which is why it is one statement about the model rather than one per binder. A structure that declares no enumerable values hands out no domain at all, so a binder over `probability` is refused rather than silently enumerated as if it were boolean.
3. **Aggregation operators** – `Op(Var1, Var2; ψ1, ψ2): expr` allows users to select any operation name that their `DeepLogFormulaFactory` supports (e.g., `sum`, `max`, `product`) and pass extra parameter formulas.
4. **Binary operator precedence** – **all binary operators share one precedence level** (no classical tiers such as `times` vs `plus`). They associate to the left (`a plus b plus c` parses as `(a plus b) plus c`). Parentheses **must** be used whenever you need to force a different evaluation order.
5. **Unary operators** – any identifier before an expression denotes a unary operation and feeds into `create_unary_node`. Parentheses without a suffix are also part of `UnaryExpr`, so `not (a or b)` parses as a unary operator whose operand is the grouped formula. Nested unary chains work because the grammar is right-recursive (`not negate leaf`). Transformations are listed as a third alternative, so expressions like `(φ)_probability times (ψ)_probability` remain valid without extra parentheses.
6. **Transformations** – appending `_structure` to a parenthesized formula turns it into a transformation node. Example: `(burglary_boolean)_probability`.
7. **Leaves** – every leaf provides both a symbol literal and its structure, which is exactly what `create_leaf_node` requires. The same spelling names a lowered module's *outputs*: the algebra a value lives in is recorded on the symbol naming it (`addition(i1,i2,3) _ probability`), not as an annotation on the module, so one module can name values in different algebras or in none. Read it back with `sole_structure(module.get_output_shape())`, or `structures(...)` for the per-symbol view. Inside a formula every module must label its outputs — an unlabelled child is refused rather than defaulted, so a forgotten label cannot silently select a conversion such as `real → probability`.
8. **Underscore usage** – because `_structure` is reserved for suffixes, predicate names that contain underscores outside parentheses should be quoted (e.g., `'=foo_bar'`) before applying the structure suffix.
9. **Structure aliases** – `_b` and `_p` act as shorthands for `_binary` and `_probabilistic`. The long forms continue to work alongside the aliases.

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

* Each `sum(Var1, Var2):` introduces an aggregation node that quantifies jointly over those variables' domains — here the boolean values `{false, true}`, inherited because neither binder is declared (note 2).
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

### Posterior (conditional probability)

A scalar posterior `P(q | e) = E[q∧e] / E[e]` is one weighted model count divided by another. The division is spelled by the name the algebra gives it — `divide` in the probability [`Semifield`](../src/deeplog/algebraic.py) — infix like every other binary operator. For `P(Burglary | Earthquake)` — joint `Burglary ∧ Earthquake`, evidence `Earthquake`:

```
(sum(Burglary, Earthquake):
      (=(Burglary,true)_boolean and =(Earthquake,true)_boolean)_probability
    times
      (=(Burglary,true)_probability times =(Earthquake,true)_probability))
divide
(sum(Earthquake):
      (=(Earthquake,true)_boolean)_probability
    times
      (=(Earthquake,true)_probability))
```

* Each side is a canonical weighted model count, so `recognize_expectation` would fold each into an `expectation`; `recognize_posterior` does this and additionally checks the evidence (`Earthquake`) is a conjunct of the numerator (`Burglary ∧ Earthquake`).
* `divide` is an ordinary binary operator here: it lowers through `create_binary_node` exactly as `times` does, and how a quotient is compiled is the compiler's business ([`split`](../src/deeplog/circuit/split.py)).
* With `P(Burglary)=0.8`, `P(Earthquake)=0.3`: `E[B∧E] = 0.24`, `E[E] = 0.3`, so the posterior is `0.8`.
