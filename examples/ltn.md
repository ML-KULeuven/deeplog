---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Logic Tensor Networks

Adapted from the LTN Tutorial: https://github.com/logictensornetworks/logictensornetworks/blob/master/tutorials/2-grounding_connectives.ipynb

```{code-cell} ipython3
from deeplog import DeepLogModuleFactory, AggregationModule, with_structure, Predicate, AlgebraicStructure, Domain
from deeplog.formula import Atom, BinaryOp, UnaryOp, Aggregation
import torch
from functools import partial
```

```{code-cell} ipython3
ltn_fuzzy = AlgebraicStructure(
    name='fuzzy',
    operator_fns={
        "and": lambda a, b: a * b,
        "or": lambda x, y: x + y - x * y,
        "implies": lambda x, y: 1 - x + x * y,
        "not": lambda x: 1.0 - x,
    },
)
```

```{code-cell} ipython3
def generalized_mean(x, p):
    return torch.mean(x**p, dim=1)**(1/p)

def _agg(name, op):
    return lambda child, vars, _params, domains: AggregationModule(child, vars, domains, name=name, op=op)

exists = _agg('exists', lambda x: generalized_mean(x, 6))
forall = _agg('forall', lambda x: 1 - generalized_mean(1-x, 4))
```

```{code-cell} ipython3
class EqualityPredicate(Predicate):
    functor = 'eq'
    arity = 2
    structure = 'fuzzy'

    def forward_predicate(self, x: torch.Tensor, y:torch.Tensor):
        return torch.exp(-torch.norm(x - y, dim=1))
```

```{code-cell} ipython3
x, y = ('x',), ('y',)

# One factory builds *and* lowers. Connectives/quantifiers are assembled as a
# formula AST (below); `ltn_factory.compile(formula)` then materializes it into a
# fuzzy circuit (using `structures`) and lowers it to a module (wiring in the
# predicate builders and enumerating the quantifiers over the variable domains).
ltn_factory = DeepLogModuleFactory(
    structures = {'fuzzy': ltn_fuzzy},
    aggregators = {'forall': forall, 'exists': exists},
    variables = {
        x: Domain.of_tensor(torch.randn(10, 2)),
        y: Domain.of_tensor(torch.randn(5, 2) * 2),
    },
    atom_builders = {('eq', 2, 'fuzzy'): EqualityPredicate},
)
```

```{code-cell} ipython3
# Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
# And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
# Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
# Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
# Forall = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMeanError(p=2),semantics="forall")
# Exists = ltn.Wrapper_Quantifier(ltn.fuzzy_ops.Aggreg_pMean(p=5),semantics="exists")
# Eq = ltn.Predicate.Lambda(lambda args: tf.exp(-tf.norm(args[0]-args[1],axis=1)))
```

```{code-cell} ipython3
Not = lambda a: UnaryOp('not', a)
And = lambda a, b: BinaryOp('and', a, b)
Or = lambda a, b: BinaryOp('or', a, b)
Implies = lambda a, b: BinaryOp('implies', a, b)
Equiv = lambda a, b: And(Implies(a, b), Implies(b, a))
Forall = lambda vars, a: Aggregation('forall', tuple(vars), (), a)
Exists = lambda vars, a: Aggregation('exists', tuple(vars), (), a)
Eq = lambda vars: Atom(with_structure(('eq', *vars), 'fuzzy'))
```

## Evaluating connectives directly

The connectives and quantifiers (`Not`, `And`, `Or`, `Implies`, `Equiv`, `Forall`, `Exists`) build a *formula AST* — cheap, inert data. To evaluate one on concrete tensors, compile it with `ltn_factory.compile(formula)` to get a callable `DeepLogModule`, then pass input tensors.

```{code-cell} ipython3
# Evaluate the Eq predicate on a single pair of inputs
eq_module = ltn_factory.compile(Eq([x, y]))
# Inputs need shape (batch, num_variables, features) — here (1, 1, 2)
out = eq_module(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[0.5, 0.5]]]))
print("Eq([0.5,0.5], [0.5,0.5]) =", out.item())
```

```{code-cell} ipython3
# Implies(Eq(x,y), Eq(x,y)) — in Reichenbach fuzzy logic, p → p is not a strict tautology
node = Implies(Eq([x, y]), Eq([x, y]))
module = ltn_factory.compile(node)
out = module(torch.tensor([[[0.5, 0.5]]]), torch.tensor([[[1.0, 0.0]]]))
print("Implies(Eq(x,y), Eq(x,y)) =", out.item())
```

```{code-cell} ipython3
# And, Or, Not on Eq predicates
and_node = And(Eq([x, y]), Eq([x, y]))
or_node = Or(Eq([x, y]), Eq([x, y]))
not_node = Not(Eq([x, y]))

x_val = torch.tensor([[[1.0, 0.0]]])
y_val = torch.tensor([[[0.5, 0.5]]])

for label, n in [("And", and_node), ("Or", or_node), ("Not", not_node)]:
    m = ltn_factory.compile(n)
    print(f"{label}: {m(x_val, y_val).item():.4f}")
```

```{code-cell} ipython3
# Equiv is defined as And(Implies(p,q), Implies(q,p))
equiv_node = Equiv(Eq([x, y]), Eq([x, y]))
module = ltn_factory.compile(equiv_node)
out = module(torch.tensor([[[0.3, 0.7]]]), torch.tensor([[[0.1, 0.9]]]))
print("Equiv(Eq(x,y), Eq(x,y)) =", out.item())
```

```{code-cell} ipython3
# A quantified formula is just inert AST until compiled
Forall([x], Eq([x, y]))
```

```{code-cell} ipython3
# Forall over x leaves y free, so the compiled module takes one input (y)
ltn_factory.compile(Forall([x], Eq([x, y])))(torch.tensor([[[0.1, -0.2]]]))
```

```{code-cell} ipython3
Forall([x, y], Eq([x, y]))
```

```{code-cell} ipython3
# Both variables bound → no free inputs
ltn_factory.compile(Forall([x, y], Eq([x, y])))()
```

```{code-cell} ipython3
ltn_factory.compile(Exists([x, y], Eq([x, y])))()
```

```{code-cell} ipython3
# Nested quantifiers: forall x . exists y . Eq(x, y)
ltn_factory.compile(Forall([x], Exists([y], Eq([x, y]))))()
```
