# Examples

Executable notebooks that teach and exercise DeepLog. Every notebook here runs as part of the default `pytest` suite via `pytest-nbmake`, so they double as integration tests — if a rename breaks one, CI catches it.

## Start here

The two curated reading paths on the docs site are the fastest way in; both pick a subset of the notebooks below and sequence them.

- **ML path** — `shape` → `deeplogmodule` → `formula_to_module` → `semantic_loss`. For readers comfortable with PyTorch who want to understand what DeepLog adds.
- **NeSy path** — `symbol` → `shape` → `predicates` → `01_aggregation_basics` → `03_free_variables_and_batching` → `formula_to_module` → `mnist_addition`. For readers comfortable with symbolic reasoning who want to see how the pieces compile to differentiable modules.

Outside the paths, the notebooks below are organised by concept — read the ones you need.

## Foundations

| Notebook | What it teaches |
|---|---|
| [`symbol/`](symbol/symbol.ipynb) | The `Symbol` type: DeepLog's lightweight tagged-tuple representation of anything symbolic. |
| [`shape/`](shape/shape.ipynb) | `SymTensor` and how symbolic shapes let DeepLog validate module composition. |
| [`deeplogmodule/`](deeplogmodule/deeplogmodule.ipynb) | `DeepLogModule` — the shape-aware `torch.nn.Module` subclass everything downstream builds on. |
| [`composition/`](composition/composition.ipynb) | Combining modules using `Sequential` and `ModuleCircuit`, and handling automatic shape transformations. |

## Core concepts

| Notebook | What it teaches                                                                                     |
|---|-----------------------------------------------------------------------------------------------------|
| [`formula_to_module/`](formula_to_module/formula_to_module.ipynb) | Compiling a logical formula straight into a runnable `DeepLogModule` via `parse_formula_to_module`. |
| [`predicates/`](predicates/predicates.ipynb) | Predicate modules: how symbolic atoms become executable tensor operations.                          |
| [`01_aggregation_basics/`](01_aggregation_basics/01_aggregation_basics.ipynb) | Aggregation syntax, finite domains, and how DeepLog builds aggregation modules.                     |
| [`03_free_variables_and_batching/`](03_free_variables_and_batching/03_free_variables_and_batching.ipynb) | Free variables are module inputs.                                                                   |
| [`circuits/`](circuits/circuits.ipynb) | The `Circuit` DAG and `to_module()`.                                                                |
| [`circuit_transformation/`](circuit_transformation/circuit_transformation.ipynb) | Transforming circuits between algebraic structures (boolean → probability, etc.).                   |
| [`language/`](language/language.ipynb) | Tour of the textual DeepLog formula language and its parser.                                        |
| [`dimacs/`](dimacs/dimacs_cnf.ipynb) | Parsing DIMACS CNF input into DeepLog formulas.                                                     |

## Applications

| Notebook | What it teaches |
|---|---|
| [`semantic_loss/`](semantic_loss/semantic_loss.ipynb) | A full ML training pipeline that uses a DeepLog formula as a differentiable loss term (semantic loss). **Slow.** |
| [`mnist_addition/`](mnist_addition/mnist_addition.ipynb) | The classic NeSy experiment: classify pairs of MNIST digits by their sum. **Slow.** |
| [`ltn/`](ltn/ltn.ipynb) | Reproducing a subset of the Logic Tensor Networks tutorial on top of DeepLog. |

## Running

Most notebooks run in a few seconds. The two marked **Slow** download MNIST and train a model; they're gated behind `--slow` in the pytest runner:

```bash
pytest examples/                  # all fast notebooks (+ tests/)
pytest examples/ --slow           # include semantic_loss and mnist_addition
pytest examples/mnist_addition    # run one notebook only
```

Needs the `deeplog[examples]` extra (`pip install -e ".[examples]"`).
