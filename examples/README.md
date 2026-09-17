# Examples

Executable notebooks that teach and exercise DeepLog. Every notebook here runs as part of the default `pytest` suite via `pytest-nbmake`, so they double as integration tests — if a rename breaks one, CI catches it.

## Start here

The two curated reading paths on the docs site are the fastest way in; both pick a subset of the notebooks below and sequence them.

- **ML path** — `shape` → `deeplogmodule` → `formula_to_module` → `semantic_loss`. For readers comfortable with PyTorch who want to understand what DeepLog adds.
- **NeSy path** — `symbol` → `shape` → `predicates` → `01_aggregation_basics` → `03_free_variables_and_batching` → `formula_to_module` → `problog` → `deepproblog`. For readers comfortable with symbolic reasoning who want to see how the pieces compile to differentiable modules.

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
| [`ast_and_rewrites/`](ast_and_rewrites/ast_and_rewrites.ipynb) | The materialized formula AST, `fold`/`map_children`, and the `recognize_expectation`/`recognize_posterior` rewrite passes. |
| [`predicates/`](predicates/predicates.ipynb) | Predicate modules: how symbolic atoms become executable tensor operations.                          |
| [`01_aggregation_basics/`](01_aggregation_basics/01_aggregation_basics.ipynb) | Aggregation syntax, finite domains, and how DeepLog builds aggregation modules.                     |
| [`03_free_variables_and_batching/`](03_free_variables_and_batching/03_free_variables_and_batching.ipynb) | Free variables are module inputs.                                                                   |
| [`circuits/`](circuits/circuits.ipynb) | The `Circuit` DAG and `to_module()`.                                                                |
| [`circuit_transformation/`](circuit_transformation/circuit_transformation.ipynb) | Transforming circuits between algebraic structures (boolean → probability, etc.).                   |
| [`language/`](language/language.ipynb) | Tour of the textual DeepLog formula language and its parser.                                        |

## Applications

| Notebook | What it teaches |
|---|---|
| [`problog/`](problog/problog.ipynb) | Running ProbLog programs: probabilistic facts, rules, queries, and conditioning on evidence with `:- ` integrity constraints (`P(q \| e)`). |
| [`deepproblog/`](deepproblog/deepproblog.ipynb) | Neural predicates: a fact's probability comes from a network instead of a constant (an annotated disjunction compiled via the MV-SDD backend), building up to the classic MNIST-addition experiment. **Slow.** |
| [`semantic_loss/`](semantic_loss/semantic_loss.ipynb) | A full ML training pipeline that uses a DeepLog formula as a differentiable loss term (semantic loss). **Slow.** |
| [`ltn/`](ltn/ltn.ipynb) | Reproducing a subset of the Logic Tensor Networks tutorial on top of DeepLog. |

## Running

Most notebooks run in a few seconds. The two marked **Slow** download MNIST and train a model, so a full run takes minutes:

```bash
pytest examples/                        # every notebook
pytest examples/deepproblog             # run one notebook only
DEEPLOG_FAST_DEV_RUN=1 pytest examples/  # one batch per fit, the way CI runs them
```

Needs the `pydeeplog[examples]` extra (`pip install -e ".[examples]"`).

No cell may be tagged `raises-exception`: nbmake would pass it whatever it raised. A cell that demonstrates an error catches that exception itself. `tests/test_examples.py` refuses the tag.

## In the browser

Every notebook has an **Open in Colab** badge under its title. It opens the notebook as released at the version in `pyproject.toml`, from that tag on the GitHub mirror, and the notebook's first code cell installs the same version (`pydeeplog[examples]==<version>`) when DeepLog is not importable. Where DeepLog is installed the cell does nothing, and the docs site hides it.

A new notebook needs both: the badge, pointing at its own path, and the install cell before its first code cell. Bumping the version in `pyproject.toml` means bumping it in every badge and install cell. `tests/test_examples.py` checks the badge, the cell and the version, and fails until they agree. The badge of a notebook added since the last release opens only once the next release is out.
