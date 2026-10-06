# DeepLog
[![GitHub Release](https://img.shields.io/github/v/release/ML-KULeuven/deeplog?display_name=tag)](https://github.com/ML-KULeuven/deeplog/releases)
[![PyPI version](https://img.shields.io/pypi/v/pydeeplog)](https://pypi.org/project/pydeeplog/)
[![License](https://img.shields.io/github/license/ML-KULeuven/deeplog)](LICENSE)
[![DOI](https://zenodo.org/badge/901879738.svg)](https://doi.org/10.5281/zenodo.20408413)

DeepLog is an operational framework for building neurosymbolic (NeSy) systems. Instead of presenting a monolithic stack, DeepLog provides high-performance building blocks that plug directly into PyTorch-first workflows so you can compose differentiable learning and symbolic reasoning with predictable interfaces.

## Why DeepLog?

- **Symbol-first modeling** – declare symbolic tensors and shapes, then let DeepLog check interfaces between modules before you wire them into larger systems.
- **Logic-aware modules** – compile formulas into differentiable modules, wrap existing PyTorch components, and connect them to multiple reasoning backends.
- **Backend flexibility** – start with the pure Python engine or enable the Janus/SWI-Prolog backend for lower latency inference without changing your training code.
- **Batteries included** – tutorial notebooks, example projects (semantic loss, MNIST addition, …), and a Sphinx site walk you from “hello world” to custom NeSy stacks.
- **Research to production** – typing, validation, and extensive tests keep advanced reasoning systems trustworthy when you move from prototypes to production workloads.

## Installation


DeepLog publishes optional extras so you can extend the base install as needed:

| Extra | Description |
| ----- | ----------- |
| `pydeeplog[examples]` | Adds interactive notebook tooling (Jupyter) plus Lightning/torchvision/torchmetrics for tutorials. |
| `pydeeplog[janus_engine]` | Installs the Janus SWI-Prolog bridge for the highest performance Prolog backend. |
| `pydeeplog[mvsdd]` | Installs MV-SDD, the knowledge compiler for formulas that test several values of one variable, such as a digit classifier's ten classes in MNIST addition. Without it, such an expectation is enumerated. |
| `pydeeplog[tests]` | Adds pytest, coverage and supporting utilities for contributors. |
| `pydeeplog[site]` | Installs the documentation/notebook toolchain (Sphinx, PyData theme, myst-nb, Jupytext, ipykernel, Lightning/torchvision/torchmetrics, …). |

Combine extras as needed, for example `pip install ".[examples,tests]"`.

### Optional: Janus/SWI-Prolog backend

The Janus backend depends on a local SWI-Prolog installation. On Ubuntu you can install it with:

```bash
sudo apt-get install software-properties-common
sudo apt-add-repository ppa:swi-prolog/stable && sudo apt-get update
sudo apt-get install swi-prolog
```

Once SWI-Prolog is available, install the Python bindings via `pip install "pydeeplog[janus_engine]"`.

### Install from source

```bash
git clone https://github.com/ML-KULeuven/deeplog.git
cd deeplog
pip install -e ".[examples,tests]"
```

Editable installs refresh automatically when you change the source tree, which is handy when contributing.

## Quick start

```python
import torch
from deeplog import SymTensor, WrappedModule, parse_symbol

# Wrap a plain PyTorch head with symbolic input/output shapes so DeepLog can
# validate every tensor that flows through it.
digits = SymTensor([parse_symbol("digit_a"), parse_symbol("digit_b")])
sigmoid_head = WrappedModule(torch.nn.Sigmoid(), digits, digits, name="sigmoid_head")

print(sigmoid_head(torch.tensor([[1.0, -2.0]])))
# tensor([[0.7311, 0.1192]])
```

`WrappedModule` validates tensor shapes against the symbolic specification, so interface mismatches surface as Python errors instead of silent shape bugs when you wire modules into larger reasoning systems.

For a taste of the symbolic side, compile a logical formula straight into a differentiable module:

```python
from deeplog import parse_formula_to_module

# Count satisfying assignments of A ∨ B over booleans (= 3: TT, TF, FT):
# cast into the reals, where true is 1, and sum.
module = parse_formula_to_module(
    "sum(A): sum(B): (=(A,true)_boolean or =(B,true)_boolean)_real"
)
print(int(module()))  # 3
```

See `examples/` for end-to-end notebooks (MNIST addition, semantic loss, LTN, …) and the full API reference on the docs site.

## Documentation & tutorials

- **Landing page & docs** – All documentation lives under [`site/`](site/). Build it locally with the steps in [Building the documentation site](#building-the-documentation-site).
- **Notebooks** – The notebooks in [`examples/`](examples/) teach DeepLog concept by concept; they are listed [below](#notebooks). The docs site sequences a selection of them into two reading paths, one for ML practitioners and one for neurosymbolic developers, and each notebook's page there has an **Open in Colab** badge.
- **API reference** – Generated automatically via `sphinx-autoapi`, covering symbols, shapes, modules, and engine utilities.

### Notebooks

#### Foundations

| Notebook | What it teaches |
|---|---|
| [`symbols_and_shapes`](examples/symbols_and_shapes.md) | `Symbol`, the tuple that names terms, atoms and values with their algebra, and `SymTensor`, which names every entry of a tensor. |
| [`deeplogmodule`](examples/deeplogmodule.md) | `DeepLogModule` — the shape-aware `torch.nn.Module` subclass everything downstream builds on. |
| [`composition`](examples/composition.md) | Combining modules using `Sequential` and `compose_modules`, and handling automatic shape transformations. |

#### Core concepts

| Notebook | What it teaches                                                                                     |
|---|-----------------------------------------------------------------------------------------------------|
| [`formula_to_module`](examples/formula_to_module.md) | A constraint compiled into a `DeepLogModule` with `parse_formula_to_module`: fed a batch, differentiated, wired to a network, and computed by counting, enumeration or sampling. |
| [`formula_ast`](examples/formula_ast.md) | The formula AST: reading it, building it from Python, and the proofs a grounder returns as ASTs. |
| [`predicates`](examples/predicates.md) | Predicate modules: how symbolic atoms become executable tensor operations.                          |
| [`aggregation_basics`](examples/aggregation_basics.md) | Aggregation syntax, finite domains, aggregation operators, and expectations under a distribution.   |
| [`free_variables_and_batching`](examples/free_variables_and_batching.md) | Free variables are module inputs.                                                                   |
| [`circuits`](examples/circuits.md) | The `Circuit` DAG and `to_module()`.                                                                |
| [`language`](examples/language.md) | Tour of the textual DeepLog formula language and its parser.                                        |

#### Applications

| Notebook | What it teaches |
|---|---|
| [`problog`](examples/problog.md) | Running ProbLog programs: probabilistic facts, rules, queries, and conditioning on evidence with `:- ` integrity constraints (`P(q \| e)`). |
| [`deepproblog`](examples/deepproblog.md) | Neural predicates: a fact's probability comes from a network instead of a constant (an annotated disjunction compiled via the MV-SDD backend), building up to the classic MNIST-addition experiment, counted exactly or estimated by sampling. **Slow.** |
| [`semantic_loss`](examples/semantic_loss.md) | A full ML training loop that uses a DeepLog formula as a differentiable loss term (semantic loss). **Slow.** |
| [`ltn`](examples/ltn.md) | Reproducing a subset of the Logic Tensor Networks tutorial on top of DeepLog. |

## Development workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for code quality and testing guidelines.

Public bug reports, questions, and feature requests live on GitHub. Development and code review happen privately, and tagged releases (`v*`) are mirrored to the public GitHub repository after release.

## Building the documentation site

The landing page and docs live under `site/` (Sphinx + PyData theme). To preview them locally:

1. Install the site tooling (from the repo root):

   ```bash
   pip install -r site/requirements.txt
   ```

2. Build the static HTML:

   ```bash
   (cd site && make html)
   ```

3. Serve the generated pages from `site/build/html`. Any static file server works, e.g.:

   ```bash
   python -m http.server --directory site/build/html 8000
   ```

Visit `http://localhost:8000` to browse the site. Tools like `sphinx-autobuild` can provide live reloads, but the steps above are the canonical way to build the site.

## Support & community

- **Bugs & questions** – Open an issue on [GitHub](https://github.com/ML-KULeuven/deeplog/issues) and include reproduction steps plus relevant version information.
- **Security reports** – Please do **not** file public issues for security vulnerabilities; instead reach the maintainers privately (see `CONTRIBUTING.md` for details).
- **Roadmap discussions** – Feature proposals, architectural questions, and broader discussions are welcome via issues.

## Contributing

We welcome contributions of all kinds—bug reports, docs, examples, and new modules. Review the [contributing guide](CONTRIBUTING.md) for coding standards, triage practices, and tips for a smooth review cycle. If you are looking for a first issue, check the tracker for `good first issue` and `help wanted` labels.

## License

DeepLog is released under the [LGPL-2.1 license](LICENSE) © DTAI Research Group, KU Leuven.
