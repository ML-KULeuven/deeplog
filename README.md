# DeepLog
[![GitHub Release](https://img.shields.io/github/v/release/ML-KULeuven/deeplog?display_name=tag)](https://github.com/ML-KULeuven/deeplog/releases)
[![PyPI version](https://img.shields.io/pypi/v/pydeeplog)](https://pypi.org/project/pydeeplog/)
[![License](https://img.shields.io/github/license/ML-KULeuven/deeplog)](LICENSE)

DeepLog is an operational framework for building neurosymbolic (NeSy) systems. Instead of presenting a monolithic stack, DeepLog provides high-performance building blocks that plug directly into PyTorch-first workflows so you can compose differentiable learning and symbolic reasoning with predictable interfaces.

## Why DeepLog?

- **Symbol-first modeling** – declare symbolic tensors and shapes, then let DeepLog check interfaces between modules before you wire them into larger systems.
- **Logic-aware modules** – compile formulas into differentiable modules, wrap existing PyTorch components, and connect them to multiple reasoning backends.
- **Backend flexibility** – start with the pure Python engine or enable the Janus/SWI-Prolog backend for lower latency inference without changing your training code.
- **Batteries included** – tutorial notebooks, example projects (semantic loss, MNIST addition, …), and a Sphinx site walk you from “hello world” to custom NeSy stacks.
- **Research to production** – typing, validation, and extensive tests keep advanced reasoning pipelines trustworthy when you move from prototypes to production workloads.

## Installation


DeepLog publishes optional extras so you can extend the base install as needed:

| Extra | Description |
| ----- | ----------- |
| `pydeeplog[examples]` | Adds interactive notebook tooling (Jupyter) plus Lightning/torchvision/torchmetrics for tutorials. |
| `pydeeplog[janus_engine]` | Installs the Janus SWI-Prolog bridge for the highest performance Prolog backend. |
| `pydeeplog[tests]` | Adds pytest, coverage and supporting utilities for contributors. |
| `pydeeplog[site]` | Installs the documentation/notebook toolchain (Sphinx, PyData theme, myst-nb, nbconvert, ipykernel, Lightning/torchvision/torchmetrics, …). |

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

### Docker images (CPU/GPU)

The root `Dockerfile` builds a CPU CI base via a `DEVICE` build arg (default CPU). For local builds:

```bash
docker build -t deeplog:cpu --build-arg DEVICE=cpu .
```

CI builds a CPU image from this Dockerfile. The devcontainer image is built from the same base.

The same image definition is used in CI. For local validation, build it with the command above and then run the contributor checks from [CONTRIBUTING.md](CONTRIBUTING.md) in your working tree.

## Quick start

```python
import torch
from deeplog import SymTensor, TransformationModule, parse_symbol

# Declare a symbolic batch of two digits and map real-valued logits to probabilities.
digits = SymTensor([parse_symbol("digit_a"), parse_symbol("digit_b")])
sigmoid_head = TransformationModule(digits, "real", "probability")

print(sigmoid_head(torch.tensor([[1.0, -2.0]])))
# tensor([[0.7311, 0.1192]])
```

`TransformationModule` validates tensor shapes against the symbolic specification, so interface mismatches surface as Python errors instead of silent shape bugs when you wire modules into larger reasoning pipelines.

For a taste of the symbolic side, compile a logical formula straight into a differentiable module:

```python
from deeplog import parse_formula_to_module

# Count satisfying assignments of A ∨ B over booleans (= 3: TT, TF, FT).
module = parse_formula_to_module(
    "sum(A): sum(B): =(A,true)_boolean or =(B,true)_boolean"
)
print(int(module()))  # 3
```

See `examples/` for end-to-end notebooks (MNIST addition, semantic loss, DIMACS CNF, LTN, …) and the full API reference on the docs site.

## Documentation & tutorials

- **Landing page & docs** – All documentation lives under [`site/`](site/). Build it locally with the steps in [Building the documentation site](#building-the-documentation-site).
- **Notebooks** – Reproduces the semantic loss workflow, DeepLog module deep dives, MNIST addition with DeepProbLog, and more. Launch them from the `examples/` directory after installing `deeplog[examples]`.
- **API reference** – Generated automatically via `sphinx-autoapi`, covering symbols, shapes, modules, and engine utilities.

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

Visit `http://localhost:8000` to browse the site. Tools like `sphinx-autobuild` can provide live reloads, but the above workflow is the canonical build pipeline.

## Support & community

- **Bugs & questions** – Open an issue on [GitHub](https://github.com/ML-KULeuven/deeplog/issues) and include reproduction steps plus relevant version information.
- **Security reports** – Please do **not** file public issues for security vulnerabilities; instead reach the maintainers privately (see `CONTRIBUTING.md` for details).
- **Roadmap discussions** – Feature proposals, architectural questions, and broader discussions are welcome via issues.

## Contributing

We welcome contributions of all kinds—bug reports, docs, examples, and new modules. Review the [contributing guide](CONTRIBUTING.md) for coding standards, triage practices, and tips for a smooth review cycle. If you are looking for a first issue, check the tracker for `good first issue` and `help wanted` labels.

## License

DeepLog is released under the [LGPL-2.1 license](LICENSE) © DTAI Research Group, KU Leuven.
