# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/deeplog/`: `symbol.py` / `shape.py` (symbolic names and layouts),
  `algebraic.py` (the algebraic taxonomy), `module/` (torch modules with shape validation),
  `circuit/` (the integer-graph engine and its passes), `formula/` (AST, factories, passes),
  `grounding/` (plain-Prolog SLD resolution), `systems/` (DeepProbLog and other front-ends).
- Tests are under `tests/deeplog/`, mirroring the source tree. Use existing helpers like
  `tests/deeplog/module/_utils.py`.
- Docs and examples: `docs/`, `site/`, `examples/` notebooks. Packaging config in
  `pyproject.toml`, `requirements.txt`.

## Build, Test, and Development Commands
- Run targeted tests: `pytest tests/deeplog/module/test_simplify_module.py`.
- Run full suite: `pytest`. `testpaths` is `["tests", "examples"]` and `--nbmake` is on by
  default, so **the example notebooks are part of the test suite** and execute on every run.
- Optional backends (`pysdd`, `pymvsdd`, `klay`, `janus-swi`) gate large parts of the suite.
  Run in an environment that has them, or a passing run proves less than it looks.
- `pre-commit` runs ruff, ruff-format and pyright; pyright resolves its interpreter from
  whichever `python` is on `PATH` at commit time, so the project environment must be active.

## Coding Style & Naming Conventions
- Python 3.12+ typing (`tuple[...]`, `list[...]`); prefer the `as_tuple` helper for uniform
  tuple handling. 4-space indentation.
- `__init__.py` holds the package docstring, re-exports and `__all__` — never a definition.
  A package needing a shared entry point names a file for its job (`dispatch.py`), and
  `common.py` / `utils.py` / `helpers.py` are not job names.
- Docstrings are published: `site/source/conf.py` points autoapi at `src/deeplog`. They state
  what the code does, what the arguments mean and what a caller must guarantee. Design
  rationale goes in the CHANGELOG or the merge request. State a fact at one level — usually
  the module — and let the class and methods under it defer.
- Minimal inline comments, and only for non-obvious logic.

## Design Rules

These are the rules the 4.0.0 rework was an application of. They are ordered by how often
they decide a question, not by importance.

### 1. A fact lives in the layer it is about
If a layer carries a flag, override, annotation or class stating something about a
*different* layer, the concept is mis-modelled. Fix the model; do not add the annotation.
Removed under this rule: `Circuit.deterministic` (a fact about the pass that produced the
circuit), `structure_override` (a rewrite, stored as a compile-time mode),
`AlgebraicStructure.circuit_operators` (the algebra answering a compiler's question),
`DeepLogModule.structure` (a module describing its values — and one slot cannot describe
columns in different algebras), `Semifield.eps` (one dtype's numerics pinned on the algebra).
Before adding one, ask which layer the fact is about and whether that layer can establish it
by construction.

### 2. Establish, don't assert
A property a later step depends on is *produced* by an earlier one, never declared alongside
the data. `knowledge_compile` makes a circuit deterministic and decomposable, which is what
makes reading `or` as a semiring sum exact, so nothing records how to read the nodes. Passes
share one contract — `(circuit, node_map)` — so ordering is the only thing carrying a
precondition, and passes compose.

### 3. Cut, don't forbid
The representation admits everything the theory admits; making it executable is the
executor's problem. A `Semifield` declares an invertible product without knowing that no
backend has a node for a quotient: `circuit/split.py` cuts the graph at such an operator,
applies the algebra's own `operator_fns` entry to the compiled operands, and compiles the
region above bounded at the cut. Never restrict what may be *built* to suit what some backend
can walk.

### 4. The type hierarchy mirrors the theory
`Algebra` (a complement) and `Semifield` (a multiplicative inverse) are independent extensions
of `Semiring`, composed by multiple inheritance, because a chain would assert that division
implies negation. Each public class names exactly one axiom set; one that names no new axiom
stays private. The regression tests are mathematical statements — `isinstance(BOOLEAN,
Semifield)` is `False`.

### 5. Never fall back to a plausible wrong answer
A default is acceptable only where it degrades to slow-but-correct — an unregistered structure
routing to the generic evaluator. It is never acceptable where it degrades to fast-but-wrong.
Every silent case 4.0.0 removed returned numbers: a Klay semiring table defaulting to `real`
discarded a custom structure's `operator_fns`; a missing structure label routed through a
`real → probability` cast; a blanket `.to(torch.float32)` hook absorbed integer truth values;
`eps=1e-12` stored as exactly zero in float16; naming the same root twice collapsed two
outputs into one. Prefer a raise that names the object and the alternative.

**A semantic claim is keyed by identity, not by a name.** `register_klay_semiring` is keyed by
structure object, because "this structure's product and sum *are* that semiring's" is not
something spelling the operators `and`/`or` can establish.

### 6. Derive values from what they are about; no tuning knobs
The division floor is `torch.finfo(dtype).tiny`, not a constant. A value's dtype comes from
its algebra, not from the argument that produced it. A compiled circuit evaluates in its
input's dtype. Where behaviour genuinely varies, supply the whole function (`division_fn`) —
a scalar knob beside it is a representable state in which the knob is ignored.

### 7. Recognition is optional; the general path is mandatory
Passes in `formula/passes.py` are recognition-only: a `divide` that `recognize_posterior` does
not recognize still lowers, as a plain division. What a sub-formula *means* (`passes.py`) is
separate from *how it is computed* (`strategies.py`). Any fast path must be deletable without
making a result wrong.

### 8. One word, one operation
`knowledge_compile` is circuit→circuit, `lower` is circuit→module, and `compile` is the whole
pipeline. Four `compile_<backend>` functions returning two different types is the failure
mode this names.

### 9. A compiled artefact speaks the user's names
Input slots and output columns are named for the user's atoms and for what they compute — an
MV-SDD categorical slot by its annotated-disjunction branch atom, not a synthetic `@cat`; a
quotient's column by the division, not relabelled to its numerator. Mint an `@`-name only
where there is no user-side referent at all, such as a padding slot.

### 10. A rename is a rename
No aliases, no deprecation shims, no back-compat layer. Update every call site; the CHANGELOG
carries the migration and the major version carries the cost. A refactor is finished when the
sweep is: grep every removed or renamed name across `src/ tests/ docs/ examples/`, resolve
Sphinx cross-references against the live package, and delete what the change orphaned — a
class nothing constructs, an exception nothing raises.

## Testing Guidelines
- Framework: `pytest`. Place new unit tests beside the module they cover, mirroring `src/`.
- Name tests descriptively (`test_simplify_duplicate_inputs`) and assert both shape and tensor
  content where relevant.
- A conceptual claim gets a test that states it: `isinstance(BOOLEAN, Semifield) is False`
  tests rule 4 the way an output comparison never could.
- Run affected tests before raising a merge request, and the notebooks when the change touches
  a documented API.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits, imperative and scoped — `fix(formula): …`,
  `refactor(circuit)!: …` where the `!` marks a breaking change.
- Every commit must pass on its own; do not leave a tree that only builds after the next one.
- MRs: include a brief summary, affected areas, and test evidence. Link issues when
  applicable.
- The CHANGELOG entry carries the reasoning the docstrings do not — what was wrong before,
  why this is the fix, and what a caller must change.

## Architecture Notes
- **Shapes**: `SymTensor` declares symbolic layouts; `construct_transformation` builds reshape
  paths between shapes. A value's algebra is recorded on the symbol naming it — read it back
  with `sole_structure(shape)` or `structures(shape)`.
- **Modules**: `DeepLogModule` wraps torch modules with shape validation, `Sequential`
  composes them, `simplify_module` trims inputs and prepends necessary transforms.
- **Formula**: `parse_formula_to_module` compiles textual formulas; `DeepLogModuleFactory`
  interprets a `FormulaNode` AST, and `fold` re-emits a tree through any factory.
- **Pipeline**: a boolean circuit is rewritten by `knowledge_compile` into a d-DNNF, then by
  `transform_circuit` into the target algebra, then by `to_module` (`circuit/lower/`) into a
  torch module. `backends.select_backend` reads the structure alone and never the graph.
- **Notebooks**: when referring to API types, link to their import path so docs render
  correctly (e.g. `[DeepLogModule](deeplog.module.deeplog_module.DeepLogModule)`).
