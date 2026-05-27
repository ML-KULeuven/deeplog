# DeepLog Module: Goal and Capabilities

DeepLog is a neurosymbolic (NeSy) toolbox that layers symbolic structure on PyTorch without replacing your stack. The DeepLog module layer wraps tensors and torch modules with symbolic shapes so you can compose differentiable learning, logical formulas, and backend reasoning engines with predictable interfaces.

## What the module layer is for
- Guard PyTorch pipelines with symbolic input/output shapes (`SymTensor`) so interface errors are caught early.
- Turn logical formulas into executable modules that drop into training loops alongside ordinary heads and losses.
- Bridge multiple reasoning backends (pure Python, optional Janus/SWI-Prolog) without changing model code.

## Core ideas
- **Symbols → SymTensor → DeepLogModule**: symbolic labels index tensor positions; modules declare `get_input_shape`/`get_output_shape` and auto-validate tensors at runtime.
- **Automatic reshaping**: `construct_transformation` and `Sequential` insert flattening, broadcasting, indexing, and tuple selection so mismatched symbolic layouts are reconciled automatically.
- **Circuits as IR**: `ModuleCircuit` evaluates DAGs of modules, filling in missing producers and ordering nodes topologically.
- **Formula compilation**: `parse_formula_to_module` lowers textual formulas through `DeepLogModuleFactory` into circuits of predicates, transformations, and aggregations.

## Two complementary views (per tutorial)
- **Symbolic wrapper around Torch**: start with symbols/shapes, wrap torch modules as `DeepLogModule`, and rely on automatic shape validation/reshaping when composing.
- **Tensorizing formulas**: use the DeepLog language, predicates, and aggregation to compile symbolic formulas into executable modules that drop into training or inference pipelines.

## Built-in capabilities
- **Shape-aware wrappers**
  - `DeepLogModule` base class adds validation and graph rendering hooks.
  - `WrappedModule` wraps arbitrary callables with declared shapes; optional `vmap` execution.
  - `simplify_module` and `reshape_*` helpers deduplicate/seed inputs and adjust public signatures.
- **Reshape/transformation ops**
  - `IndexingTransform`, `TupleIndexTransform`, `FlattenTransform`, and `BroadcastModule` build reshape paths between symbolic layouts.
  - `TransformationModule` maps between logical structures (e.g., real → probability via sigmoid) while preserving symbolic labels.
- **Predicate modules**
  - `Predicate` base handles mixing constants/symbols and batching over argument bindings.
  - Built-ins: `ProbabilityPredicate` (prob/logprob labels), `EqualityPredicate` over finite domains, `SumsPredicate` (x+y=z), `NetworkPredicate` for torch-backed predicates, and `IndexPredicate` to read earlier outputs by symbolic index.
- **Aggregation**
  - `AggregationModule` performs reductions (e.g., sum) over variable domains, broadcasting assignments and reshaping outputs consistently.
- **Circuits and composition**
  - `ModuleCircuit` stitches modules into DAGs, caches intermediate tensors, and emits tensors in symbolic order.
  - `Sequential` chains modules; `insert_transformations` keeps neighbor modules shape-compatible.

## How it fits in an ML workflow
1. Declare symbols and shapes for model heads (e.g., class logits as `SymTensor`).
2. Wrap heads with `DeepLogModule` (manually or via `WrappedModule`).
3. Parse constraints or formulas with `parse_formula_to_module`; the factory injects predicate modules, aggregation, and structure transforms, returning a `DeepLogModule`.
4. Compose everything with `Sequential` or `ModuleCircuit`; DeepLog auto-inserts reshape transforms and validates tensors during forward passes.
5. Train normally: constraints can act as regularizers (semantic loss), auxiliary heads, or logical evaluators; swap inference backends when needed without touching model code.

## Example touchpoints in the repo
- Quick start in `README.md` shows a symbol-aware sigmoid head.
- Formula compilation pipeline in `src/deeplog/formula/deeplogmodulefactory.py`.
- Shape utilities and symbolic tensors in `src/deeplog/shape.py`.
- Module primitives in `src/deeplog/module/` (reshape, predicates, aggregation, circuits).
- Tutorials and notebooks under `site/` (semantic loss, MNIST addition, language/predicate demos).

Use this layer whenever you need PyTorch modules that understand symbolic structure, can be validated automatically, and can be composed with logical reasoning components without leaving the torch ecosystem.
