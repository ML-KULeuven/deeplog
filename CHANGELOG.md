# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.5] - 2026-06-17

### Fixed
- `compile_to_module` no longer raises `ValueError: Ambiguous probability mapping` when distinct labeled atoms share arguments (e.g. `nn1(x1) :: a(x1).` and `nn2(x1) :: b(x1).`). The boolean-to-probability leaf mapping is now built directly from the engine's atom labels, which is unambiguous, instead of being reconstructed by an argument-overlap heuristic.

### Changed
- `deeplog.formula.distribution.build_leaf_mapping` now takes the engine `labels` map (boolean atom → probability label atom) and builds the leaf mapping directly. The argument-overlap heuristic, along with the internal `build_probability_distribution` and `factorize` helpers, has been removed. The `expectation` aggregation builder no longer accepts a probability-formula parameter.

## [3.0.4] - 2026-06-04

### Fixed
- Plain (non-`Semiring`) `AlgebraicStructure`s with custom `operator_fns` now compile through the generic per-operator evaluator instead of the Klay semiring fast path, so a custom connective (e.g. a fuzzy `or` of `a + b - a*b`) is honored rather than silently replaced by the semiring sum/product.
- `parse_formula_to_module` now parses formulas of bare-atom leaves (`name_struct`, no predicate arguments); previously such a leaf in operator position was mis-lexed as an operator (`Unknown operator '...'`).
- A leaf-only formula now compiles to an identity (pass-through) module, and a numeric-constant leaf (e.g. `1.0_fuzzy`) to a constant module — both via a one-node circuit — instead of raising.

## [2.2.0] - 2026-04-08

### Added
- Circuit transformation API: `transform()` converts a circuit from one algebraic structure to another with automatic operator mapping between semirings; `transform_nodes()` batch-transforms multiple `CircuitNode` objects sharing the same circuit in a single pass; `CircuitNode.transform()` provides per-node convenience.
- `Semiring` and `Algebra` subclasses of `AlgebraicStructure` that expose named roles (`product`, `sum`, `negation`) used by the automatic operator mapping in `transform()`.
- Compilation pipeline for DeepProbLog engine results: `compile_to_module()` performs end-to-end compilation (probability distribution, expectation aggregation, batch circuit transformation, predicate module composition); `build_probability_distribution()` builds a factorized probability circuit from engine atom labels.
- `ExpectationNode` lazy wrapper that defers boolean-to-probability circuit transformation until batch compilation, enabling efficient shared-subgraph traversal.

### Changed
- `AggregationModule` now accepts a `variables: list[Symbol]` parameter (plural) for joint multi-variable aggregation, replacing the previous single-variable interface.
- `expectation` aggregation builder now creates `ExpectationNode` instances that support batch transformation via `transform_nodes`, instead of eagerly transforming each formula individually.

## [2.1.0] - 2026-03-02

### Added
- Added `expectation` aggregation operator for computing expected values of boolean formulas over probabilistic inputs (e.g., `expectation(Var1, Var2): formula`).

## [2.0.0] - 2026-02-23

### Added
- `LogProbabilityPredicate` class for predicates that output in log-probability space, split from `ProbabilityPredicate`.
- Test suite for circuit constants (`test_circuit_constants.py`) verifying that constants map to neutral elements.

### Changed
- **BREAKING:** `ProbabilityPredicate` no longer accepts a `structure` parameter to switch between probability and log-probability outputs. Use `LogProbabilityPredicate` for log-probability outputs instead.
- **BREAKING:** `NetworkPredicate` is no longer a class. Use `get_network_predicate(functor, arity, structure, module)` to create a predicate class that wraps a neural network module.
- **BREAKING:** `IndexPredicate` has been removed. Indexing functionality is now built into `get_network_predicate`.
- **BREAKING:** `Predicate.get_factory()` has been removed. Use `functools.partial` to create predicate factories with preset parameters.
- **BREAKING:** `WMCModuleFactory` has been renamed to `DeepLogModuleFactory`. Use `atom_builders` dict instead of `predicate_factories` list.
- **BREAKING:** `factory.to_multiroot_module()` has been replaced with `to_module(*nodes, names=answers, deterministic=True)`.
- Circuit constants (both simple like `('0',)` and structured like `('_', ('0',), ('probability',))`) now correctly map to neutral elements (true/false nodes) instead of being treated as inputs.
- `::` annotated atoms from DeepProbLog (e.g., `classifier(I,N) :: digit(I,N)`) now correctly wire the neural predicate outputs to circuit inputs.
- `NetworkPredicate.forward_predicate` now converts index tensors to `torch.long` before indexing.

### Fixed
- Fixed circuit input shapes including constants that should be neutral elements.
- Fixed shape mismatch between `NetworkPredicate` outputs and circuit inputs for `::` annotated atoms.
- Fixed notebooks to use `reshape_input` for explicit input ordering instead of relying on internal ordering.

### Removed
- `IndexPredicate` class (functionality merged into `get_network_predicate`).
- `Predicate.get_factory()` classmethod (use `functools.partial` instead).

## [1.10.0] - 2026-02-23

### Added
- Added LTN (Logic Tensor Networks) demo example notebook (`examples/ltn/ltn.ipynb`) showcasing neuro-symbolic integration with MNIST addition.
- Introduced `DeepLogCircuitNode` wrapper for tracking circuit nodes with attached leaf modules.
- Added `ExpectationModule` for computing expectations over probabilistic distributions.
- Added `with_structure` helper for structure-aware formula handling.
- Added `algebraic.py` module with algebraic utilities.
- Added utility functions in `util.py` with corresponding tests.

### Fixed
- Fixed aggregation module handling for correct tensor operations.
- Fixed unary operators in formula processing.

### Changed
- Reworked `DeepLogModuleFactory` to operate on `DeepLogCircuitNode` wrappers with improved caching and structure awareness.
- Refactored circuit and module creation flow for better factory integration.
- Updated predicate modules (`builtin_predicates.py`, `predicate.py`) with improved batching and structure handling.
- Refactored `KlayCircuit` and circuit base class for cleaner node management.
- Updated formula modules and module circuit for improved factory compatibility.
- Renamed `test_translation.py` to `test_deeplog_formula_modules.py` for clarity.
- Bumped numpy dependency to `>=2.4.0`.

### Removed
- Removed `test_predicate_get_factory.py` (functionality consolidated elsewhere).
- Removed `test_engine_with_klay.py` integration tests (pending WMC rework).

## [1.9.1] - 2026-01-16

### Added

### Fixed

### Changed
- Switched textual aggregation syntax to `op(binders; params): body`, replacing the legacy `agg(op, Var):` form.
- Updated docs publishing so the main branch builds at the root, previews use per-branch prefixes, and the version switcher points main to `/`.
- Limited multiversion builds to the latest patch tag for each major.minor series.

### Removed

## [1.9.0] - 2026-01-12

### Added
- Added builder protocol types and default predicate/circuit registries under `deeplog.formula.deeplogmodulefactory` so downstream projects can register shared atom and circuit builders ahead of factory creation.
- Exposed formula factory types and helpers (`DeepLogModuleFactory`, `SymbolicFormulaFactory`, `build_domains`, `get_variable_domain`, `strip_literal_structure`, `get_structure`) from `deeplog.formula` to simplify imports.

### Changed
- Refactored `DeepLogModuleFactory` to live under `deeplog.formula.deeplogmodulefactory`, operate on `CircuitNode` wrappers, and cache circuits per structure while allowing `register_predicate_factory`/`register_circuit_factory` overrides.
- `Circuit`/`CircuitNode` conversion now tracks attached leaf modules, materializes missing predicate modules when building a `ModuleCircuit`, and accepts extra modules and explicit root names when emitting modules.
- Updated examples and notebooks to the new factory layout and naming so tutorials mirror the refactored API.

### Fixed

### Removed
- Disabled the weighted model counting factory helpers and DeepProbLog engine tests pending a rework of the WMC flow.

## [1.8.0] - 2025-11-27

### Added
- Introduced `WMCFactoryMixin` plus `SymbolicWMCFactory` and `WMCModuleFactory` so WMC semantics can be reused across symbolic and module pipelines.
- Added WMC parsing helper `parse_formula_to_wmc` structure selection and refreshed example notebooks (`mnist_addition`, `exactly_one`) to compile constraints via WMC.
- Engines now automatically register a positive-only `ProbabilityPredicate` when constructing their WMC factory, and `DeepLogModuleFactory` exposes `add_predicate_factory` for post-init registration.

### Changed
- Engines now always compile through `WMCModuleFactory`, with EngineFactory enforcing the WMC path and mapping boolean surface operators to the chosen probability/logprobability semiring.
- Engine unit/integration tests now rely on the WMC factories, and the DeepProbLog language docs describe the WMC-based engine semantics.

### Fixed
- The exactly-one constraint notebook now builds boolean formulas and evaluates them through WMC, avoiding probabilistic-surface inconsistencies.

### Changed
- The textual formula parser now ignores full whitespace (not just inline spaces), so indented multi-line rules parse correctly and the `formula_to_module` tutorial feeds those strings straight into `DeepLogModuleFactory` without manual tuple construction.
- `ProbabilityPredicate` accepts bare probability labels (and auto-converts them when the module emits log-probabilities), with regression tests covering both probability and log-probability modes.
- `DeepLogModuleFactory` takes symbolic domain names instead of raw tensors for `variable_domains` (and the argument is now optional for boolean-only formulas), defaulting to a built-in `boolean` domain while allowing per-factory overrides; `AggregationModule` now expands enumerations by mapping domain symbols to their indices so factories no longer pass ad-hoc tensors.
- Circuit selection is now structure-aware: pass an optional `circuit_factories={'boolean': ..., 'probability': ...}` mapping to override the defaults (`boolean` → `KlayCircuit`, `probability`/`logprobability` → `DDKlayCircuit`), and omit the parameter entirely when the defaults suffice.
- Added `parse_formula_to_module`, which instantiates a default `DeepLogModuleFactory` (or uses the provided one) and returns a finalized `DeepLogModule`, eliminating boilerplate for notebook-style experiments.
- `DeepLogModuleFactory` automatically registers default equality and probability predicates when they are absent, so Boolean literals and `p/2` facts work out-of-the-box.
- `formula_to_module` now requires a configured `DeepLogModuleFactory` (instead of accepting raw domain/predicate mappings), so callers explicitly decide how formulas are compiled.
- Example notebooks (model counting, predicates, exactly_one, MNIST addition) were refreshed to rely on those defaults, removing redundant factory setup code.

## [1.6.0] - 2025-11-25

### Added

### Changed
- Example notebooks and integration tests register `NetworkPredicate`/`IndexPredicate` factories instead of relying on `NetworkModule`, keeping the symbolic shapes consistent with the new label structure.
- Engines/tests now expect `EngineFactory.get_boolean` to wrap labels as `_` literals, and predicate unit tests validate the updated probability symbols.

### Fixed

### Removed
- Removed the `Parallel` container module; multi-branch wiring should be expressed through explicit modules or circuits instead of the deprecated helper.
- Deleted the legacy `container.py` shim now that `Sequential` and `ModuleCircuit` live in dedicated modules.
- Removed `NetworkModule`; wrap torch.nn.Modules with `WrappedModule` and explicit symbolic shapes instead.

## [1.5.0] - 2025-11-24

### Added
- Introduced the `Predicate` abstract base class together with a dedicated `predicate_modules` package so built-in predicates (`SumsPredicate`, `EqualityPredicate`, `ProbabilityPredicate`, `NetworkPredicate`, and the new `IndexPredicate`) automatically handle batching, constant extraction, and optional deduplication – custom predicates now only override `_get_constant` and `forward_predicate`.
- Added a `positive_only` option to `ProbabilityPredicate` for engines that only materialize positive literals, ensuring probability and log-probability tensors can be produced without wiring negated atoms.

### Changed
- `DeepLogModuleFactory` now accepts predicate factories produced via `Predicate.get_factory`, so predicate modules self-describe their `(functor, arity, structure)` signatures. Tests and integration code were updated accordingly.
- Rewrote all built-in predicate implementations (and the predicate example notebook/tests) on top of the new base class: `BooleanEqualityPredicate` is now the more general `EqualityPredicate`, `NetworkPredicate` deduplicates repeated groundings, and predicates expose consistent symbolic shapes for reshaping utilities.
- `predicate_factories` are now keyed by `(functor, arity, structure)` and `DeepLogModuleFactory` recursively scans predicate inputs to instantiate any referenced predicate modules, so formulas that read predicate outputs (e.g., via `IndexPredicate`) automatically wire the necessary modules.
- `WrappedModule.__repr__` now reports the wrapped callable name together with the declared shapes and the `IndexClassifier` helper accepts flat integer tensors, simplifying debugging of compiled circuits and predicate-driven tests.
- Circuits no longer require structured literal tuples: `KlayCircuit`/`DDKlayCircuit` accept bare symbols, expose neutral elements directly, and DeepLog module creation now unwraps predicate inputs/outputs accordingly.
- `TransformationModule` outputs use the canonical `_` literal form so downstream components see the expected `(name, structure)` tuples.
- Unified the `KlayCircuit` and `DDKlayCircuit` unit tests into a single backend-parameterized suite so both implementations share coverage and expectations.

### Fixed
- `ModuleCircuit` seeds the implicit empty `SymTensor([])` input so compiled circuits that refer to the neutral tensor no longer crash when evaluated.
- Predicate batching fixes ensure `SumsPredicate`, `NetworkPredicate`, and probability predicates preserve dtypes, respect empty argument lists, and stop recomputing identical atoms.

### Removed
- Deleted `tests/circuit/test_dd_klay_circuit.py` after moving its scenarios into the consolidated test module.

## [1.4.0] - 2025-11-16

### Added
- Added square-bracket list parsing to `str_to_symbol`, DeepProbLog program parsers, and their tests so `[a,b|T]` is accepted throughout the stack and materialized as nested `cons/2` terms.
- Documented list syntax in the DeepProbLog language guide and cross-linked it from the Resources page.
- Introduced a Lark-based textual parser (`text_parser_lark.py`) that implements the documented grammar, plugs directly into `DeepLogFormulaFactory`, and is covered by the updated parser tests; `lark` is now a project dependency.

### Changed
- Replaced the “DeepLog engine module reference” card in the Resources page with a “DeepProbLog language guide” entry and hid the standalone language page from the top navigation.
- Reworked `formula_to_module` to compile formulas through `DeepLogModuleFactory`/`KlayCircuit` via the shared symbolic visitor to remove the bespoke segmentation logic.
- Renamed `formula/iterator.py` to `formula/symbolic_visitor.py` and updated imports so the visitor name reflects its purpose.
- The exactly-one example notebook now constructs constraint modules inline with the textual parser and reorganizes its cells to share the parsed constraint and dataloaders between training, testing, and profiling workflows. It also shows constraint adherence.

## [1.3.0] - 2025-11-15

### Added
- Added logprobability support to `KlayCircuit`, `DDKlayCircuit`, `DeepLogModuleFactory`, and the integration tests so circuits and engines can run entirely in the log semiring.
- Introduced log-aware predicate factories and predicate tests, plus optional structure selection for DeepProbLog engines.

### Changed
- `ProbabilityPredicate` can now emit logprobability tensors and automatically converts constant labels to log space while validating inputs.
- Engine constructors (`SimpleEngine`, `JanusEngine`) accept a `structure` flag and `EngineFactory` enforces valid structures.

## [1.2.0] - 2025-11-14

### Added
- Published the MNIST addition walkthrough in the tutorial site as a neuro-symbolic developer use case, showcasing a full DeepProbLog pipeline.

### Changed
- `Circuit.set_root` now requires a symbolic output name and `to_module()` no longer takes arguments, enabling consistent multi-root modules across `KlayCircuit`, `DDKlayCircuit`, and `DeepLogModuleFactory`.

## [1.1.0] - 2025-11-13

### Added
- Added unary `not`/`negate` operators to `KlayCircuit`, `DDKlayCircuit`, and `DeepLogModuleFactory` so boolean and probabilistic literals can be negated end-to-end through the DeepProbLog engine.
- Expanded regression coverage (circuits, module factory, engines, integration) and now run the integration suite against the `dd_klay` backend to verify the new negation semantics.

### Changed
- DeepProbLog `EngineFactory.negate` now requests the `'negate'` operator for probabilistic formulas, ensuring the correct semiring is used when flipping probabilities.
- Refreshed `examples/exactly_one/exactly_one.ipynb` to reflect the corrected negation flow used in the tutorial.

### Fixed
- `DDKlayCircuit` now keeps structured atom names intact when creating literals, preventing clashes after neutral elements are requested.
- `DeepLogModuleFactory.create_unary_node` now rejects negation of already composed modules with a clear error, avoiding silently incorrect formulas.

## [1.0.0] - 2025-11-12

### Added
- Added `DDKlayCircuit` for decision diagram-based circuit compilation.
- Added `DeepLogFactory` and `DeepLogModuleFactory` for streamlined formula and module creation.
- Added new `Circuit` base class and refactored circuit implementations.
- Added comprehensive test suites for `KlayCircuit` and `DDKlayCircuit`.
- Added `ProbabilityPredicate` improvements and new test coverage.

### Changed
- **BREAKING:** Renamed `deeplog_formula` module to `formula` - all imports must be updated.
- **BREAKING:** Refactored formula and circuit architecture - removed old formula builder API and backends (klay, pysdd, simple).
- **BREAKING:** Replaced `Formula` and `CategoricalFormula` with factory-based approach.
- **BREAKING:** Simplified `Symbol` and `Program` typing and conversion logic.
- Updated `SimpleEngine` and `JanusEngine` to work with new factory system.
- Migrated circuit backends from `circuit/backends/klay.py` to `KlayCircuit` and `DDKlayCircuit`.
- Updated klaycircuits version requirement in dependencies.
- Updated example notebooks (formula_to_module.ipynb, language.ipynb).

### Removed
- **BREAKING:** Removed @symbolfunction decorator.
- Removed old formula backends (`backends/klay.py`, `backends/pysdd.py`, `backends/simple.py`).
- Removed old `formula/builder.py`, `formula/formula.py`, and `formula/categorical_formula.py`.
- Removed old `circuit/backends/` directory and legacy Klay backend.
- Removed `examples/formula/formula.ipynb` notebook.
- Removed `deeplog_formula` module (consolidated into `formula`).

## [0.5.1] - 2025-10-21

### Changed
- JanusEngine now performs compilation after finishing each proof.
- WeightedFormula now keeps a single weight dictionary in the Factory.
- KlayCircuit is now pickleable. The klay.Circuit is no longer stored in its state.

### Fixed
- Filtered unused literals out of `KlayCircuit` input shapes while zero-padding their weights to keep the underlying circuit evaluation stable.

## [0.5.0] - 2025-10-15

### Added
- Introduced the `deeplog_formula` package with DPL compilation, stratification, visualisation, and iterator utilities.
- Added a formula builder API together with aggregation, transformation, and predicate modules (`BooleanEqualityPredicate`, `ProbabilityPredicate`, `SumsPredicate`, `NetworkPredicate`, `IndexedNetworkPredicate`) for constructing DeepLog formulas.
- Added `IndexClassifier` and other testing helpers, plus comprehensive unit tests covering formula compilation, translation, stratification, and predicate behaviour.

### Changed
- Migrated DeepProbLog components under `deeplog.systems.deepproblog` and refactored module containers/reshape logic to auto-wire submodules and handle missing transformations.
- `DeepLogModule` now validates input/output shapes automatically and supports Graphviz rendering; `NetworkPredicate` produces tensors with an explicit domain axis while the new `IndexedNetworkPredicate` allows direct probability lookup.
- Updated dependencies (`klay` → `klaycircuits`, added `graphviz`) and refreshed example notebooks and Docker configuration.

### Fixed
- Improved reshape transforms, container wiring, and flattening logic to prevent shape mismatches when composing module circuits.
- Fixed bug in calculation of mgu.

## [0.4.0] - 2024-12-19

### Added
- Added NetworkModule

### Changed
- All DeepLogModules now expect an implicit first batch dimension.

### Fixed
- Fixed implementation of Parallel and added test.
- Fixed FlattenTransform and added test.

## [0.3.1] - 2024-12-05

### Changed
- Updated example notebooks with internal feedback.
- `symbol_to_pretty_string` now prints queries and constraints without false in the head.

### Fixed
- Fixed a bug in SimpleEngine where answers to the same query were not grouped.
- Spaces in symbol parsing are now ignored.

## [0.3.0] - 2024-11-04

### Added
- @symbol_function decorator

### Changed
- Changed how shapes work.
- Updated shape.ipynb
- Creating Symbols and SymTensor now generally accepts string inputs througsh @symbol_function decorator.
