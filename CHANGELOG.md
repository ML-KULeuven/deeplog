# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.1] - 2026-09-11

### Added
- Every example notebook opens in Google Colab from an **Open in Colab** badge under its title. Both the badge and the notebook's first code cell are pinned to the version in `pyproject.toml`: the badge opens the notebook at that release's tag on the GitHub mirror, and the cell installs `pydeeplog[examples]==<version>` when DeepLog is not importable. The cell does nothing where DeepLog is installed, and is hidden on the docs site. The docs of every release therefore open that release's notebook and run it against that release. `tests/test_examples.py` checks that each badge points at its own notebook at that version and that each notebook starts with the install cell, so a version bump fails the suite until the notebooks follow.

### Changed
- The numpy floor drops from 2.4.0 to 2.0.2, the version Colab and Kaggle ship. Nothing in DeepLog needs a newer numpy — the full suite and every notebook pass on 2.0.2 — and the higher floor made installing into those runtimes upgrade a numpy they had already imported, which Colab answers by asking for a session restart.

### Fixed
- The LTN notebook's quantifier cells run again. 4.0.0 made `DeepLogModuleFactory(variables=...)` take `Domain`s, but the notebook still passed raw tensors, so `Forall` and `Exists` raised `AttributeError: 'Tensor' object has no attribute 'as_tensor'`. Nothing noticed, because those four cells had been tagged `raises-exception` since the notebook was written, though they never raised until then. nbmake passes such a cell whatever it raises, and the docs build does not fail on notebook errors, so the published 4.0.0 page showed four tracebacks. The domains are now `Domain.of_tensor(...)`, the tags are gone, and `tests/test_examples.py` refuses the tag in any example notebook.

## [4.0.0] - 2026-09-09

### Added
- **One module for N formulas.** `DeepLogModuleFactory.compile(*nodes)` takes any number of formula ASTs and returns one module with a column per formula, in the order given. Before, N formulas meant N `compile` calls — N circuit factories, N knowledge compilations, and a copy of every predicate module per formula — even where they shared subformulas, atoms, or the boolean circuit they count. The engine path already had the batching (`compile_to_module` builds co-resident lumps and hands them to `lower_circuit_nodes`); the AST path now reaches the same mechanism rather than a second one. Both folds run over every root under one memo, so a subformula the roots share *as an object* is built once and the atoms they share are one circuit leaf; when every root constructs to a lump of one circuit they are lowered through `lower_circuit_nodes`, so the boundary is folded once, one knowledge compilation serves every count over that circuit, and each predicate module is evaluated once. Any other mix — roots in different algebras, a root that stayed an enumeration — is lowered root by root and composed column-wise, where a shared sub-module is built once but still evaluated once per root enclosing it: slower, never wrong. `compile(node)` is unchanged, output shape included. Two roots that would name the same column are refused rather than silently collapsing into one, and sharing between *separately parsed* formulas reaches only their atoms, since each parse interns into its own hash-cons table.

- `deeplog.formula.ast.fold` takes the optional `memo` its circuit-side counterpart `fold_circuit` already had, and is now the whole fold — the private `_fold` that two modules imported across package boundaries is gone. `children(node)` joins it as the shallow projection companion to `map_children`, for the walks that read children without rewriting them. `deeplog.formula.circuit_node.lump_name(node)` is the one spelling of the name a lump compiles under, which three sites had written by hand and two of them carried a docstring warning that drift is silent.

- Operators with no circuit form are now lowered as tensor operations instead of being rejected (glab #137), and an operator with no *node* form is compiled across rather than kept out. Which of the two applies is not configured anywhere and not asked of the algebra: an operator over operands that could not be circuit-represented (a connective or a division over a quantifier, or over two separately lowered modules) is applied to their already-reduced outputs by the new `ElementwiseModule` (`deeplog.module.ElementwiseModule`), which aligns the operands onto their symbol-union input and combines them column-wise, broadcasting a single column across the rest. This makes connectives over quantifiers compile — `And(Forall x . φ, Forall y . ψ)` in the LTN fuzzy setting previously raised `NotImplementedError`. `ColumnwiseModule` is the single-module dual: it evaluates one module once and applies the operator across selections of its output columns, and now names its own output columns when asked (`names=`) instead of always borrowing the first group's.

- Circuits are **compiled across** an operator the backend has no node for, rather than the operator being kept out of the circuit — the new `deeplog.circuit.split`. A semifield's `divide` is the standing case: knowledge compilation and Klay build their graph out of a semiring's product, sum and complement, and a quotient is none of those. So the graph is *cut* there: the operand subgraphs are compiled as roots of one ordinary compilation, the algebra's own `operator_fns` entry combines their output columns, and the region above the cut is the same circuit compiled again, *bounded* at the cuts, which it treats as input slots fed by those columns. A division is therefore an elementwise operation *on circuits*, and what sits above one is still a circuit on the backend the algebra selected — where previously everything above a division stayed symbolic to the root. Nesting needs no extra machinery: the operands are compiled by the same entry point, which cuts them in turn. `Circuit.get_operator` accordingly accepts every operator its algebra defines, `divide` included.

  Bounding a compile is `frontier`, a new optional `{node_id: symbol}` argument on `Circuit.to_module` and on the traversals the backends walk (`Circuit.iter_topological` / `reachable_leaves` / `reachable_constants` / `flatten_chains`, and `Graph.iter_topological` / `flatten_chains` beneath them): those nodes are yielded but not descended into, and they are input slots of the region above them under the given symbols. It is what makes a partition compile without being copied into a circuit of its own.


- `Semifield` (`deeplog.Semifield`) — a `Semiring` whose product is invertible, so `divide` is defined. `PROBABILITY` and `LOGPROBABILITY` are now semifields. The algebra declares the quotient and says nothing about what compiles it — no backend walk has a node for one, so it is cut (`deeplog.circuit.split`) — and the denominator floor that keeps impossible evidence finite lives on the algebra rather than on the module performing the division, and log-space division is subtraction, floored at `finfo.min` rather than at the linear floor transported through `log` — the two representations have different floors by design, since log space exists to hold probabilities far below the linear one, and clamping a log denominator at `log(tiny)` (`-87.3` in float32) silently corrupts any program with more than ~126 independent facts. The floor is **derived from the operands' dtype** (`torch.finfo(dtype).tiny`) rather than pinned: for a numerator in `[0, 1]` that is the smallest denominator the dtype can divide by without overflowing, whereas the `1e-12` it replaces was a float32 constant that stores as *exactly zero* in float16 — the clamp was a silent no-op there and impossible evidence came back as `inf`/`nan`. There is no floor *setting*: `DEFAULT_DIVISION_EPS`, `DEFAULT_RATIO_EPS` and the `eps` argument of `Semifield` are all removed. An algebra whose division differs at all supplies the whole `division_fn`, which subsumes a one-scalar knob and cannot silently disagree with it — `eps` alongside `division_fn` was a representable state in which `eps` was ignored. In float32 the floor relaxes from `1e-12` to `1.18e-38`, so a denominator between the two is no longer clamped.

- Scalar posterior recognition: `P(q | e) = E[q∧e] / E[e]` compiles as a `divide` of two `expectation` aggregations. Division is an operator of the probability `Semifield` like any other, so it becomes an ordinary circuit node; no backend walk has a node for a quotient, so `deeplog.circuit.split` cuts the graph there and applies the algebra's `divide` across the compiled columns (denominator clamped away from zero, so impossible evidence stays finite). `recognize_posterior` rewrites `divide(WMC, WMC)` into `divide(expectation, expectation)` when both operands are expectations/canonical weighted model counts and the denominator evidence is a conjunct of the numerator `q ∧ e`; it runs ahead of `recognize_expectation` in `DEFAULT_PASSES`. Like every other pass it is recognition-only — a `divide` it does not recognize is left alone and still lowers, as a plain division of its two operands. Probability domain only for now.
- Numeric-constant facts (`0.6 :: fact`) are now baked into the PySDD-compiled arithmetic circuit, matching the MV-SDD backend: a boolean leaf whose `EngineResult.labels` entry is a numeric constant is pre-filled rather than left a runtime input, so a program of only numeric facts compiles to a module with no inputs (only neural-network-backed labels stay inputs). A fully-baked, input-less module is callable bare (`module()`): `ModuleCircuit` evaluates a single constant row when handed no tensors, and a `WrappedModule` pre-hook synthesises the empty `(1, 0)` batch channel.
- `to_module` accepts its roots as a single `{name: node}` mapping, the shape multi-root producers (grounder proofs, engine result formulas) already have, alongside the existing positional `CircuitNode`s with `names`. The mapping form forbids `names` and takes its output order from insertion order, exactly as the positional form does; a mapping mixed with other positional roots raises. Naming the same node twice now raises `Duplicate root` instead of silently collapsing the two outputs into one.
- `to_dict(tensors, shape)` — the read-out counterpart of shape validation. It splits a module's result along its symbolic shape into a `{Symbol: tensor}` mapping with the batch dimension preserved, so `to_dict(module(x), module.get_output_shape())` reads a module's outputs by name instead of by index. The result is a `SymbolDict`, a plain `dict` whose `repr` prints pretty symbols and formats scalars compactly, so interactive readouts stay legible. Raises `ShapeMismatchException` when the tensors do not conform to `shape`, and `ValueError` when `shape` names the same symbol twice. Exported from the top-level package.
- DeepProbLog conditional queries: a program's `:- body.` integrity constraints now condition every query on evidence, so `Solver.get_query_result` returns the posterior `P(q | e) = E[q∧e] / E[e]` instead of the prior `P(q)`. A constraint `:- body` reads as "`body` must not hold" (the operator dual of the `?- q` query directive), so positive evidence is written `:- not(e).` and negative evidence `:- e.`. The engine builds the shared evidence `e = ⋀ᵢ ¬(bodyᵢ)` on the same source circuit as the query proofs — so each numerator `q∧e` and the denominator `e` co-reside and share leaves — and carries it on the new `EngineResult.evidence` field; `compile_to_module` transforms numerators and the shared denominator in one batched WMC pass, lowers all of them in *one* knowledge compilation, and divides with the new `PosteriorModule` (`deeplog.module.PosteriorModule`), so the shared circuit and the predicate modules feeding it are evaluated once for all answers rather than once per answer. Conditional and unconditional results have the same shape — one `SymTensor` with a column per query answer — so adding a constraint to a program does not reshape its result. With no constraints, `evidence` is `None` and the plain `P(q)` path is unchanged. New `is_constraint` / `get_constraint_body` accessors in `deeplog.grounding.prolog` (re-exported from `deeplog.systems.deepproblog`).
- Every formula AST node renders itself, so a parsed formula can be read without navigating it by attribute. `str(node)` is the formula's surface syntax — the fold through `SymbolicFormulaFactory`, so it is the text the parser reads back — while `repr(node)` gives the node structure that text parses into, one line of `Kind(datum, children...)`, and `node.tree()` gives that same structure indented, one line per node. The three live on `deeplog.formula.ast.FormulaNodeDisplay`, which every node kind inherits. A `CircuitNode` has no surface syntax, so `str` spells it as its circuit handle (`circuit_probability_0#12`) and the structural views show the formulas its boundary defers, never its compiled interior. `SymbolicFormulaFactory.create_atom` is total over untagged atoms as a result — it renders a bare symbol as itself instead of raising, which is what a lump's boundary atoms are.
- `AstFactory` (`deeplog.formula.AstFactory`, re-exported from the top-level package) — the identity interpreter of the formula signature: each eliminator is the constructor of the node kind it eliminates, so `fold(node, AstFactory())` reproduces `node`. It is what a factory-driven producer — a grounder, a DIMACS reader — builds through to obtain a formula it can inspect, rewrite and print, rather than one already lowered. The AST was previously reachable only by parsing text, so a test that wanted to assert a grounder's proof *structure* had to render it and compare strings; both grounder suites now assert the formula itself.

### Changed
- **A boolean identity is absorbed where the proof is built, not where it is printed.** `ProofBuilder.conjoin` / `disjoin` drop an operand equal to `get_true()` / `get_false()`, so the identity that seeds every proof fold (`reduce(builder.disjoin, proofs, builder.get_false())`, the `defaultdict(builder.get_false)` accumulator, a closed fact's `true`) disappears once a real proof arrives — which is what the fold meant, and what `JanusGrounder` already did on the Prolog side, where an identity is emitted only for the empty `and([])` / `or([])`. `SymbolicFormulaFactory.create_binary_node` no longer folds neutral elements: a text printer performed an algebraic rewrite no other factory performed, so a `SimpleGrounder` proof that really was `false ∨ a` *printed* as `a` while compiling to a circuit with the dead node still in it, and the engine-parity suite compared the two grounders through that printer and could not see the difference. Compiled circuits lose those nodes; every value is unchanged.
- **A structure declares its roles, and a circuit crosses by them.** `AlgebraicStructure.roles` is the operator name a structure spells each role it declares with: `{}` for a bare structure, product and sum for a `Semiring`, plus negation for an `Algebra` and division for a `Semifield`, merged along the class chain exactly as `operator_fns` already is. `transform_circuit` infers its operator mapping from the roles *both* structures declare, so a semifield's `divide` crosses to another semifield like any other operator — transforming a formula containing a division from `probability` to `logprobability` was refused before, though both spell it `divide` and the target's `division_fn` is exactly the log-space subtraction the transform wanted. The two eager refusals are gone with it: a pair that was not two semirings, and a source whose negation the target does not declare, were rejected before the graph was looked at. Whether a missing counterpart matters is a question about the nodes there are, so it is answered at the node that uses one — `Cannot map node type 'not' from 'boolean' to 'plain': 'plain' declares no negation` — and a transform whose subgraph never uses the missing operator now simply runs. The constants an axiom names get the same treatment: `AlgebraicStructure.identities` (`{"zero": ..., "one": ...}` on a `Semiring`) is what a named constant crosses by, so the constant mapping intersects two declarations rather than testing for a semiring pair. `deeplog.circuit.backends.circuit_roles` selects from the same table instead of enumerating the taxonomy a second time, which is what let `Semifield` be added without division crossing.
- **A circuit is a sub-algebra of the formula AST, and folds through the same `DeepLogFormulaFactory`.** `deeplog.formula.ast.fold` is the AST's catamorphism; circuits had none, so every consumer re-implemented one — a topological walk over a `dict[int, T]` accumulator with its own per-node-kind dispatch, in `build_diagram`, `transform_circuit` and `lower_generic` alike. A circuit needs only three of the algebra's six eliminators: `get_symbol_name` already reports one symbol for a leaf, a named constant and a numeric constant alike, so all three are `create_atom`; and every operator node is unary or binary, because every one is built through `get_operator` with one or two children (the n-ary form exists only as `flatten_chains`' view for Klay, never as a stored node). Nothing in a circuit corresponds to `create_transformation`, `create_aggregation` or `embed_circuit`. So `Circuit.fold(roots, algebra, *, frontier=None, memo=None)` joins `transform` and `to_module` as a verb on the engine, `deeplog.circuit.fold.fold_circuit` is the free function, and **`build_diagram` is deleted** — the role dispatch it hand-rolled now lives in `DiagramAlgebra`, the algebra that cares. `transform_circuit` keeps its signature, `node_map` semantics and `into=` threading exactly (the fold's `memo` *is* `into=`'s node map, the same injection point `_fold`'s memo is on the AST side), and `lower_generic` folds with the slot index as its carrier. The construction side is shared too: `CircuitAlgebra` (`deeplog.circuit.CircuitAlgebra`) is the formula algebra whose carrier is a node id of one given circuit, and both `transform_circuit` and `DiagramEmitter` build through it rather than each calling `get_leaf_node` / `get_operator` themselves — the emitter lost `leaf`, `conjoin`, `disjoin`, `negate`, `constant` and `_fold` as implementations, keeping them only as the vocabulary its backend walks call. `lower_klay` stays hand-written on purpose: `flatten_chains` exists so it can emit `or_node([a,b,c,d])` as one shallow node, and a binary eliminator would rebuild the deep chain that pass removes. An algebra that only ever meets circuit nodes raises on the three eliminators a circuit has no node for — deliberately not by giving the ABC raising defaults, which would have cost `@abstractmethod` on the AST side. A circuit now folds through `SymbolicFormulaFactory` to its formula text, which is the sub-algebra claim demonstrated rather than asserted.
- **Knowledge compilation emits into the algebra that will count it.** `knowledge_compile(circuit, roots, *, variables=None, structure=None, leaf_mapping=None)`: `structure=None` keeps today's meaning — compile into the source's own algebra — while a given `structure` emits the compiled diagram straight into it, renaming each atom through `leaf_mapping`. The rewrite preserves roles, so reading them in another algebra is exactly what transforming the compiled circuit afterwards would have done; `DiagramEmitter` already resolved its operators by role from *its target's* structure, so pointing it at the probability circuit emits `times`/`plus`/`negate` with no other change. `transform_expectation_to_probability` therefore makes one call instead of two, and the boolean d-DNNF circuit it used to build, walk once and discard is never built. `transform_circuit` stays public and unchanged: it is still the pass for retagging an *uncompiled* circuit, where the reading is exact only after compilation.
- **BREAKING:** **Variables are modelled explicitly, and a domain is declared rather than reconstructed.** The DeepLog theory gives every variable a domain — a *regular* variable its own, a *reification* variable the values of the algebraic structure it is associated with (Def 8/12) — and the codebase had neither. It had two disconnected halves instead: `DeepLogModuleFactory.variables` mapped a binder to a raw `torch.Tensor` and silently fell back to `defaultdict(lambda: torch.tensor([False, True]))`, so an undeclared binder over *any* algebra was enumerated as if it were boolean; and multi-valued variables were a side channel, `categoricals: dict[Symbol, tuple[str, int | str]]`, threaded from `Solver` through `EngineResult`, `compile_to_module` and `transform_expectation_to_probability` into `knowledge_compile`, where `mvsdd.py` then *reconstructed* the domain as `max(value_idx) + 1` plus a `"none"` residual. Non-contiguous value indices therefore oversized the domain silently.

  `AlgebraicStructure` now carries the set of values Definition 1 says it has — `A_R`, the labels a formula over it takes, so `BOOLEAN.values == (("false",), ("true",))`, while `PROBABILITY` / `LOGPROBABILITY` / `MPE` / `REAL` declare `None`, because their values are not finitely enumerable and a stand-in would be enumerated as if it were the right one. The new `deeplog.variable` module holds the concept: `Domain` with two constructors — `Domain.of(values)` for named values (position is identity) and `Domain.of_tensor(t)` for values with no names of their own, such as images — plus `Domain.of_structure(structure)`, and `Variable(name, domain)`. Each domain answers only the view it has: `TensorDomain.values` raises rather than inventing names.

  - `DeepLogModuleFactory(variables=...)` now takes `Mapping[Symbol, Domain]`, not tensors: `{("N1",): Domain.of_tensor(torch.arange(10))}`. The `defaultdict` is gone. An undeclared binder is a reification variable and inherits the values of the new `reification=` structure (default `BOOLEAN`) — one statement about the model, in Def 12's terms, instead of a constant hidden in a lambda. It is a *declaration*, not a derivation, because a reification variable's structure is genuinely independent of the atoms it appears in: the paper's own `burglary[B]_probability` reifies a **boolean** `B`, and in DeepLog's WMC encoding `Burglary` occurs in both `=(Burglary,true)_boolean` and `p(Burglary)_probability`. A binder over a structure that declares no values now raises instead of quietly enumerating `{false, true}`.
  - `knowledge_compile(circuit, roots, variables=...)` replaces `categoricals=`, and so do `transform_expectation_to_probability(..., variables=...)` and `EngineResult.variables` (was `EngineResult.categoricals`). A `Variable` is `(name, domain)` and owns **no atoms**: an atom is what a predicate applied to terms yields, and a term may be a variable (Def 8), so a variable is linked to an atom by *occurring* in it — the new `deeplog.variable.VariableAtoms`, described below.
  - **An annotated disjunction is recognized, not destroyed and rebuilt — and a variable is linked to an atom by *occurring* in it.** Previously the AD was split into n facts up front with every label wrapped as `("@cat", label, ("@cat_id_<n>",), ("<i>",))` purely so the structure survived grounding, then unwrapped afterwards to reassemble it. The base grounder path now keeps what the disjunction declares (`ad.split_annotated_disjunctions`) and reads it back off the ground atoms (`ad.instantiate`), matching them with the `calculate_mgu` that already reattaches labels. What a disjunction declares is an **occurrence** — the atom with the variable's term position left open — and the domain that position ranges over: `digit(i1,0); ...; digit(i1,9)` is `digit(i1,_)` over `0 … 9`. Nothing pairs a value with an atom, because Definition 8 builds an atom out of terms and a term may be a variable: the atom asserting a value is what substituting it into an occurrence *yields* (`variable.atom_asserting`), so no value needs an atom of its own, one variable may occur in several atoms, and one atom may be an occurrence of several variables. `ValueAtoms` (a `Mapping[Variable, Mapping[int, Symbol]]`, which could hold only one atom per value and forbade an atom asserting two variables' values) is therefore replaced by **`VariableAtoms`**, a `Mapping[Variable, tuple[Symbol, ...]]`; `indicated_values` derives the atom-keyed view by substitution and returns *all* the pairs an atom asserts, and MV-SDD raises where it cannot compile two variables on one leaf — a statement about that backend, not about what a model may say. Two atoms belong to the same variable when they match under the same binding of the occurrence's free arguments, so an AD written over a free image variable yields **one variable per image** rather than collapsing every grounding into one. Branches that share no argument position are unrelated atoms, which the theory has no variable *term* for: the whole atom is the position, so the occurrence is bare and the values are the branch atoms themselves. The `@cat` tag survives only where it is load-bearing, and now lives there: the k-best prover's `engine.pl` matches on it to hoist negation over a variable it has not yet decided, which it must do *during* search, so the functor, `create_categorical_label` and `expand_annotated_disjunctions` moved into `kbest/kbest.py` beside the renderer that emits them. `ad.py` is the disjunction as the base path sees it and nothing else; `is_categorical_label` / `unwrap_categorical_label` are gone, since the unwrapping happens in Prolog.
  - `mvsdd.py` is one uniform path over those variables. A reachable leaf that asserts no known value becomes a two-valued variable built on the spot, so `boolean_leaves`, `variable_of_leaf`, the second seeding loop and `_leaf_of_boolean_variable` are gone, along with `branch_values`, `residual_index`, the `max(numeric) + 1` sizing and the `int | str` value type. Every value no reachable atom asserts collapses into a **single residual slot** read back as the complement of the asserted ones — collapsing them is what keeps it exact, since the formula cannot tell them apart and a slot each would read back as the complement more than once and double count. A binary variable's `false` and an AD's residual outcome are now the same case rather than two mechanisms.
  - Backend selection is derived rather than a proxy for "was anything declared": **MV-SDD iff some variable has two or more asserted values**, since a variable with at most one already *is* a plain SDD variable. Compiling the same program with a partly-proved disjunction therefore stays on SDD, where before any declaration forced the multi-valued compiler.
  - **A disjunction can be written non-ground**, as DeepProbLog's own neural annotation: `nn(m_digit, [X], Y, [0..9]) :: digit(X,Y).` declares outright what the enumerated form implies — `Y` is the variable and `[0..9]` its domain — and *is* `m_digit(X,0)::digit(X,0); … ; m_digit(X,9)::digit(X,9).`, same branches, same labels, same probability. Ranges (`0..9`) expand; the declared variable must occur exactly once in the atom, or the annotation is refused rather than given a position to guess at. Both spellings reach `ad.declare_*` and produce the same declaration, so nothing downstream — including the k-best rendering — distinguishes them.
  - **One weighted-model-count site.** `compile_to_module` no longer counts its lumps itself and then hands the results to the lowering; it lowers each lump as the *expectation* of it, exactly as an `expectation` written in the textual language is lowered, so `batched_lowering` is the only place a count happens. `lower_circuit_nodes` and `batched_lowering` gained `leaf_mapping` and `variables` to carry what the engine knows and a count cannot derive — Definition 12's α, and where each variable occurs; both default to the general reading, which is what a textual formula means. That closes a silent hole: the lowering's count could previously be called with neither, so an annotated disjunction reaching it compiled as if its branches were independent — `0.3::x(1); 0.7::x(2).` with `q :- x(1). q :- x(2).` counted `0.79` where exclusivity gives `1.0`. **A fully baked module now declares no inputs at all**, not even the empty `(batch, 0)` channel, and is called as `module()` — the contract its own docstring already stated.
  - `ModuleCircuit` declared its external inputs with `SymTensor(symbol)` where a symbol is itself a tuple, so `SymTensor` read it as an array *of the symbol's parts* and produced a scalar shape (the ambiguity its docstring warns about). No submodule declares that shape, so every single-symbol input looked unproduced and earned a synthesized transformation — a correct one, reconciling two spellings of the same thing, but it changed the tensor's rank and so corrupted any consumer that wanted the plain shape. The boundary is now declared in the spelling its consumer uses.
  - `deeplog.Domain`, `deeplog.SymbolicDomain`, `deeplog.TensorDomain` and `deeplog.Variable` are exported.


- **BREAKING:** `Semifield` is no longer a subclass of `Algebra`, and no algebra answers questions about circuit backends any more. Two changes with one cause — the taxonomy was making a claim that is false, and the backend was reading its own capabilities off the taxonomy.

  The hierarchy now composes the way the algebraic taxonomy itself does, by multiple inheritance rather than by a chain: `Algebra` (a complement) and `Semifield` (a multiplicative inverse) are *independent* extensions of `Semiring`, and a structure with both declares both as bases. The old chain asserted that division implies negation, and permitted a `Semifield` built on min/max, which has no multiplicative inverses — `isinstance(BOOLEAN, Semifield)` is now correctly `False`. `PROBABILITY` and `LOGPROBABILITY` are instances of a private `_ComplementedSemifield(Algebra, Semifield)`: a class has to exist because Python reifies an axiom set as a type, and sharing one between the two beats a singleton class each, but it names no new axiom so it stays out of the public taxonomy. Every structure class is now a keyword-only dataclass registering its own operator in a cooperative `__post_init__`, so each level declares only its own fields instead of re-forwarding the whole parameter list; equality is by identity, making a structure usable as a dictionary key. Existing keyword construction (`Semiring(name=..., product=...)`) is unchanged; positional construction, and passing `negation=` to `Semifield`, are not.

  `AlgebraicStructure.circuit_operators` is **removed** and nothing replaces it: which operators a circuit may carry is no longer a question anyone asks. A circuit carries every operator its algebra defines, and the new `deeplog.circuit.backends` answers only what each walk can build — `routed_operators(structure, backend)` and `circuit_roles(structure)`, keyed by **role** rather than by name, so a structure's product, sum and complement are the same three nodes whether they are spelled `and`/`or`/`not` or `times`/`plus`/`negate`, and the `KC_OPS`/`KLAY_OPS` name tables are gone along with the walks' hardcoded dispatch. Anything else is cut at compile time (`deeplog.circuit.split`) rather than excluded at construction time, which is what lets the algebra stay a description of the algebra: a `Semifield` declares that its product is invertible without knowing that nothing has a node for the quotient.

- **BREAKING:** A structure's named constants are spelled as the values they denote, and the spelling is checked. `BOOLEAN`'s identities are `("false",)` / `("true",)` — the two values it already declares — instead of `("0",)` / `("1",)`, and `LOGPROBABILITY`'s are `("-inf",)` / `("0",)` instead of the probability-space `("0",)` / `("1",)` it inherited. The invariant is that a named constant's symbol resolves, through its own structure's `constant_fn`, to the element it names, and `AlgebraicStructure.__post_init__` raises when it does not — a structure can no longer declare an identity that nothing can evaluate. That closes a silent-wrong-numbers hole in the generic evaluator, which materialised every identity as a literal `torch.ones` / `torch.zeros` whatever the structure said: a custom log-space semiring evaluated `times(a, one)` as `a + 1` and `plus(a, zero)` as `logaddexp(a, 0)`. An identity is now baked in like any other constant and the `_TRUE` / `_FALSE` step kinds are gone. No built-in structure changes numerically — Klay reads its identities from its own semiring, and every other built-in identity *is* 1.0 or 0.0 — but formulas change spelling: the boolean identities are `false_boolean` / `true_boolean` in the textual grammar and `("false",)` / `("true",)` in a circuit. `0_boolean` / `1_boolean` now raise instead of quietly meaning something slightly different — `BOOLEAN`'s `constant_fn` rejects a numeral, because the structure enumerates its values and a number is not one of them. A symbol that is *not* a number still resolves to `None`, since not being a constant is a different answer from being an ill-formed one.

- **BREAKING:** A circuit has one kind of constant node, and a structure no longer keeps a registry of roles. `"zero"` / `"one"` are gone as *node types*: every constant is a `constant` node carrying the symbol it was written as and the value its structure resolved that to — which only became honest once an identity resolved to the right number (previous entry). Which constants a backend can bind natively is a fact about the backend, so it lives there: Klay binds the two identities to `false_node()` / `true_node()` and pre-fills every other constant as an input slot, the diagram compilers bind them to the manager's `false()` / `true()`, and the generic evaluator bakes all of them alike. `Circuit.reachable_constants` therefore reports the identities too — a backend with a primitive drops them — and it now mirrors `reachable_leaves`, returning `{symbol: node_id}` keyed by the structure-tagged name instead of `{node_id: value}`, with the values read from `Circuit.constant_values`. That asymmetry only existed because a constant had no symbol to be keyed by; Klay consequently names its pre-filled slots with the symbol the constant was written as, rather than re-rendering the float. `Circuit.constant_nodes` is keyed by symbol rather than by role string or `str(value)`; `Circuit.zero_node` / `one_node` are unchanged in meaning but resolve through the structure's declared `zero` / `one`. `Circuit.get_symbol_name` returns a constant's own spelling instead of a re-rendered float, so `("0",)` no longer comes back as `("0.0",)`, and constants dedup by symbol rather than by value. **`AlgebraicStructure.named_constants` and `symbol_to_role` are removed**, with nothing replacing them: the identities are `Semiring.zero` / `.one`, and each consumer that needs them reads them off the narrowed type. `transform_circuit` builds a constant mapping once per call beside its operator mapping, so a constant still crosses by the role it plays rather than by its value — boolean's zero becomes log space's `-inf`, not `0.0` — and the "role has no equivalent in target" error is unreachable now that both sides are semirings. `DiagramEmitter` raises a named error when its source structure declares no product and sum, where it previously died on a `KeyError`.

- **BREAKING:** The Klay semiring lookup raises instead of falling back to `real`. `SEMIRINGS` — a name-keyed table with a `("real", False)` default — is replaced by a registry keyed by *structure object* (`register_klay_semiring`). Klay applies a fixed semiring and never consults `operator_fns`, so eligibility has to mean "this structure's product and sum *are* that semiring's", which a name cannot establish: a custom structure whose operators happened to be spelled `and`/`or` previously passed the Klay compatibility test and was then evaluated in the real semiring with its own `operator_fns` silently discarded — no error, wrong numbers. Unregistered structures now route to the generic evaluator, which honours `operator_fns` directly. A `structure_override` still resolves by name, and still passes an unrecognised name to Klay verbatim as a semiring identifier. Relatedly, a deterministic circuit over a structure with no product and sum now fails at compile with a message naming the structure, instead of dying inside `walk_and_combine` with `Unknown node_type`.

- **BREAKING:** A module's algebraic structure now lives on its **output symbols**, and the module-level annotation is gone. `DeepLogModule.structure` and the `HasStructure` protocol (`get_structure()`) are removed, along with every per-class override of them (`ModuleCircuit`, `Sequential`, `AbstractAggregationModule`'s settable `structure` property, `_StructureCast`); `compose_modules` and `ModuleCircuit.__init__` no longer take a `structure` argument. A module annotation was the wrong granularity: one slot cannot describe a module whose columns live in different algebras (the DeepProbLog conditional path lowers N numerators *and* the shared evidence count together), cannot describe a module in none (a wrapped network, a reshape transform), and cannot describe a cast, whose inputs and outputs are in *different* algebras. Which algebra a value lives in is a property of the value, so it is recorded where the value is named: the `("_", inner, (structure,))` tag that circuit leaves, `Predicate` outputs and the textual grammar (`p(goal,label)_probability`) already used. Read it back with the new `sole_structure(shape)` — the one structure shared by every symbol, `None` when all are bare, raising when the shape mixes — or `structures(shape)` for the per-symbol view; `module.get_structure()` becomes `sole_structure(module.get_output_shape())`. Compiled circuit roots, aggregation binder names (`sum(binders(X),q) _ probability`), elementwise columns (`divide(q1,z) _ probability`) and DeepProbLog query answers (`addition(i1,i2,3) _ probability`) are now labelled, so `to_dict(module(x), module.get_output_shape())` is keyed by the labelled atom. New symbol helpers `structure_of` (total; replaces the raising `get_structure`), `without_structure` (tolerant unwrap) and `retag`. Consequences of deriving rather than storing: `reshape(module, output=...)` no longer produces a module whose structure read raises, `compose_modules` no longer silently drops the structure on its single-module path, and an `ElementwiseModule` can no longer disagree with the symbols it names.

- **BREAKING:** Inside a formula, every module must label its output symbols; a missing label is an error, never a default. `DeepLogModuleFactory`'s eliminators (`create_unary_node` / `create_binary_node` / `create_transformation` / `create_aggregation`) raise `NotImplementedError` naming the module when handed a child with bare outputs. In particular the `real -> probability` cast is selected by a **declared** `real` label — a network output that wants `torch.sigmoid` labels itself `with_structure(symbol, "real")` — and never by the absence of a tag, so a forgotten label cannot quietly route through a real conversion and return plausible wrong numbers. `real` is now a registered `AlgebraicStructure` (`deeplog.REAL`) defining no operators, so the name is discoverable and nothing can be computed in it. `DeepLogModuleFactory` gained a `transformations={(from, to): builder}` constructor argument, so a per-factory cast can be registered without touching the global registry. The module layer stays tolerant: `AggregationModule`, `ElementwiseModule`, `Sequential` and `ModuleCircuit` propagate whatever labels they are given, including none, since composing `DeepLogModule`s directly is a use outside the formula layer where the operator function is supplied explicitly. `DeepLogModuleFactory.create_atom` applies the same rule to its *input*: an unlabelled symbol raises `ValueError: Invalid atom`, the refusal `CircuitFactory.create_atom` already makes one fold earlier, where it previously returned `None` and left the symbol as an external input. Its `None` now means one thing only — a labelled leaf with no registered builder, which is also how a baked constant reaches it, no predicate being registered for a numeral.

- **BREAKING:** A compiled circuit's outputs, and its input slots under a `structure_override`, are labelled consistently by all four backends. Previously `pysdd` retagged only inputs, `mvsdd` only outputs, and `klay` neither; both used a positional `(sym[0], sym[1], (override,))` rewrite that corrupted any arity-2 compound name (an MV-SDD root named `digit(i1,3)` became `digit(i1,probability)`). The retag now happens once in `deeplog.circuit.compile.to_module` for the roots, and on the slots through `with_structure`, so a synthetic slot name (`@cat`, `@bool_false`) is wrapped rather than overwritten. Compiling a boolean circuit with `structure_override="probability"` therefore declares probability-valued inputs and outputs, which is what the caller actually feeds and reads.

- **BREAKING:** `deeplog.module.PosteriorModule` is removed, and with it the reserved `("@evidence",)` output symbol the DeepProbLog conditional path used to mint so the division could tell the shared evidence count apart from the query answers. The conditional path still lowers every numerator `E[q∧e]` and the shared denominator `E[e]` in one knowledge compilation and still evaluates that joint module exactly once per forward — now through the general `ColumnwiseModule`, which selects column groups *by symbol*. Only the answer columns are relabelled to their query atoms; the denominator keeps the positional name the lowering gave it, which is already outside the user's atom space, so no synthetic atom is created at all. The division itself is `PROBABILITY.get_operator_fn("divide")` — the algebra's operator, not arithmetic hand-written in the compile path.
- **BREAKING:** Knowledge compilation is a **circuit-to-circuit pass**, not a backend, and a weighted model count is two ordinary rewrites. `deeplog.circuit.knowledge_compile.knowledge_compile(circuit, roots, variables=None)` takes a formula and returns an equivalent circuit that is *deterministic* (a disjunction's branches are mutually exclusive) and *decomposable* (a conjunction's operands share no variables) — same `(new_circuit, node_map)` contract as `transform_circuit`, and composable with it. A count is that pass followed by the transform, in that order, because the order is what makes the transform true: reading `or` as a semiring sum claims `P(a) + P(b)`, which is false in general and exact on a d-DNNF.

  The old design ran those two steps backwards. `transform_circuit(boolean → probability)` renamed `and`/`or`/`not` to `times`/`plus`/`negate` on an arbitrary formula, producing a circuit whose `plus` node did not mean `plus` — for `a ∨ b` it meant `a + b − ab` — and `Circuit.deterministic` was the flag that told the compiler not to read those nodes literally. The knowledge-compilation backends then renamed the operators straight back to `&`/`|`/`~` to build their diagram. So **`Circuit.deterministic` is gone**, along with its constructor argument, the `deterministic=` parameter of `transform_circuit` / `Circuit.transform` / `transform_nodes` / `transform_expectation_to_probability`, and the `deterministic=True` the default boolean circuit builder used to declare. Nothing records a *reading* any more: the property the transform needs is established by the pass that produces its input. `AlgebraicStructure.idempotent` is removed with it: that flag's only reader was the `deterministic=not structure.idempotent` default, and one boolean per structure could not have stated the fact in any case — idempotence is a property of an operator, and `MPE`'s sum (`max`) is idempotent where its product is not.

  `select_backend(circuit)` is correspondingly just the algebra: Klay when a Klay semiring implements the structure, the generic evaluator when none does. It no longer takes the declared variables, and neither does `to_module` — declaring a multi-valued variable selects the multi-valued *compiler*, where the engine already knows it, rather than a backend. `structure_override` is removed from `to_module` and every backend, and with it `register_klay_alias`: evaluating a circuit in another algebra is `transform_circuit` into that algebra, which is a rewrite you can inspect rather than a compile-time mode. `deeplog.MPE` is now a registered `Algebra` (max-product, verified against Klay's `mpe` semiring), so the most probable explanation is an ordinary transform target.

  The old `compile/pysdd.py` and `compile/mvsdd.py` are replaced by `knowledge_compile/sdd.py` and `knowledge_compile/mvsdd.py`, which walk the compiled diagram into a `Circuit` instead of straight into a Klay graph. `deeplog.circuit.compile` is now evaluation only. Consequences of the d-DNNF being an ordinary circuit:

  - **A weighted model count in a custom algebra is possible at all.** The counted circuit is compiled by whichever evaluator its structure selects, so a formula can be counted in a structure Klay does not implement — the generic evaluator honours its `operator_fns`. Previously a count could only ever be a Klay semiring.
  - **Constant labels need no special case.** `build_leaf_mapping` no longer skips numeric labels to keep them out of a diagram that could not hold a constant node; `0.6 :: fact` maps to a constant symbol, folds to a constant node, and Klay's existing prefill bakes it. The label-baking machinery in both knowledge-compilation backends is gone.
  - **Annotated-disjunction declarations are not re-keyed.** `EngineResult.variables` names the boolean atoms the grounder emitted as its domain values, which is exactly what the pass consumes, so `_transformed_categoricals` is deleted.
  - **`not` over a compound compiles.** Knowledge compilation *is* the NNF pass, so a formula Klay could not negate is no longer a limitation of the counted path (glab #138 now covers only as-written circuits).
  - **Two equivalent roots are one node.** Compilation is canonical, so a query and its evidence can compile to the same node; `to_module` gives each name its own output column instead of rejecting the duplicate.

- **BREAKING:** `expectation` is lowered as a boundary, not a rewrite. `CircuitFactory` places a leaf in the probability circuit fed by the boolean lump it counts (the mechanism a cross-structure cast already used), and the lowering gathers the feeders that count the same circuit so they share one knowledge compilation — `CircuitFactory._transformed` and its `_transform_targets` accumulation are gone. A compiled formula whose root is an expectation is therefore a *composition*, and like every other composition it declares one input channel per symbol rather than one packed tensor; `reshape(module, input=SymTensor([...]))` packs it, as the surrounding tests and notebooks already did for ordering.

- `EqualityPredicate` and `SumsPredicate` return a float in the default dtype rather than a bool mask and an integer. Their outputs are values of the `boolean` algebra, whose operators are `min`/`max`/`1 - x` over numbers, so the dtype belongs to that algebra and not to the arguments that produced the value — `SumsPredicate` took it from `z`, which is a domain encoding and is routinely `torch.long`. Both only ever worked because every consumer was a circuit that cast its inputs; that cast is now gone (see below).
- **BREAKING:** A compiled circuit evaluates in its input's dtype. Every Klay-compiled circuit previously returned `float32` whatever it was handed, because the compile layer attached an unconditional `.to(torch.float32)` forward pre-hook to the Klay module; the generic evaluator pinned `float32` on its numeric constants for the same reason. Both are gone — Klay is already dtype-polymorphic, and the pin existed only to absorb the integer truth values `SumsPredicate` used to emit. A `float64` model now stays in `float64` end to end, which is what makes the semifield's dtype-derived division floor (`torch.finfo(dtype).tiny`) reachable through a compiled circuit rather than only through the algebra in isolation. Feeding a non-floating tensor to a compiled circuit is no longer silently converted.
- **BREAKING:** `deeplog.circuit.compile` is renamed `deeplog.circuit.lower`, and its backend entry points `compile_klay` / `compile_generic` / `compile_across_cuts` become `lower_klay` / `lower_generic` / `lower_across_cuts`. Two different operations were sharing the word: `knowledge_compile` rewrites a circuit into an equivalent d-DNNF *circuit* (a pass), while this package turns a circuit into a *torch module* (the end of the pipeline) — so `compile_sdd` and `compile_klay` read as siblings while returning different types. `lower` is the word the formula layer already uses for the same step one altitude up (`formula.deeplogmodulefactory.lower`). The user-facing verb is unchanged: `Circuit.to_module`, `DeepLogModuleFactory.compile` and `compile_to_module` keep their names, since a whole formula-to-module pipeline is fairly called a compilation; it is the two passes inside it that needed distinct names. `deeplog.circuit.generic` moves to `deeplog.circuit.lower.generic`, so both backends sit beside the `to_module` dispatcher, and `compile.common` is gone: `to_module` is the package itself and the rest of that module was Klay-only.
- **BREAKING:** `deeplog.module.RatioModule` is removed. A ratio was a module because division had no circuit form; now that `divide` is an ordinary circuit node cut at compile time (`deeplog.circuit.split`), nothing constructs one — a posterior over co-resident counts lowers through `ColumnwiseModule`, and a division over two separately lowered modules through `ElementwiseModule`, both applying the algebra's own `division_fn`. The denominator floor is unchanged, since it lives on the algebra. Two behaviours went with the class: a ratio no longer relabels its output to the numerator's symbols (the column is named for what it computes), and dividing in an algebra with no invertible product now fails on the missing `divide` operator rather than on a missing axiom.

- **BREAKING:** The plain-Prolog grounder is now a first-class citizen in `deeplog.grounding`, decoupled from DeepProbLog's probabilistic semantics (glab #131). SLD resolution — turning a logic program + query into a boolean proof AST — lives in `deeplog.grounding.prolog` (`SimpleGrounder`, `JanusGrounder`, and the Prolog-scoped `PrologGrounder` ABC they share), knows *only* plain Prolog, and interprets no ProbLog semantics: its single extension point is the set of **open predicates** whose facts become proof leaves (every other fact collapses to `true`) — a semantics-free notion in the spirit of ASP `#external` / abducibles. The grounder builds the proof structure straight through a `DeepLogFormulaFactory` via a thin `ProofBuilder` (`get_true` / `get_false` / `leaf` / `conjoin` / `disjoin` / `negate`) and returns `dict[ground_atom, formula]` — no labels, no `::`, no annotated disjunctions. DeepProbLog is now a thin layer over it (`deeplog.systems.deepproblog.Solver(grounder)`): it parses `::` / AD syntax, declares the probabilistic predicates open, grounds through the chosen grounder, then reattaches labels and declares the annotated disjunctions' variables **post-hoc** by matching each ground leaf against its declaration. `EngineResult`, `compile_to_module`, and `str_to_rule(s)` are re-exported from their new homes; the annotated-disjunction / label transforms (`expand_annotated_disjunctions`, `remove_labeled_rules`) and the k-best-internal `ProbabilisticFactory` are *not* part of the public API — they live on their submodules for the layer's own use. Renames (no back-compat shims): `SimpleEngine` → `deeplog.grounding.SimpleGrounder`; `JanusEngine` → `deeplog.grounding.JanusGrounder`; `KBestJanusEngine` → `deeplog.systems.deepproblog.KBestJanusGrounder`; `Engine` → `deeplog.grounding.prolog.PrologGrounder`. The old `EngineFactory` is removed — its boolean-construction half is `ProofBuilder` and its label-recording half became DeepProbLog's post-hoc interpretation (k-best keeps a probability-aware `ProbabilisticFactory`, since it needs scalar probabilities *during* search). `SimpleGrounder` no longer enforces annotated-disjunction mutual exclusivity at proof-construction time (it now matches `JanusGrounder`); the mutex is enforced structurally by MV-SDD compilation instead, which `compile_to_module` selects by carrying `EngineResult.variables` through the probability transform, so compiled probabilities are unchanged. Canonical call: `Solver(SimpleGrounder()).get_query_result(program, CircuitFactory())`. `symbol_to_prolog_str` (Symbol → Prolog source) is now public on `deeplog.grounding.prolog`, so a prover that renders its own dialect of a program — the k-best grounder does — overrides `JanusGrounder._rules_to_janus_code` instead of reimplementing program assertion.
- **BREAKING:** A predicate's input shape asks for each *distinct* symbol once per argument position, instead of once per evaluation. Ground atoms sharing an argument — `digit(i1,0) … digit(i1,9)`, the usual "one classifier, N mutually exclusive values" pattern — are distinct evaluations over the *same* input, so a network predicate over twenty such atoms now declares `([i1 i2], [])` rather than twenty duplicated slots, and callers pass one row per distinct symbol. `simplify_module` already presented that interface and is now a no-op on it; code that fed the duplicated interface directly must drop the duplicates. `Predicate` gains the `distinct_arguments` class attribute: positions listed there are handed to `forward_predicate` unexpanded, one row per distinct symbol, and the predicate maps its result back over the evaluations with `evaluation_slots(j)`; listing a position that carries constants raises. Materialization for every other position is now a gather from the distinct rows rather than an allocate-and-scatter.
- `_NetworkPredicate` opts its first argument into `distinct_arguments`, so the wrapped module runs once per distinct input rather than once per ground atom. It was a row-selector — evaluating the network for each atom and keeping one row of its output — so MNIST-Addition ran the CNN 10× per image and discarded 9/10 of each result. On a batch of 8 image pairs that is 16 CNN rows instead of 160, ~7× less perception work, with identical outputs and gradients.
- MV-SDD categorical input slots are named by their annotated-disjunction branch atom (`digit(i1,3)`) rather than a synthetic `("@cat", cat_id, value)` name, so a compiled module's interface speaks the user's atoms — matching how boolean slots are already named. Only padding slots, domain values that no leaf maps to, keep the `@cat` name. Constant-baking is correspondingly scoped to categorical slots: a boolean indicator slot and its `@bool_false` companion stay free runtime inputs, since the runtime input *is* the truth indicator and there is no label-based constant to substitute.
- Split the example notebooks along the ProbLog → DeepProbLog seam. The former `examples/deepproblog/` notebook (probabilistic facts, rules, queries, and conditioning on evidence) is now `examples/problog/`, and a new `examples/deepproblog/` notebook teaches *neural* predicates — a fact's probability comes from a network instead of a constant — building from a tiny worked example up to the MNIST-addition capstone (which absorbs and replaces the standalone `examples/mnist_addition/` notebook). The capstone declares each image's digit as an annotated disjunction and compiles through the MV-SDD compiler (`transform_expectation_to_probability(..., variables=...)`), so the per-image digit probabilities form one mutually-exclusive distribution and the sum is their exact convolution — rather than the independent approximation of the previous `compile_to_module` walkthrough. This adds an optional `mv-sdd` (`pymvsdd`) dependency to the `examples`/`tests`/`site` extras and `requirements.txt`.

### Fixed
- `JanusGrounder` no longer carries a builder across queries. Its tabled lattice aggregation (`disjoin_formulas/3`) has no argument for the factory and reads a global `factory/1` fact, which a query that ended early — an exception, an abandoned solution generator — left asserted; the next query then asserted a second fact and the aggregation resolved the *stale* one, building that query's disjunctions through the previous run's factory. `prove_query` now clears the fact on the way in as well as on the way out. Only observable when two runs use different factories, so it stayed invisible while every caller rendered to text.
- Prolog module names no longer collide across Janus engines. Each engine class caches its consulted programs separately (a subclass renders a different dialect of the same program), but named them `program_<n>` from the size of *its own* cache — while a Prolog module name is global, so `JanusGrounder` and `KBestJanusGrounder` both called their first program `program_0` and the second consult silently replaced the first, leaving the first engine's cache pointing at another engine's program (`unknown_procedure/2` on a predicate it had just asserted). The counter is now shared across the hierarchy, as `engine_counter` already was.
- Proving the goal `true` now yields the identity of conjunction instead of the boolean `false` constant, which zeroed every clause body it appeared in (`q :- true, a.` gave `P(q) = 0`). Carried over from `SimpleEngine`, so it affected released versions too.
- `logp` reads a probability label as `-inf` only when it is actually zero. The threshold was a pinned `1e-12`, so a written label between `0` and `1e-12` — an ordinary float, whose log is perfectly representable — was silently flattened to an impossible fact instead of its true log. Like the semifield's division floor, a bound on a value belongs to the value's own type and not to a constant chosen for one of them.
- `_IndexingTransform` and `_TupleIndexingTransform` now emit an empty output channel as a canonical `(batch, 0)` tensor. Gathering with zero indices instead dragged the input's subtensor dimensions along, so reshaping onto a shape containing an empty `SymTensor` — for instance the constant-only argument position of a binary predicate — produced a wrongly-shaped channel.

## [3.0.5] - 2026-06-17

### Added
- A materialized formula AST in `deeplog.formula`: `FormulaNode` (`Atom`, `UnaryOp`, `BinaryOp`, `Transformation`, `Aggregation`, `CircuitNode`), `fold` (the catamorphism re-emitting a tree through any factory), `map_children`, and `parse_formula_to_ast`. Existing factories (`SymbolicFormulaFactory`, `DeepLogModuleFactory`) are now interpreters of this AST.
- `recognize_expectation` — an AST→AST pass that normalizes a hand-written weighted model count (`sum` over a boolean formula cast to probability, times a factorized distribution) into the efficient `expectation` aggregation when the probability factor uses exactly the same atoms retagged from `boolean` to `probability`.
- `inputs(node)` — a free function giving a derived, transparent view of what feeds a compiled lump: its reachable named circuit leaves, as `Atom`s.
- `Circuit.transform(roots, target_structure, …)` — the "transform" verb as a method on the engine (thin wrapper over `transform_circuit`).
- `Circuit.reachable_leaves(roots)` / `Circuit.reachable_leaf_names(roots)` — the leaves reachable from the given roots (by-symbol map / topological-order name list).

### Changed
- **BREAKING:** `CircuitNode`, `to_module`, and `transform_nodes` are now formula concepts living in `deeplog.formula` (and re-exported from the top-level `deeplog` package). They are no longer importable from `deeplog.circuit`, which is now a pure integer-graph engine. Update imports to `from deeplog.formula import …` (or `from deeplog import …`).
- **BREAKING:** `CircuitNode` is now pure, frozen `(circuit, node)` data, like its symbolic AST siblings — it carries no behaviour and no longer implements the `SupportsToModule` / `HasStructure` protocols. Operating on a lump is a free function: `inputs(node)` for its boundary, `to_module(node, …)` / `transform_nodes(node, …)` to compile or transform, and `node.circuit.structure` to read its structure. The former handle methods are removed (see Removed). The circuit's leaves are the single source of truth for a lump's boundary; cross-structure feeders are recorded against the leaf symbol they re-leaf under (the old `children` field and the `note_origin` hook on `DeepLogFormulaFactory` are gone).
- **BREAKING:** Compiling a sub-root of a shared circuit now scopes input slots, predicate feeders, and expectation leaf mappings to the leaves *reachable* from the compiled roots, instead of every leaf ever added to the circuit.
- **BREAKING (manual factory building):** when you drive a `DeepLogModuleFactory` directly (`factory.create_atom(...)`, `create_binary_node`, …) and hold raw `CircuitNode`s, lower them with the free `lower_circuit_nodes(factory, *nodes)` (or lower a formula AST with `factory.compile(ast)`) — lowering composes the predicate and cross-structure boundary. `parse_formula_to_module(text)` is unchanged (it lowers internally).
- Structure casts (`create_transformation`) no longer compose their boundary eagerly: they return an open spine, and a compiled lump is *transparent* to the lowering fold — its boundary (a predicate leaf per `create_atom`, a cast spine per `create_transformation`) is lowered through the ordinary eliminators and composed with the circuit interior when the lump is lowered. Each circuit leaf is lowered independently (per-leaf, not one batched forward pass).
- DeepProbLog compilation now builds boolean-to-probability leaf mappings directly from the engine `labels` map (boolean atom -> probability label atom). The argument-overlap heuristic, along with the internal probability-distribution factorization helpers, has been removed. The `expectation` aggregation builder no longer accepts probability-formula parameters; handwritten WMC recognition only permits exact same-atom structure retagging.

### Removed
- **BREAKING:** `CircuitNode`, `to_module`, and `transform_nodes` are no longer re-exported from `deeplog.circuit`; import them from `deeplog.formula` or the top-level `deeplog` package.
- **BREAKING:** The `CircuitNode` handle methods `.to_module()`, `.transform_circuit()`, `.get_structure()`, and the `.root` accessor. A lump is now pure data: use the free functions `to_module(node, …)` / `transform_nodes(node, …)`, read `node.circuit.structure`, and use the node directly instead of `.root`. Factory-produced nodes are lowered with `lower_circuit_nodes(factory, *nodes)`.

### Fixed
- An `expectation` over a boolean circuit containing cross-structure boundary leaves now raises a clear error instead of silently mis-counting (the leaf-symbol rewrite would detach the boundary from its feeder).
- `compile_to_module` no longer raises `ValueError: Ambiguous probability mapping` when distinct labeled atoms share arguments (e.g. `nn1(x1) :: a(x1).` and `nn2(x1) :: b(x1).`). The boolean-to-probability leaf mapping is now built directly from the engine's atom labels, which is unambiguous, instead of being reconstructed by an argument-overlap heuristic.

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
