# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/deeplog/` (core shapes, modules, circuits, formula factory).
- Tests are under `tests/deeplog/` (unit) and `tests/integration/` (end-to-end). Use existing helpers like `tests/deeplog/module/_utils.py`.
- Docs and examples: `docs/`, `site/`, `examples/` notebooks. Packaging config in `pyproject.toml`, `requirements.txt`.

## Build, Test, and Development Commands
- Run targeted tests: `PYTHONPATH=src pytest tests/deeplog/module/test_simplify_module.py`.
- Run full suite: `PYTHONPATH=src pytest`.

## Coding Style & Naming Conventions
- Python 3.12+ typing style with `tuple[...]`/`list[...]`, prefer `as_tuple` helper for uniform tuple handling.
- Indentation: 4 spaces; keep lines readable; minimal inline comments unless clarifying non-obvious logic.

## Testing Guidelines
- Framework: `pytest`. Place new unit tests alongside feature modules (e.g., `tests/deeplog/module/` for `reshape`/`sequential` changes).
- Name tests descriptively (`test_simplify_duplicate_inputs`) and assert both shape and tensor content where relevant.
- Run affected tests before raising PRs (`PYTHONPATH=src pytest path/to/test_file.py`).

## Commit & Pull Request Guidelines
- Commits: keep messages imperative and scoped (e.g., "Add simplify_module empty-input fast path").
- PRs: include a brief summary, affected areas, and test evidence (`pytest …`). Link issues when applicable. Include screenshots only for docs/UI changes (rare here).

## Architecture Notes
- Shapes: `SymTensor` declares symbolic layouts; `construct_transformation` builds reshape paths between shapes.
- Modules: `DeepLogModule` wraps torch modules with shape validation; `Sequential` composes them; `simplify_module` trims inputs and prepends necessary transforms.
- Circuits/Formula: `formula_to_module` compiles symbolic formulas via `DeepLogModuleFactory`; aggregation nodes spawn aggregation modules automatically.
- Notebooks: when referring to API types, link to their import path so docs render correctly (e.g., `[DeepLogModule](deeplog.module.deeplog_module.DeepLogModule)`).
