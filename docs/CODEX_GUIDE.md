# CODEX Guide for Future Changes

## Change philosophy (minimal diffs)

- Prefer additive changes over refactors.
- Do not move or rename existing modules unless explicitly requested.
- Mark older paths as legacy rather than deleting them during harness migration.

## Reproducibility requirements

- Keep experiment behavior config-driven from `cfg/project.yaml` (or explicitly documented config extensions).
- Always use explicit seeds for any stochastic behavior.
- Ensure each run writes to run-scoped output directories:
  - Results: `experiments/results/<run_id>/`
  - Plots: `experiments/plots/<run_id>/`

## Output conventions

- Required per run:
  - `metrics.csv` (or `metrics.jsonl`) in results directory.
  - At least one `.png` plot in plots directory.
- Prefer per-method/per-defense plot files to avoid cluttered combined plots.

## Testing expectations

- Keep at least one automated smoke test that executes end-to-end and validates artifact creation.
- Keep default smoke settings tiny so `pytest` completes quickly.
- If adding new run modes, include deterministic tests for config + outputs.

## Documentation expectations

- Update `docs/STATUS.md` when behavior, scope, or legacy status changes.
- Keep README top-level quickstart aligned with the current smoke command.
