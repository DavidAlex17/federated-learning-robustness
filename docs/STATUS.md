# Project Status

## What currently works

- A fast, config-driven smoke-run entrypoint exists at `experiments/run_smoke.py`, backed by reusable `experiments/runner.py`.
- The smoke run uses `cfg/project.yaml` defaults (clients, rounds, seed) with bounded tiny settings for fast execution.
- Smoke artifacts are written under run-specific directories:
  - `experiments/results/<run_id>/metrics.csv`
  - `experiments/plots/<run_id>/smoke_synth_val_acc.png`
- Metrics schema is documented in `experiments/metrics_schema.md` and used by the plotting path.
- A pytest smoke test (`tests/test_smoke.py`) validates end-to-end execution and artifact creation.

## Legacy / outdated components

- The existing Flower-based experiment path in `experiments/baseline_fl.py` remains a prototype path and is **not** the CI smoke-run path.
- Existing static artifacts under `experiments/results/` and `experiments/plots/` are legacy examples and not treated as reproducible benchmark outputs.
- Defense and attack settings in `cfg/project.yaml` are currently broader than the smoke harness scope and should be considered staged configuration for future full experiments.

## Next milestones

1. Add config-driven method variants for baseline/attack/defense runs with dedicated per-method output folders under each `run_id`.
2. Wire full FL and PID-style exclusion path into a reproducible run orchestration script while preserving smoke speed defaults.
3. Expand plotting to method-separated figures and add summary comparison plots without clutter.
4. Add additional tests for config override behavior and deterministic seeds.

## Assumptions (Step 2)

- The smoke run is intentionally lightweight and uses a deterministic synthetic loop so tests remain fast and robust in constrained environments.
- Full Flower/FEMNIST execution is preserved as legacy/prototype code and not required for Step 1 completion.
