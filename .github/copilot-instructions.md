## Purpose

Short, actionable guidance for AI coding agents working in this repository. Focus on the project's architecture, developer workflows, config conventions, and concrete edit/run examples.

## Big-picture architecture (what to know first)

- Top-level layout:
  - `framework/aggregators.py` — pure NumPy aggregation functions (`fedavg`, `trimmed_mean`, `multi_krum`). The only library module in `framework/`.
  - `experiments/methods/fedavg.py` — the complete FL backend: `MLP`, `MnistClient`, `RobustFedStrategy` (aggregation + attack injection + PID defense), and all debug writers.
  - `experiments/runner.py` — dispatch layer: `run_experiment(config, run_id, out_dir_results, out_dir_plots, method)`.
  - `experiments/run_fedavg.py` — CLI entrypoint for all FL runs (aggregator, attack, defense, partition overrides).
  - `experiments/run_smoke.py` — fast synthetic smoke entrypoint (no real FL, used by CI).
  - `cfg/project.yaml` + `cfg/load_config.py` — all config defaults and path resolution.
- Data: raw MNIST lives under `data/MNIST/`. Results written to `experiments/results/<run_id>/`, plots to `experiments/plots/<run_id>/`.
- Flow for a real FL run: `run_fedavg.py` → `load_cfg()` → `run_experiment(method="fedavg")` → `fedavg.run_fedavg()` → Flower simulation with `RobustFedStrategy` → writes `metrics.csv`, `run_meta.yaml`, `agg_debug.csv`, `attack_debug.csv`, `defense_debug.csv`, `partition_debug.csv`.

## Key integration points (where to edit)

- **Aggregation logic** (FedAvg / TrimmedMean / Multi-Krum): `framework/aggregators.py` for the math; `RobustFedStrategy._aggregate_custom()` in `experiments/methods/fedavg.py` for dispatch.
- **Attack injection**: `apply_signflip_attack()` and `MnistClient.fit()` in `experiments/methods/fedavg.py`.
- **Defense / anomaly scoring**: `RobustFedStrategy._apply_pid_exclusion()` in `experiments/methods/fedavg.py`; PID helpers `update_pid_score()`, `cosine_direction_error()`, `select_top_k_by_score()` are module-level functions in the same file.
- **Partitioning**: `_partition_indices_iid()` and `_dirichlet_partition_indices()` in `experiments/methods/fedavg.py`.
- **Config defaults**: `cfg/project.yaml` (source of truth) + `cfg/load_config.py` (validates and fills missing keys). Always add new knobs in both.
- **CLI flags**: `experiments/run_fedavg.py` — add `argparse` args here and forward them into `cfg` before calling `run_experiment`.

## How to run (concrete commands)

```bash
# Activate environment (devcontainer already does this)
source .venv/bin/activate

# Fast synthetic smoke run (used by CI / pytest)
PYTHONPATH=. python experiments/run_smoke.py --run-id my-smoke --clean

# Real FL run — clean baseline
PYTHONPATH=. python experiments/run_fedavg.py --run-id clean-fedavg --rounds 5 --aggregator fedavg

# With sign-flip attack (25% malicious clients)
PYTHONPATH=. python experiments/run_fedavg.py --run-id atk-fedavg --rounds 5 \
  --attack --malicious-fraction 0.25

# Attack + PID defense
PYTHONPATH=. python experiments/run_fedavg.py --run-id pid-atk-fedavg --rounds 5 \
  --attack --malicious-fraction 0.25 --defense --k-exclude 2

# Non-IID (Dirichlet α=0.1) partition
PYTHONPATH=. python experiments/run_fedavg.py --run-id niid-clean --rounds 5 \
  --partition dirichlet --alpha 0.1

# Summarize all runs (compact table)
PYTHONPATH=. python experiments/summarize_runs.py --minimal

# Run tests
PYTHONPATH=. python -m pytest -q
```

## Project-specific conventions and patterns

- **Reproducibility**: always seed with `cfg["seed"]`; propagate to numpy, torch, python random. Attack client selection seeds from `attack.seed`.
- **Config**: use `cfg.load_config.load()` — it converts `data_dir`/`results_dir`/`plots_dir` to absolute paths. Never construct paths manually.
- **Schema**: `metrics.csv` columns are fixed: `round, client_fraction, train_loss, val_loss, val_acc, time_round_sec`. Do not add columns.
- **Debug artifacts**: extra data goes into separate CSVs (`agg_debug.csv`, `attack_debug.csv`, `defense_debug.csv`, `partition_debug.csv`), not into `metrics.csv`.
- **Run-scoped outputs**: every run writes under `experiments/results/<run_id>/` and `experiments/plots/<run_id>/`. Never write to a shared path.
- **PYTHONPATH**: always run scripts with `PYTHONPATH=.` from repo root (or use the Makefile targets).
- **Output dirs**: call `Path(...).mkdir(parents=True, exist_ok=True)` before writing any file.

## Concrete examples to copy from

- Adding a new experiment script: copy `experiments/run_fedavg.py`, add CLI flags, override `cfg` dict before calling `run_experiment`.
- Adding a new aggregator: add a function to `framework/aggregators.py`, add a unit test in `tests/test_aggregators.py`, add the dispatch branch in `RobustFedStrategy._aggregate_custom()`.
- Adding a new defense: implement inside `RobustFedStrategy` in `experiments/methods/fedavg.py`, add a config key to `cfg/project.yaml` + `cfg/load_config.py`, and write a unit test.

## Dependencies & environment

- Key packages: `torch`, `torchvision`, `flwr[simulation]`, `numpy`, `matplotlib`, `pyyaml`, `pytest`. See `requirements.txt`.
- Dev Container (Python 3.11) is the recommended environment. If not using it: `pip install -r requirements.txt`.

## Quick troubleshooting hints

- Import errors: run with `PYTHONPATH=.` from repo root.
- Wrong paths: check `cfg/project.yaml` and confirm `cfg.load_config.load()` is called (it makes paths absolute).
- Non-determinism: confirm `cfg["seed"]` is passed through to all RNG sites.

## Files to reference when making changes

- `cfg/project.yaml`, `cfg/load_config.py` — config defaults and path resolution
- `framework/aggregators.py` — aggregation math (fedavg, trimmed_mean, multi_krum)
- `experiments/methods/fedavg.py` — FL backend (model, client, strategy, attack, defense, partitioning, debug writers)
- `experiments/runner.py` — experiment dispatch
- `experiments/run_fedavg.py` — canonical CLI entrypoint
- `experiments/run_smoke.py` — fast smoke entrypoint (copy for new quick-check scripts)
- `Makefile` — convenient targets

---

If any of these areas are unclear or you'd like the instructions to include more examples (for instance, sample code snippets to implement a PID aggregator), tell me which part to expand and I'll iterate.
