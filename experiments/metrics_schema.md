# Metrics CSV Schema

This repository uses a single, method-agnostic per-round schema for `metrics.csv`.

## Required columns

1. `round` (int): 1-indexed training round number.
2. `client_fraction` (float): fraction of clients participating in the round.
3. `train_loss` (float, optional/nullable): training loss summary for the round.
4. `val_loss` (float, optional/nullable): validation loss summary for the round.
5. `val_acc` (float): validation accuracy summary for the round.
6. `time_round_sec` (float): wall-clock seconds consumed by the round.

## Notes

- Output format for Step 2 is CSV (`metrics.csv`) under `experiments/results/<run_id>/`.
- Plotting tools consume this schema and produce one plot per method under `experiments/plots/<run_id>/`.
