from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "pytest-smoke"


def test_smoke_run_produces_artifacts() -> None:
    cmd = [
        sys.executable,
        "experiments/run_smoke.py",
        "--run-id",
        RUN_ID,
        "--clean",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    metrics_path = REPO_ROOT / "experiments" / "results" / RUN_ID / "metrics.csv"
    plot_path = REPO_ROOT / "experiments" / "plots" / RUN_ID / "accuracy.png"

    assert metrics_path.exists(), "metrics.csv was not created"
    assert plot_path.exists(), "accuracy.png was not created"

    with metrics_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows, "metrics.csv should contain at least one data row"
