import csv
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import yaml
from flask import Flask, abort, jsonify, render_template, request, send_file

app = Flask(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"

_BATCH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")
_SCORE_RE = re.compile(r"(\d+):([0-9.]+)\(cos=(-?[0-9.]+)\)")

def _batch_dirs() -> list[Path]:
    """Return sorted list of timestamped batch directories (oldest first)."""
    if not RESULTS_DIR.exists():
        return []
    return sorted(
        p for p in RESULTS_DIR.iterdir()
        if p.is_dir() and _BATCH_RE.match(p.name)
    )


def _find_run_dir(run_id: str, batch: str | None = None) -> tuple[str, Path]:
    """Return (batch_name, run_path) for the given run_id.

    Searches the most recent batch by default; honours the `batch` filter when
    provided.  Aborts with 400/404 on invalid or missing inputs.
    """
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        abort(400, "Invalid run_id")

    batches = _batch_dirs()

    if batch:
        if not _BATCH_RE.match(batch):
            abort(400, "Invalid batch name")
        for b in batches:
            if b.name == batch:
                rp = b / run_id
                if rp.is_dir():
                    return b.name, rp
        abort(404)

    # Most recent batch that contains this run
    for b in reversed(batches):
        rp = b / run_id
        if rp.is_dir():
            return b.name, rp

    abort(404)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/visualize")
def visualize():
    return render_template("visualize.html")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/runs")
def api_runs():
    """Return all run IDs grouped by batch timestamp (newest first)."""
    runs = []
    for batch_dir in reversed(_batch_dirs()):
        for run_dir in sorted(batch_dir.iterdir()):
            if run_dir.is_dir():
                runs.append({"run_id": run_dir.name, "batch": batch_dir.name})
    return jsonify({"runs": runs})


@app.route("/api/summary")
def api_summary():
    """Return one summary row per run: final metrics + meta settings."""
    rows = []
    for batch_dir in reversed(_batch_dirs()):
        for run_dir in sorted(batch_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            metrics_path = run_dir / "metrics.csv"
            meta_path = run_dir / "run_meta.yaml"

            if not metrics_path.exists():
                continue

            # Read last row of metrics
            with open(metrics_path, newline="") as f:
                last_row = None
                for last_row in csv.DictReader(f):
                    pass

            if last_row is None:
                continue

            meta: dict = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = yaml.safe_load(f) or {}

            partition = meta.get("partition") or {}
            attack = meta.get("attack") or {}
            defense = meta.get("defense") or {}

            rows.append({
                "run_id": run_dir.name,
                "batch": batch_dir.name,
                "aggregator": meta.get("aggregator", ""),
                "partition": partition.get("type", ""),
                "partition_alpha": partition.get("alpha", ""),
                "attack_enabled": bool(attack.get("enabled", False)),
                "attack_type": attack.get("type", ""),
                "malicious_fraction": attack.get("malicious_fraction", 0),
                "defense_enabled": bool(defense.get("enabled", False)),
                "defense_type": defense.get("type", ""),
                "k_exclude": defense.get("k_exclude", ""),
                "rounds": meta.get("rounds", ""),
                "clients": meta.get("clients", ""),
                "final_val_acc": float(last_row.get("val_acc", 0)),
                "final_val_loss": float(last_row.get("val_loss", 0)),
            })

    return jsonify(rows)


def _run_response(run_id: str, batch_name: str, run_dir: Path):
    """Build the JSON response for a single run."""
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        abort(404)

    metrics = []
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            metrics.append({
                "round": int(row["round"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
                "time_round_sec": float(row["time_round_sec"]),
            })

    meta: dict = {}
    meta_path = run_dir / "run_meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

    return jsonify({"run_id": run_id, "batch": batch_name, "meta": meta, "metrics": metrics})


@app.route("/api/run/<batch_id>/<run_id>")
def api_run_path(batch_id: str, run_id: str):
    """Return per-round metrics — batch and run_id both in the URL path."""
    batch_name, run_dir = _find_run_dir(run_id, batch_id)
    return _run_response(run_id, batch_name, run_dir)


@app.route("/api/run/<run_id>")
def api_run(run_id: str):
    """Return per-round metrics (legacy: batch as query param)."""
    batch = request.args.get("batch")
    batch_name, run_dir = _find_run_dir(run_id, batch)
    return _run_response(run_id, batch_name, run_dir)


@app.route("/api/defense/<batch_id>/<run_id>")
def api_defense(batch_id: str, run_id: str):
    """Return precision/recall per round from defense_debug.csv."""
    batch_name, run_dir = _find_run_dir(run_id, batch_id)
    debug_path = run_dir / "defense_debug.csv"
    if not debug_path.exists():
        abort(404)
    rows = []
    with open(debug_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("precision") == "" or row.get("recall") == "":
                continue
            rows.append({
                "round": int(row["round"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
            })
    return jsonify({"run_id": run_id, "batch": batch_name, "defense": rows})


@app.route("/api/defense_scores/<batch_id>/<run_id>")
def api_defense_scores(batch_id: str, run_id: str):
    """Return per-client PID scores and cosine similarity per round."""
    batch_name, run_dir = _find_run_dir(run_id, batch_id)
    debug_path = run_dir / "defense_debug.csv"
    if not debug_path.exists():
        abort(404)

    meta: dict = {}
    meta_path = run_dir / "run_meta.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f) or {}

    malicious_ids: list = meta.get("attack", {}).get("malicious_ids") or []

    rounds = []
    with open(debug_path, newline="") as f:
        for row in csv.DictReader(f):
            excluded_raw = row.get("excluded_ids", "") or ""
            excluded_ids = [int(x) for x in excluded_raw.split("|") if x.strip()]

            clients = []
            for m in _SCORE_RE.finditer(row.get("scores", "") or ""):
                clients.append({
                    "client_id": int(m.group(1)),
                    "score": float(m.group(2)),
                    "cos": float(m.group(3)),
                })

            rounds.append({
                "round": int(row["round"]),
                "excluded_ids": excluded_ids,
                "clients": clients,
            })

    return jsonify({
        "run_id": run_id,
        "batch": batch_name,
        "malicious_ids": malicious_ids,
        "rounds": rounds,
    })


@app.route("/api/run_batch", methods=["POST"])
def api_run_batch():
    """Accept run config JSON, write a temp batch YAML, launch run_batch.py."""
    cfg = request.get_json(silent=True)
    if not cfg or not isinstance(cfg, dict):
        abort(400, "Invalid JSON body")

    try:
        rounds    = max(1, min(200, int(cfg.get("rounds", 20))))
        clients   = max(1, min(100, int(cfg.get("clients", 10))))
        part_type = str(cfg.get("partition_type", "iid"))
        if part_type not in ("iid", "dirichlet"):
            abort(400, "Invalid partition_type")
        alpha              = max(0.01, min(10.0, float(cfg.get("alpha", 0.1))))
        attack_enabled     = bool(cfg.get("attack_enabled", False))
        malicious_fraction = max(0.05, min(0.9, float(cfg.get("malicious_fraction", 0.25))))
        defense_enabled    = bool(cfg.get("defense_enabled", False))
        k_exclude          = max(1, min(10, int(cfg.get("k_exclude", 2))))
        kp                 = max(0.0, min(10.0, float(cfg.get("Kp", 1.0))))
    except (TypeError, ValueError):
        abort(400, "Invalid field value")

    run_id = f"ui-{uuid.uuid4().hex[:8]}"

    run_def: dict = {
        "run_id": run_id,
        "rounds": rounds,
        "clients": clients,
        "partition": {"type": part_type},
        "attack": {"enabled": attack_enabled},
        "defense": {"enabled": defense_enabled},
    }
    if part_type == "dirichlet":
        run_def["partition"]["alpha"] = alpha
    if attack_enabled:
        run_def["attack"]["type"] = "signflip"
        run_def["attack"]["malicious_fraction"] = malicious_fraction
    if defense_enabled:
        run_def["defense"]["k_exclude"] = k_exclude
        run_def["defense"]["Kp"] = kp

    temp_yaml = PROJECT_ROOT / "experiments" / f"_ui_batch_{uuid.uuid4().hex[:8]}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.safe_dump({"runs": [run_def]}, f)

    subprocess.Popen(
        [sys.executable,
         str(PROJECT_ROOT / "experiments" / "run_batch.py"),
         "--batch", str(temp_yaml)],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
    )

    return jsonify({"status": "started", "run_id": run_id})


# ---------------------------------------------------------------------------
# Static plot serving
# ---------------------------------------------------------------------------

@app.route("/plots/<path:image_path>")
def serve_plot(image_path: str):
    """Serve PNG plots from experiments/plots/ with path-traversal protection."""
    plots_root = PLOTS_DIR.resolve()
    target = (PLOTS_DIR / image_path).resolve()

    if not target.is_relative_to(plots_root):
        abort(403)
    if not target.is_file():
        abort(404)

    return send_file(target)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
