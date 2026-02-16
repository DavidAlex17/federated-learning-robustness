import os
import yaml

# Directory this file lives in (…/cfg)
BASE_DIR = os.path.dirname(__file__)


def validate_and_fill_defaults(cfg):
    """Ensure required config keys exist and fill lightweight defaults."""
    cfg = dict(cfg or {})
    assumptions = []

    defaults = {
        "seed": 42,
        "rounds": 5,
        "clients": 10,
        "client_fraction": 1.0,
        "output_root": "experiments",
    }

    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
            assumptions.append(f"missing '{key}' -> using default {value}")

    plot_cfg = cfg.get("plot", {})
    if not isinstance(plot_cfg, dict):
        plot_cfg = {}
        assumptions.append("invalid 'plot' section -> using defaults")

    if "dpi" not in plot_cfg:
        plot_cfg["dpi"] = 120
        assumptions.append("missing 'plot.dpi' -> using default 120")

    if "figsize" not in plot_cfg:
        plot_cfg["figsize"] = [6, 4]
        assumptions.append("missing 'plot.figsize' -> using default [6, 4]")

    cfg["plot"] = plot_cfg

    if "results_dir" not in cfg:
        cfg["results_dir"] = os.path.join(cfg["output_root"], "results")
        assumptions.append("missing 'results_dir' -> using output_root/results")

    if "plots_dir" not in cfg:
        cfg["plots_dir"] = os.path.join(cfg["output_root"], "plots")
        assumptions.append("missing 'plots_dir' -> using output_root/plots")

    if assumptions:
        print("[config] assumptions:")
        for note in assumptions:
            print(f"  - {note}")

    return cfg


def load(path=None):
    """
    Load project configuration and resolve relative paths.
    If path is None, loads cfg/project.yaml next to this file.
    """
    if path is None:
        path = os.path.join(BASE_DIR, "project.yaml")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = validate_and_fill_defaults(cfg)

    for key in ("data_dir", "results_dir", "plots_dir", "output_root"):
        if key in cfg:
            cfg[key] = os.path.abspath(cfg[key])

    return cfg
