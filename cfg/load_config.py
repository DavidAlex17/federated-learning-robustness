import os
import yaml

# Directory this file lives in (…/cfg)
BASE_DIR = os.path.dirname(__file__)


def validate_and_fill_defaults(cfg):
    """Ensure required config keys exist and fill lightweight defaults."""
    cfg = dict(cfg or {})
    assumptions = []

    defaults = {
        "method": "smoke_synth",
        "seed": 42,
        "rounds": 5,
        "clients": 10,
        "client_fraction": 1.0,
        "output_root": "experiments",
        "local_epochs": 1,
        "batch_size": 32,
        "lr": 0.01,
    }

    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
            assumptions.append(f"missing '{key}' -> using default {value}")

    dataset_cfg = cfg.get("dataset", {})
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
        assumptions.append("invalid 'dataset' section -> using defaults")

    dataset_defaults = {
        "name": "mnist",
        "train_samples_cap": 1000,
        "test_samples_cap": 200,
    }
    for key, value in dataset_defaults.items():
        if key not in dataset_cfg:
            dataset_cfg[key] = value
            assumptions.append(f"missing 'dataset.{key}' -> using default {value}")
    cfg["dataset"] = dataset_cfg

    partition_cfg = cfg.get("partition", {})
    if not isinstance(partition_cfg, dict):
        partition_cfg = {}
        assumptions.append("invalid 'partition' section -> using defaults")

    partition_defaults = {"type": "iid", "alpha": 0.1}
    for key, value in partition_defaults.items():
        if key not in partition_cfg:
            partition_cfg[key] = value
            assumptions.append(f"missing 'partition.{key}' -> using default {value}")

    cfg["partition"] = partition_cfg

    server_cfg = cfg.get("server", {})
    if not isinstance(server_cfg, dict):
        server_cfg = {}
        assumptions.append("invalid 'server' section -> using defaults")

    if "aggregator" not in server_cfg:
        server_cfg["aggregator"] = "fedavg"
        assumptions.append("missing 'server.aggregator' -> using default fedavg")
    if "trim_ratio" not in server_cfg:
        server_cfg["trim_ratio"] = 0.1
        assumptions.append("missing 'server.trim_ratio' -> using default 0.1")
    if "f" not in server_cfg:
        server_cfg["f"] = 1
        assumptions.append("missing 'server.f' -> using default 1")
    if "m" not in server_cfg:
        server_cfg["m"] = None

    cfg["server"] = server_cfg

    attack_cfg = cfg.get("attack", {})
    if not isinstance(attack_cfg, dict):
        attack_cfg = {}
        assumptions.append("invalid 'attack' section -> using defaults")

    attack_defaults = {
        "enabled": False,
        "type": "signflip",
        "malicious_fraction": 0.0,
        "scale": 1.0,
        "target": "update",
    }
    for key, value in attack_defaults.items():
        if key not in attack_cfg:
            attack_cfg[key] = value
            assumptions.append(f"missing 'attack.{key}' -> using default {value}")
    if "seed" not in attack_cfg:
        attack_cfg["seed"] = int(cfg.get("seed", 42))
        assumptions.append("missing 'attack.seed' -> using global seed")

    cfg["attack"] = attack_cfg

    defense_cfg = cfg.get("defense", {})
    if not isinstance(defense_cfg, dict):
        defense_cfg = {}
        assumptions.append("invalid 'defense' section -> using defaults")

    defense_defaults = {
        "enabled": False,
        "type": "pid_exclusion",
        "k_exclude": 1,
        "Kp": 1.0,
        "Ki": 0.0,
        "Kd": 0.0,
        "warmup_rounds": 0,
    }
    for key, value in defense_defaults.items():
        if key not in defense_cfg:
            defense_cfg[key] = value
            assumptions.append(f"missing 'defense.{key}' -> using default {value}")

    cfg["defense"] = defense_cfg

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
