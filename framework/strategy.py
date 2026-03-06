"""RobustFedStrategy: configurable aggregation with optional PID-based Byzantine defense."""

from __future__ import annotations

import time

import numpy as np
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from framework.aggregators import fedavg, multi_krum, trimmed_mean
from framework.defense import cosine_direction_error, select_top_k_by_score, update_pid_score


class RobustFedStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        client_fraction: float,
        aggregator_cfg: dict,
        attack_cfg: dict,
        defense_cfg: dict,
        malicious_ids: set[int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.client_fraction = client_fraction
        self.aggregator = aggregator_cfg.get("aggregator", "fedavg")
        self.trim_ratio = float(aggregator_cfg.get("trim_ratio", 0.1))
        self.f = int(aggregator_cfg.get("f", 1))
        self.m = aggregator_cfg.get("m", None)
        if self.m is not None:
            self.m = int(self.m)

        self.attack_cfg = attack_cfg
        self.malicious_ids = malicious_ids

        self.defense_cfg = defense_cfg
        self.defense_enabled = bool(defense_cfg.get("enabled", False))
        self.k_exclude = int(defense_cfg.get("k_exclude", 1))
        self.threshold = float(defense_cfg.get("threshold", 0.5))
        self.integral_decay = float(defense_cfg.get("integral_decay", 1.0))
        self.kp = float(defense_cfg.get("Kp", 1.0))
        self.ki = float(defense_cfg.get("Ki", 0.0))
        self.kd = float(defense_cfg.get("Kd", 0.0))
        self.warmup_rounds = int(defense_cfg.get("warmup_rounds", 0))
        self.pid_state: dict[int, dict] = {}

        self.round_metrics: list[dict] = []
        self.agg_debug: list[dict] = []
        self.attack_debug: list[dict] = []
        self.defense_debug: list[dict] = []
        self._round_starts: dict[int, float] = {}
        self._train_loss_by_round: dict[int, float] = {}
        self._round_base_params: dict[int, list[np.ndarray]] = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_starts[server_round] = time.perf_counter()
        self._round_base_params[server_round] = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    def _aggregate_custom(self, server_round: int, results):
        ndarrays_by_client = [parameters_to_ndarrays(res.parameters) for _, res in results]
        vectors = [np.concatenate([a.ravel() for a in arrs]) for arrs in ndarrays_by_client]
        n_clients_total = len(vectors)

        if self.aggregator == "fedavg":
            weights = [res.num_examples for _, res in results]
            agg_vec = fedavg(vectors, weights=weights)
            n_used = n_clients_total
            n_selected = n_clients_total
        elif self.aggregator == "trimmed_mean":
            agg_vec = trimmed_mean(vectors, trim_ratio=self.trim_ratio)
            k = int(np.floor(self.trim_ratio * n_clients_total))
            n_used = max(1, n_clients_total - 2 * k)
            n_selected = n_used
        elif self.aggregator == "multi_krum":
            neighbor_count = n_clients_total - self.f - 2
            if neighbor_count <= 0:
                agg_vec = fedavg(vectors, weights=None)
                n_used = n_clients_total
                n_selected = n_clients_total
            else:
                m = self.m if self.m is not None else neighbor_count
                m = max(1, min(m, n_clients_total))
                agg_vec = multi_krum(vectors, f=self.f, m=m)
                n_used = n_clients_total
                n_selected = m
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

        template = ndarrays_by_client[0]
        rebuilt, offset = [], 0
        for arr in template:
            size = int(np.prod(arr.shape))
            rebuilt_arr = agg_vec[offset : offset + size].reshape(arr.shape).astype(arr.dtype)
            rebuilt.append(rebuilt_arr)
            offset += size

        self.agg_debug.append(
            {
                "round": server_round,
                "aggregator": self.aggregator,
                "n_clients_total": n_clients_total,
                "n_clients_used": n_used,
                "n_selected": n_selected,
                "trim_ratio": self.trim_ratio if self.aggregator == "trimmed_mean" else "",
            }
        )
        return ndarrays_to_parameters(rebuilt)

    def _apply_pid_exclusion(self, server_round: int, results):
        if not results:
            return results, [], {}, {}

        base = self._round_base_params.get(server_round)
        if base is None:
            return results, [], {}, {}
        base_vec = np.concatenate([a.ravel() for a in base])

        client_ids = [int(res.metrics.get("client_id", idx)) for idx, (_, res) in enumerate(results)]
        updates = []
        for _, fit_res in results:
            vec = np.concatenate([a.ravel() for a in parameters_to_ndarrays(fit_res.parameters)])
            updates.append(vec - base_vec)
        updates_arr = np.asarray(updates)
        ref = np.median(updates_arr, axis=0)

        scores = {}
        cosines = {}
        for cid, upd in zip(client_ids, updates):
            error, cosine = cosine_direction_error(upd, ref)
            score, new_state = update_pid_score(error, self.pid_state.get(cid, {}), self.kp, self.ki, self.kd, self.integral_decay)
            self.pid_state[cid] = new_state
            scores[cid] = score
            cosines[cid] = cosine

        excluded_ids = []
        if self.defense_enabled and server_round > self.warmup_rounds:
            k = max(0, min(self.k_exclude, len(results) - 1))
            candidates = select_top_k_by_score(scores, k)
            excluded_ids = [cid for cid in candidates if scores[cid] > self.threshold]

        kept = []
        for pair, cid in zip(results, client_ids):
            if cid not in excluded_ids:
                kept.append(pair)
        if not kept:
            kept = [results[0]]

        return kept, excluded_ids, scores, cosines

    def aggregate_fit(self, server_round, results, failures):
        total_examples, weighted_loss = 0, 0.0
        participating_ids = []
        for _, fit_res in results:
            n = fit_res.num_examples
            total_examples += n
            weighted_loss += n * float(fit_res.metrics.get("train_loss", 0.0))
            if "client_id" in fit_res.metrics:
                participating_ids.append(int(fit_res.metrics["client_id"]))
        self._train_loss_by_round[server_round] = weighted_loss / max(total_examples, 1)

        malicious_participating = sorted([cid for cid in participating_ids if cid in self.malicious_ids])
        self.attack_debug.append(
            {
                "round": server_round,
                "malicious_fraction": float(self.attack_cfg.get("malicious_fraction", 0.0)),
                "malicious_count": len(malicious_participating),
                "total_clients_sampled": len(participating_ids),
                "malicious_ids_participated": "|".join(str(cid) for cid in malicious_participating),
            }
        )

        filtered_results, excluded_ids, scores, cosines = self._apply_pid_exclusion(server_round, results)

        n_total = len(results)
        n_excluded = len(excluded_ids)
        malicious_in_round = set(malicious_participating)
        excluded_set = set(excluded_ids)
        if bool(self.attack_cfg.get("enabled", False)):
            tp = len(excluded_set & malicious_in_round)
            fp = len(excluded_set - malicious_in_round)
            fn = len(malicious_in_round - excluded_set)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            tp = fp = fn = ""
            precision = recall = ""

        score_str = "|".join(f"{cid}:{scores[cid]:.6f}(cos={cosines.get(cid, 1.0):.6f})" for cid in sorted(scores.keys()))
        self.defense_debug.append(
            {
                "round": server_round,
                "n_total": n_total,
                "n_excluded": n_excluded,
                "excluded_ids": "|".join(str(cid) for cid in sorted(excluded_ids)),
                "malicious_ids_in_round": "|".join(str(cid) for cid in sorted(malicious_in_round)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "scores": score_str,
            }
        )

        if not filtered_results:
            return None, {}
        params = self._aggregate_custom(server_round, filtered_results)
        return params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
        total_examples, weighted_acc = 0, 0.0
        for _, eval_res in results:
            n = eval_res.num_examples
            total_examples += n
            weighted_acc += n * float(eval_res.metrics.get("val_acc", 0.0))
        val_acc = weighted_acc / max(total_examples, 1)

        time_sec = time.perf_counter() - self._round_starts.get(server_round, time.perf_counter())
        self.round_metrics.append(
            {
                "round": server_round,
                "client_fraction": round(self.client_fraction, 6),
                "train_loss": round(self._train_loss_by_round.get(server_round, 0.0), 6),
                "val_loss": round(float(agg_loss or 0.0), 6),
                "val_acc": round(val_acc, 6),
                "time_round_sec": round(time_sec, 6),
            }
        )
        if isinstance(agg_metrics, dict):
            agg_metrics["val_acc"] = val_acc
        return agg_loss, agg_metrics
