"""Flower NumPyClient for MNIST federated learning."""

from __future__ import annotations

import numpy as np
import flwr as fl
from torch.utils.data import Subset

from framework.attack import apply_signflip_attack
from model.mlp import MLP, _evaluate, _get_params, _set_params, _train_one_epoch


class MnistClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        train_subset: Subset,
        test_subset: Subset,
        config: dict,
        malicious_ids: set[int],
        attack_cfg: dict,
    ) -> None:
        self.cid = cid
        self.is_malicious = cid in malicious_ids and bool(attack_cfg.get("enabled", False))
        self.attack_type = str(attack_cfg.get("type", "signflip"))
        self.attack_scale = float(attack_cfg.get("scale", 1.0))

        self.device = "cpu"
        self.model = MLP().to(self.device)
        self.lr = float(config.get("lr", 0.01))
        self.local_epochs = int(config.get("local_epochs", 1))
        self.batch_size = int(config.get("batch_size", 32))

        from torch.utils.data import DataLoader
        self.train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

    def get_parameters(self, config):
        return _get_params(self.model)

    def fit(self, parameters, config):
        old_params = [np.array(p, copy=True) for p in parameters]
        _set_params(self.model, old_params)
        losses = [_train_one_epoch(self.model, self.train_loader, self.lr, self.device) for _ in range(self.local_epochs)]
        mean_loss = float(np.mean(losses)) if losses else 0.0
        new_params = _get_params(self.model)

        if self.is_malicious and self.attack_type == "signflip":
            returned_params = apply_signflip_attack(old_params, new_params, self.attack_scale)
        else:
            returned_params = new_params

        return returned_params, len(self.train_loader.dataset), {
            "train_loss": mean_loss,
            "client_id": self.cid,
            "is_malicious": int(self.is_malicious),
        }

    def evaluate(self, parameters, config):
        _set_params(self.model, parameters)
        val_loss, val_acc = _evaluate(self.model, self.test_loader, self.device)
        return float(val_loss), len(self.test_loader.dataset), {"val_acc": float(val_acc)}
