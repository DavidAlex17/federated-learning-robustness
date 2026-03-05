"""MLP model definition and local training / evaluation helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _get_params(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def _set_params(model: nn.Module, params: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state, strict=True)


def _train_one_epoch(model: nn.Module, loader: DataLoader, lr: float, device: str) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        batch_n = x.size(0)
        loss_sum += loss.item() * batch_n
        n += batch_n
    return loss_sum / max(n, 1)


def _evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)
            batch_n = x.size(0)
            loss_sum += loss.item() * batch_n
            correct += (preds == y).sum().item()
            n += batch_n
    return loss_sum / max(n, 1), correct / max(n, 1)
