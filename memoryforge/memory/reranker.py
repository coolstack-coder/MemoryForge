"""A small PyTorch reranker trained on logged retrieval outcomes.

Each retrieval event is a (features, label) pair where the label says whether
the lesson's pattern was avoided *and* the attempt passed. The model only
switches on when there is enough signal, and even then it is blended with the
hand-written formula rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .retrieval import FEATURES

MIN_EVENTS = 120
MIN_PER_CLASS = 15


class RerankMLP(nn.Module):
    def __init__(self, n_features: int = len(FEATURES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


@dataclass
class Reranker:
    model: RerankMLP
    mean: np.ndarray
    std: np.ndarray
    weight: float
    val_auc: float
    formula_auc: float
    n_events: int

    def score(self, f: dict) -> float:
        x = (np.array([f[k] for k in FEATURES], dtype=np.float32) - self.mean) / self.std
        with torch.no_grad():
            return float(torch.sigmoid(self.model(torch.from_numpy(x)[None]))[0])

    def blend(self, f: dict, formula_score: float) -> float:
        return self.weight * self.score(f) + (1 - self.weight) * formula_score

    def save(self, path: str | Path) -> None:
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "weight": self.weight, "val_auc": self.val_auc,
                    "formula_auc": self.formula_auc, "n_events": self.n_events}, path)

    def report(self) -> dict:
        return {"n_events": self.n_events, "val_auc": round(self.val_auc, 3),
                "formula_auc": round(self.formula_auc, 3), "blend_weight": round(self.weight, 3)}


def events_to_xy(events: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [e for e in events if not e["withheld"] and e["passed"] is not None and e["features"]]
    X = np.array([[e["features"][k] for k in FEATURES] for e in rows], dtype=np.float32)
    y = np.array([float(e["passed"] and not e["pattern_hit"]) for e in rows], dtype=np.float32)
    formula = np.array([e["score"] for e in rows], dtype=np.float32)
    return X, y, formula


def train_reranker(events: list[dict], *, epochs: int = 200, seed: int = 0) -> Reranker | None:
    """Train on events; return None when the data cannot support a model."""
    X, y, formula = events_to_xy(events)
    if len(y) < MIN_EVENTS or y.sum() < MIN_PER_CLASS or (1 - y).sum() < MIN_PER_CLASS:
        return None
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = idx[:cut], idx[cut:]
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mean) / std
    xt, yt = torch.from_numpy(Xn[tr]), torch.from_numpy(y[tr])
    model = RerankMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    pos_weight = torch.tensor((1 - y[tr]).sum() / max(1.0, y[tr].sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = model(torch.from_numpy(Xn[va])).numpy()
        auc = _auc(val, y[va])
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    formula_auc = _auc(formula[va], y[va])
    # trust the model only as far as it beats the formula on held-out events
    weight = float(np.clip((best_auc - formula_auc) * 4 + 0.3, 0.0, 0.7)) if best_auc > formula_auc else 0.0
    return Reranker(model, mean, std, weight, best_auc, formula_auc, len(y))
