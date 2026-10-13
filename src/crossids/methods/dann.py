"""Domain-adversarial neural network (Ganin et al., 2016) on CPU with PyTorch."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import Domain, Method, MethodUnavailable, pool_sources

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None

if torch is not None:

    class _GradientReversal(torch.autograd.Function):
        """Identity in the forward pass, scaled negated gradient in the backward pass."""

        @staticmethod
        def forward(ctx, x, lam):
            ctx.lam = lam
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.lam * grad, None


class Dann(Method):
    """Shared feature extractor with a label head and a gradient-reversed domain head."""

    name = "dann"
    transductive = True
    supports_inductive = False

    def __init__(self, seed: int = 0, epochs: int = 20, batch_size: int = 512, lr: float = 1e-3, hidden: int = 128):
        if torch is None:
            raise MethodUnavailable("torch is not installed; the dann method is skipped")
        super().__init__(seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.scaler: StandardScaler | None = None
        self.features = None
        self.label_head = None

    def fit(self, sources: Sequence[Domain], target_pool: np.ndarray | None = None) -> "Dann":
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        X, y = pool_sources(sources)
        self.scaler = StandardScaler().fit(X)
        Xs = torch.from_numpy(self.scaler.transform(X).astype(np.float32))
        ys = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        d = Xs.shape[1]
        h = self.hidden
        self.features = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.label_head = nn.Linear(h, 1)
        domain_head = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Linear(64, 1))
        params = [*self.features.parameters(), *self.label_head.parameters(), *domain_head.parameters()]
        optimiser = torch.optim.Adam(params, lr=self.lr)
        n_pos = float(ys.sum())
        n_neg = float(len(ys) - n_pos)
        label_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0)))
        domain_loss = nn.BCEWithLogitsLoss()
        Xt = None
        if target_pool is not None:
            Xt = torch.from_numpy(self.scaler.transform(np.asarray(target_pool, dtype=np.float64)).astype(np.float32))
        steps_per_epoch = math.ceil(len(Xs) / self.batch_size)
        total_steps = self.epochs * steps_per_epoch
        step = 0
        self.features.train()
        self.label_head.train()
        domain_head.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xs), generator=generator)
            for b in range(steps_per_epoch):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                h_s = self.features(Xs[idx])
                loss = label_loss(self.label_head(h_s), ys[idx])
                if Xt is not None:
                    progress = step / max(total_steps, 1)
                    lam = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
                    t_idx = torch.randint(0, Xt.shape[0], (len(idx),), generator=generator)
                    h_t = self.features(Xt[t_idx])
                    h_all = torch.cat([h_s, h_t])
                    domain_target = torch.cat([torch.zeros(len(h_s), 1), torch.ones(len(h_t), 1)])
                    dom_logits = domain_head(_GradientReversal.apply(h_all, lam))
                    loss = loss + domain_loss(dom_logits, domain_target)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                step += 1
        self.features.eval()
        self.label_head.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.features is None or self.label_head is None or self.scaler is None:
            raise RuntimeError("method is not fitted")
        Z = torch.from_numpy(self.scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32))
        with torch.no_grad():
            logits = self.label_head(self.features(Z)).squeeze(1)
            return torch.sigmoid(logits).numpy().astype(np.float64)
