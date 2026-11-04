"""Meta-train SchemaFlow on all datasets except the target (leave-one-dataset-out)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from .episodes import Domain, collate, draw_episode
from .model import SchemaFlow
from .native import load_native
from .text import embed_texts

KEYS = ["unswnb15", "toniot", "insdn", "cicids2017", "ctu13"]


def load_domains(keys, data_dir="data", notext=False, seed=0):
    doms = {}
    rng = np.random.default_rng(1234)
    for k in keys:
        X, y, names, texts = load_native(k, data_dir, seed)
        emb = embed_texts(texts)
        if notext:  # control: features are anonymous, one random vector per (dataset, column)
            emb = rng.standard_normal(emb.shape).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        doms[k] = Domain(k, X, y, names, emb)
    return doms


def train(doms, sources, steps=2500, batch=16, lr=1e-3, seed=0, log_every=250):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    model = SchemaFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0, run = time.time(), []
    for step in range(1, steps + 1):
        eps = [draw_episode(rng, doms[sources[int(rng.integers(len(sources)))]]) for _ in range(batch)]
        vs, ys, sm, vq, e, fm, yq = collate(eps)
        lp = model(vs, ys, sm, vq, e, fm)
        loss = Fn.nll_loss(lp.reshape(-1, 2), yq.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(loss.item())
        if step % log_every == 0:
            print(f"step {step} loss {np.mean(run[-log_every:]):.4f} {time.time()-t0:.0f}s", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--notext", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sf")
    a = ap.parse_args()
    doms = load_domains(KEYS, notext=a.notext)
    src = [k for k in KEYS if k != a.target]
    m = train(doms, src, a.steps, seed=a.seed)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), f"{a.out}/{a.target}{'_notext' if a.notext else ''}_s{a.seed}.pt")
