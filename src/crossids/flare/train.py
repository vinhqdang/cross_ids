"""Meta-training of FLARE (or ProtoNet) on the LODO sources of one target (DESIGN.md §14).

Example::

    python -m crossids.flare.train --target usbids1 --seed 0 --steps 60000 --out runs/flare/usbids1_s0

Validation episodes come from the held-out 30% of the meta-training datasets only; the target is never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .data import (
    AugConfig,
    EpisodeConfig,
    EpisodeSampler,
    SyntheticSampler,
    collate,
    load_split,
    meta_train_keys,
    pools_from_split,
    target_keys,
)
from ..schema import CORE_FEATURES
from .model import NAMED_CONFIGS, EpisodeModel, batch_to_torch, build_model, save_checkpoint

log = logging.getLogger("crossids.flare")

ABLATIONS = (None, "noA1", "noA3", "rowmlp", "synthetic")
VAL_SEED_OFFSET = 10_000
DEFAULT_STEPS = 60_000


def episode_settings(ablation: str | None, base: EpisodeConfig | None = None) -> tuple[EpisodeConfig, FlareConfig]:
    """Episode config and architecture overrides of an ablation (None = the full method)."""
    ecfg = base or EpisodeConfig()
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}")
    aug: AugConfig = ecfg.aug
    arch_mixing = "attention"
    if ablation == "noA1":
        aug = replace(aug, sign_flip=False, warp=False)
    elif ablation == "noA3":
        aug = replace(aug, composite=False)
    elif ablation == "rowmlp":
        arch_mixing = "rowmlp"
    return replace(ecfg, aug=aug), arch_mixing


def make_streams(target: str, seed: int, ablation: str | None, data_dir: str | Path, ecfg: EpisodeConfig,
                 n_val: int, max_features: int = 13):
    """Training sampler and a fixed list of validation episodes for one target and seed."""
    if ablation == "synthetic":
        train = SyntheticSampler(ecfg, seed, max_features=max_features)
        val = SyntheticSampler(ecfg, VAL_SEED_OFFSET + seed, max_features=max_features).batch(n_val)
        return train, val, []
    keys = meta_train_keys(target)
    if not keys:
        raise ValueError(f"no meta-training datasets for target {target}")
    splits = [load_split(k, seed, data_dir, features=CORE_FEATURES) for k in keys]
    train = EpisodeSampler([pools_from_split(s, "train") for s in splits], ecfg, seed)
    val = EpisodeSampler([pools_from_split(s, "test") for s in splits], ecfg, VAL_SEED_OFFSET + seed).batch(n_val)
    return train, val, keys


def _lr_factor(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def validate(model: EpisodeModel, val_batches: list[dict], device) -> dict[str, float]:
    """Mean query BCE and mean per-episode ROC-AUC over validation episodes."""
    model.eval()
    losses, aucs = [], []
    for t in val_batches:
        logits = model.forward_episode(t["x"], t["feat_pad"], t["row_valid"], t["is_query"], t["label"]).float()
        for b in range(logits.shape[0]):
            m = (t["is_query"][b] & t["row_valid"][b]).cpu().numpy()
            y = t["truth"][b].cpu().numpy()[m]
            s = logits[b].cpu().numpy()[m]
            losses.append(float(F.binary_cross_entropy_with_logits(torch.from_numpy(s), torch.from_numpy(y))))
            if 0 < y.sum() < len(y):
                aucs.append(float(roc_auc_score(y, s)))
    model.train()
    return {"val_bce": float(np.mean(losses)), "val_auc": float(np.mean(aucs)) if aucs else float("nan")}


def fit(model: EpisodeModel, sampler, val_episodes: list, *, steps: int, out: Path | None = None,
        lr: float = 3e-4, weight_decay: float = 0.01, warmup: int = 2000, clip: float = 1.0,
        batch_size: int = 8, device="cpu", eval_every: int | None = None, log_every: int = 100,
        val_batch_size: int = 8) -> dict:
    """Train with AdamW and a warm-up + cosine schedule. Keeps the checkpoint with the lowest validation BCE.

    Returns a summary dict. Writes ``best.pt``, ``last.pt``, ``log.jsonl`` and ``summary.json`` when ``out`` is set.
    """
    model.to(device).train()
    warmup = min(warmup, max(1, steps // 10))
    eval_every = eval_every or min(1000, max(1, steps // 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, steps, warmup))

    use_cuda = torch.device(device).type == "cuda"
    amp_dtype = None
    scaler = None
    if use_cuda:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:  # T4 has no bf16; fp16 with loss scaling is the equivalent mixed-precision mode
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")

    val_batches = []
    for i in range(0, len(val_episodes), val_batch_size):
        val_batches.append(batch_to_torch(collate(val_episodes[i:i + val_batch_size]), device))

    log_fh = None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        log_fh = open(out / "log.jsonl", "w", encoding="utf-8")

    def write(rec: dict) -> None:
        if log_fh is not None:
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()

    best = {"val_bce": float("inf"), "step": -1}
    t_start = time.perf_counter()
    loss_acc, n_acc = 0.0, 0
    params = list(model.parameters())
    for step in range(steps):
        batch = batch_to_torch(collate(sampler.batch(batch_size)), device)
        amp = torch.autocast(device_type=torch.device(device).type, dtype=amp_dtype, enabled=amp_dtype is not None)
        with amp:
            logits = model.forward_episode(batch["x"], batch["feat_pad"], batch["row_valid"], batch["is_query"],
                                           batch["label"])
        mask = batch["is_query"] & batch["row_valid"]
        loss = F.binary_cross_entropy_with_logits(logits.float()[mask], batch["truth"][mask])
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
        sched.step()
        loss_acc += float(loss.detach())
        n_acc += 1
        if (step + 1) % log_every == 0 or step == 0:
            rec = {"step": step + 1, "loss": loss_acc / n_acc, "lr": sched.get_last_lr()[0],
                   "elapsed": time.perf_counter() - t_start}
            write(rec)
            log.info("step %d loss %.4f lr %.2e (%.1fs)", rec["step"], rec["loss"], rec["lr"], rec["elapsed"])
            loss_acc, n_acc = 0.0, 0
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            metrics = validate(model, val_batches, device)
            is_best = metrics["val_bce"] < best["val_bce"]
            if is_best:
                best = {"val_bce": metrics["val_bce"], "step": step + 1, "val_auc": metrics["val_auc"]}
                if out is not None:
                    save_checkpoint(out / "best.pt", model, step=step + 1, **metrics)
            if out is not None:
                save_checkpoint(out / "last.pt", model, step=step + 1, **metrics)
            write({"step": step + 1, **metrics, "best": is_best})
            log.info("eval step %d val_bce %.4f val_auc %.4f%s", step + 1, metrics["val_bce"], metrics["val_auc"],
                     " (best)" if is_best else "")
    elapsed = time.perf_counter() - t_start
    summary = {
        "kind": model.kind,
        "steps": steps,
        "batch_size": batch_size,
        "device": str(device),
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, steps),
        "amp": str(amp_dtype) if amp_dtype is not None else "fp32",
        "best_step": best["step"],
        "best_val_bce": best["val_bce"],
        "best_val_auc": best.get("val_auc"),
        "config": model.cfg.__dict__ if hasattr(model, "cfg") else {},
    }
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        log_fh.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.train", description="Meta-train FLARE on the LODO sources")
    p.add_argument("--target", required=True, choices=target_keys())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--out", default=None, help="run directory (default runs/flare/<target>_s<seed>)")
    p.add_argument("--ablation", choices=[a for a in ABLATIONS if a], default=None)
    p.add_argument("--model", choices=["flare", "protonet"], default="flare")
    p.add_argument("--config", choices=sorted(NAMED_CONFIGS), default="default")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-episodes", type=int, default=64)
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    out = Path(args.out or f"runs/flare/{args.target}_s{args.seed}")
    ecfg, mixing = episode_settings(args.ablation)
    cfg = NAMED_CONFIGS[args.config]
    cfg = replace(cfg, feature_mixing=mixing, max_features=max(cfg.max_features, len(CORE_FEATURES)))
    if args.model == "protonet":
        cfg = replace(cfg, feature_mixing="attention")
    train, val, keys = make_streams(args.target, args.seed, args.ablation, args.data_dir, ecfg, args.val_episodes,
                                    max_features=cfg.max_features)
    log.info("target %s seed %d ablation %s model %s config %s on %s; meta-train: %s",
             args.target, args.seed, args.ablation, args.model, args.config, device, ", ".join(keys) or "synthetic")
    torch.manual_seed(args.seed)
    model = build_model(args.model, cfg)
    meta = {"target": args.target, "seed": args.seed, "ablation": args.ablation, "meta_train": keys}
    summary = fit(model, train, val, steps=args.steps, out=out, lr=args.lr, weight_decay=args.weight_decay,
                  warmup=args.warmup, clip=args.clip, batch_size=args.batch_size, device=device,
                  eval_every=args.eval_every, log_every=args.log_every)
    summary.update(meta)
    (out / "run.json").write_text(json.dumps({**meta, "args": vars(args)}, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = run(parse_args(argv))
    log.info("done: best step %s val_bce %.4f, %.3f s/step", summary["best_step"], summary["best_val_bce"],
             summary["seconds_per_step"])


if __name__ == "__main__":
    main()
