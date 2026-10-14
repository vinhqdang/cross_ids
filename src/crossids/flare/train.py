"""Meta-training of FLARE (or ProtoNet) on the LODO sources of one target (DESIGN.md §14).

Example::

    python -m crossids.flare.train --target usbids1 --seed 0 --out runs/flare/usbids1_s0

Validation episodes come from the held-out 30% of the meta-training datasets only; the target is never touched.
Running again with the same ``--out`` resumes from ``last.pt`` (written every 1000 steps and at the end).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..schema import CORE_FEATURES
from .data import (
    AugConfig,
    EpisodeConfig,
    EpisodeSampler,
    EpisodeStream,
    SyntheticSampler,
    batch_arrays,
    collate,
    load_split,
    meta_train_keys,
    pools_from_split,
    target_keys,
)
from .model import NAMED_CONFIGS, EpisodeModel, batch_to_torch, build_model, save_checkpoint

log = logging.getLogger("crossids.flare")

ABLATIONS = (None, "noA1", "noA3", "rowmlp", "synthetic")
VAL_SEED_OFFSET = 10_000
DEFAULT_STEPS = 20_000
DEFAULT_WARMUP = 1000
DEFAULT_QUERY = 256
DEFAULT_VAL_EVERY = 500
DEFAULT_LOG_EVERY = 50
CKPT_EVERY = 1000
BENCH_WARMUP = 5


def episode_settings(ablation: str | None, query_size: int = DEFAULT_QUERY, warp: bool = False,
                     context_max: int = 0):
    """Episode config and architecture mixing of an ablation (None = the full method).

    A1 is the sign flip. ``warp`` switches on the monotone warp (a no-op under rank normalisation, kept for tests).
    ``context_max`` > 0 adds U ~ U{0..context_max} unlabelled context rows per episode (§20.1, FLARE-U).
    """
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}")
    aug = AugConfig(warp=warp)
    mixing = "attention"
    if ablation == "noA1":
        aug = replace(aug, sign_flip=False, warp=False)
    elif ablation == "noA3":
        aug = replace(aug, composite=False)
    elif ablation == "rowmlp":
        mixing = "rowmlp"
    return EpisodeConfig(query_size=query_size, aug=aug, context_max=context_max), mixing


def build_training_data(target: str, seed: int, ablation: str | None, data_dir: str | Path, ecfg: EpisodeConfig,
                        n_val: int, max_features: int = 13):
    """Returns (pools or None, synthetic flag, validation episodes, meta-train keys)."""
    if ablation == "synthetic":
        val = SyntheticSampler(ecfg, VAL_SEED_OFFSET + seed, max_features=max_features).batch(n_val)
        return None, True, val, []
    keys = meta_train_keys(target)
    if not keys:
        raise ValueError(f"no meta-training datasets for target {target}")
    splits = [load_split(k, seed, data_dir, features=CORE_FEATURES) for k in keys]
    pools = [pools_from_split(s, "train") for s in splits]
    val = EpisodeSampler([pools_from_split(s, "test") for s in splits], ecfg, VAL_SEED_OFFSET + seed).batch(n_val)
    return pools, False, val, keys


def _lr_factor(step: int, steps: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def select_precision(device: torch.device) -> tuple[torch.dtype | None, bool, str]:
    """(autocast dtype, use GradScaler, description). bf16 only on CUDA with native support (capability >= 8)."""
    if device.type != "cuda":
        return None, False, "fp32 (cpu)"
    major, minor = torch.cuda.get_device_capability(device)
    if torch.cuda.is_bf16_supported() and major >= 8:
        return torch.bfloat16, False, f"bf16 autocast (capability {major}.{minor})"
    return torch.float16, True, f"fp16 autocast + GradScaler (capability {major}.{minor}, no native bf16)"


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: torch.as_tensor(v).to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def validate(model: EpisodeModel, val_batches: list[dict]) -> dict[str, float]:
    """Mean query BCE and mean per-episode ROC-AUC over the validation episodes."""
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


def read_resume_step(out: Path | None) -> int:
    """Step stored in ``out/last.pt`` (0 when there is no run to resume)."""
    if out is None or not (Path(out) / "last.pt").exists():
        return 0
    return int(torch.load(Path(out) / "last.pt", map_location="cpu", weights_only=False).get("step", 0))


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}


def _set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _save_last(path: Path, model, opt, sched, scaler, step: int, best: dict, meta: dict) -> None:
    tmp = path.with_suffix(".tmp")
    save_checkpoint(tmp, model, step=step, context_max=int(meta.get("context_max", 0)), train_state={
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best": best,
        "rng": _rng_state(),
        "meta": meta,
    })
    os.replace(tmp, path)


def fit(model: EpisodeModel, source, val_episodes: list, *, steps: int, out: Path | None = None,
        lr: float = 3e-4, weight_decay: float = 0.01, warmup: int = DEFAULT_WARMUP, clip: float = 1.0,
        batch_size: int = 8, device="cpu", val_every: int = DEFAULT_VAL_EVERY, log_every: int = DEFAULT_LOG_EVERY,
        ckpt_every: int = CKPT_EVERY, compile: bool = False, bench: int = 0, val_batch_size: int = 8,
        meta: dict | None = None) -> dict:
    """Train with AdamW and a warm-up + cosine schedule; keep the checkpoint with the lowest validation BCE.

    ``source`` is either a sampler with a ``batch(n)`` method (episodes built in-process) or an iterable of
    collated batches (a DataLoader over :class:`EpisodeStream`). If ``out`` holds a ``last.pt`` the run resumes
    from it. ``bench=N`` runs N timed steps after a 5-step warm-up, without validation or checkpoints.
    """
    device = torch.device(device)
    meta = meta or {}
    total = bench + BENCH_WARMUP if bench else steps
    warmup = min(warmup, max(1, total // 10))
    model.to(device).train()
    ctx_max = int(meta.get("context_max", 0))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr_factor(s, total, warmup))
    amp_dtype, use_scaler, precision = select_precision(device)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    log.info("precision: %s", precision)
    forward = torch.compile(model.forward_episode, dynamic=True) if compile else model.forward_episode

    step = 0
    best = {"val_bce": float("inf"), "step": -1, "val_auc": None}
    if out is not None and not bench:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        last = out / "last.pt"
        if last.exists():
            state = torch.load(last, map_location=device, weights_only=False)
            model.load_state_dict(state["state_dict"])
            ts = state["train_state"]
            opt.load_state_dict(ts["optimizer"])
            sched.load_state_dict(ts["scheduler"])
            if scaler is not None and ts.get("scaler") is not None:
                scaler.load_state_dict(ts["scaler"])
            _set_rng_state(ts["rng"])
            best = ts["best"]
            step = int(state["step"])
            log.info("resuming %s from step %d", out, step)

    val_batches = [batch_to_torch(collate(val_episodes[i:i + val_batch_size]), device)
                   for i in range(0, len(val_episodes), val_batch_size)]
    it = None if hasattr(source, "batch") else iter(source)
    log_fh = None
    if out is not None and not bench:
        log_fh = open(out / "log.jsonl", "a", encoding="utf-8")

    def write(rec: dict) -> None:
        if log_fh is not None:
            log_fh.write(json.dumps(rec) + "\n")
            log_fh.flush()

    t_start = time.perf_counter()
    t_bench = None
    loss_acc, n_acc = 0.0, 0
    params = list(model.parameters())
    start_step = step
    while step < total:
        if it is None:
            batch = _to_device(batch_arrays(source.batch(batch_size)), device)
        else:
            batch = _to_device(next(it), device)
        amp = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)
        with amp:
            logits = forward(batch["x"], batch["feat_pad"], batch["row_valid"], batch["is_query"], batch["label"])
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
        step += 1
        if bench:
            if step == BENCH_WARMUP:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_bench = time.perf_counter()
            continue
        loss_acc += float(loss.detach())
        n_acc += 1
        if step % log_every == 0 or step == start_step + 1:
            rec = {"step": step, "loss": loss_acc / n_acc, "lr": sched.get_last_lr()[0],
                   "elapsed": time.perf_counter() - t_start}
            write(rec)
            log.info("step %d loss %.4f lr %.2e (%.1fs)", step, rec["loss"], rec["lr"], rec["elapsed"])
            loss_acc, n_acc = 0.0, 0
        if step % val_every == 0 or step == total:
            metrics = validate(model, val_batches)
            is_best = metrics["val_bce"] < best["val_bce"]
            if is_best:
                best = {"val_bce": metrics["val_bce"], "step": step, "val_auc": metrics["val_auc"]}
                if out is not None:
                    save_checkpoint(out / "best.tmp", model, step=step, context_max=ctx_max, **metrics)
                    os.replace(out / "best.tmp", out / "best.pt")
            write({"step": step, **metrics, "best": is_best})
            log.info("eval step %d val_bce %.4f val_auc %.4f%s", step, metrics["val_bce"], metrics["val_auc"],
                     " (best)" if is_best else "")
            if out is not None and (step % ckpt_every == 0 or step == total):
                _save_last(out / "last.pt", model, opt, sched, scaler, step, best, meta)
    elapsed = time.perf_counter() - t_start
    bench_sps = None
    if bench:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        bench_sps = (time.perf_counter() - t_bench) / bench
        log.info("bench: %d steps after %d warm-up, mean %.3f s/step (batch %d, %s)", bench, BENCH_WARMUP,
                 bench_sps, batch_size, precision)
    summary = {
        "kind": model.kind,
        "steps": total,
        "resumed_from": start_step,
        "batch_size": batch_size,
        "device": str(device),
        "precision": precision,
        "compile": compile,
        "seconds": elapsed,
        "seconds_per_step": elapsed / max(1, total - start_step),
        "bench_seconds_per_step": bench_sps,
        "best_step": best["step"],
        "best_val_bce": best["val_bce"],
        "best_val_auc": best.get("val_auc"),
        "config": model.cfg.__dict__ if hasattr(model, "cfg") else {},
        **meta,
    }
    if out is not None and not bench:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if log_fh is not None:
        log_fh.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m crossids.flare.train", description="Meta-train FLARE on the LODO sources")
    p.add_argument("--target", required=True, choices=target_keys())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--out", default=None, help="run directory (default runs/flare/<target>_s<seed>); resumes if it has last.pt")
    p.add_argument("--ablation", choices=[a for a in ABLATIONS if a], default=None)
    p.add_argument("--model", choices=["flare", "protonet"], default="flare")
    p.add_argument("--config", choices=sorted(NAMED_CONFIGS), default="default")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda when available)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--query-size", type=int, default=DEFAULT_QUERY)
    p.add_argument("--warp", action="store_true", help="enable the A1 monotone warp (no-op under rank normalisation)")
    p.add_argument("--context-max", type=int, default=0,
                   help="FLARE-U (§20.1): up to this many unlabelled context rows per episode, U ~ U{0..N} (0 = off)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    p.add_argument("--val-episodes", type=int, default=64)
    p.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers generating episodes")
    p.add_argument("--compile", action="store_true", help="torch.compile the model for training (off by default)")
    p.add_argument("--bench", type=int, default=0, metavar="N", help="time N training steps (after 5 warm-up) and exit")
    p.add_argument("--threads", type=int, default=2)
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    out = None if args.bench else Path(args.out or f"runs/flare/{args.target}_s{args.seed}")
    resume_step = read_resume_step(out)
    ecfg, mixing = episode_settings(args.ablation, query_size=args.query_size, warp=args.warp,
                                    context_max=args.context_max)
    cfg = NAMED_CONFIGS[args.config]
    cfg = replace(cfg, feature_mixing=mixing, max_features=max(cfg.max_features, len(CORE_FEATURES)))
    pools, synthetic, val, keys = build_training_data(args.target, args.seed, args.ablation, args.data_dir, ecfg,
                                                      args.val_episodes, max_features=cfg.max_features)
    stream = EpisodeStream(pools, ecfg, args.seed, batch_size=args.batch_size, stream=resume_step,
                           synthetic=synthetic, max_features=cfg.max_features)
    source = DataLoader(stream, batch_size=None, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    log.info("target %s seed %d ablation %s model %s config %s on %s; meta-train: %s; query %d; workers %d",
             args.target, args.seed, args.ablation, args.model, args.config, device, ", ".join(keys) or "synthetic",
             args.query_size, args.num_workers)
    torch.manual_seed(args.seed)
    model = build_model(args.model, cfg)
    meta = {"target": args.target, "seed": args.seed, "ablation": args.ablation, "meta_train": keys,
            "model_kind": args.model, "query_size": args.query_size, "num_workers": args.num_workers,
            "stream": resume_step, "context_max": args.context_max}
    summary = fit(model, source, val, steps=args.steps, out=out, lr=args.lr, weight_decay=args.weight_decay,
                  warmup=args.warmup, clip=args.clip, batch_size=args.batch_size, device=device,
                  val_every=args.val_every, log_every=args.log_every, compile=args.compile, bench=args.bench,
                  meta=meta)
    if out is not None:
        (out / "run.json").write_text(json.dumps({**meta, "args": vars(args)}, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    summary = run(parse_args(argv))
    if summary["bench_seconds_per_step"] is not None:
        print(f"BENCH mean_seconds_per_step={summary['bench_seconds_per_step']:.4f} steps={summary['steps'] - BENCH_WARMUP}")
    else:
        log.info("done: best step %s val_bce %.4f, %.3f s/step", summary["best_step"], summary["best_val_bce"],
                 summary["seconds_per_step"])


if __name__ == "__main__":
    main()
