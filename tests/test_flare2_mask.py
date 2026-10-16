"""FLARE-2 unsupported-query handling: the loss mask (``--loss-mask-unsupported``, run suffix ``_f2m``), the
``--ensure-support`` sampler option and the ``flare2-m`` registrations. With both switches off the behaviour is the
original one (covered by ``test_flare2.py``, which must keep passing unchanged).
"""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from crossids.flare import evaluate, train
from crossids.flare import evaluate_class as ec
from crossids.flare.data import AugConfig, ClassPools, Episode, EpisodeConfig, TaskConfig, TaskSampler, collate
from crossids.flare.model import NEG_LOGIT, TINY_CONFIG, Flare2Model, batch_to_torch, load_model

torch.set_num_threads(2)

D = 5
ROOT = Path(__file__).resolve().parents[1]
TINY_F2 = replace(TINY_CONFIG, slots=16)


def _tiny(seed: int = 0) -> Flare2Model:
    torch.manual_seed(seed)
    return Flare2Model(TINY_F2)


def _episode(support_per_class, query_per_class, slot_of, seed: int = 0) -> Episode:
    """Class episode with the given support and query row counts per class (class c is row label c)."""
    rng = np.random.default_rng(seed)
    s_y = np.concatenate([np.full(n, c) for c, n in enumerate(support_per_class)]).astype(np.int64)
    q_y = np.concatenate([np.full(n, c) for c, n in enumerate(query_per_class)]).astype(np.int64)
    X = rng.normal(size=(len(s_y) + len(q_y), D)).astype(np.float32)
    return Episode(x=X, label=np.concatenate([s_y, -np.ones(len(q_y), np.int64)]),
                   truth=np.concatenate([s_y, q_y]), n_support=len(s_y), n_context=0,
                   meta={"binary_task": False}, n_classes=len(support_per_class),
                   slot_of=np.asarray(slot_of, dtype=np.int64), n_slots=16)


def _forward(model, batch):
    with torch.no_grad():
        return model.forward_episode(batch["x"], batch["feat_pad"], batch["row_valid"], batch["is_query"],
                                     batch["label"])


def _reference_loss(model, episodes, batch) -> torch.Tensor:
    """Cross-entropy written out by hand: supported query rows only, softmax over the supported slots of the episode."""
    logits = _forward(model, batch)
    terms = []
    for b, e in enumerate(episodes):
        q_cols = np.flatnonzero(batch["is_query"][b].numpy() & batch["row_valid"][b].numpy())
        n_s = e.n_support
        supported = sorted(set(np.unique(e.label[:n_s]).tolist()))
        slots = [int(e.slot_of[c]) for c in supported]
        for i, col in enumerate(q_cols):
            c = int(e.truth[n_s + i])
            if c not in supported:
                continue
            z = logits[b, col, slots].float()
            terms.append(F.cross_entropy(z[None], torch.tensor([slots.index(int(e.slot_of[c]))])))
    return torch.stack(terms).mean()


def test_unsupported_query_rows_are_left_out_of_the_loss_and_softmax():
    model = _tiny(0)
    # episode 1: class 2 has no support row (its slot is active, its two query rows are unsupported)
    e1 = _episode([3, 3, 0], [4, 3, 2], slot_of=[6, 1, 9], seed=1)
    e2 = _episode([2, 2], [5, 1], slot_of=[4, 12], seed=2)  # fully supported episode in the same batch
    episodes = [e1, e2]
    batch = batch_to_torch(collate(episodes))
    logits = _forward(model, batch)
    mask_loss = train.class_loss(logits, batch, mask_unsupported=True)
    ref = _reference_loss(model, episodes, batch)
    assert torch.allclose(mask_loss, ref, atol=1e-5), (float(mask_loss), float(ref))
    # the masked loss does not depend on the (meaningless) logits of the unsupported slot
    perturbed = logits.clone()
    unsup_slot = int(e1.slot_of[2])
    perturbed[0, :, unsup_slot] += 123.0
    assert torch.allclose(train.class_loss(perturbed, batch, mask_unsupported=True), mask_loss, atol=1e-5)


def test_default_loss_is_huge_with_unsupported_rows_and_identical_without_them():
    model = _tiny(1)
    e1 = _episode([3, 3, 0], [4, 3, 2], slot_of=[6, 1, 9], seed=1)
    batch = batch_to_torch(collate([e1]))
    logits = _forward(model, batch)
    # the model scores the unsupported slot at NEG_LOGIT: that is the cause of the ~3.6e6 training losses
    unsup_slot = int(e1.slot_of[2])
    assert float(logits[0, batch["is_query"][0] & batch["row_valid"][0], unsup_slot].max()) < NEG_LOGIT / 2
    assert float(train.class_loss(logits, batch)) > 1e6  # documented: loss with the switch off
    # an episode without unsupported rows gives the same loss with and without the mask
    e2 = _episode([2, 2], [5, 1], slot_of=[4, 12], seed=2)
    b2 = batch_to_torch(collate([e2]))
    l2 = _forward(model, b2)
    assert torch.allclose(train.class_loss(l2, b2), train.class_loss(l2, b2, mask_unsupported=True), atol=1e-5)


def _pools(seed: int = 0, n_datasets: int = 2, n_classes: int = 5, rows: int = 300) -> list[ClassPools]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_datasets):
        benign = rng.normal(0, 1, (rows * 3, D)).astype(np.float32)
        attack = [rng.normal(c + i, 1.0, (rows, D)).astype(np.float32) for c in range(n_classes)]
        out.append(ClassPools(f"ds{i}", benign, [f"att{c}" for c in range(n_classes)], attack))
    return out


def test_real_size_episodes_give_an_order_one_loss_with_the_mask_and_a_huge_one_without():
    # default episode shape (k log-uniform 1..32, cap 160, 256 queries) and default A4 label noise (2%)
    cfg = EpisodeConfig(query_size=256)
    episodes = TaskSampler(_pools(), cfg, seed=5).batch(64)
    batch = batch_to_torch(collate(episodes))
    model = _tiny(3)
    logits = _forward(model, batch)
    frac = train.unsupported_fraction(batch)
    assert frac > 0, "the batch should contain unsupported query rows for this test to mean anything"
    loss_on = float(train.class_loss(logits, batch, mask_unsupported=True))
    loss_off = float(train.class_loss(logits, batch))
    assert loss_on < 20, loss_on
    assert loss_off > 1e6, loss_off  # documented: with the switch off, the same batch gives ~frac * 1e9


def test_validation_metrics_exclude_unsupported_rows_when_masked():
    cfg = EpisodeConfig(query_size=256)
    batch = batch_to_torch(collate(TaskSampler(_pools(), cfg, seed=5).batch(64)))  # has unsupported rows (asserted)
    model = _tiny(4)
    assert train.unsupported_fraction(batch) > 0
    on = train.validate_classes(model, [batch], mask_unsupported=True)
    off = train.validate_classes(model, [batch])
    assert on["frac_unsupported_queries"] == off["frac_unsupported_queries"] > 0
    assert on["val_bce"] < 20 and off["val_bce"] > 1e6
    assert np.isfinite(on["val_acc"])


def test_ensure_support_gives_every_class_a_support_row_and_default_does_not():
    aug = AugConfig(sign_flip=False, warp=False, subset=False, composite=False, label_noise=True, noise_rate=1.0)
    cfg = EpisodeConfig(k_min=1, k_max=32, query_size=24, aug=aug)
    guarded = TaskSampler(_pools(2), cfg, seed=7, tasks=TaskConfig(ensure_support=True)).batch(40)
    plain = TaskSampler(_pools(2), cfg, seed=7, tasks=TaskConfig()).batch(40)
    for e in guarded:  # every class of the episode has a support row whose label is its true class
        s_true, s_lab = e.truth[: e.n_support], e.label[: e.n_support]
        correct = set(np.unique(s_true[s_lab == s_true]).tolist())
        assert set(np.unique(e.truth[e.n_support + e.n_context:]).tolist()) <= correct
    unsupported = [float(np.mean([int(t) not in set(e.label[: e.n_support].tolist()) for t in e.truth[e.n_support:]]))
                   for e in plain]
    assert max(unsupported) > 0  # with every support label noisy, some classes lose all their support rows


def test_ensure_support_is_off_by_default_and_leaves_episode_draws_unchanged():
    cfg = EpisodeConfig(query_size=32)
    pools = _pools(3)
    default = TaskSampler(pools, cfg, seed=11).batch(5)
    explicit = TaskSampler(pools, cfg, seed=11, tasks=TaskConfig(ensure_support=False)).batch(5)
    for a, b in zip(default, explicit):
        assert np.array_equal(a.label, b.label) and np.array_equal(a.x, b.x)
    assert TaskConfig().ensure_support is False and TaskConfig().loss_mask_unsupported is False


def test_fit_with_the_mask_logs_the_fraction_and_the_checkpoint_loads_without_the_flag(tmp_path):
    cfg = EpisodeConfig(query_size=64)
    src = TaskSampler(_pools(4), cfg, seed=0, tasks=TaskConfig(loss_mask_unsupported=True))
    val = TaskSampler(_pools(5), cfg, seed=1, tasks=TaskConfig(loss_mask_unsupported=True)).batch(4)
    model = _tiny(5)
    train.fit(model, src, val, steps=2, out=tmp_path, batch_size=2, device="cpu", val_every=2, log_every=1,
              warmup=1, meta={"slot_perm": True}, mask_unsupported=True)
    recs = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    train_rec = [r for r in recs if "loss" in r][-1]
    val_rec = [r for r in recs if "val_bce" in r][-1]
    assert 0.0 <= train_rec["frac_unsupported_queries"] <= 1.0
    assert 0.0 <= val_rec["frac_unsupported_queries"] <= 1.0
    assert val_rec["val_bce"] < 20
    assert load_model(tmp_path / "best.pt").kind == "flare2"  # no flag needed to load or evaluate


def test_run_name_and_flare2_m_registrations():
    base = dict(target="iot23", seed=0, model="flare2", val_mode=None, tasks="mixed", no_slot_perm=False,
                context_max=0)
    assert train.run_name(Namespace(**base)) == "iot23_s0_f2"  # unchanged default
    assert train.run_name(Namespace(**base, loss_mask_unsupported=True)) == "iot23_s0_f2m"
    assert train.run_name(Namespace(**{**base, "tasks": "binary"}, loss_mask_unsupported=True)) == "iot23_s0_f2m_binary"
    assert train.run_name(Namespace(**base, ensure_support=True)) == "iot23_s0_f2_ens"
    assert evaluate.FLARE_VARIANTS["flare2-m"] == ("_f2m", None)
    assert ec.CHECKPOINT_SUFFIX["flare2-m"] == "_f2m"
    assert "flare2-m" in ec.METHODS and "flare2-m" in ec.NEURAL


BASE_BINARY = ("catboost-target rf lr knn catboost-ft catboost-fulllabel catboost-lodo catboost-pairwise-mean "
               "catboost-pairwise-best")


@pytest.mark.parametrize("variant, expected", [
    ("f2m", ("iot23_s0_f2m", "runs/flare/eval_iot23_f2m_s0.csv", "runs/flare/class_iot23_f2m_s0.csv", "flare2-m",
             "catboost rf lr knn flare2-m", "--model flare2 --loss-mask-unsupported")),
    ("f2", ("iot23_s0_f2", "runs/flare/eval_iot23_all_s0.csv", "runs/flare/class_iot23_s0.csv",
            f"{BASE_BINARY} flare2", "catboost rf lr knn flare2", "--model flare2")),
])
def test_colab_variant_tables_name_the_fixed_variant(variant, expected):
    script = (
        f'. "{ROOT}/scripts/flare_colab_common.sh"; output_names iot23 0 {variant}; '
        'printf "%s|%s|%s|%s|%s|%s" "$RUN" "$EVAL_CSV" "$CLASS_CSV" "$EVAL_METHODS" "$CLASS_METHODS" "$TRAIN_ARGS"'
    )
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout
    assert tuple(out.split("|")) == expected
