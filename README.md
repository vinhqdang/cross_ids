# cross_ids

Cross-dataset flow-based intrusion detection. Models are trained on labelled
source networks and evaluated on a target network whose labels are never used
for training, model selection or threshold tuning. The task is binary detection
(benign = 0, attack = 1).

The benchmark covers six producers and three flow-extractor families, reports
every result by transfer type, and includes **RIFT** (Rank-Invariant Flow
Transfer), a method that treats covariate shift, spurious cues and label shift
separately, and **RIFT-ST**, an uncertainty-gated self-training extension. The
full specification is in `DESIGN.md`.

## Datasets

| Key | Producer | Extractor (family) | Source files | Labels |
|---|---|---|---|---|
| `cicids2017` | UNB CIC (Canada) | CICFlowMeter (`cic`) | HF `bvsam/cic-ids-2017`, `machine_learning/*.parquet` | `Label != BENIGN` |
| `cicids2018` | UNB CIC + CSE (Canada) | CICFlowMeter (`cic`) | HF `c01dsnap/CIC-IDS2018`, 9 daily CSVs (the 4 GB Thursday-20-02 file is excluded) | `Label != Benign`; repeated header rows dropped |
| `usbids1` | Univ. of Sannio (Italy) | CICFlowMeter, patched (`sannio`) | `idsdata.ding.unisannio.it` `USB-IDS-1-{TRAINING,VALIDATION,TEST}.csv.gz` | `Label != BENIGN` |
| `insdn` | Univ. College Dublin (Ireland) | CICFlowMeter (`ucd`) | `aseados.ucd.ie` `InSDN_DatasetCSV.zip` (Normal / OVS / metasploitable-2) | `Label != Normal` |
| `unswnb15` | UNSW Canberra (Australia) | Argus + Bro (`unsw`) | HF `Mouwiya/UNSW-NB15`, `UNSW_NB15_training-set.csv` | `label` |
| `toniot` | UNSW Canberra (Australia) | Zeek conn.log (`unsw`) | HF `codymlewis/TON_IoT_network`, `train_test_network.csv` | `label` |
| `iot23` | Stratosphere Lab, CTU (Czechia) | Zeek conn.log (`ctu`) | HF `19kmunz/iot-23-preprocessed-allcolumns`, 3 parquet shards | `label` not starting with `Benign` |
| `ctu13` | Stratosphere Lab, CTU (Czechia) | Argus bi-flows (`ctu`) | `mcfp.felk.cvut.cz`, 13 scenarios (`CTU-Malware-Capture-Botnet-42..54`) | `Botnet` → 1, `Normal` → 0, `Background` dropped |

Rows with a missing or empty label are dropped and never mapped to attack; the
prepare log and metadata count them as unlabelled. Rows with missing feature
values are dropped and counted per label. Zeek `-` and
`(empty)` values are read as 0 for byte and duration fields.

`attack_type` keeps the raw label string for per-attack recall. For CTU-13 it is
normalised to `normal` or `botnet-s{N}`, where `N` is the scenario number from
the file name (each scenario is one malware family). Transfer types:
`same-family` (for example `cicids2017` → `cicids2018`), `same-extractor`
(different family, same extractor) and `cross-extractor`.

## Common Flow Schema (CFS)

Features are mapped to the same physical meaning and units before learning.
Four tiers are defined in `crossids.schema` (`DATASET_TIERS` lists the tiers
each dataset supports):

| Tier | Datasets | Features |
|---|---|---|
| `minimal` | all 8 | `duration`, `total_pkts`, `fwd_bytes`, `bwd_bytes` and derived `total_bytes`, `bytes_per_s`, `pkts_per_s`, `bytes_per_pkt`, `fwd_bwd_byte_ratio` (9 features). CTU-13 supports only this tier. |
| `core` | 7 (all except CTU-13) | minimal base features plus per-direction packet and byte features (13 features) |
| `extended` | 5 (CICFlowMeter datasets, InSDN, UNSW-NB15) | core plus `fwd_iat_mean`, `bwd_iat_mean` (15 features) |
| `cicflow` | 4 (CICFlowMeter datasets only) | extended plus `fwd_init_win`, `bwd_init_win` (17 features). Window sizes are not comparable to Argus window fields, so they are used only within CICFlowMeter. |

Ports and protocol are excluded. Durations are converted to seconds, negative
sentinel values (-1) are clipped to 0, and every feature receives
`log1p(max(x, 0))`. Each source has its own spelling table (for example
`Total Fwd Packets` / `Tot Fwd Pkts` / `Total Fwd Packet`), matched
case-insensitively. Remaining semantic gaps between extractors are part of the
cross-extractor shift and are not corrected.

### Sampling, distinct rows and the prior

Harmonisation reads raw files in chunks with only the needed columns. Each
cleaned row gets a seeded 64-bit hash of its rounded base features combined
with its label, mapped to `u` in `[0, 1)`. Per label, a bottom-k sketch keeps the
`max_rows` rows with the smallest `u`. Duplicate rows share a hash and occupy one
slot, so the sketch is a uniform sample of **distinct** rows.

- The KMV estimate `(k - 1) / u_(k)` gives the number of distinct rows per label
  over the whole file (an exact count when the file has fewer than `k` distinct rows).
- The **natural prior** is defined on distinct rows: `pi = D_attack / (D_attack + D_benign)`.
  Repeated attack flows therefore do not inflate it. The raw prior (all rows) is
  recorded next to it in the metadata.
- The cached table has `n = min(max_rows, D_benign + D_attack)` rows with
  `round(n * pi)` attack rows. If a class is short, `n` shrinks so the prior is
  kept; the other class is never back-filled.
- Rows with the same features and different labels are kept.

`data/processed/{key}.json` records, per dataset: raw, valid and NaN-dropped
counts per label, the distinct estimate per label, `prior_raw`,
`prior_distinct`, the cached counts and `prior_cached`.

## RIFT

`RIFT = DN(rank, benign anchor) + IFS + HGB + calibration + LSC`

1. **Domain-wise normalisation (DN).** Each domain is normalised by its own
   label-free statistics (Gaussian rank transform by default; `robust` and
   `standard` are alternatives; `global` is one scaler fitted on the training
   sources). With the benign anchor, a source's statistics are fitted on its
   benign rows only. On the target, the anchor is the **prior-aware
   pseudo-benign set**: the lowest-scoring `gamma * (1 - pi_t)` fraction of the
   pool (`gamma = 0.8`, at least 5% of the pool), where `pi_t` comes from EM. The
   anchor is refit and rescored for two iterations. Labels are not used.
   Inductive mode normalises the target with pooled source statistics.
2. **Invariance-driven feature selection (IFS).** With two or more sources, a
   feature is kept only if its univariate AUC strength has the same sign in
   every source and magnitude at least `tau = 0.05`. If fewer than
   `min_features` pass, the top-up draws only from sign-consistent features.
   With one source and a target pool, the 25% of features with the largest KS
   distance are dropped (never below six).
3. **Classifier and calibration.** `HistGradientBoostingClassifier` with balanced
   class weights, isotonic calibration on a held-out 20% of the sources.
4. **Label-shift correction (LSC, transductive only).** The Saerens–Latinne–
   Decaestecker EM estimates the target attack fraction from unlabelled pool
   features and re-weights the posteriors. The decision is `p >= 0.5` after the
   correction. Ranking metrics (ROC-AUC, PR-AUC) use `p + 1e-6 * raw_score`, so
   isotonic plateaus do not create ties.

**RIFT-ST** (`rift-st`, transductive only) adds uncertainty-gated self-training.
Target rows whose adjusted posterior is at least 0.9 (or at most 0.1) are
pseudo-labelled, class proportions follow the estimated `pi_t`, at most 20% of
the pool is used per round, and the HGB is retrained on sources plus these rows
for two rounds. Target labels are never used.

Ablations: `rift-noDN` (global scaler), `rift-noAnchor`, `rift-noIFS`,
`rift-noLSC`, `rift-robust`, `rift-standard`, `rift-poolInit` (first prior estimate
from a normaliser fitted on the whole pool instead of the pooled source normaliser).

### Known limitation: label-shift estimate under per-domain normalisation

EM assumes pure label shift: the class-conditional distributions are the same
in source and target. Per-domain normalisation fitted on the target pool changes
the feature marginals whenever the attack fraction changes, so it breaks that
assumption. The default RIFT starts from the pooled source benign-anchored
normaliser (`prior_init="source"`) and then replaces the target map by an
anchor-matched quantile transport: the target anchor set (lowest-scoring
`gamma * (1 - pi_t)` rows) is mapped onto the source benign rows truncated at the
same score quantile. Measured on synthetic data with identical class-conditionals
(6000 pool rows, three seeds, true target attack fraction about 0.20 and 0.80):

| variant | true 0.20 | true 0.80 |
| --- | --- | --- |
| `rift` (default), final `pi_t` | 0.23 to 0.32 | 0.86 to 0.95 |
| `rift` first estimate (`pi_init`, source normaliser) | 0.20 to 0.22 | 0.78 to 0.81 |
| `rift-poolInit` (`prior_init="pool"`) | 0.19 to 0.26 | 0.08 to 0.18 |
| `rift-noDN` (global scaler) | 0.20 to 0.22 | 0.78 to 0.81 |

The transport halves the upward drift of the earlier fitted-normaliser version
but does not meet the 0.07 tolerance. Diagnostics: a transport fitted on the true
benign pool rows truncated at the same score quantile estimates 0.21 and 0.80,
so the mapping itself is correct. The shortfall comes from the estimated anchor
set, whose purity is 0.97 at true 0.2 and 0.80 at true 0.8, and whose cut point
depends on the estimated prior. The default RIFT test for this check is an
expected failure (`test_default_rift_prior_under_pure_label_shift_within_tolerance`
in `tests/test_methods.py`) and is not loosened. The benchmark reports
`est_prior` and `prior_abs_error` for every method, so this bias is visible in
the results and is not hidden by the summary.

## Baselines

All baselines use the CFS features with a global `StandardScaler` fitted on the
sources, threshold 0.5:

- `lr`, `rf`, `mlp`: logistic regression, random forest, MLP (128, 64).
- `hgb`: the HGB classifier used by RIFT, without DN, IFS or LSC.
- `catboost`: CatBoost with balanced class weights and early stopping on a 10%
  source split. Its train directory is a temporary folder, so nothing is written
  to the working directory.
- `xgboost`: XGBoost with `scale_pos_weight` set from the source class ratio and
  early stopping on a 10% source split.
- `coral`: CORAL (Sun et al., 2016) with the `+I` regulariser: sources are whitened
  with `C_s + I` and re-coloured with `C_t + I` (no mean shift), then HGB. The
  identity is approximate: the relative error is about 0.2% when source variance is
  ten times the identity and about 20% at unit variance, which is the regime of
  standardised features.
- `dann`: domain-adversarial network (PyTorch, CPU) with a gradient-reversal
  domain head and the lambda ramp of Ganin et al. Skipped with a warning if
  torch is missing.

`device: cuda` switches XGBoost to `device="cuda"` and CatBoost to
`task_type="GPU"`. `threads` caps CatBoost, XGBoost, random forest and the
BLAS/OpenMP pools.

## Protocol

Each dataset is split 70/30, stratified and seeded. Experiments:

- `indomain`: source train, target test (upper bound, no pool).
- `pairwise`: every ordered pair (42 pairs for the seven core datasets, 56 for
  the eight minimal datasets).
- `lodo`: leave one dataset out, with the rest as sources.
- `fewshot`: pairwise reference (not label-free). A CatBoost model trained on the
  sources is continued (`init_model`) on a seeded, class-stratified `k%` of the
  target train split, for `fewshot_k` (default 0.1, 1 and 5). The row also stores
  `shift_score`, the mean Isolation Forest anomaly score of the target pool under
  a model fitted on the source features.

Modes: `modes: [transductive, inductive]` runs both in one pass. Transductive
methods receive the target train split **features only** as the unlabelled pool
(disjoint from the evaluated test rows). Inductive mode gives no pool; methods
that need one (`rift-st`, `coral`, `dann`) are skipped there.

Seeds 0, 1, 2 are reported as mean ± std. Metrics: ROC-AUC, PR-AUC, F1, balanced
accuracy, MCC, detection rate, false-positive rate, estimated vs. true target
prior, absolute prior error, and per-attack recall. With a single-class target,
ROC-AUC, PR-AUC, balanced accuracy and MCC are NaN.

Results are resumable. A run is identified by
`(schema, mode, method, experiment, scenario, seed)`. Each row carries a
`config_hash` of the experiment-relevant configuration. On restart the CSV
header is checked against the current version and a clear error is raised on a
mismatch. Per-attack rows are written before the run row.

## Configuration

`configs/default.yaml` runs the full benchmark. Important keys:

| Key | Meaning |
|---|---|
| `datasets` | dataset keys (datasets that do not support `schema` are skipped) |
| `max_rows` | cached rows per dataset (distinct-row sample) |
| `schema` | `minimal`, `core`, `extended` or `cicflow` |
| `modes` | `transductive`, `inductive` |
| `methods`, `experiments` | what to run |
| `fewshot_k` | labelled budgets in percent for the few-shot reference |
| `device` | `cpu` or `cuda` (XGBoost and CatBoost) |
| `threads` | thread cap, or `null` for all cores |
| `seeds`, `test_size`, `data_dir`, `results_dir` | as named |

`configs/smoke.yaml` runs the whole pipeline on synthetic domains.

## Installation

```bash
pip install -e .
```

Python 3.11 or newer. Dependencies are listed in `requirements.txt`. The
`crossids` console script is installed with the package.

## Usage

```bash
crossids download [--datasets cicids2017 usbids1 ...]   # raw files -> data/raw/{key}/ (resumable)
crossids prepare  [--datasets ...] --config configs/default.yaml   # harmonise, sample, cache -> data/processed/{key}.parquet
crossids run      --config configs/default.yaml [--datasets ...] [--methods ...] [--experiments ...] [--modes ...]
crossids report   --results results/
crossids synth    --out data/synthetic                  # synthetic domains as parquet
```

`crossids run` is resumable. Each run is appended to `results/runs.csv` (and its
per-attack recall to `results/per_attack.csv`) right after it finishes; rows
already present are skipped on restart.

A quick end-to-end check on synthetic domains (no downloads):

```bash
crossids run --config configs/smoke.yaml     # writes results/smoke/
crossids report --results results/smoke
```

Tests use only synthetic data and run with `pytest` (about 75 seconds on four
cores).

## Results

`crossids run` writes `results/runs.csv` and `results/per_attack.csv`.
`crossids report` writes `results/summary.csv` and `results/summary.md`, with one
table per mode, experiment and transfer type.

`results/quick_catboost/` contains exploratory CatBoost runs. Their samples are
water-filled per attack type, so their priors are not natural priors and their
numbers are not benchmark results; each file says so at the top.

## Part II: FLARE

FLARE (Few-shot Learning of Attack Rules across Environments, DESIGN.md §11–19)
is an in-context detector meta-trained on the other core-tier datasets (leave one
dataset out). At test time it receives k labelled target flows (at least one per
class) and the unlabelled target flows, and it predicts the attack probability of
every target flow without any gradient step on the target. Code lives in
`src/crossids/flare/`:

| Module | Contents |
|---|---|
| `data.py` | Part I 70/30 split per seed (same `split_tables` call, so test rows match the benchmark), per-episode Gaussian-rank normalisation, episode sampler with augmentations A1–A4 (switchable), padded batches, `EpisodeStream` for DataLoader workers with per-worker seeded generators |
| `model.py` | FLARE encoder (§13), ProtoNet baseline, sign-flip test-time ensemble, checkpoints |
| `train.py` | Meta-training: AdamW, 1000-step warm-up and cosine schedule, validation every 500 steps on 64 held-out source episodes, `last.pt` every 1000 steps with automatic resume, `--bench` timing |
| `evaluate.py` | Shared support draws (`supports_<target>_s<seed>.json`), FLARE variants, ProtoNet, active cold start, CatBoost, RF, LR, kNN, source-pretrained CatBoost, TabPFN when available, full-label CatBoost reference, per-attack recall at k = 16 |
| `al.py` | Active cold start (§15): k-center on label-free embeddings, then entropy times distance; raw-feature k-center |
| `report.py` | Learning curves, Wilcoxon tests against target-only CatBoost, per-attack-family recall, Part I reference line |

**Invariance and augmentation.** The model sees, for every feature, the rank of
each row within its episode. It is therefore invariant to any strictly monotone
per-feature transform by construction, and a monotone warp of the raw values
changes nothing. Augmentation A1 is the per-feature sign flip (p = 0.5). The
monotone warp exists behind `--warp` but is off by default. The model is
invariant to feature order, so the test-time ensemble (E = 8) varies only the
sign flips. With `E = 1` the prediction is the plain forward pass.

Usage (run from the repository root; the default device is CUDA when available,
pass `--device cpu` for the tiny configuration):

```bash
# meta-train FLARE for one target and seed (default: 6 layers, width 128, 4 heads, 20k steps, query 256)
python -m crossids.flare.train --target usbids1 --seed 0 --out runs/flare/usbids1_s0
# same command again resumes from runs/flare/usbids1_s0/last.pt
# timing only: 3 timed steps after a 5-step warm-up, no validation or checkpoints
python -m crossids.flare.train --target usbids1 --bench 3 --device cpu

# ablations (seed 0): no sign flip (noA1), no composite tasks (noA3), row MLP instead of feature
# attention (rowmlp), synthetic random-function episodes only (synthetic); and ProtoNet
python -m crossids.flare.train --target usbids1 --ablation noA1 --out runs/flare/usbids1_s0_noA1
python -m crossids.flare.train --target usbids1 --model protonet --out runs/flare/usbids1_s0_protonet

# few-shot evaluation: k in {4,...,128}, 10 shared support draws, metrics on the Part I target test split
python -m crossids.flare.evaluate --target usbids1 --seeds 0 1 2 --runs-dir runs/flare --out runs/flare/eval_usbids1.csv

# summary: learning curves per target and averaged, Wilcoxon tests, per-attack-family recall at k = 16
python -m crossids.flare.report --runs runs/flare --out results/flare
```

Training options: `--num-workers` (DataLoader workers that generate episodes,
default 2; the episode sequence is deterministic for a given seed and worker count),
`--query-size` (256), `--val-every` (500), `--log-every` (50), `--compile`
(`torch.compile`, off by default). On CUDA, bf16 autocast is used only when the
device supports it natively (compute capability 8.0 or newer); otherwise fp16
autocast with a gradient scaler is used. The chosen mode is logged.

The evaluation CSV has one row per (method, target, k, draw, seed) with ROC-AUC,
PR-AUC, balanced accuracy and F1. `catboost-fulllabel` is the full-label upper
reference, stored once per seed with k = 0. `per_attack_<target>.csv` holds
per-attack-type recall at k = 16. Rows are appended as they are produced, and rows
already present are skipped on restart. Support draws are stored in
`supports_<target>_s<seed>.json` and reused by every method. Test-time normalisation
uses the unlabelled target test features, as the episode normalisation does during
meta-training.

The report reads the Part I table `results/runs.csv` read-only for the best
label-free method per target (LODO, transductive, seed 0), and writes CSVs and
figures under `results/flare/`.

Meta-training on a GPU is run by `scripts/flare_colab.sh`. It packages `src/`, the
project metadata and the cached parquet files, creates a Colab session with
`google-colab-cli`, and runs each job detached on the VM with a log file. The
local side polls every job with short calls and downloads its outputs when it
finishes. A relaunched job resumes from its `last.pt`, and a finished job is
skipped:

```bash
scripts/flare_colab.sh              # all stages: package setup train eval report stop
FLARE_DRY_RUN=1 scripts/flare_colab.sh train   # print the colab commands without running them
```

Tests for FLARE run on CPU with the tiny configuration:

```bash
pytest tests/test_flare.py
```
