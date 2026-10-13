# Cross-Dataset Intrusion Detection: Design

## 1. Problem

Flow-based intrusion detectors reach >99% accuracy when trained and tested on
the same dataset, then drop sharply when moved to another network. This project
studies the **cross-dataset** setting: train on one or more labelled *source*
datasets and evaluate on a *target* dataset whose labels are never used for
training, model selection or threshold tuning.

Task: **binary detection** (benign = 0, attack = 1). Attack taxonomies differ
across datasets, so the multi-class problem is not well posed across them.

Settings:

| Setting | Train | Target access |
|---|---|---|
| In-domain (reference) | A train split | A test split (upper bound) |
| Pairwise transfer | A | B, no target data at all (inductive) |
| Pairwise transductive | A | B unlabelled pool (features only) |
| Leave-one-dataset-out (LODO) | all but B | B (inductive or transductive) |

## 2. Datasets

CIC-IDS2017 and CSE-CIC-IDS2018 come from the same lab (UNB CIC), the same
testbed design and the same extractor, so transfer between them says little
about real cross-network generalisation. The benchmark therefore spans
**six producers and three flow-extractor families**, and every result is also
reported by *transfer type*: same lab, same extractor / different lab, or
different extractor.

| Key | Producer | Extractor | Mirror | Files |
|---|---|---|---|---|
| `cicids2017` | UNB CIC (Canada) | CICFlowMeter | HF `bvsam/cic-ids-2017` | `machine_learning/*.parquet` |
| `cicids2018` | UNB CIC + CSE (Canada) | CICFlowMeter | HF `c01dsnap/CIC-IDS2018` | all CSVs except the 4 GB `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` |
| `usbids1` | Univ. of Sannio (Italy) | CICFlowMeter (patched) | `https://idsdata.ding.unisannio.it/datasets/USBIDS1TTV/` | `USB-IDS-1-TRAINING.csv.gz`, `USB-IDS-1-VALIDATION.csv.gz`, `USB-IDS-1-TEST.csv.gz` |
| `unswnb15` | UNSW Canberra (Australia) | Argus + Bro | HF `Mouwiya/UNSW-NB15` | `UNSW_NB15_training-set.csv` |
| `toniot` | UNSW Canberra (Australia) | Zeek conn.log | HF `codymlewis/TON_IoT_network` | `train_test_network.csv` |
| `iot23` | Stratosphere Lab, CTU (Czechia) | Zeek conn.log | HF `19kmunz/iot-23-preprocessed-allcolumns` | `data/train-*.parquet` |
| `ctu13` | Stratosphere Lab, CTU (Czechia) | Argus bi-flows | `https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-{42..54}/detailed-bidirectional-flow-labels/*.binetflow` | 13 scenarios, saved as `data/raw/ctu13/scenario{N}-{file}.binetflow` |
| `insdn` | University College Dublin (Ireland) | CICFlowMeter | `https://aseados.ucd.ie/datasets/SDN/InSDN_DatasetCSV.zip` | zip with Normal / OVS / metasploitable-2 CSVs, saved as `data/raw/insdn/InSDN_DatasetCSV.zip` |

Note: USB-IDS-1 does contain benign flows (45 708 of ~722 000 rows in the
TEST file, ~6%), so it is a high-attack-prior target, which stresses label-shift
handling.

`family` metadata: `cicids2017`, `cicids2018` → `cic`; `usbids1` → `sannio`;
`unswnb15`, `toniot` → `unsw`; `iot23`, `ctu13` → `ctu`; `insdn` → `ucd`. `extractor` metadata:
`cicflowmeter`, `argus`, `zeek`. A pair is `same-family`, `same-extractor`
(different family, same extractor) or `cross-extractor`.

HF URL pattern: `https://huggingface.co/datasets/{repo}/resolve/main/{path}`.
Raw files go to `data/raw/{key}/`; the harmonised, sampled table is cached at
`data/processed/{key}.parquet`. `data/` is git-ignored. The dataset registry is
a dict in `datasets.py`, so new datasets are added by writing one loader and
one column mapping.

Loading rules:
- Strip whitespace from column names and resolve them through a per-extractor
  alias table (CIC 2017, CIC 2018 and USB-IDS-1 / CICFlowMeter-v4 use three
  different spellings, e.g. `Total Fwd Packets` / `Tot Fwd Pkts` /
  `Total Fwd Packet`, `Init_Win_bytes_forward` / `Init Fwd Win Byts` /
  `FWD Init Win Bytes`). Matching is case-insensitive.
- CIC-2018 CSVs contain repeated header rows; drop rows where
  `Label == "Label"`. Read large CSVs in chunks (`chunksize=200_000`, only the
  needed columns) and keep a per-class reservoir so memory stays bounded.
- Replace ±inf with NaN, coerce to numeric, drop rows with NaN in base
  features, drop exact duplicate rows of (features, label).
- Labels → binary: CIC / USB-IDS-1 `Label` not equal to `BENIGN`
  (case-insensitive) → 1; UNSW-NB15 `label`; ToN-IoT `label`; IoT-23 `label`
  string starting with `Benign` (case-insensitive) → 0, otherwise 1;
  InSDN `Label` equal to `Normal` → 0, otherwise 1; CTU-13 `Label` containing
  `Botnet` → 1, containing `Normal` → 0, and **Background flows are dropped**
  (they are unlabelled mixed traffic, the standard CTU-13 practice). The raw
  label / `attack_cat` / `type` / `detailed-label` string is kept as
  `attack_type` for per-attack recall.
- Deduplication and sampling (one streaming pass, memory bounded): each
  cleaned row gets a 64-bit hash of its rounded harmonised base features plus
  its binary label, mapped to `u in [0, 1)` (seeded). Per label, keep the
  `max_rows` rows with the smallest `u` (bottom-k / KMV sketch). Because
  duplicates share one hash they occupy one slot, so this is a uniform sample
  of **distinct** rows, and the KMV estimate `(k - 1) / u_(k)` gives the number
  of distinct rows per label over the full file. The *natural prior* is defined
  on distinct rows: `pi = D_attack / (D_attack + D_benign)`. The cached table
  has `n = min(max_rows, D_benign + D_attack)` rows with `round(n * pi)` attack
  rows taken from the attack sketch; if a class is short, `n` shrinks so the
  prior is preserved instead of back-filling the other class. Rows with the
  same features but different labels are kept (inherent label noise). The
  prepare metadata records raw, valid, dropped (NaN) and distinct counts per
  label.
- CTU-13 `attack_type` is normalised to `normal` or `botnet-s{scenario}`
  (each scenario is one malware family), not the raw flow label.

## 3. Common Flow Schema (CFS)

The extractors export different features. Cross-dataset learning is only
meaningful on features with the **same physical meaning and units**. CFS has
three tiers.

**Minimal tier (all eight datasets).** Base features `duration`,
`total_pkts`, `fwd_bytes`, `bwd_bytes`, with derived `total_bytes`,
`bytes_per_s`, `pkts_per_s`, `bytes_per_pkt = total_bytes / max(total_pkts, 1)`
and `fwd_bwd_byte_ratio`. CTU-13 (Argus binetflow) only supports this tier:
`duration = Dur`, `total_pkts = TotPkts`, `fwd_bytes = SrcBytes`,
`bwd_bytes = TotBytes - SrcBytes` (clipped at 0).

**Core tier (all datasets except CTU-13).** Five base features:

| CFS feature | Unit | CICFlowMeter | Argus (UNSW-NB15) | Zeek (ToN-IoT, IoT-23) |
|---|---|---|---|---|
| `duration` | s | `Flow Duration` / 1e6 | `dur` | `duration` (`-` → 0) |
| `fwd_pkts` | count | `Total Fwd Packets` | `spkts` | `src_pkts` / `orig_pkts` |
| `bwd_pkts` | count | `Total Backward Packets` | `dpkts` | `dst_pkts` / `resp_pkts` |
| `fwd_bytes` | bytes | `Total Length of Fwd Packets` | `sbytes` | `src_bytes` / `orig_bytes` (`-` → 0) |
| `bwd_bytes` | bytes | `Total Length of Bwd Packets` | `dbytes` | `dst_bytes` / `resp_bytes` (`-` → 0) |

Derived core features, computed after harmonisation so they are consistent by
construction:

- `total_pkts = fwd_pkts + bwd_pkts`, `total_bytes = fwd_bytes + bwd_bytes`
- `bytes_per_s = total_bytes / max(duration, 1e-6)`
- `pkts_per_s = total_pkts / max(duration, 1e-6)`
- `fwd_bytes_per_pkt = fwd_bytes / max(fwd_pkts, 1)`
- `bwd_bytes_per_pkt = bwd_bytes / max(bwd_pkts, 1)`
- `fwd_bwd_pkt_ratio = fwd_pkts / (bwd_pkts + 1)`
- `fwd_bwd_byte_ratio = fwd_bytes / (bwd_bytes + 1)`

**Extended tier (CICFlowMeter datasets incl. InSDN + UNSW-NB15).** Core plus
`fwd_iat_mean`, `bwd_iat_mean`.

**CICFlow tier (the four CICFlowMeter datasets only).** Extended plus
`fwd_init_win`, `bwd_init_win`. Argus `swin`/`dwin` are not byte counts like
CICFlowMeter's initial window bytes (mostly 0 or 255), so window features are
only comparable within CICFlowMeter. They are also the top CatBoost features in
the CIC-family quick experiment, i.e. a likely testbed artefact.

Source columns:

| CFS feature | Unit | CICFlowMeter | Argus (UNSW-NB15) |
|---|---|---|---|
| `fwd_iat_mean` | s | `Fwd IAT Mean` / 1e6 | `sinpkt` / 1e3 |
| `bwd_iat_mean` | s | `Bwd IAT Mean` / 1e6 | `dinpkt` / 1e3 |
| `fwd_init_win` | bytes | `Init_Win_bytes_forward` | (not used) |
| `bwd_init_win` | bytes | `Init_Win_bytes_backward` | (not used) |

Ports and protocol are excluded: the UNSW-NB15 train/test CSVs have no port
columns and the CIC-IDS2017 ML CSVs have no protocol column. Negative values
(CIC uses -1 for "not available") are clipped to 0. Remaining semantic gaps
(e.g. payload bytes vs. IP bytes between extractors) are documented in the
README as an irreducible part of the cross-extractor shift.

Every feature then receives `log1p(max(x, 0))`. The schema module exposes
`CORE_FEATURES`, `EXTENDED_FEATURES`, `features_for(tier)`,
`DATASET_TIERS` (which tiers a dataset supports) and
`harmonise(df, dataset_key) -> DataFrame` returning all features the dataset
supports plus `label`, `attack_type`, `dataset`. An experiment uses
`schema: minimal` (all 8 datasets), `schema: core` (default, 7 datasets),
`schema: extended` (5 datasets) or `schema: cicflow` (4 datasets); datasets not supporting the chosen tier are
skipped with a log message. `MINIMAL_FEATURES` is exposed as well.

## 4. Proposed method: RIFT (Rank-Invariant Flow Transfer)

Three sources of cross-dataset failure are addressed separately:

1. **Covariate (marginal) shift.** Capture tools, link speeds and traffic mixes
   change feature scales between networks.
2. **Spurious, dataset-specific cues.** Features that separate classes in one
   lab testbed by accident (e.g. a fixed TCP window of the attack machine).
3. **Label (prior) shift.** The attack fraction ranges from under 20% to
   over 90% across datasets, so a threshold of 0.5 fitted on the source is miscalibrated
   on the target.

### 4.1 Domain-wise normalisation (DN)

Each domain *d* gets its own normaliser fitted on its own feature matrix, with
no labels:
- `rank`: empirical CDF per feature, `u = (rank - 0.5) / n`, then mapped through
  the standard-normal quantile function (Gaussian rank transform). Evaluate on
  new points by interpolation over stored quantiles (1 000 quantiles).
- `robust`: `(x - median_d) / IQR_d` (IQR floored at 1e-6).
- `standard`: z-score per domain.
- `global`: one scaler fitted on the training sources only (the conventional
  baseline; no target statistics).

Benign-anchored variant (`anchor="benign"`): on **source** domains the
normaliser is fitted on benign rows only, because benign traffic is the stable
"background" of a network. On the target, where labels are unknown, the anchor
set is chosen with the estimated target prior. A normaliser fitted on the whole
pool absorbs the label shift (it maps a 20%-attack target onto the source
mixture), so the first prior estimate must not come from it. Two stages:
(1) initial `pi_t`: map the pool with the pooled *source* benign-anchored
normaliser (`prior_init="source"`), score, run EM; (2) the anchor set is the
`gamma * (1 - pi_t)` fraction of pool rows with the lowest score
(`gamma = 0.8`, a purity margin; at least 5% of the pool) → transport the
target into the source representation by **anchor-matched quantile transport**
→ rescore → re-estimate `pi_t` by EM, for `n_anchor_iter` (default 2)
iterations.

Anchor-matched quantile transport. Fitting a target normaliser directly on the
anchor set is biased: the anchor set is benign traffic *truncated* at a score
quantile, while the source normaliser was fitted on all benign rows. The
reference set is therefore truncated by the same rule on the source: `R` =
pooled source benign rows (in the shared normalised space) whose model score is
at or below the `gamma` quantile of source benign scores. For each feature,
target value `x` maps to `F_R^{-1}(F_A(x))`, where `F_A` is the empirical CDF of
the target anchor set (raw harmonised features, 1 000 quantiles, linear tails as
in the rank normaliser) and `F_R^{-1}` the quantile function of `R`. Under pure
label shift both `A` and `R` are benign truncated at the same benign quantile,
so the transport is exact; under covariate shift it is a monotone per-feature
alignment of benign backgrounds. The model, IFS and calibration stay in source
space; nothing is retrained. Ablation `rift-poolInit` uses
`prior_init="pool"` (first estimate from a normaliser fitted on all pool rows). A median split would anchor on
attacks whenever the target attack prior exceeds 0.5 (e.g. USB-IDS-1, ~94%).
`anchor_purity` (fraction of anchor rows that are truly benign) is logged as a
diagnostic only and never used by the method.

Inductive mode (no target pool): target is normalised with the pooled source
statistics; transductive mode: with its own statistics as above.

### 4.2 Invariance-driven feature selection (IFS)

For each source domain *d* and feature *j*, compute the univariate ROC-AUC
`a_dj` of feature *j* (after DN) for the label. Define signed strength
`s_dj = a_dj - 0.5`.

- **Multi-source (≥2 sources):** keep *j* iff `sign(s_dj)` is identical for all
  sources and `min_d |s_dj| >= tau` (default `tau = 0.05`). If fewer than
  `min_features` pass, top up with the strongest **sign-consistent**
  features only; sign-inconsistent features are never re-admitted. Weight
  `w_j = min_d |s_dj| / max_d |s_dj|` (agreement ratio) is stored for analysis.
- **Single source, transductive:** compute the Kolmogorov–Smirnov statistic
  between the source and target (normalised) marginals for each feature;
  drop the `drop_frac` (default 0.25) features with the largest KS, never
  dropping below `min_features` (default 6).
- **Single source, inductive:** keep all features.

### 4.3 Classifier

`HistGradientBoostingClassifier` (max_iter 300, learning_rate 0.1,
max_leaf_nodes 31, l2_regularization 1.0, early stopping on a 10% validation
split, `class_weight="balanced"`). Probabilities are calibrated with isotonic
regression on a held-out 20% split of the source data (sources pooled).

### 4.4 Test-time label-shift correction (LSC)

Transductive only. Saerens–Latinne–Decaestecker EM:
- `pi_s` = attack fraction in the calibrated source training data (after
  class balancing the calibrated model represents the empirical source prior;
  use the calibration split prior).
- Iterate: `p'_i = (pi_t/pi_s) p_i / ((pi_t/pi_s) p_i + ((1-pi_t)/(1-pi_s))(1-p_i))`,
  `pi_t = mean(p'_i)`, until `|Δpi_t| < 1e-6` or 100 iterations. Clip `pi_t`
  to `[0.001, 0.999]`.
- Final decision: `p'_i >= 0.5`.
- EM assumes pure label shift; under residual covariate shift its estimate is
  biased. Every run stores `est_prior` and `true_prior`, and the report gives
  the mean absolute prior error per method. Threshold-free metrics (ROC-AUC,
  PR-AUC) are computed on `p + 1e-6 * raw_score` to break the ties that
  isotonic calibration introduces.

### 4.4b Uncertainty-gated self-training (`rift-st`)

Transductive extension. After RIFT predicts the target pool, pseudo-label the
most confident rows: per class, the rows whose adjusted posterior is most
extreme, keeping class proportions equal to `pi_t` and at most
`st_frac = 0.2` of the pool; rows are only accepted if
`max(p', 1 - p') >= 0.9`. Retrain the HGB on source rows plus pseudo-labelled
target rows (target rows weighted `st_weight = 1.0`), recalibrate on the source
calibration split, re-run EM, repeat `st_rounds = 2` times. Target labels are
never used.

### 4.5 Reference setting: few-shot target fine-tuning

Not part of the label-free benchmark, reported separately as an upper
reference. An unsupervised shift score (Isolation Forest fitted on source
features, mean anomaly score of the target pool) quantifies how far the target
is from the source; a CatBoost model trained on the sources is then continued
(`init_model`) on a labelled budget of `k%` of the target train split
(k ∈ {0.1, 1, 5}). This quantifies how many target labels the label-free
methods are worth.

### 4.6 Ablations

`RIFT = DN(rank, benign-anchor) + IFS + HGB + calibration + LSC`. Ablations
switch off one component at a time: `-DN` (global scaler), `-anchor`,
`-IFS`, `-LSC`, and `DN=robust` / `DN=standard`.

## 5. Baselines

All baselines use the CFS features with a global `StandardScaler` fitted on the
sources, threshold 0.5:

- `lr`: LogisticRegression (balanced, max_iter 2000).
- `rf`: RandomForest (300 trees, balanced_subsample, n_jobs=-1).
- `hgb`: same HGB as RIFT, no DN/IFS/LSC.
- `catboost`: CatBoostClassifier (500 iterations, depth 6, auto class
  weights "Balanced", early stopping on a 10% source validation split,
  thread_count = all cores, verbose 0). This is the "naive" reference: it
  reaches ~0.999 ROC-AUC in-domain on CIC-IDS2017 and is known to collapse
  across datasets; the benchmark quantifies that collapse.
- `xgboost`: XGBClassifier (500 trees, depth 6, learning_rate 0.1,
  hist tree method, `scale_pos_weight` = benign/attack ratio of the sources,
  early stopping on a 10% source validation split). `device` comes from the
  config key `device: cpu|cuda` (default cpu); CatBoost uses
  `task_type="GPU"` when `device: cuda`.
- `mlp`: sklearn MLPClassifier (128, 64), early stopping.
- `coral`: CORAL (Sun et al., 2016) — `C_s + I`, `C_t + I` (the paper's
  regulariser), whiten source covariance, re-colour with target covariance,
  no mean shift (transductive), then HGB.
- `dann`: Domain-adversarial network (Ganin et al., 2016) in PyTorch, CPU:
  feature extractor 2×128 ReLU, label head, domain head with gradient-reversal
  (λ ramp `2/(1+exp(-10p)) - 1`), Adam 1e-3, 20 epochs, batch 512.
  Transductive (uses unlabelled target). Torch import is optional; the method
  is skipped with a warning if torch is missing.

## 6. Metrics

On the target test set: ROC-AUC, PR-AUC (average precision), F1 (attack
class), balanced accuracy, MCC, detection rate (recall), false-positive rate,
and estimated vs. true target prior for LSC. Per-attack-type recall is saved
separately. Threshold-dependent metrics are computed at the method's own
decision rule (never tuned on target labels).

## 7. Protocol

- Each dataset is split 70/30 stratified into `train` / `test` (seeded).
  Sources train on `train`; the target is evaluated on `test`; the transductive
  unlabelled pool is the target `train` split **features only** (disjoint from
  the evaluated rows).
- Experiments: in-domain (one per dataset), pairwise (all ordered pairs,
  42 for the seven core-tier datasets), LODO (one per dataset). Each pair carries its
  `transfer_type` (`same-family`, `same-extractor`, `cross-extractor`) and
  the summary aggregates by it.
- Seeds: 0, 1, 2. Report mean ± std.
- `modes: [transductive, inductive]` runs both in one pass; inductive methods
  get no pool, transductive-only methods are skipped in inductive mode.
- The resume key is `(schema, mode, method, experiment, scenario, seed)`; a
  config hash is stored per row and the header is validated on resume.
- Few-shot reference (§4.5) is the experiment `fewshot` (pairwise only).
- Results: `results/runs.csv` (one row per method × experiment × seed),
  `results/summary.csv`, `results/summary.md` (markdown tables), per-attack
  recall in `results/per_attack.csv`.

## 8. Package layout

```
pyproject.toml            # package "crossids", src layout, entry point `crossids`
requirements.txt
configs/default.yaml      # datasets, max_rows, seeds, methods, experiments
src/crossids/
  __init__.py
  schema.py               # CFS definition, aliases, harmonise()
  datasets.py             # download, load, cache, sample, synthetic generator
  normalize.py            # DomainNormalizer (rank/robust/standard/global, anchor)
  selection.py            # invariance + KS selection
  labelshift.py           # EM prior estimation, posterior adjustment
  methods/
    __init__.py           # registry: name -> factory
    base.py               # interface: fit(sources, target_pool=None), predict_proba(X_target, domain="target")
    baselines.py          # lr, rf, hgb, mlp
    coral.py
    dann.py
    rift.py               # RIFT + ablation flags
  metrics.py
  experiment.py           # protocols, seeds, result writing
  report.py               # summary tables
  cli.py                  # crossids download | run | report | synth
tests/                    # pytest, synthetic data only, fast (<60 s total)
```

Method interface:

```python
class Method:
    transductive: bool
    def fit(self, sources: list[Domain], target_pool: np.ndarray | None) -> "Method": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...   # P(attack), target rows
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

`Domain` is a dataclass `(name, X: np.ndarray, y: np.ndarray, meta: DataFrame)`.

## 9. Synthetic data (tests and smoke runs)

`make_synthetic_domains(n_domains=3, n=4000, seed)` returns domains that share
one labelling mechanism on latent features but differ by per-domain
monotone feature distortions (scale, offset, power), a different attack prior
per domain (e.g. 0.2, 0.5, 0.7) and one spurious feature that is predictive in
one domain only. Tests check that RIFT beats the `hgb` baseline on
synthetic cross-domain ROC-AUC/balanced accuracy, that EM recovers a known
prior within 0.05, and that rank normalisation is invariant to monotone
transforms.
