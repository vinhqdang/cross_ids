# cross_ids

Cross-dataset flow-based network intrusion detection: a benchmark of why transfer between datasets fails, and a study of what a few labelled target flows can recover.
Models are trained on labelled source networks and evaluated on a target network. In the label-free benchmark the target labels are never used for training, model selection or threshold tuning; in the few-label study the target contributes only $k$ labelled flows. The task is binary detection (benign = 0, attack = 1), with attack-class results for the few-label study.

The repository holds one manuscript, `paper/ids.tex`, built from the results in `results/`:

1. **Benchmark.** Seven public datasets (eight with CTU-13), five laboratories, three flow extractors; in-domain, pairwise (42 ordered pairs) and leave-one-dataset-out (LODO) settings; eight baselines and **RIFT**, a domain-invariant method of our own, with ablations and oracle diagnostics. Result: transfer fails, RIFT does not repair it, and single-attack-family transfer is also near chance.
2. **Few-label study.** **FLARE-2**, a transformer meta-trained on tasks from the other datasets, against tree ensembles and **TabPFN** (versions 3 and 3.5), with three meta-training seeds, attack-class results, ablations and controls. Result: FLARE-2 matches the trees on binary ROC-AUC and TabPFN is slightly better on every target.

`DESIGN.md` is the specification of both parts, `docs/` holds the related-work notes, and `HANDOFF.md` records the state of the project, the pitfalls found on the development machine and what is open.

## Repository layout

| Path | Contents |
|---|---|
| `src/crossids/` | Benchmark package: schema, datasets, sampling, normalisation, selection, label shift, methods (`methods/`), experiment runner, analysis, controls, few-shot reference |
| `src/crossids/flare/` | Few-label study: episodes and tasks (`data.py`), models (`model.py`), meta-training (`train.py`), evaluation (`evaluate.py`, `evaluate_class.py`), reports (`report.py`, `report_class.py`), active selection (`al.py`) |
| `configs/` | `default.yaml` (full benchmark), `smoke.yaml` (synthetic domains) |
| `scripts/` | Statistics (`stats_flare.py`, `stats_seeds.py`, `stats_tabpfn.py`), table and figure generators for the paper (`make_tables_benchmark.py`, `make_tables_fewlabel.py`), controls (`family_transfer.py`, `incontext_controls.py`, `diag_noperm.py`), the early exploratory `quick_catboost.py` |
| `results/` | Raw per-run results and derived tables, all in git (see Results) |
| `paper/` | `ids.tex`, `refs.bib`, `tables/` and `figures/` generated from `results/`; `springer/` holds the Springer Nature template version |
| `tests/` | Unit and integration tests (synthetic data only) |
| `data/`, `runs/` | Raw and cached datasets, checkpoints and logs; git-ignored |

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
cleaned row gets a seeded 64-bit hash of all harmonised features the dataset provides (rounded to
1e-6) combined with its label, mapped to `u` in `[0, 1)`. Per label, a bottom-k sketch keeps the
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
`crossids` console script is installed with the package. Optional extras: `pip install -e .[tabpfn]` for the TabPFN baselines,
`.[tabpfn-api]` for the API backend, `.[windows]` for the Windows certificate store and `.[test]` for pytest.

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
`crossids report` writes `results/summary.csv` and `results/summary.md`, with one table per mode, experiment and transfer type, and `crossids analyze` writes the tables and figures of `results/analysis/`.

| Folder | Contents |
|---|---|
| `results/runs.csv`, `per_attack.csv`, `summary.*` | Benchmark runs: 3 seeds, 4 experiments, 17 methods and variants, plus the two RIFT oracle diagnostics |
| `results/analysis/` | Benchmark analysis (method ranking, Friedman tests, feature signs, pairwise matrices, family analysis, few-shot curve, shift score, prior estimation) |
| `results/controls/` | Oracle search, nearest-neighbour concept test, dataset separability, fair few-label comparison, single-family transfer (`family_transfer.csv`) |
| `results/flare/raw/` | Few-label evaluation CSVs per target and seed: FLARE-2 and baselines (`eval_*_all_s*.csv`, `class_*_s*.csv`), loss-masked and ablation variants, TabPFN (`*_tabpfn_s*.csv`), normalisation and in-context controls (`rankctl_*`, `incontext_s1.csv`, `noperm_diagnostic_log.txt`) |
| `results/flare/stats/` | Paired statistics: `stats.md` (seed 0, five baselines), `seeds*.csv` (three seeds), `tabpfn*.csv` |
| `results/quick_catboost/` | Exploratory CatBoost runs. Their samples are water-filled per attack type, so their priors are not natural priors and their numbers are not benchmark results; each file says so at the top |

## Few-label study: FLARE-2 and TabPFN

FLARE-2 (DESIGN.md Part II) is an in-context detector meta-trained on the other core-tier datasets (leave one dataset out). At test time it receives $k$ labelled target flows (at least one per class) and the unlabelled target flows and predicts without any gradient step on the target. It uses random label slots with a class-equivariant matching readout, so one model serves binary detection and attack-type classification. Checkpoints are selected on a held-out source dataset.

```bash
# meta-train FLARE-2 for one target and seed (6 layers, width 128, 4 heads, 20k steps, 256 query rows; resumes from last.pt)
python -m crossids.flare.train --target usbids1 --seed 0 --steps 20000 --device cuda --model flare2 --out runs/flare/usbids1_s0_f2
# variants: --loss-mask-unsupported (run suffix _f2m), --no-slot-perm (_f2_noperm), --val-mode insource (_f2_insrc), --tasks binary

# binary evaluation: k in {4,...,128}, 10 shared support draws, metrics on the benchmark target test split
python -m crossids.flare.evaluate --target usbids1 --seeds 0 --methods flare2 rf catboost-target catboost-ft lr knn \
    --device cuda --out runs/flare/eval_usbids1_all_s0.csv --supports-dir runs/flare
# attack-class evaluation (per-class detection and N-way); --ks, --draws and --max-query bound the work
python -m crossids.flare.evaluate_class --target usbids1 --seeds 0 --methods flare2 catboost rf lr knn --device cuda \
    --out runs/flare/class_usbids1_s0.csv
```

Evaluation CSVs have one row per (method, target, k, draw, seed); rows are appended as they are produced and skipped on restart, support draws are stored in `supports_<target>_s<seed>.json` and shared by every method, and `--max-query N` scores every method on a fixed stratified subsample of the test rows. Normalisation controls are the methods `flare2-rank0` (ranks over the support only) and `flare2-rank256` (support and 256 random query rows). A control on the use of the support labels is `scripts/incontext_controls.py`, and `scripts/diag_noperm.py` explains the no-permutation ablation.

**TabPFN.** `tabpfn-v2`, `tabpfn-v3` and `tabpfn-v3.5` run the released weights of the `tabpfn` package on the same support and query rows. Versions other than v2 need a Prior Labs license accepted for your account, and a token in the `TABPFN_TOKEN` environment variable or in the git-ignored file `runs/tabpfn_token.txt` (never commit it). The weights are under a non-commercial research license; state the version in any paper. `TABPFN_BACKEND=api` sends the support and query rows to the Prior Labs service instead (daily quota, models may change over time), and `TABPFN_CHUNK` sets the query rows per predict call to save GPU memory.

**RIFT diagnostics.** `rift-oraclePrior` and `rift-oracleAnchor` read the labels of the target pool and are not label-free methods; they are registered in `ALL_METHOD_NAMES` but not in the default method list, and `crossids analyze` leaves them out. `scripts/family_transfer.py` runs the single-attack-family transfer control.

## Reproducing the paper

```bash
python -m pytest -q -p no:warnings tests                       # all tests
python scripts/stats_flare.py && python scripts/stats_seeds.py && python scripts/stats_tabpfn.py
python scripts/make_tables_benchmark.py && python scripts/make_tables_fewlabel.py   # tables and figures in paper/
cd paper && pdflatex ids && bibtex ids && pdflatex ids && pdflatex ids
```

The Springer Nature journal version (template `sn-jnl`, numbered references, one `.tex` file with the tables inlined) is generated from `paper/ids.tex`:

```bash
python scripts/make_springer.py                                  # writes paper/springer/ids_sn.tex, refs.bib, figures/
cd paper/springer && pdflatex ids_sn && bibtex ids_sn && pdflatex ids_sn && pdflatex ids_sn
```

The class file and bibliography style in `paper/springer/` come from the Springer Nature template package. The Declarations section has placeholders (funding, competing interests, author contribution) to be completed by the author before submission.

Training needs a GPU (about 45 minutes per model on an RTX-class GPU with 16 GB; two models at once can exhaust the memory and slow both). The benchmark runs on CPU. Results in `results/` are enough to regenerate all tables, figures and statistics without retraining.

Notes for Windows: import `pyarrow` before `torch` (the package does it); run downloads through `truststore` if the certificate store is missing; tests use `shutil.which("bash")` because `subprocess` otherwise resolves `bash` to the WSL launcher.
