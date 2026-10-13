# cross_ids

Cross-dataset flow-based intrusion detection. Models are trained on labelled
source networks and evaluated on a target network whose labels are never used
for training, model selection or threshold tuning. The task is binary detection
(benign = 0, attack = 1).

The benchmark covers six producers and three flow-extractor families, reports
every result by transfer type, and includes **RIFT** (Rank-Invariant Flow
Transfer), a method that separates covariate shift, spurious cues and label
shift. The full specification is in `DESIGN.md`.

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

`attack_type` keeps the raw label string for per-attack recall. Transfer types:
`same-family` (for example `cicids2017` → `cicids2018`), `same-extractor`
(different family, same extractor) and `cross-extractor`.

## Common Flow Schema (CFS)

Features are mapped to the same physical meaning and units before learning.
Three tiers are defined in `crossids.schema`:

- **minimal** (all eight datasets): `duration`, `total_pkts`, `fwd_bytes`,
  `bwd_bytes` and derived `total_bytes`, `bytes_per_s`, `pkts_per_s`,
  `bytes_per_pkt`, `fwd_bwd_byte_ratio`. CTU-13 supports only this tier.
- **core** (seven datasets, all except CTU-13): five base features plus eight
  derived per-direction features (13 in total).
- **extended** (five datasets: the CICFlowMeter datasets, InSDN and UNSW-NB15):
  core plus `fwd_iat_mean`, `bwd_iat_mean`, `fwd_init_win`, `bwd_init_win`.

Ports and protocol are excluded. Durations are converted to seconds, negative
sentinel values (-1) are clipped to 0, `-` in Zeek files is read as 0, and
every feature receives `log1p(max(x, 0))`. Each source has its own spelling
table (for example `Total Fwd Packets` / `Tot Fwd Pkts` / `Total Fwd Packet`),
matched case-insensitively. Remaining semantic gaps between extractors are part
of the cross-extractor shift and are not corrected.

Harmonisation reads raw files in chunks with only the needed columns and keeps
a bottom-k random sample per binary label, so memory stays bounded. Exact
duplicate (features, label) rows are removed inside the retained sample pool; a
full-data duplicate count is not computed.

## RIFT

`RIFT = DN(rank, benign anchor) + IFS + HGB + calibration + LSC`

1. **Domain-wise normalisation (DN).** Each source domain is normalised by its
   own label-free statistics (Gaussian rank transform by default; `robust` and
   `standard` are alternatives). With the benign anchor, a source's statistics
   are fitted on its benign rows only. On the target, the anchor is the set of
   pseudo-benign pool rows (score below the median), refined over two
   iterations. Inductive mode normalises the target with pooled source
   statistics.
2. **Invariance-driven feature selection (IFS).** With two or more sources, a
   feature is kept only if its univariate AUC strength has the same sign in
   every source and magnitude at least `tau = 0.05`. With one source and a
   target pool, the features with the largest KS distance are dropped.
3. **Classifier and calibration.** `HistGradientBoostingClassifier` with balanced
   class weights, isotonic calibration on a held-out 20% of the sources.
4. **Label-shift correction (LSC, transductive only).** The Saerens–Latinne–
   Decaestecker EM estimates the target attack fraction from unlabelled pool
   features and re-weights the posteriors.

Ablations (`rift-noDN`, `rift-noAnchor`, `rift-noIFS`, `rift-noLSC`,
`rift-robust`, `rift-standard`) switch off or swap one component.

Baselines: `lr`, `rf`, `hgb`, `catboost`, `mlp` (global `StandardScaler`, threshold
0.5), `coral` (CORAL alignment with the target pool, then HGB) and `dann` (a
domain-adversarial network in PyTorch; skipped with a warning if torch is
missing).

Methods never use target labels for fitting, calibration, feature selection or
thresholds. A transductive method only receives target-pool **features** (the
target train split); evaluation is on the disjoint target test split.

## Protocol

Each dataset is split 70/30, stratified and seeded. Experiments:

- `indomain`: source train, target test (upper bound, no pool).
- `pairwise`: every ordered pair (42 pairs for the seven core datasets, 56 for
  the eight minimal datasets).
- `lodo`: leave one dataset out, with the rest as sources.

Seeds 0, 1, 2 are reported as mean ± std. Metrics: ROC-AUC, PR-AUC, F1, balanced
accuracy, MCC, detection rate, false-positive rate, estimated vs. true target
prior, and per-attack recall.

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
crossids run      --config configs/default.yaml [--datasets ...] [--methods ...] [--experiments ...]
crossids report   --results results/
crossids synth    --out data/synthetic                  # synthetic domains as parquet
```

`crossids run` is resumable. Each (method, experiment, scenario, seed) row is
appended to `results/runs.csv` (and its per-attack recall to
`results/per_attack.csv`) right after the fit, and rows already present are
skipped on restart.

A quick end-to-end check on synthetic domains (no downloads, under two minutes):

```bash
crossids run --config configs/smoke.yaml     # writes results/smoke/
crossids report --results results/smoke
```

Tests use only synthetic data and run with `pytest`.

## Results

Results are produced by `crossids run`. The summary tables are written by
`crossids report` to `results/summary.md` and `results/summary.csv`.
