# HANDOFF: state of the project and what is left

Repo: https://github.com/vinhqdang/cross_ids, branch `main` (push there, no other branch).
Two manuscripts share this repository:

| Paper | Folder | Content |
|---|---|---|
| Few-label transfer (FLARE-2 versus tabular foundation models) | `paper/flare` | Part II of `DESIGN.md`, results in `results/flare` |
| Cross-dataset benchmark and negative results (RIFT) | `paper/benchmark` | Part I of `DESIGN.md`, results in `results/runs.csv`, `results/analysis`, `results/controls` |

## 0. Rules from the owner (follow exactly)

- Git identity before every commit: `git config user.name "Vinh"` and `git config user.email "dqvinh87@gmail.com"`.
- No mention of Claude, Anthropic, AI, or automation anywhere (commits, code comments, paper, files). No `Co-Authored-By` or `Signed-off-by` trailers.
- Commit only verified work. Run the tests first. Push to `main`. Check `git status` shows no unintended files before pushing.
- Research author details (paper only): Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam, vinh.dq4@buv.edu.vn, ORCID 0000-0002-3877-8024.
- Report negative results as they are. Do not rewrite a claim to match a hoped-for outcome.
- Never commit secrets. The Prior Labs token lives in `runs/tabpfn_token.txt` (git-ignored) or in the `TABPFN_TOKEN` environment variable.

## 1. Setup (Windows, conda env `py313`, one NVIDIA GPU with 16 GB)

```bash
pip install -e . && pip install tabulate matplotlib scipy pytest truststore tabpfn tabpfn-client
python -m pytest -q -p no:warnings tests            # all must pass
crossids download && crossids prepare --config configs/default.yaml   # data/ (git-ignored, about 7 GB)
```

Pitfalls found on this machine:

- `pyarrow` must be imported before `torch` (otherwise `read_parquet` segfaults). `src/crossids/__init__.py` does it.
- SSL errors on downloads: run through `truststore` (`truststore.inject_into_ssl()`); `runs/dl.py` is a local wrapper (not in git).
- `subprocess` resolves `bash` to the WSL launcher; tests use `shutil.which("bash")`.
- Two training jobs at once use about 15 GB of the 16 GB GPU; evaluation spikes then spill into system RAM (a job slows down 100 times or dies silently). Run one training job at a time, or two with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a retry loop (`runs/queue2.sh`).
- Do not use `pkill -f` or `pgrep -f` with a pattern that appears in your own command line.
- `runs/` (checkpoints, logs, queue scripts) and `data/` are not in git. The seed-0 main FLARE-2 checkpoints (`*_s0_f2`) are not on this machine, so controls that need them were run on seed 1.

## 2. State of the results (all raw CSVs in git)

| What | Where | Status |
|---|---|---|
| FLARE-2 + baselines, binary (k=4..128, 10 draws) and class level (k per class 2..16, 5 draws), 8 targets, meta-training seeds 0, 1, 2 | `results/flare/raw/eval_*_all_s{0,1,2}.csv`, `class_*_s{0,1,2}.csv` | done |
| Loss-masked FLARE-2 (`flare2-m`), seed 0, 8 targets | `eval_*_f2m_s0.csv`, `class_*_f2m_s0.csv` | done |
| Ablations (no slot permutation, in-source validation, binary-only tasks) on CTU-13 and CIC-IDS2017, seed 0 | `eval_{ctu13,cicids2017}_{f2noperm,f2insrc,f2mbinary}_s0.csv` and class files | done |
| TabPFN v3 (seed 0) and v3.5 (seeds 0-2), binary | `eval_*_tabpfn_s*.csv` | done (v3/v3.5 run locally) |
| TabPFN-3.5, class level, seed 0 | `class_*_tabpfn_s0.csv` | done |
| Normalisation control, in-context controls, no-permutation diagnostic (seed-1 models) | `rankctl_*_s1.csv`, `incontext_s1.csv`, `noperm_diagnostic_log.txt` | done |
| Statistics | `scripts/stats_flare.py` (seed 0, five-baseline family), `stats_seeds.py` (3 seeds, binary and class), `stats_tabpfn.py` | rerun after new data |
| Benchmark (Part I), 3 seeds, all experiments | `results/runs.csv`, `results/analysis`, `results/controls` | done |

Headline numbers: see the two manuscripts (they are generated from these files by `scripts/make_paper_tables.py` and `scripts/make_benchmark_tables.py`).
Main findings to keep honest: FLARE-2 is on par with RF and CatBoost on binary ROC-AUC (3 seeds: +0.001 and -0.001); TabPFN v3 and v3.5 are higher than FLARE-2 on every target (3.5: -0.014 over three seeds, 95% interval [-0.018, -0.011]); the seed-0 advantage over source-pretrained CatBoost at k>=32 does not hold over three seeds; the class-level gains (per-class AUC +0.011, lower FPR) come with lower recall and lower N-way balanced accuracy.

## 3. Bugs found and fixed (keep in the papers' limitations)

- Queries whose class has no support row get a target logit of -1e9 (loss about 1e6). In one binary-only run validation picked a collapsed checkpoint. Fixed in `train.py` (validation scores supported queries only); the loss mask `--loss-mask-unsupported` removes the artefact from training (`flare2-m`, mean AUC 0.851 versus 0.850 over 8 targets). The main runs are unmasked.
- `scripts/stats_flare.py` and `make_paper_tables.py` used to pool all seeds and all result files; they now restrict the main tables to seed 0 and to the main result files.
- A queue script once copied partial result files (`&&` after a failing command). Result files are now copied only after the run succeeds.

## 4. State of the TODO list

Done: class-level TabPFN-3.5 comparison (`results/flare/raw/class_*_tabpfn_s0.csv`, stats in `results/flare/stats/tabpfn_class.csv`; seed 0, five draws, k<=16; five targets scored on a stratified subsample of at most 5000 query rows, checked on IoT-23); RIFT oracle diagnostics (`rift-oraclePrior`, `rift-oracleAnchor`, in `results/runs.csv`); same-family transfer (`results/controls/family_transfer.csv`); both manuscripts audited and reviewed once and corrected.

Dropped on purpose (code and tests exist, not run): the TabPFN control variants `tabpfn-v3.5-rank` and `tabpfn-v3.5-src` (`python -m crossids.flare.evaluate --methods tabpfn-v3.5-rank tabpfn-v3.5-src ...`). The owner decided TabPFN is only an added baseline.

Open, optional:
1. A final review round of both manuscripts after the last edits (class-level TabPFN paragraph in `paper/flare`, oracle and family paragraphs in `paper/benchmark` were checked by the author only).
2. Tuned baselines, TabPFN fine-tuning, leave-laboratory-out evaluation, PR-AUC and TPR at fixed low FPR, per-attack-type cross-laboratory analysis for the FLARE paper, a one-model-for-all-targets variant, feature-semantics audit and dedup/feature-tier sensitivity for the benchmark paper.
3. TabPFN: version 3 only for seed 0; class level only for version 3.5 and seed 0.
4. References: IDS entries checked against Crossref (`docs/related_cross_dataset.md`); general ML entries checked against Crossref, DBLP/arXiv or publisher pages (PFN: ICLR 2022 confirmed by the owner); Saerens against Crossref; TabPFN-3 cites its model card. The Layeghy and Cantone summaries were checked against the arXiv versions, not the journal versions.
5. Before any release: rotate the Hugging Face and Prior Labs keys that were shared in the working session, and delete `runs/tabpfn_token.txt`.

## 5. Do not

- Do not claim FLARE-2 beats tuned tree ensembles or TabPFN; the data say parity with trees and a loss to TabPFN.
- Do not use unmasked binary-only ablation results from before the checkpoint-selection fix (discarded).
- Do not commit `data/`, `runs/`, `*.aux`, `*.log`, tokens.
- TabPFN weights are under a non-commercial research license; state the version and license in any paper or release.
