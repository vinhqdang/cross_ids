# HANDOFF: what is done, what to do next

Repo: https://github.com/vinhqdang/cross_ids, branch `main` (push there, no other branch).
Project: cross-dataset network intrusion detection. Method: FLARE-2 (in-context few-label transformer). Design: `DESIGN.md` (Part I = benchmark + RIFT, Part II = FLARE / FLARE-2). Manuscript: `paper/flare/main.tex`.

## 0. Rules from the owner (follow exactly)

- Git identity before every commit: `git config user.name "Vinh"` and `git config user.email "dqvinh87@gmail.com"`.
- No mention of Claude, Anthropic, AI, or automation anywhere (commits, code comments, paper, files). No `Co-Authored-By` or `Signed-off-by` trailers.
- Commit only verified work. Run the tests first. Push to `main`. Check `git status` shows no unintended files before pushing.
- Research author details (paper only): Quang-Vinh Dang, British University Vietnam, Hung Yen, Vietnam, vinh.dq4@buv.edu.vn, ORCID 0000-0002-3877-8024.
- Report negative results as they are. Do not rewrite a claim to match a hoped-for outcome.

## 1. Setup on the new machine

```bash
git clone https://github.com/vinhqdang/cross_ids.git
cd cross_ids
pip install -e . && pip install tabulate matplotlib scipy pytest
python -m pytest -q -p no:warnings tests/test_flare.py tests/test_flare2.py tests/test_flare2_mask.py   # all must pass
crossids prepare --config configs/default.yaml     # downloads + caches data/processed/*.parquet (git-ignored, large)
```

Not in git (you must recreate): `data/` (run `crossids prepare`), `runs/` (all checkpoints, logs, per-run CSVs). Only the result CSVs in `results/` are in git. A new model must be trained again (about 1 hour on a T4, hopeless on CPU). TeX: `texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended` to build the paper.

## 2. State of the results (all in git)

| What | Where | Status |
|---|---|---|
| FLARE-2 + baselines, binary (k=4..128, 10 draws) and class level (k per class 2..16, 5 draws), 8 targets, seed 0 | `results/flare/raw/eval_*.csv`, `class_*_s0.csv` | done: cicids2017, cicids2018, ctu13, insdn, iot23, toniot, unswnb15, usbids1 |
| Paired statistics (exact Wilcoxon over 8 targets, Holm, bootstrap CI) | `scripts/stats_flare.py` -> `results/flare/stats/` | done, rerun after new data |
| Paper tables and figures | `scripts/make_paper_tables.py` -> `paper/flare/tables`, `paper/flare/figures` | done for main results; ablation table NOT written yet |
| Manuscript draft | `paper/flare/main.tex`, `refs.bib`, `ablation_section.tex` (placeholder) | 11 pages, compiles |
| Ablation, binary-only tasks + loss mask (`flare2-m-binary`) | `results/flare/raw/eval_{ctu13,cicids2017}_f2mbinary_s0.csv`, `class_*` | done on ctu13 and cicids2017 |
| Reference with loss mask (`flare2-m`) | `results/flare/raw/eval_ctu13_f2m_s0.csv`, `class_ctu13_f2m_s0.csv` | done on ctu13; **cicids2017 missing** |

Headline numbers (seed 0, mean over 8 targets, binary ROC-AUC, FLARE-2 minus baseline): vs RF +0.004 (CI -0.003..+0.010, not significant), vs target-only CatBoost -0.001 (not significant), vs LR +0.075 and kNN +0.110 (8/8 targets, p_Holm 0.039), vs source-pretrained CatBoost fine-tuned: significant only for k>=32 (7/8 targets, p_Holm 0.047). Class level: per-class ROC-AUC higher than RF and CatBoost on 7/8 targets (p=0.016 uncorrected), FPR lower by 0.05 but recall lower by 0.03-0.04 at threshold 0.5; N-way macro-F1 higher on average but only 3/8 targets, not significant. See `results/flare/stats/stats.md`.

Ablation so far (do not over-read, 1 seed, 2 targets):
- CTU-13 binary ROC-AUC k=4/16/64/128: f2m 0.594/0.707/0.792/0.842; f2m-binary-only 0.563/0.671/0.779/0.824.
- CIC-2017 (vs f2 without mask): f2 0.785/0.861/0.965/0.975; f2m-binary-only 0.776/0.856/0.949/0.962. N-way macro-F1 at k=16: f2 0.504, f2m-binary-only **0.566** (binary-only is better there).
- So label-space tasks give a small binary gain, and no class-level gain on CIC-2017. The paper must not say they are "the key ingredient" unless the full ablation supports it. The draft text in `main.tex` Sect. 4 ("Meta-training episodes") still contains that framing: fix it from the data.

## 3. Bug found and fixed (explains old numbers, keep in the paper's limitations)

Queries whose class has no support row (2% label noise at k=1) get a target logit of -1e9, so their loss is about 1e9. In validation of the binary-only run this dominated `val_bce`, and checkpoint selection picked the worst, collapsed checkpoint (AUC about 0.50). Fixed in `src/crossids/flare/train.py` (`validate_classes` now scores supported queries only). The main `f2` runs are unaffected because their validation episodes had 0 unsupported queries (logged `frac_unsupported_queries`). The first binary-only results (unmasked) were discarded. Training loss values of about 1e6 in old `f2` logs are this same artifact; the loss mask (`--loss-mask-unsupported`, variant `f2m`) removes it and gave results within 0.01 ROC-AUC of `f2` on CTU-13.

## 4. TODO, in this order

### A. Finish the ablation runs (seed 0, targets `ctu13` and `cicids2017`)

Needed (GPU, about 1 h train + 30 min evaluation each):

| Run | Train flags | Eval method name | Status |
|---|---|---|---|
| f2m on cicids2017 | `--model flare2 --loss-mask-unsupported` | `flare2-m` | missing (was running on a Colab session that may be gone) |
| no slot permutation, ctu13 + cicids2017 | `--model flare2 --no-slot-perm` | `flare2-noperm` | missing |
| in-source validation (no held-out source), ctu13 + cicids2017 | `--model flare2 --val-mode insource` | `flare2-insrc` | missing |
| optional: f2 seeds 1 and 2, ctu13 + cicids2017 | `--model flare2 --seed 1` (and 2) | `flare2` | missing |

Local GPU commands (replace `TARGET`, `VAR` suffix, `METHOD` per the table; run dir naming: `_f2m`, `_f2_noperm`, `_f2_insrc`):

```bash
T=ctu13   # then cicids2017
python -m crossids.flare.train --target $T --seed 0 --steps 20000 --device cuda --num-workers 2 \
  --model flare2 --no-slot-perm --out runs/flare/${T}_s0_f2_noperm
python -m crossids.flare.evaluate --target $T --seeds 0 --methods flare2-noperm --device cuda \
  --out runs/flare/eval_${T}_f2noperm_s0.csv --supports-dir runs/flare
python -m crossids.flare.evaluate_class --target $T --seeds 0 --methods flare2-noperm --device cuda \
  --out runs/flare/class_${T}_f2noperm_s0.csv
cp runs/flare/eval_${T}_f2noperm_s0.csv runs/flare/class_${T}_f2noperm_s0.csv results/flare/raw/
```

Training resumes from `last.pt` if the run directory already exists. Training flags per variant are also in `scripts/flare_colab_common.sh` (`variant_config`).

Sanity check before trusting the new machine: the support draws must match the old ones. Evaluate a cheap baseline and compare to the committed file:
`python -m crossids.flare.evaluate --target ctu13 --seeds 0 --methods rf --out /tmp/chk.csv --supports-dir runs/flare`, then compare `roc_auc` of `rf` per (k, draw) with `results/flare/raw/eval_ctu13_all_s0.csv`. They must be equal. If they differ, stop and find out why before comparing ablations with the old FLARE-2 numbers.

Colab alternative (what has been used): `scripts/flare_colab_target.sh TARGET SEED VARIANT` (variants `f2 f2binary f2noperm f2insrc f2ctx f2m f2mbinary`) and `scripts/flare_colab_targets.sh "T1 T2" SEED VARIANT`. Needs `google-colab-cli` logged in (`colab` command). Everything resumes if rerun. Pitfalls: free quota runs out ("Service Unavailable" on allocation: wait or use another Google account via `COLAB_ACCOUNT_DIR`); if the machine running the script sleeps, the local poller dies and the session list is lost, so keep it awake; the setup cell can fail silently if `pip install` fails (check `runs/colab/<tag>.log` for "No module named crossids").

### B. Ablation table and section

1. In `scripts/make_paper_tables.py`, fill `ablation_table()` (stub): read the ablation CSVs in `results/flare/raw`, produce `paper/flare/tables/ablation.tex` with rows f2, f2m, f2m-binary-only, noperm, insrc, columns binary ROC-AUC at k=4/16/64 and N-way macro-F1 at k=16 and per-class ROC-AUC at k=4, for ctu13 and cicids2017. Compare each variant with its proper reference (`f2m` for `f2m-binary`; `f2` for noperm and insrc).
2. Replace `paper/flare/ablation_section.tex` with the real section: a table plus 2-3 sentences stating exactly what the data show, including that one seed and two targets cannot support significance claims.
3. Rerun `python scripts/stats_flare.py && python scripts/make_paper_tables.py`.

### C. Fix the manuscript to match the evidence

Edit `paper/flare/main.tex`:
- Sect. 4 "Meta-training episodes" and the abstract/introduction/conclusion: remove or soften any claim that class-level tasks are the key ingredient unless the ablation supports it.
- Limitations: replace the sentence about the very negative logit by the true account in section 3 above (artifact, fixed, main runs unaffected, `f2m` within 0.01 ROC-AUC).
- Keep: abstract's "no significant gain over RF/CatBoost on binary ROC-AUC".
- References: the IDS entries in `refs.bib` were verified against Crossref (`docs/related_cross_dataset.md`). The general ML and dataset entries added later (CatBoost, TabPFN, PFN, matching/prototypical nets, MAML, Set Transformer, DANN, CORAL, CIC-IDS, UNSW-NB15, CTU-13, InSDN, ToN-IoT, IoT-23) were written from memory and are NOT verified: check each against Crossref (volume, pages, year) before submission.
- Build: `cd paper/flare && pdflatex main && bibtex main && pdflatex main && pdflatex main`. No undefined references, no overfull boxes.
- Ask for an independent review of the draft (a different model or a person) and fix what they find.

### D. Optional, after A-C

- FLARE-2 seeds 1-2 on all 8 targets (about 16 more GPU-hours) to put a seed-level interval on the main table.
- The benchmark paper (Part I, 8 datasets, RIFT negative results, concept-shift controls). Material: `results/analysis/`, `results/controls/`, `DESIGN.md` Part I, `docs/related_cross_dataset.md`. The Part I seed-2 benchmark was paused at about 40% (`results/runs.csv`); see `README.md` for the commands.
- Check whether the loss-mask variant `f2m` should become the default for all targets (needs `f2m` on the other 6 targets, 8 GPU-hours).

## 5. Do not

- Do not use unmasked binary-only results; they came from the checkpoint-selection bug.
- Do not compare ablation runs with old `f2` numbers until the support-draw check in section 4A passes.
- Do not claim FLARE-2 beats tuned tree ensembles. The data say parity on binary ROC-AUC.
- Do not commit `data/`, `runs/`, `*.aux`, `*.log`.
- Do not use `pkill -f` or `pgrep -f` with a pattern that appears in your own command line: it kills or matches your own shell.
