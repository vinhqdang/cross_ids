#!/usr/bin/env bash
# FLARE on Google Colab (GPU): packages the project, meta-trains every (target, seed) job plus the ablations and
# ProtoNet, runs the few-shot evaluation per target and downloads the outputs of every finished job.
#
# Needs the google-colab-cli (`colab`) with a logged-in account. Each job is skipped when its outputs already exist
# locally, so rerunning the script resumes at the first unfinished (target, seed) job. A job that is interrupted
# restarts from its beginning (checkpoints are per job, not per step).
#
# Usage:
#   scripts/flare_colab.sh [stage ...]             stages: package setup train eval stop (default: all, in order)
#   FLARE_DRY_RUN=1 scripts/flare_colab.sh train   print the colab commands without running them
#
# Environment (defaults in brackets):
#   FLARE_SESSION [flare]     Colab session name             FLARE_GPU [T4]        accelerator for `colab new`
#   FLARE_STEPS [60000]       meta-training steps            FLARE_SEEDS ["0 1 2"] meta-training seeds
#   FLARE_TARGETS [all 8]     FLARE_ABLATIONS ["noA1 noA3 rowmlp synthetic"]    ablations, seed 0 only
#   FLARE_PROTONET [1]        also meta-train ProtoNet per (target, seed)
#   FLARE_TIMEOUT [14400]     seconds allowed for one training job, or one evaluation target
#
# Outputs (local): runs/flare/<target>_s<seed>[_<ablation>|_protonet]/{best.pt,last.pt,log.jsonl,summary.json},
#                  runs/flare/eval_<target>.csv and runs/flare/supports_<target>_s<seed>.json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${FLARE_SESSION:-flare}"
GPU="${FLARE_GPU:-T4}"
STEPS="${FLARE_STEPS:-60000}"
SEEDS="${FLARE_SEEDS:-0 1 2}"
TARGETS="${FLARE_TARGETS:-cicids2017 cicids2018 usbids1 unswnb15 toniot iot23 insdn ctu13}"
ABLATIONS="${FLARE_ABLATIONS:-noA1 noA3 rowmlp synthetic}"
TIMEOUT="${FLARE_TIMEOUT:-14400}"
PROTONET="${FLARE_PROTONET:-1}"
DRY="${FLARE_DRY_RUN:-0}"

BUILD="$ROOT/build/flare"
TARBALL="$BUILD/crossids-flare.tar.gz"
RUNS="$ROOT/runs/flare"
REMOTE_ROOT="/content/flare"
JOBS="$BUILD/jobs"

run() {  # run <command...>: executes, or prints when FLARE_DRY_RUN=1
  if [[ "$DRY" == "1" ]]; then
    printf 'DRY: %s\n' "$*"
  else
    "$@"
  fi
}

stage_package() {
  local stage
  stage="$(mktemp -d "$BUILD/stage.XXXXXX")"
  mkdir -p "$stage/data/processed"
  cp -r src pyproject.toml README.md LICENSE "$stage/"
  find "$stage/src" -name __pycache__ -prune -exec rm -rf {} + ; rm -rf "$stage"/src/*.egg-info
  cp scripts/flare_requirements.txt "$stage/requirements.txt"
  cp data/processed/*.parquet "$stage/data/processed/"
  tar --exclude=__pycache__ --exclude='*.egg-info' -czf "$TARBALL" -C "$stage" .
  rm -rf "$stage"
  echo "packaged $TARBALL ($(du -h "$TARBALL" | cut -f1))"
}

# remote_job <name> <setup-shell-command-or-empty> <archive-paths> <command...>
# Writes a Python file for `colab exec` that runs the command in the project folder and archives the outputs.
remote_job() {
  local name="$1" setup="$2" archive="$3"; shift 3
  local file="$JOBS/$name.py"
  mkdir -p "$JOBS"
  python3 - "$file" "$name" "$setup" "$archive" "$REMOTE_ROOT" "$@" <<'PY'
import json, pathlib, sys
path, name, setup, archive, root, *cmd = sys.argv[1:]
body = f'''import os, subprocess, pathlib
os.makedirs("/content/out", exist_ok=True)
if {setup!r}:
    subprocess.run({setup!r}, shell=True, check=True)
cmd = {json.dumps(cmd)}
print("+", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd={root!r}, check=True, env={{**os.environ, "PYTHONUNBUFFERED": "1"}})
subprocess.run("tar -czf /content/out/{name}.tgz -C {root} {archive}", shell=True, check=True)
print("archived /content/out/{name}.tgz", flush=True)
'''
pathlib.Path(path).write_text(body)
PY
  echo "$file"
}

fetch_job() {  # fetch_job <name>: download the archive of a finished remote job and unpack it locally
  run colab download "/content/out/$1.tgz" "$BUILD/$1.tgz" -s "$SESSION"
  run tar -xzf "$BUILD/$1.tgz" -C "$ROOT"
}

stage_setup() {
  if [[ "$DRY" == "1" ]] || ! colab ls -s "$SESSION" /content >/dev/null 2>&1; then
    run colab new -s "$SESSION" --gpu "$GPU"
  fi
  run colab upload "$TARBALL" /content/crossids-flare.tar.gz -s "$SESSION"
  mkdir -p "$JOBS"
  cat > "$JOBS/setup.py" <<'PY'
import os, subprocess, sys
def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
sh("rm -rf /content/flare && mkdir -p /content/flare && tar -xzf /content/crossids-flare.tar.gz -C /content/flare")
sh(f"{sys.executable} -m pip install -q -r /content/flare/requirements.txt || true")
sh(f"{sys.executable} -m pip install -q -e /content/flare")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
PY
  run colab exec -s "$SESSION" --timeout 1800 -f "$JOBS/setup.py"
}

train_job() {  # train_job <target> <seed> <suffix> <extra args...>
  local target="$1" seed="$2" suffix="$3"; shift 3
  local name="${target}_s${seed}${suffix}"
  if [[ -f "$RUNS/$name/summary.json" ]]; then
    echo "skip (done) $name"; return
  fi
  echo "train $name"
  local file
  file="$(remote_job "$name" "" "runs/flare/$name" \
          python -m crossids.flare.train --target "$target" --seed "$seed" --steps "$STEPS" \
          --out "runs/flare/$name" --device cuda "$@")"
  run colab exec -s "$SESSION" --timeout "$TIMEOUT" -f "$file"
  fetch_job "$name"
}

stage_train() {
  for t in $TARGETS; do
    for s in $SEEDS; do
      train_job "$t" "$s" ""
      if [[ "$PROTONET" == "1" ]]; then train_job "$t" "$s" "_protonet" --model protonet; fi
    done
    for a in $ABLATIONS; do
      train_job "$t" 0 "_$a" --ablation "$a"
    done
  done
}

stage_eval() {
  # checkpoints from earlier sessions are pushed once so the evaluation can run on this session
  if [[ -d "$RUNS" ]]; then
    run tar -czf "$BUILD/checkpoints.tgz" -C "$ROOT" --exclude='last.pt' runs/flare
    run colab upload "$BUILD/checkpoints.tgz" /content/checkpoints.tgz -s "$SESSION"
  fi
  for t in $TARGETS; do
    if [[ -f "$RUNS/eval_done_$t" ]]; then echo "skip (done) eval $t"; continue; fi
    echo "eval $t"
    local name="eval_$t"
    local file
    file="$(remote_job "$name" "mkdir -p /content/flare/runs && tar -xzf /content/checkpoints.tgz -C /content/flare 2>/dev/null || true" \
            "runs/flare/eval_$t.csv runs/flare/supports_${t}_s*.json" \
            python -m crossids.flare.evaluate --target "$t" --seeds $SEEDS --runs-dir runs/flare \
            --out "runs/flare/eval_$t.csv" --supports-dir runs/flare --device cuda)"
    run colab exec -s "$SESSION" --timeout "$TIMEOUT" -f "$file"
    fetch_job "$name"
    run touch "$RUNS/eval_done_$t"
  done
}

stage_stop() {
  run colab stop -s "$SESSION"
}

main() {
  local stages=("$@")
  if [[ ${#stages[@]} -eq 0 ]]; then stages=(package setup train eval stop); fi
  mkdir -p "$BUILD" "$RUNS"
  for st in "${stages[@]}"; do
    case "$st" in
      package) stage_package ;;
      setup) stage_setup ;;
      train) stage_train ;;
      eval) stage_eval ;;
      stop) stage_stop ;;
      *) echo "unknown stage: $st" >&2; exit 2 ;;
    esac
  done
}

main "$@"
