#!/usr/bin/env bash
# FLARE on Google Colab (GPU): packages the project, runs every (target, seed) training job, the ablations and
# ProtoNet, evaluates each target, then builds the report locally.
#
# Each job starts detached on the VM (nohup, log file on the VM). The local side polls it with short
# `colab exec --timeout 60` calls and downloads the job's archive when it finishes, so no call blocks for long.
# Training resumes from last.pt when a job is relaunched after the VM or session was lost. Rerunning this script
# skips jobs whose outputs already exist locally.
#
# Needs the google-colab-cli (`colab`) with a logged-in account.
#
# Usage:
#   scripts/flare_colab.sh [stage ...]             stages: package setup train eval report stop (default: all)
#   FLARE_DRY_RUN=1 scripts/flare_colab.sh train   print the colab commands without running them
#
# Environment (defaults in brackets):
#   FLARE_SESSION [flare]      Colab session name            FLARE_GPU [T4]           accelerator for `colab new`
#   FLARE_STEPS [20000]        training steps per run        FLARE_SEEDS ["0 1 2"]    meta-training seeds
#   FLARE_TARGETS [all 8]      FLARE_ABLATIONS ["noA1 noA3 rowmlp synthetic"]       ablations (seed 0 only)
#   FLARE_PROTONET [1]         also meta-train ProtoNet      FLARE_QUERY [256]        query size
#   FLARE_WORKERS [2]          DataLoader workers            FLARE_COMPILE [0]        1 = --compile
#   FLARE_POLL [60]            seconds between polls         FLARE_BENCHMARK [results] Part I folder for the report
#
# Outputs (local): runs/flare/<target>_s<seed>[_<ablation>|_protonet]/ (best.pt, last.pt, log.jsonl, summary.json),
#                  runs/flare/eval_<target>.csv, runs/flare/per_attack_<target>.csv, runs/flare/supports_*.json,
#                  results/flare/ (report).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SESSION="${FLARE_SESSION:-flare}"
GPU="${FLARE_GPU:-T4}"
STEPS="${FLARE_STEPS:-20000}"
SEEDS="${FLARE_SEEDS:-0 1 2}"
TARGETS="${FLARE_TARGETS:-cicids2017 cicids2018 usbids1 unswnb15 toniot iot23 insdn ctu13}"
ABLATIONS="${FLARE_ABLATIONS:-noA1 noA3 rowmlp synthetic}"
PROTONET="${FLARE_PROTONET:-1}"
QUERY="${FLARE_QUERY:-256}"
WORKERS="${FLARE_WORKERS:-2}"
COMPILE="${FLARE_COMPILE:-0}"
POLL="${FLARE_POLL:-60}"
BENCHMARK="${FLARE_BENCHMARK:-results}"
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
  mkdir -p "$BUILD"
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
sh("rm -rf /content/flare && mkdir -p /content/flare /content/out && tar -xzf /content/crossids-flare.tar.gz -C /content/flare")
sh(f"{sys.executable} -m pip install -q -r /content/flare/requirements.txt || true")
sh(f"{sys.executable} -m pip install -q -e /content/flare")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
PY
  run colab exec -s "$SESSION" --timeout 60 -f "$JOBS/setup.py"
}

# launch_job <name> <archive-paths> <command...>
# Writes a launcher that starts the command detached on the VM (idempotent: nothing happens if it is running or done).
launch_job() {
  local name="$1" archive="$2"; shift 2
  local file="$JOBS/launch_$name.py"
  mkdir -p "$JOBS"
  python3 - "$file" "$name" "$archive" "$REMOTE_ROOT" "$@" <<'PY'
import json, pathlib, shlex, sys
path, name, archive, root, *cmd = sys.argv[1:]
body = f'''import os, subprocess, pathlib
out = pathlib.Path("/content/out"); out.mkdir(parents=True, exist_ok=True)
rc, pid = out / "{name}.rc", out / "{name}.pid"
def alive(p):
    try:
        os.kill(int(p.read_text()), 0); return True
    except Exception:
        return False
if rc.exists():
    print("DONE already", rc.read_text().strip())
elif pid.exists() and alive(pid):
    print("RUNNING", pid.read_text().strip())
else:
    script = "cd {root} && {' '.join(shlex.quote(c) for c in cmd)}; s=$?; " \\
             "tar -czf /content/out/{name}.tgz -C {root} {archive}; echo $s > /content/out/{name}.rc"
    logf = open(out / "{name}.log", "a")
    p = subprocess.Popen(["bash", "-c", script], stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    pid.write_text(str(p.pid))
    print("LAUNCHED", p.pid, flush=True)
'''
pathlib.Path(path).write_text(body)
PY
  run colab exec -s "$SESSION" --timeout 60 -f "$file"
}

# poll_job <name>: prints DONE <rc> when the job has finished, otherwise the last log lines
poll_job() {
  local name="$1"
  local file="$JOBS/poll_$name.py"
  mkdir -p "$JOBS"
  cat > "$file" <<PY
import pathlib, subprocess
out = pathlib.Path("/content/out")
rc = out / "$name.rc"
if rc.exists():
    print("DONE", rc.read_text().strip(), flush=True)
else:
    log = out / "$name.log"
    tail = subprocess.run(["tail", "-n", "2", "$REMOTE_ROOT/runs/flare/$name/log.jsonl"],
                          capture_output=True, text=True).stdout
    print("RUNNING", tail.strip(), (log.read_text()[-200:] if log.exists() else ""), flush=True)
PY
  run colab exec -s "$SESSION" --timeout 60 -f "$file"
}

# run_job <name> <archive-paths> <command...>: launch, poll until finished, download and unpack the archive
run_job() {
  local name="$1" archive="$2"; shift 2
  launch_job "$name" "$archive" "$@"
  if [[ "$DRY" == "1" ]]; then
    run colab download "/content/out/$name.tgz" "$BUILD/$name.tgz" -s "$SESSION"
    return
  fi
  while true; do
    local status
    status="$(colab exec -s "$SESSION" --timeout 60 -f "$JOBS/poll_$name.py" 2>&1 || true)"
    if grep -q "^DONE" <<<"$status"; then
      echo "$status" | grep "^DONE" | sed "s/^/$name: /"
      break
    fi
    echo "$name: $(grep -v '^$' <<<"$status" | tail -n 1 | cut -c1-160)"
    sleep "$POLL"
  done
  run colab download "/content/out/$name.tgz" "$BUILD/$name.tgz" -s "$SESSION"
  run tar -xzf "$BUILD/$name.tgz" -C "$ROOT"
}

train_job() {  # train_job <target> <seed> <suffix> <extra args...>
  local target="$1" seed="$2" suffix="$3"; shift 3
  local name="${target}_s${seed}${suffix}"
  if [[ -f "$RUNS/$name/summary.json" ]]; then
    echo "skip (done) $name"; return
  fi
  local extra=()
  if [[ "$COMPILE" == "1" ]]; then extra+=(--compile); fi
  echo "train $name"
  run_job "$name" "runs/flare/$name" \
    python -m crossids.flare.train --target "$target" --seed "$seed" --steps "$STEPS" --query-size "$QUERY" \
    --num-workers "$WORKERS" --out "runs/flare/$name" --device cuda "${extra[@]}" "$@"
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
  # checkpoints from earlier sessions are pushed once so the evaluation can run on this session (no last.pt)
  if [[ -d "$RUNS" ]]; then
    run tar -czf "$BUILD/checkpoints.tgz" -C "$ROOT" --exclude='last.pt' runs/flare
    run colab upload "$BUILD/checkpoints.tgz" /content/checkpoints.tgz -s "$SESSION"
  fi
  for t in $TARGETS; do
    if [[ -f "$RUNS/eval_done_$t" ]]; then echo "skip (done) eval $t"; continue; fi
    echo "eval $t"
    run_job "eval_$t" "runs/flare/eval_$t.csv runs/flare/per_attack_$t.csv runs/flare/supports_${t}_s*.json" \
      bash -c "mkdir -p /content/flare/runs/flare && tar -xzf /content/checkpoints.tgz -C /content/flare 2>/dev/null; \
      python -m crossids.flare.evaluate --target $t --seeds $SEEDS --runs-dir runs/flare \
      --out runs/flare/eval_$t.csv --supports-dir runs/flare --device cuda"
    run touch "$RUNS/eval_done_$t"
  done
}

stage_report() {
  run python -m crossids.flare.report --runs "$RUNS" --out "$ROOT/results/flare" --benchmark "$BENCHMARK"
}

stage_stop() {
  run colab stop -s "$SESSION"
}

main() {
  local stages=("$@")
  if [[ ${#stages[@]} -eq 0 ]]; then stages=(package setup train eval report stop); fi
  mkdir -p "$BUILD" "$RUNS"
  for st in "${stages[@]}"; do
    case "$st" in
      package) stage_package ;;
      setup) stage_setup ;;
      train) stage_train ;;
      eval) stage_eval ;;
      report) stage_report ;;
      stop) stage_stop ;;
      *) echo "unknown stage: $st" >&2; exit 2 ;;
    esac
  done
}

main "$@"
