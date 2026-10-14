#!/usr/bin/env bash
# Train one FLARE model on a Colab GPU and evaluate it, keeping the kernel busy
# for the whole job (free Colab runtimes are reclaimed when the kernel is idle,
# so training runs as a foreground cell rather than a detached process).
#
# Usage: scripts/flare_colab_job.sh TARGET SEED [VARIANT] [STEPS]
#   VARIANT: flare (default) | ctx (FLARE-U, section 20.1) | noA1 | noA3 | rowmlp | synthetic | protonet
#            f2 (FLARE-2, section 21) | f2binary | f2noperm | f2insrc | f2ctx (its ablations and context variant)
#   SKIP_EVAL=1 trains only (the run directory is downloaded, evaluation happens locally with EVAL_METHODS).
# Outputs: runs/flare/<target>_s<seed>[_variant]/ and runs/flare/eval_<target>_<variant>_s<seed>.csv
set -euo pipefail

TARGET=$1
SEED=$2
VARIANT=${3:-flare}
STEPS=${4:-20000}
GPU=${GPU:-T4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$ROOT/runs/colab}
mkdir -p "$WORK" "$ROOT/runs/flare"

case $VARIANT in
  flare) SUFFIX=""; TRAIN_ARGS=""; EVAL_METHODS="flare flare-E1 flare-al flare-kcenter-raw" ;;
  ctx) SUFFIX="_ctx"; TRAIN_ARGS="--context-max 512"; EVAL_METHODS="flare-u" ;;
  protonet) SUFFIX="_protonet"; TRAIN_ARGS="--model protonet"; EVAL_METHODS="protonet" ;;
  f2) SUFFIX="_f2"; TRAIN_ARGS="--model flare2"; EVAL_METHODS="flare2" ;;
  f2binary) SUFFIX="_f2_binary"; TRAIN_ARGS="--model flare2 --tasks binary"; EVAL_METHODS="flare2-binary" ;;
  f2noperm) SUFFIX="_f2_noperm"; TRAIN_ARGS="--model flare2 --no-slot-perm"; EVAL_METHODS="flare2-noperm" ;;
  f2insrc) SUFFIX="_f2_insrc"; TRAIN_ARGS="--model flare2 --val-mode insource"; EVAL_METHODS="flare2-insrc" ;;
  f2ctx) SUFFIX="_f2_ctx"; TRAIN_ARGS="--model flare2 --context-max 512"; EVAL_METHODS="flare2-ctx" ;;
  noA1|noA3|rowmlp|synthetic) SUFFIX="_$VARIANT"; TRAIN_ARGS="--ablation $VARIANT"; EVAL_METHODS="flare-$VARIANT" ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

RUN="${TARGET}_s${SEED}${SUFFIX}"
SESSION="j-${RUN//_/-}"
LOG="$WORK/$RUN.log"
PKG="$WORK/pkg.tgz"
EVAL_CSV="runs/flare/eval_${TARGET}_${VARIANT}_s${SEED}.csv"

SKIP_EVAL=${SKIP_EVAL:-0}  # 1: train on the GPU only and evaluate locally
if [[ -f "$ROOT/runs/flare/$RUN/summary.json" && ( "$SKIP_EVAL" == 1 || -f "$ROOT/$EVAL_CSV" ) ]]; then
  echo "$RUN already done"; exit 0
fi

remote() {  # remote TIMEOUT SHELL_COMMAND
  python3 - "$1" "$2" <<'PY' | colab exec -s "$SESSION" --timeout "$1" >> "$LOG" 2>&1
import sys
cmd = sys.argv[2]
print("import subprocess, sys")
print(f"r = subprocess.run({cmd!r} + ' 2>&1', shell=True, capture_output=True, text=True)")
print("print(r.stdout[-6000:])")
print("print('RC', r.returncode)")
print("assert r.returncode == 0")
PY
}

cleanup() { colab stop -s "$SESSION" >> "$LOG" 2>&1 || true; }
trap cleanup EXIT

echo "== $RUN start $(date -u +%FT%TZ)" >> "$LOG"
(cd "$ROOT" && tar czf "$PKG" --exclude=__pycache__ pyproject.toml src data/processed)
colab new -s "$SESSION" --gpu "$GPU" >> "$LOG" 2>&1
colab upload -s "$SESSION" "$PKG" /content/pkg.tgz >> "$LOG" 2>&1
# Resume support: ship an existing partial run directory back to the VM.
if [[ -d "$ROOT/runs/flare/$RUN" ]]; then
  (cd "$ROOT" && tar czf "$WORK/$RUN.partial.tgz" "runs/flare/$RUN")
  colab upload -s "$SESSION" "$WORK/$RUN.partial.tgz" /content/partial.tgz >> "$LOG" 2>&1
fi
remote 900 "cd /content && rm -rf cx && mkdir cx && tar xzf pkg.tgz -C cx && cd cx && (test -f /content/partial.tgz && tar xzf /content/partial.tgz || true) && pip install -q -e . catboost"
# Mirror checkpoints locally while training, so a reclaimed VM can be resumed.
mirror() {
  mkdir -p "$ROOT/runs/flare/$RUN"
  while true; do
    sleep 600
    for f in last.pt best.pt log.jsonl run.json; do
      colab download -s "$SESSION" "/content/cx/runs/flare/$RUN/$f" "$ROOT/runs/flare/$RUN/$f.tmp" >> "$LOG" 2>&1 \
        && mv "$ROOT/runs/flare/$RUN/$f.tmp" "$ROOT/runs/flare/$RUN/$f" || rm -f "$ROOT/runs/flare/$RUN/$f.tmp"
    done
  done
}
mirror & MIRROR=$!
trap 'kill $MIRROR 2>/dev/null || true; cleanup' EXIT
[[ -f "$ROOT/runs/flare/$RUN/summary.json" ]] || remote 14400 "cd /content/cx && python -m crossids.flare.train --target $TARGET --seed $SEED --steps $STEPS --device cuda --num-workers 2 $TRAIN_ARGS --out runs/flare/$RUN"
kill $MIRROR 2>/dev/null || true
if [[ "$SKIP_EVAL" == 1 ]]; then
  remote 600 "cd /content/cx && tar czf /content/out.tgz runs/flare/$RUN"
else
remote 7200 "cd /content/cx && python -m crossids.flare.evaluate --target $TARGET --seeds $SEED --methods $EVAL_METHODS --device cuda --threads 2 --out $EVAL_CSV --supports-dir runs/flare"
remote 600 "cd /content/cx && tar czf /content/out.tgz runs/flare/$RUN $EVAL_CSV runs/flare/supports_${TARGET}_s${SEED}.json"
fi
colab download -s "$SESSION" /content/out.tgz "$WORK/$RUN.out.tgz" >> "$LOG" 2>&1
(cd "$ROOT" && tar xzf "$WORK/$RUN.out.tgz")
echo "== $RUN done $(date -u +%FT%TZ)" >> "$LOG"
