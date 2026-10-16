#!/usr/bin/env bash
# Remote half of scripts/flare_colab_target.sh. Runs inside one Colab cell, from the unpacked package root:
# train (resumes from last.pt; skipped when summary.json exists), binary evaluation, class-level evaluation, archive.
# Usage: bash scripts/flare_colab_remote.sh TARGET SEED VARIANT [STEPS]
# At the end it writes $OUT_DIR/out_TAG.tgz and $OUT_DIR/DONE_TAG (containing the exit code). OUT_DIR defaults to /content.
set -uo pipefail
TARGET=$1
SEED=$2
VARIANT=${3:-f2}
STEPS=${4:-20000}
OUT_DIR=${OUT_DIR:-/content}
cd "$(dirname "$0")/.."
. scripts/flare_colab_common.sh
output_names "$TARGET" "$SEED" "$VARIANT" || exit 2

step() {  # step NAME command...
  local name=$1
  shift
  echo "== $name $(date -u +%FT%TZ)"
  "$@"
  local rc=$?
  echo "== $name exit $rc $(date -u +%FT%TZ)"
  return $rc
}

pipeline() {
  if [[ -f runs/flare/$RUN/summary.json ]]; then
    echo "== train skipped: runs/flare/$RUN/summary.json exists"
  else
    step train python3 -m crossids.flare.train --target "$TARGET" --seed "$SEED" --steps "$STEPS" \
      --device cuda --num-workers 2 $TRAIN_ARGS --out "runs/flare/$RUN" || return
  fi
  step eval python3 -m crossids.flare.evaluate --target "$TARGET" --seeds "$SEED" --methods $EVAL_METHODS \
    --device cuda --threads 2 --out "$EVAL_CSV" --supports-dir runs/flare || return
  step class python3 -m crossids.flare.evaluate_class --target "$TARGET" --seeds "$SEED" \
    --methods $CLASS_METHODS --tasks perclass nway --ks 2 4 8 16 --draws 5 \
    --device cuda --threads 2 --out "$CLASS_CSV" || return
}

archive() {
  shopt -s nullglob
  local -a items=()
  [[ -d runs/flare/$RUN ]] && items+=("runs/flare/$RUN")
  items+=(runs/flare/supports_"${TARGET}"_s"${SEED}"*.json runs/flare/class_supports_"${TARGET}"_s"${SEED}"*.json)
  shopt -u nullglob
  local f
  for f in "$EVAL_CSV" "$CLASS_CSV"; do
    [[ -e $f ]] && items+=("$f")
  done
  if [[ ${#items[@]} -eq 0 ]]; then echo "nothing to archive"; return 1; fi
  tar czf "$OUT_DIR/out_$TAG.tgz" "${items[@]}" || return 1
  echo "archived ${#items[@]} items to $OUT_DIR/out_$TAG.tgz"
}

pipeline
rc=$?
if ! archive; then
  [[ $rc == 0 ]] && rc=1
fi
echo "$rc" > "$OUT_DIR/DONE_$TAG"
echo "== finished rc=$rc $(date -u +%FT%TZ)"
exit $rc
