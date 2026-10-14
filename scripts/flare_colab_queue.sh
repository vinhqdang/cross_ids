#!/usr/bin/env bash
# Train FLARE models on Colab one after another (GPU only; evaluation runs locally).
# Usage: scripts/flare_colab_queue.sh "TARGET:SEED:VARIANT ..."
set -uo pipefail
cd "$(dirname "$0")/.."
for job in $1; do
  IFS=: read -r t s v <<< "$job"
  for attempt in 1 2 3; do
    SKIP_EVAL=1 scripts/flare_colab_job.sh "$t" "$s" "$v" && [[ -f "runs/flare/${t}_s${s}$([[ $v == flare ]] || echo _$v)/summary.json" ]] && break
    echo "retry $job ($attempt)" >> runs/colab/queue.log
    sleep 300
  done
done
