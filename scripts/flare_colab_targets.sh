#!/usr/bin/env bash
# Runs scripts/flare_colab_target.sh for several targets one after another, for one Colab account per invocation
# (set COLAB_ACCOUNT_DIR to pick the account). Targets whose archive was already extracted are skipped.
# Usage: scripts/flare_colab_targets.sh "T1 T2 ..." [SEED] [VARIANT]
# Example: COLAB_ACCOUNT_DIR=$HOME/.config/colab-cli-acc1 scripts/flare_colab_targets.sh "cicids2018 ctu13"
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TARGETS=${1:?usage: $0 "T1 T2 ..." [SEED] [VARIANT]}
SEED=${2:-0}
VARIANT=${3:-f2}
. scripts/flare_colab_common.sh
failed=()
for t in $TARGETS; do
  output_names "$t" "$SEED" "$VARIANT" || exit 2
  if [[ -f runs/flare/.colab_done_$TAG ]]; then
    echo "skip $t: runs/flare/.colab_done_$TAG exists"
    continue
  fi
  echo "== $(date -u +%FT%TZ) target $t seed $SEED variant $VARIANT ${COLAB_ACCOUNT_DIR:+account $COLAB_ACCOUNT_DIR}"
  if ! "$ROOT/scripts/flare_colab_target.sh" "$t" "$SEED" "$VARIANT"; then
    failed+=("$t")
  fi
done
echo "== $(date -u +%FT%TZ) finished; failed: ${failed[*]:-none}"
[[ ${#failed[@]} -eq 0 ]]
