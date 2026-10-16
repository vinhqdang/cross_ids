#!/usr/bin/env bash
# FLARE-2 on Colab for one (target, seed). One remote cell trains (resuming when possible), runs the binary and the
# class-level evaluation and archives the results, so the cell finishes even when this local side is gone.
# The local side only launches that cell and polls it. It can be restarted at any time: running the same command
# again attaches to the existing session (no second session, no relaunch) and keeps polling.
#
# Usage: scripts/flare_colab_target.sh TARGET SEED [VARIANT]
#   VARIANT: f2 (default) | f2binary | f2noperm | f2insrc | f2ctx
# Environment:
#   COLAB_ACCOUNT_DIR   run under an alternate Colab account (see scripts/flare_colab_common.sh); unset = default account
#   GPU [T4]  STEPS [20000]  POLL [300] seconds between checks  MAX_ATTEMPTS [3] session losses before giving up
#   DRY_RUN=1  print the plan and the colab commands instead of running them
#   PACK_ONLY=1  build the upload package and stop
#   WORK [runs/colab]  local folder for logs, the package, the cell files and downloads
# Outputs (local, extracted from the archive): runs/flare/<TARGET>_s<SEED><SUFFIX>/, eval and class CSVs,
#   supports JSONs; runs/flare/.colab_done_<TAG> marks a finished target.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TARGET=${1:?usage: $0 TARGET SEED [VARIANT]}
SEED=${2:?usage: $0 TARGET SEED [VARIANT]}
VARIANT=${3:-f2}
GPU=${GPU:-T4}
STEPS=${STEPS:-20000}
POLL=${POLL:-300}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
DRY_RUN=${DRY_RUN:-0}
WORK=${WORK:-$ROOT/runs/colab}
. scripts/flare_colab_common.sh
output_names "$TARGET" "$SEED" "$VARIANT"

NAME="tg-${RUN//_/-}"          # Colab session name for this job
OLD="j-${RUN//_/-}"            # session name used by scripts/flare_colab_job.sh (must not run at the same time)
PKG="$WORK/pkg_$TAG.tgz"
LOG="$WORK/$TAG.log"
SETUP="$WORK/$TAG.setup.py"
CELL="$WORK/$TAG.cell.py"
RUN_DIR="runs/flare/$RUN"
DONE_MARK="runs/flare/.colab_done_$TAG"
mkdir -p "$WORK" runs/flare

log() { echo "[$(date -u +%FT%TZ)] $TAG: $*" | tee -a "$LOG"; }
session_exists() {  # judged by the listing text only (the CLI's exit status is not used)
  local listing
  listing=$(colab sessions 2>/dev/null) || true
  grep -qF "[$1] " <<< "$listing"
}

package() {  # builds $PKG: the code, the Part I benchmark runs, and this target's existing run and supports files
  shopt -s nullglob
  local -a extra=()
  [[ -d $RUN_DIR ]] && extra+=("$RUN_DIR")
  extra+=(runs/flare/supports_"$TARGET"_s"$SEED"*.json runs/flare/class_supports_"$TARGET"_s"$SEED"*.json)
  shopt -u nullglob
  if [[ $DRY_RUN == 1 ]]; then
    echo "DRY: tar czf $PKG pyproject.toml src data/processed results/runs.csv scripts/flare_colab_common.sh scripts/flare_colab_remote.sh ${extra[*]}"
    return 0
  fi
  tar czf "$PKG" --exclude=__pycache__ --exclude='*.egg-info' --exclude='*.tmp' \
    pyproject.toml src data/processed results/runs.csv scripts/flare_colab_common.sh scripts/flare_colab_remote.sh \
    "${extra[@]}"
  log "packaged $PKG ($(du -h "$PKG" | cut -f1))"
}

write_cell_files() {
  cat > "$SETUP" <<PY
import subprocess, sys
def sh(cmd):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        raise SystemExit(r.returncode)
sh("rm -rf /content/flare_$TAG && mkdir -p /content/flare_$TAG && tar -xzf /content/pkg_$TAG.tgz -C /content/flare_$TAG")
sh(f"cd /content/flare_$TAG && {sys.executable} -m pip install -q -e . catboost")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
PY
  cat > "$CELL" <<PY
import subprocess
from pathlib import Path
TAG = "$TAG"
done = Path(f"/content/DONE_{TAG}")
done.unlink(missing_ok=True)
with open(f"/content/log_{TAG}.txt", "a") as log:
    log.write("== cell start\\n")
    log.flush()
    r = subprocess.run(["bash", "scripts/flare_colab_remote.sh", "$TARGET", "$SEED", "$VARIANT", "$STEPS"],
                       cwd=f"/content/flare_{TAG}", stdout=log, stderr=subprocess.STDOUT)
    log.write(f"== cell exit {r.returncode}\\n")
if not done.exists():
    done.write_text(str(r.returncode))
print("cell exit", r.returncode, "DONE", done.read_text().strip(), flush=True)
PY
}

create_and_launch() {  # new session, upload, setup cell, then the pipeline cell detached from this process
  write_cell_files
  package || return 1  # rebuilt for every attempt, so a retry carries the mirrored last.pt
  colab stop -s "$NAME" >/dev/null 2>&1 || true
  log "creating session $NAME (GPU $GPU)"
  colab new -s "$NAME" --gpu "$GPU" >> "$LOG" 2>&1 || return 1
  colab upload -s "$NAME" "$PKG" "/content/pkg_$TAG.tgz" >> "$LOG" 2>&1 || return 1
  log "setup cell (pip install -e . catboost)"
  colab exec -s "$NAME" --timeout 1800 -f "$SETUP" >> "$LOG" 2>&1 || return 1
  log "launching the pipeline cell (train, binary eval, class eval, archive)"
  # fd 9 (the lock) is closed for the detached child, so the lock is released when this script exits
  colab exec -s "$NAME" --timeout 43200 -f "$CELL" >> "$LOG" 2>&1 9>&- &
  disown $! 2>/dev/null || true
}

mirror() {  # copies the resumable state of the run back to this machine (best effort)
  local f
  mkdir -p "$RUN_DIR"
  for f in last.pt best.pt log.jsonl summary.json; do
    if colab download -s "$NAME" "/content/flare_$TAG/$RUN_DIR/$f" "$RUN_DIR/$f.tmp" >/dev/null 2>&1 && [[ -s "$RUN_DIR/$f.tmp" ]]; then
      mv -f "$RUN_DIR/$f.tmp" "$RUN_DIR/$f"
    else
      rm -f "$RUN_DIR/$f.tmp"
    fi
  done
  colab download -s "$NAME" "/content/log_$TAG.txt" "$WORK/$TAG.remote.log" >/dev/null 2>&1 || true
}

# poll: 0 when DONE appeared (exit code in REMOTE_RC), 2 when the session is gone or its kernel stopped without DONE
poll() {
  local n=0 idle=0 st
  rm -f "$WORK/$TAG.done"
  while sleep "$POLL"; do
    n=$((n + 1))
    if ! session_exists "$NAME"; then log "session $NAME is gone"; return 2; fi
    if colab download -s "$NAME" "/content/DONE_$TAG" "$WORK/$TAG.done" >/dev/null 2>&1 && [[ -s "$WORK/$TAG.done" ]]; then
      REMOTE_RC=$(tr -d '[:space:]' < "$WORK/$TAG.done")
      return 0
    fi
    if (( n % 2 == 0 )); then mirror; fi
    if st=$(colab status -s "$NAME" 2>&1); then
      if [[ $st == *BUSY* ]]; then idle=0; else idle=$((idle + 1)); fi
    fi
    log "poll $n: $(tail -n1 "$WORK/$TAG.remote.log" 2>/dev/null | cut -c1-200) | status: $(echo "$st" | head -n1 | cut -c1-160)"
    if (( idle >= 3 )); then log "kernel not busy and no DONE for 15 min"; return 2; fi
  done
}

finish() {  # finish RC: download the archive and the log, stop the session, unpack into the repo
  local rc=$1
  colab download -s "$NAME" "/content/out_$TAG.tgz" "$WORK/$TAG.out.tgz" >> "$LOG" 2>&1 || log "download of out_$TAG.tgz failed"
  colab download -s "$NAME" "/content/log_$TAG.txt" "$WORK/$TAG.remote.log" >> "$LOG" 2>&1 || true
  colab stop -s "$NAME" >> "$LOG" 2>&1 || true
  if [[ -s "$WORK/$TAG.out.tgz" ]]; then
    if tar -tzf "$WORK/$TAG.out.tgz" | grep -Eq '^/|(^|/)\.\.(/|$)'; then
      log "refusing an archive with absolute or parent paths"; return 1
    fi
    tar -xzf "$WORK/$TAG.out.tgz" -C "$ROOT"
    log "extracted $WORK/$TAG.out.tgz into $ROOT"
  fi
  if [[ $rc == 0 ]]; then
    touch "$DONE_MARK"
    log "done (remote exit code 0)"
    return 0
  fi
  log "remote pipeline failed (exit code $rc); see $WORK/$TAG.remote.log"
  return 1
}

if [[ -f $DONE_MARK ]]; then echo "$TAG already archived ($DONE_MARK)"; exit 0; fi
if [[ ${PACK_ONLY:-0} == 1 ]]; then package; exit 0; fi
exec 9>"$WORK/$TAG.lock"
flock -n 9 || { echo "$TAG: another local process holds $WORK/$TAG.lock" >&2; exit 4; }
if session_exists "$OLD"; then
  log "session $OLD (scripts/flare_colab_job.sh) is still running; rerun this script after it has finished"
  exit 3
fi

if [[ $DRY_RUN == 1 ]]; then
  if session_exists "$NAME"; then
    log "DRY: session $NAME exists, would poll every ${POLL}s until /content/DONE_$TAG appears"
  else
    log "DRY: no session $NAME, would create one:"
    create_and_launch
    log "DRY: then poll every ${POLL}s until /content/DONE_$TAG appears (session losses retry up to $MAX_ATTEMPTS times)"
  fi
  exit 0
fi

attempt=0
REMOTE_RC=""
while true; do
  if session_exists "$NAME"; then
    log "session $NAME exists: polling, not relaunching"
  else
    attempt=$((attempt + 1))
    if (( attempt > MAX_ATTEMPTS )); then log "giving up after $MAX_ATTEMPTS attempts"; exit 1; fi
    log "attempt $attempt of $MAX_ATTEMPTS"
    if ! create_and_launch; then
      log "create or launch failed; retrying after ${POLL}s"
      colab stop -s "$NAME" >> "$LOG" 2>&1 || true
      sleep "$POLL"
      continue
    fi
  fi
  if poll; then
    finish "$REMOTE_RC"
    exit $?
  fi
  log "session lost before DONE: stopping it; the next attempt resumes from the mirrored last.pt"
  colab stop -s "$NAME" >> "$LOG" 2>&1 || true
done
