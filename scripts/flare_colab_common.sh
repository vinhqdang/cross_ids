# Shared by scripts/flare_colab_target.sh, scripts/flare_colab_targets.sh and scripts/flare_colab_remote.sh.
# Sourcing it defines variant_config, output_names and a colab wrapper that selects the Colab account.

# variant_config VARIANT: sets SUFFIX (run directory suffix), VTAG (artifact tag suffix, empty for f2), TRAIN_ARGS,
# EVAL_METHODS (binary evaluation) and CLASS_METHODS (class-level evaluation).
# f2 is the FLARE-2 model (DESIGN.md section 21); the other variants are its ablations and its context variant.
variant_config() {
  local base="catboost-target rf lr knn catboost-ft catboost-fulllabel catboost-lodo catboost-pairwise-mean catboost-pairwise-best"
  local bin
  case $1 in
    f2) SUFFIX=_f2; TRAIN_ARGS="--model flare2"; bin=flare2 ;;
    f2binary) SUFFIX=_f2_binary; TRAIN_ARGS="--model flare2 --tasks binary"; bin=flare2-binary ;;
    f2noperm) SUFFIX=_f2_noperm; TRAIN_ARGS="--model flare2 --no-slot-perm"; bin=flare2-noperm ;;
    f2insrc) SUFFIX=_f2_insrc; TRAIN_ARGS="--model flare2 --val-mode insource"; bin=flare2-insrc ;;
    f2ctx) SUFFIX=_f2_ctx; TRAIN_ARGS="--model flare2 --context-max 512"; bin=flare2-ctx ;;
    f2m) SUFFIX=_f2m; TRAIN_ARGS="--model flare2 --loss-mask-unsupported"; bin=flare2-m ;;
    *) echo "unknown FLARE-2 variant '$1' (f2 f2binary f2noperm f2insrc f2ctx f2m)" >&2; return 2 ;;
  esac
  if [[ $1 == f2 ]]; then
    VTAG=""; EVAL_METHODS="$base $bin"
  else
    VTAG="_$1"; EVAL_METHODS="$bin"
  fi
  CLASS_METHODS="catboost rf lr knn $bin"
}

# output_names TARGET SEED VARIANT: sets RUN (run directory name), TAG (archive/DONE/log name), EVAL_CSV, CLASS_CSV.
output_names() {
  variant_config "$3" || return
  RUN="${1}_s${2}${SUFFIX}"
  TAG="${1}_s${2}${VTAG}"
  if [[ -z $VTAG ]]; then
    EVAL_CSV="runs/flare/eval_${1}_all_s${2}.csv"; CLASS_CSV="runs/flare/class_${1}_s${2}.csv"
  else
    EVAL_CSV="runs/flare/eval_${1}_${3}_s${2}.csv"; CLASS_CSV="runs/flare/class_${1}_${3}_s${2}.csv"
  fi
}

# colab wrapper. Default account: the CLI's own ~/.config/colab-cli. With COLAB_ACCOUNT_DIR set, the call runs under
# that account: the CLI keeps its token at $HOME/.config/colab-cli/token.json (no option exists for the token path),
# and its sessions file is taken from --config. So HOME points at $COLAB_ACCOUNT_DIR/home, whose .config/colab-cli
# is a symlink back to COLAB_ACCOUNT_DIR. The account directory then holds token.json, sessions.json, settings.json.
# DRY_RUN=1 prints mutating calls instead of running them (`colab sessions` still runs, it only reads a local file).
colab() {
  local acc
  if [[ -n ${COLAB_ACCOUNT_DIR:-} ]]; then
    acc=$(readlink -f "$COLAB_ACCOUNT_DIR")
    mkdir -p "$acc/home/.config"
    [[ -e $acc/home/.config/colab-cli ]] || ln -s "$acc" "$acc/home/.config/colab-cli"
    if [[ ${DRY_RUN:-0} == 1 && ${1:-} != sessions ]]; then echo "DRY: COLAB_ACCOUNT_DIR=$acc colab $*"; return 0; fi
    env HOME="$acc/home" colab --config "$acc/sessions.json" "$@"
  else
    if [[ ${DRY_RUN:-0} == 1 && ${1:-} != sessions ]]; then echo "DRY: colab $*"; return 0; fi
    command colab "$@"
  fi
}
