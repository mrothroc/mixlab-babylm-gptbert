#!/usr/bin/env bash
# The 6 zero-shot tasks (BLiMP, BLiMP-supplement, EWoK, entity_tracking, COMPS, reading) for the
# faithful replica, mntp backend, FULL subsets. (Super)GLUE is a separate fine-tune step; AoA is not run.
# Usage: run_zero_shot.sh <hf_model_dir> <results_dir> <log_dir>
set -uo pipefail
# Point these at your checkout of the official BabyLM eval pipeline + a python with its deps.
P="${BABYLM_PIPELINE:?set BABYLM_PIPELINE to the babylm-eval-pipeline/strict dir}"
PY="${PYTHON:-python}"
MODEL="${1:?hf_model_dir}"
RES="${2:?results_dir}"
LOG="${3:?log_dir}"
BACKEND=mntp
mkdir -p "$RES" "$LOG"
export PYTHONPATH="$P"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
FAIL=()

run() {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] START $name"
  if (cd "$P" && "$@") > "$LOG/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $name"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $name (see $LOG/$name.log)"; FAIL+=("$name")
  fi
}

ZS() { # task data_subdir
  run "$1-$2" "$PY" -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL" --revision_name main --backend "$BACKEND" \
    --task "$1" --data_path "$P/evaluation_data/full_eval/$2" \
    --output_dir "$RES" --save_predictions
}

ZS blimp           blimp_filtered
ZS blimp           supplement_filtered
ZS ewok            ewok_filtered
ZS entity_tracking entity_tracking
ZS comps           comps
run reading "$PY" -m evaluation_pipeline.reading.run \
  --model_path_or_name "$MODEL" --revision_name main --backend "$BACKEND" \
  --data_path "$P/evaluation_data/full_eval/reading/reading_data.csv" \
  --output_dir "$RES"

echo "=========================================="
echo "[$(date +%H:%M:%S)] ZERO-SHOT BATCH DONE. failures: ${FAIL[*]:-none}"
if [ "${#FAIL[@]}" -gt 0 ]; then exit 1; fi
