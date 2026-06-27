#!/usr/bin/env bash
# (Super)GLUE fine-tune for the faithful replica, via the BabyLM finetune harness.
# Usage: run_glue.sh <hf_model_dir> <results_dir> <log_dir>
#
# NOTE: unlike the zero-shot tasks (run_zero_shot.sh), GLUE is a fine-tune — 7 tasks,
# 10-30 epochs each — so it realistically needs a GPU (we ran it on RunPod GPUs), not
# an Apple-silicon Mac. The command below is the same one the RunPod worker shelled out
# to; it runs on any single CUDA GPU (a RunPod Pod, Colab, or a local box).
#
# Macro GLUE = mean of each task's PRIMARY metric (accuracy; F1 for MRPC/QQP).
# Hyperparams (matching the reported run): lr 3e-5, seed 42, sequence_length 512,
# padding_side left, take_final.
set -uo pipefail
P="${BABYLM_PIPELINE:?set BABYLM_PIPELINE to the babylm-eval-pipeline/strict dir (data downloaded)}"
PY="${PYTHON:-python}"
MODEL="${1:?hf_model_dir}"
RES="${2:?results_dir}"
LOG="${3:?log_dir}"
mkdir -p "$RES" "$LOG"
export PYTHONPATH="$P"
FAIL=()

# task        num_labels  batch  epochs  metric_for_valid  metrics
TASKS=(
  "boolq      2  16  10  accuracy  accuracy f1 mcc"
  "multirc    2  16  10  accuracy  accuracy f1 mcc"
  "rte        2  32  10  accuracy  accuracy f1 mcc"
  "wsc        2  32  30  accuracy  accuracy f1 mcc"
  "mrpc       2  32  10  f1        accuracy f1 mcc"
  "qqp        2  32  10  f1        accuracy f1 mcc"
  "mnli       3  32  10  accuracy  accuracy"
)

for row in "${TASKS[@]}"; do
  read -r task nl bsz ep mfv metrics <<< "$row"
  echo "[$(date +%H:%M:%S)] START $task (num_labels=$nl bsz=$bsz epochs=$ep)"
  if (cd "$P" && "$PY" -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL" \
        --train_data   "evaluation_data/full_eval/glue_filtered/$task.train.jsonl" \
        --valid_data   "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --predict_data "evaluation_data/full_eval/glue_filtered/$task.valid.jsonl" \
        --task "$task" --num_labels "$nl" --batch_size "$bsz" \
        --learning_rate 3e-5 --num_epochs "$ep" --sequence_length 512 \
        --results_dir "$RES" --metrics $metrics --metric_for_valid "$mfv" \
        --seed 42 --verbose --padding_side left --take_final) > "$LOG/$task.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK    $task"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $task (see $LOG/$task.log)"; FAIL+=("$task")
  fi
done

echo "=========================================="
echo "[$(date +%H:%M:%S)] GLUE FINE-TUNE DONE. failures: ${FAIL[*]:-none}"
echo "Results under $RES/<model>/main/finetune/<task>/. Macro = mean of per-task primary metric."
if [ "${#FAIL[@]}" -gt 0 ]; then exit 1; fi
