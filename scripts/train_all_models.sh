#!/usr/bin/env bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(realpath "$script_dir/..")"
cd "$repo_root"

mkdir -p runs/batch_logs
batch_log="runs/batch_logs/train_all_$(date +%Y%m%d_%H%M%S).log"

weights_list=(yolo26n yolo12n yolo11n yolov10n yolov9c yolov8n yolov7-tiny yolov6n yolov5n)

if [ "$#" -gt 0 ]; then
    weights_list=("$@")
fi

echo "Batch started: $(date)" | tee "$batch_log"
echo "Models: ${weights_list[*]}" | tee -a "$batch_log"
echo "Eval split: test" | tee -a "$batch_log"

for weights in "${weights_list[@]}"; do
    echo "" | tee -a "$batch_log"
    echo "==> $weights started: $(date)" | tee -a "$batch_log"

    uv run python src/run_experiment.py \
        --weights "$weights" \
        --eval-split test \
        2>&1 | tee -a "$batch_log"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo "==> $weights failed: $(date)" | tee -a "$batch_log"
        exit 1
    fi

    echo "==> $weights finished: $(date)" | tee -a "$batch_log"
done

echo "" | tee -a "$batch_log"
echo "Batch finished: $(date)" | tee -a "$batch_log"
echo "Batch log: $batch_log"
