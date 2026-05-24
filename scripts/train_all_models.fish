#!/usr/bin/env fish

set script_dir (dirname (status --current-filename))
set repo_root (realpath "$script_dir/..")
cd "$repo_root"

mkdir -p runs/batch_logs
set batch_log runs/batch_logs/train_all_(date +%Y%m%d_%H%M%S).log

set weights_list yolo26n yolo12n yolo11n yolov10n yolov9c yolov8n yolov7 yolov6n yolov5n

if test (count $argv) -gt 0
    set weights_list $argv
end

echo "Batch started: "(date) | tee $batch_log
echo "Models: $weights_list" | tee -a $batch_log
echo "Eval split: test" | tee -a $batch_log

set failed_weights

for weights in $weights_list
    echo "" | tee -a $batch_log
    echo "==> $weights started: "(date) | tee -a $batch_log

    uv run python src/run_experiment.py \
        --weights $weights \
        --eval-split test \
        2>&1 | tee -a $batch_log

    if test $pipestatus[1] -ne 0
        echo "==> $weights failed: "(date) | tee -a $batch_log
        set -a failed_weights $weights
        continue
    end

    echo "==> $weights finished: "(date) | tee -a $batch_log
end

echo "" | tee -a $batch_log
echo "Batch finished: "(date) | tee -a $batch_log
if test (count $failed_weights) -gt 0
    echo "Failed models: $failed_weights" | tee -a $batch_log
    echo "Batch log: $batch_log"
    exit 1
end
echo "Batch log: $batch_log"
