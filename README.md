# YOLO Version Comparison

This project runs reproducible YOLO version comparison experiments without notebooks. Use `uv run`; do not run the system Python directly.

## Usage

### Run All Models

The fish script is the simplest way to run a batch of experiments. It uses the weight list defined inside `scripts/train_all_models.fish`, writes one run folder per model, and keeps a batch log under `runs/batch_logs/`.

```fish
fish scripts/train_all_models.fish
```

or for a normal shell like bash/zsh:

```bash
./scripts/train_all_models.sh
```

To run only specific weights, pass them after the script name:

```fish
fish scripts/train_all_models.fish yolov8n yolo26n
```

Each model is executed through:

```fish
uv run python src/run_experiment.py --weights <weights> --eval-split test
```

### Train And Test One Model

Run training and then evaluation in one command:

```bash
uv run python src/run_experiment.py --weights yolov8n --eval-split test
```

Run multiple evaluation splits after training:

```bash
uv run python src/run_experiment.py --weights yolov8n --eval-split val test
```

Official weights can omit `.pt` when they are listed in `configs/models.yaml`. For example, use `yolov8n`, not `yolov8n.pt`.

### Train Only

```bash
uv run python src/train.py --weights yolov8n
```

Train all configured models from `configs/models.yaml`:

```bash
uv run python src/train.py --all
```

Train from a local weight file:

```bash
uv run python src/train.py --weights ./weights/custom.pt
```

### Test Only

Test a trained run. The short run directory name is accepted:

```bash
uv run python src/test.py --run-dir run_yolov8n_1
```

Equivalent explicit path:

```bash
uv run python src/test.py --run-dir runs/run_yolov8n_1
```

Choose the dataset split:

```bash
uv run python src/test.py --run-dir run_yolov8n_1 --split val
uv run python src/test.py --run-dir run_yolov8n_1 --split test
uv run python src/test.py --run-dir run_yolov8n_1 --split train
```

`test.py` prints a final summary with image count, precision, recall, mAP50, mAP50-95, and the report path.

## Environment

Create and sync the project environment with `uv`:

```bash
uv venv --python 3.11
uv sync
```

Run commands through `uv run`, for example:

```bash
uv run python src/train.py --weights yolov8n --dry-run
```

Legacy YOLO repositories have different dependency expectations. Clone them under `external/` when you need to run those adapters:

```bash
git clone --branch v7.0 https://github.com/ultralytics/yolov5 external/yolov5
git clone --branch v3.0 https://github.com/meituan/YOLOv6 external/YOLOv6
git clone https://github.com/WongKinYiu/yolov7 external/yolov7
```

If a legacy repo needs a different Python version, create a separate uv-managed environment inside that repo and install its requirements according to that repo's documentation.

## Other Commands

List official weights configured in `configs/models.yaml`:

```bash
uv run python -m yolo_compare list-weights
```

Inspect commands without training or testing:

```bash
uv run python src/train.py --weights yolov8n --dry-run
uv run python src/test.py --run-dir run_yolov8n_1 --dry-run
uv run python src/run_experiment.py --weights yolov8n --eval-split test --dry-run
```

Re-split the dataset as 8:1:1 while keeping every class present in train, val, and test:

```bash
uv run python -m yolo_compare.split_dataset --dataset-root data --ratio 8 1 1 --seed 42
```

The split manifest is written to `data/split_manifest.yaml`.

## Outputs

Runs are written under:

```text
runs/run_{weight-name}_{number}
```

Example:

```text
runs/run_yolov8n_1
```

A run directory contains:

- `config.snapshot.yaml`: experiment snapshot for reproducibility.
- `snapshots/`: copied config files and fixed absolute-path dataset yaml.
- `logs/`: train and test logs.
- `train/`: training outputs, curves, train-time validation images, and weights.
- `val/`, `test/`, `train/`: independent evaluation outputs from `src/test.py --split ...`.
- `predict/{split}/`: prediction images for the requested split.
- `reports/`: Markdown and YAML metric reports.

## Project Structure

- `src/train.py`: training entrypoint.
- `src/test.py`: evaluation and prediction entrypoint.
- `src/run_experiment.py`: train then evaluate one model.
- `src/yolo_compare`: shared orchestration code.
- `src/yolo_compare/version_adapters`: version-specific behavior.
- `configs/experiment.yaml`: dataset, training, testing, and run settings.
- `configs/models.yaml`: official weight names, adapters, and legacy repo paths.
- `scripts/train_all_models.fish`: batch runner for known weights.
