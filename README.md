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

`test.py` prints a final summary with image count, precision, recall, mAP50, mAP50-95, ms/img, and the report path.

### Test On Rock5B With RKNN

The Rock5B/RK3588 test path is exposed through the `rknn` adapter. The original Rock5B project is linked as a submodule under `external/Rock5B-RKNN-Traffic-Sign-Recognition`:

```bash
git submodule update --init --recursive
```

Run RKNN evaluation from an existing training run:

```bash
python3 src/test.py --run-dir run_yolov8n_1 --adapter rknn
```

Run RKNN video recognition through the same submodule bridge:

```bash
python3 src/test.py --run-dir run_yolov8n_1 --adapter rknn --video data/videos/sample-day-1.mp4
```

The yolo-ver-comp adapter only delegates to the Rock5B submodule. The submodule finds `<run-dir>/train/weights/best.pt`, exports it with Ultralytics RKNN export when needed, then runs RKNNLite inference. The exported model is reused on later runs:

```text
runs/run_x/train/weights/best_rknn_model/best-rk3588.rknn
```

RKNN outputs are kept separate from normal YOLO test outputs:

```text
runs/run_x/test_rknn/
runs/run_x/reports/test_report_rknn.md
runs/run_x/reports/test_report_rknn.yaml
```

RKNN tests record Linux thermal-zone temperatures by default. The YAML report keeps both per-sample readings and per-sensor summaries; the Markdown report shows start/end/min/max/avg temperatures and embeds an NPU temperature curve when an `npu` thermal sensor is available.

```bash
python3 src/test.py --run-dir run_yolov8n_1 --adapter rknn --temperature-interval 0.5
python3 src/test.py --run-dir run_yolov8n_1 --adapter rknn --no-temperature-log
```

RKNN comparison reports also plot all matching thermal samples together:

```bash
uv run python scripts/compare_runs.py --backend rknn
```

All RKNN-specific conversion, image/video testing, and report generation live in the Rock5B submodule. yolo-ver-comp only prepares paths and calls `external/Rock5B-RKNN-Traffic-Sign-Recognition/src/yolo_ver_comp_bridge.py` through CLI args.

By default, yolo-ver-comp runs the bridge in the submodule's own uv project:

```bash
uv run --directory external/Rock5B-RKNN-Traffic-Sign-Recognition --group export --group runtime python src/yolo_ver_comp_bridge.py ...
```

Set `ROCK5B_RKNN_PYTHON` when the Rock5B dependencies live in another environment, such as system Python with `python3-rknnlite2`:

```bash
ROCK5B_RKNN_PYTHON=python3 uv run python src/test.py --run-dir run_yolov8n_1 --adapter rknn
```

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

Legacy YOLO repositories have different dependency expectations. When `repo_url` is configured in `configs/models.yaml`, the runner clones the repo into `external/` automatically if it is missing. You can also clone them manually:

```bash
git clone --branch v7.0 https://github.com/ultralytics/yolov5 external/yolov5
git clone --branch 0.3.0 https://github.com/meituan/YOLOv6 external/YOLOv6
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
- `reports/`: Markdown and YAML metric reports, including `ms_per_img` speed measurements when the adapter emits timing.

## Project Structure

- `src/train.py`: training entrypoint.
- `src/test.py`: evaluation and prediction entrypoint.
- `src/run_experiment.py`: train then evaluate one model.
- `src/yolo_compare`: shared orchestration code.
- `src/yolo_compare/version_adapters`: version-specific behavior.
- `configs/experiment.yaml`: dataset, training, testing, and run settings.
- `configs/models.yaml`: official weight names, adapters, and legacy repo paths.
- `scripts/train_all_models.fish`: batch runner for known weights.
