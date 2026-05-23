from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from yolo_compare.cli import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_MODELS_CONFIG,
    model_from_weights,
    test_one,
    train_one,
)
from yolo_compare.config import load_experiment_config, load_model_configs


def main() -> None:
    args = parse_args()

    experiment_path = args.config.resolve()
    models_path = args.models_config.resolve()
    experiment = load_experiment_config(experiment_path)
    models = load_model_configs(models_path)

    model = model_from_weights(args.weights, models, args.adapter)
    train_args = SimpleNamespace(
        run_number=args.run_number,
        dry_run=args.dry_run,
    )
    run_dir = train_one(train_args, experiment, model, experiment_path, models_path)

    if args.dry_run:
        print("")
        print("Dry run only. Evaluation commands:")

    for split in args.eval_split:
        eval_args = SimpleNamespace(
            run_dir=run_dir,
            split=split,
            weights=None,
            adapter=args.adapter,
            dry_run=args.dry_run,
        )
        test_one(eval_args, experiment, models, experiment_path, models_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train, then evaluation splits for one model.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--eval-split",
        choices=("train", "val", "test"),
        nargs="+",
        default=["test"],
        help="Evaluation splits to run after training. Default: test.",
    )
    parser.add_argument(
        "--adapter",
        choices=("auto", "ultralytics", "yolov5", "yolov6", "yolov7"),
        default="auto",
    )
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
