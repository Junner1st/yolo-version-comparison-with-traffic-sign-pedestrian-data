from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from yolo_compare.config import ModelConfig, load_experiment_config, load_model_configs, read_yaml
from yolo_compare.dataset import image_paths, prepare_dataset_yaml
from yolo_compare.runs import create_run_dir
from yolo_compare.snapshot import write_snapshot
from yolo_compare.version_adapters import build_adapter


DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiment.yaml")
DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")


def main(prefix_args: list[str] | None = None) -> None:
    args = parse_args(prefix_args)

    experiment_path = args.config.resolve()
    models_path = args.models_config.resolve()
    experiment = load_experiment_config(experiment_path)
    models = load_model_configs(models_path)

    if args.command == "list-weights":
        for model in models.values():
            print(f"{model.weights:16s} {model.adapter:12s}")
        return

    if args.command == "train":
        for model in selected_train_models(args, models):
            train_one(args, experiment, model, experiment_path, models_path)
        return

    if args.command == "test":
        test_one(args, experiment, models, experiment_path, models_path)
        return

    raise ValueError(f"Unsupported command: {args.command}")


def parse_args(prefix_args: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO version comparison runner.")
    add_config_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-weights")
    add_config_args(list_parser)
    list_parser.set_defaults(command="list-weights")

    train_parser = subparsers.add_parser("train")
    add_config_args(train_parser)
    train_group = train_parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument(
        "--weights",
        help="Official weight name from configs/models.yaml, or a local .pt path like ./weights/best.pt.",
    )
    train_group.add_argument("--all", action="store_true")
    train_parser.add_argument(
        "--adapter",
        choices=("auto", "ultralytics", "yolov5", "yolov6", "yolov7"),
        default="auto",
        help="Adapter for local/custom weights. Official configured weights use their configured adapter.",
    )
    train_parser.add_argument("--run-number", type=int)
    train_parser.add_argument("--dry-run", action="store_true")

    test_parser = subparsers.add_parser("test")
    add_config_args(test_parser)
    test_parser.add_argument("--run-dir", type=Path, required=True)
    test_parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        help="Dataset split to evaluate. Defaults to the split in configs/experiment.yaml.",
    )
    test_parser.add_argument(
        "--weights",
        help="Optional .pt weights to test. Defaults to <run-dir>/weights/best.pt.",
    )
    test_parser.add_argument(
        "--adapter",
        choices=("auto", "ultralytics", "yolov5", "yolov6", "yolov7"),
        default="auto",
        help="Adapter for --weights when no train snapshot exists.",
    )
    test_parser.add_argument("--dry-run", action="store_true")

    command_line = [*prefix_args, *sys.argv[1:]] if prefix_args else sys.argv[1:]
    return parser.parse_args(command_line)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)


def selected_train_models(args: argparse.Namespace, models: dict[str, ModelConfig]) -> list[ModelConfig]:
    if args.all:
        return [
            replace(model, name=weight_run_name(model.weights), weights=validate_weight_reference(model.weights, models))
            for model in models.values()
        ]
    return [model_from_weights(args.weights, models, args.adapter)]


def train_one(
    args: argparse.Namespace,
    experiment,
    model,
    experiment_path: Path,
    models_path: Path,
) -> Path:
    run_dir = create_run_dir(experiment.runs.root, model.name, args.run_number)
    prepared_yaml = prepare_dataset_yaml(
        experiment.dataset.root,
        experiment.dataset.yaml_path,
        run_dir / "snapshots" / "data.fixed.yaml",
    )
    write_snapshot(
        run_dir=run_dir,
        experiment=experiment,
        model=model,
        prepared_data_yaml=prepared_yaml,
        command_args=sys.argv,
        config_paths=[
            experiment_path,
            models_path,
            experiment.dataset.yaml_path,
            Path("pyproject.toml").resolve(),
            Path("uv.lock").resolve(),
        ],
    )

    print(f"Run directory: {run_dir}")
    adapter = build_adapter(model)
    adapter.train(experiment, prepared_yaml, run_dir, args.dry_run)
    return run_dir


def test_one(
    args: argparse.Namespace,
    experiment,
    models: dict[str, ModelConfig],
    experiment_path: Path,
    models_path: Path,
) -> None:
    run_dir = resolve_run_dir(args.run_dir, experiment.runs.root)
    if args.split:
        experiment = replace(experiment, test=replace(experiment.test, split=args.split))

    default_weights = run_dir / "train" / "weights" / "best.pt"
    if not args.dry_run and not default_weights.exists():
        default_weights = run_dir / "weights" / "best.pt"
    weights = Path(validate_weight_reference(args.weights, models)) if args.weights else default_weights
    model = model_for_test(run_dir, weights, models, args.adapter)
    prepared_yaml = prepare_dataset_yaml(
        experiment.dataset.root,
        experiment.dataset.yaml_path,
        run_dir / "snapshots" / "data.fixed.yaml",
    )
    write_snapshot(
        run_dir=run_dir,
        experiment=experiment,
        model=model,
        prepared_data_yaml=prepared_yaml,
        command_args=sys.argv,
        config_paths=[
            experiment_path,
            models_path,
            experiment.dataset.yaml_path,
            Path("pyproject.toml").resolve(),
            Path("uv.lock").resolve(),
        ],
        output_name="test.snapshot.yaml",
    )

    print(f"Run directory: {run_dir}")
    adapter = build_adapter(model)
    adapter.test(experiment, prepared_yaml, run_dir, weights, args.dry_run)
    if not args.dry_run:
        image_count = len(image_paths(experiment.dataset.root, experiment.test.split))
        print_test_summary(run_dir, experiment.test.split, image_count)


def resolve_run_dir(run_dir_arg: Path, runs_root: Path) -> Path:
    if run_dir_arg.exists():
        return run_dir_arg.resolve()

    if not run_dir_arg.is_absolute() and run_dir_arg.parent == Path("."):
        nested_run_dir = runs_root / run_dir_arg
        if nested_run_dir.exists():
            return nested_run_dir.resolve()

    return run_dir_arg.resolve()


def print_test_summary(run_dir: Path, split: str, image_count: int | None = None) -> None:
    report_path = run_dir / "reports" / f"{split}_report.yaml"
    if not report_path.exists():
        return

    report = read_yaml(report_path)
    overall = report.get("overall", {})
    image_count = report.get("image_count") or image_count
    print("")
    print(f"{split.title()} summary")
    print(f"Images: {image_count or ''}")
    print(f"Precision: {format_summary_value(overall.get('precision'))}")
    print(f"Recall: {format_summary_value(overall.get('recall'))}")
    print(f"mAP50: {format_summary_value(overall.get('map50'))}")
    print(f"mAP50-95: {format_summary_value(overall.get('map50_95'))}")
    print(f"Report: {report_path}")


def format_summary_value(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def model_from_weights(
    weights: str,
    models: dict[str, ModelConfig],
    adapter_override: str,
) -> ModelConfig:
    validated_weights = validate_weight_reference(weights, models)
    matched = configured_model_for_weights(validated_weights, models)
    adapter = matched.adapter if matched and adapter_override == "auto" else adapter_override
    if adapter == "auto":
        adapter = "ultralytics"

    if matched:
        return replace(
            matched,
            name=weight_run_name(validated_weights),
            weights=validated_weights,
            adapter=adapter,
        )

    return ModelConfig(
        name=weight_run_name(validated_weights),
        adapter=adapter,
        weights=validated_weights,
        repo_dir=None,
        config_file=None,
        runtime={"type": "current-python"},
        raw={"weights": validated_weights, "adapter": adapter},
    )


def model_for_test(
    run_dir: Path,
    weights: Path,
    models: dict[str, ModelConfig],
    adapter_override: str,
) -> ModelConfig:
    snapshot_path = run_dir / "config.snapshot.yaml"
    if snapshot_path.exists() and adapter_override == "auto":
        snapshot = read_yaml(snapshot_path)
        model_raw = snapshot["model"]
        repo_dir = model_raw.get("repo_dir")
        return ModelConfig(
            name=weight_run_name(model_raw["weights"]),
            adapter=model_raw["adapter"],
            weights=str(weights),
            repo_dir=Path(repo_dir) if repo_dir else None,
            config_file=model_raw.get("config_file"),
            runtime=model_raw.get("runtime", {"type": "current-python"}),
            raw=model_raw.get("raw", model_raw),
        )

    return model_from_weights(str(weights), models, adapter_override)


def configured_model_for_weights(
    weights: str,
    models: dict[str, ModelConfig],
) -> ModelConfig | None:
    weight_basename = Path(weights).name
    weight_stem = Path(weights).stem
    for model in models.values():
        model_weight_name = Path(model.weights).name
        model_weight_stem = Path(model.weights).stem
        if weights in {model.weights, model_weight_name, model_weight_stem}:
            return model
        if weight_basename == model_weight_name or weight_stem == model_weight_stem:
            return model
    return None


def validate_weight_reference(weights: str, models: dict[str, ModelConfig]) -> str:
    path = Path(weights)
    is_local_reference = path.is_absolute() or weights.startswith(".") or "/" in weights

    if is_local_reference and not path.exists():
        raise SystemExit(f"Local weights file does not exist: {weights}")

    if is_local_reference and path.suffix != ".pt":
        raise SystemExit(f"Local weights must be a .pt file: {weights}")

    if not is_local_reference and path.suffix and path.suffix != ".pt":
        raise SystemExit(f"Official weights must be a .pt weight name or stem: {weights}")

    matched = configured_model_for_weights(weights, models) if not is_local_reference else None
    if not is_local_reference and matched is None:
        available = ", ".join(model.weights for model in models.values())
        raise SystemExit(
            f"Unknown official weights: {weights}\n"
            f"Available official weights: {available}\n"
            f"You may omit .pt, for example: yolo26n\n"
            f"For a local file, use a path such as ./weights/{Path(weights).stem}.pt"
        )

    if is_local_reference:
        return str(path.resolve())
    assert matched is not None
    return matched.weights


def weight_run_name(weights: str | Path) -> str:
    return Path(str(weights)).stem
