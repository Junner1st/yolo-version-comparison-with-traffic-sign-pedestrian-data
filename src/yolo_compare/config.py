from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    yaml_path: Path


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    imgsz: int
    batch: int
    workers: int
    patience: int
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    augment: bool
    amp: bool
    device: str
    save_period: int
    seed: int


@dataclass(frozen=True)
class TestConfig:
    split: str
    conf: float
    iou: float
    max_images: int


@dataclass(frozen=True)
class RunsConfig:
    root: Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    train: TrainConfig
    test: TestConfig
    runs: RunsConfig
    raw: dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    adapter: str
    weights: str
    repo_dir: Path | None
    config_file: str | None
    runtime: dict[str, Any]
    raw: dict[str, Any]


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = read_yaml(path)
    base_dir = path.parent.parent.resolve()

    dataset_raw = raw["dataset"]
    train_raw = raw["train"]
    test_raw = raw["test"]
    runs_raw = raw["runs"]

    return ExperimentConfig(
        dataset=DatasetConfig(
            root=resolve_path(dataset_raw["root"], base_dir),
            yaml_path=resolve_path(dataset_raw["yaml"], base_dir),
        ),
        train=TrainConfig(**train_raw),
        test=TestConfig(**test_raw),
        runs=RunsConfig(root=resolve_path(runs_raw["root"], base_dir)),
        raw=raw,
    )


def load_model_configs(path: Path) -> dict[str, ModelConfig]:
    raw = read_yaml(path)
    base_dir = path.parent.parent.resolve()
    models: dict[str, ModelConfig] = {}

    for name, item in raw["models"].items():
        repo_dir = item.get("repo_dir")
        models[name] = ModelConfig(
            name=name,
            adapter=item["adapter"],
            weights=item["weights"],
            repo_dir=resolve_path(repo_dir, base_dir) if repo_dir else None,
            config_file=item.get("config_file"),
            runtime=item.get("runtime", {"type": "current-python"}),
            raw=item,
        )

    return models
