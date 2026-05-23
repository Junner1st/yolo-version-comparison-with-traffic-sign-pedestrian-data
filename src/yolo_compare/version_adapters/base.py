from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path

from yolo_compare.config import ExperimentConfig, ModelConfig


class VersionAdapter(ABC):
    def __init__(self, model: ModelConfig) -> None:
        self.model = model

    @abstractmethod
    def train(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        dry_run: bool,
    ) -> None:
        pass

    @abstractmethod
    def test(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights: Path | None,
        dry_run: bool,
    ) -> None:
        pass


def runtime_python(model: ModelConfig) -> list[str]:
    runtime_type = model.runtime.get("type", "current-python")
    if runtime_type == "current-python":
        return [sys.executable]
    if runtime_type == "uv-run":
        if model.repo_dir is None:
            raise ValueError(f"{model.name} uses uv-run but has no repo_dir.")
        return ["uv", "run", "--directory", str(model.repo_dir), "python"]
    if runtime_type == "venv-python":
        python_path = model.runtime["python"]
        return [python_path]
    raise ValueError(f"Unsupported runtime type for {model.name}: {runtime_type}")


def require_repo(model: ModelConfig) -> Path:
    if model.repo_dir is None:
        raise ValueError(f"{model.name} requires repo_dir in configs/models.yaml.")
    if not model.repo_dir.exists():
        raise FileNotFoundError(
            f"{model.name} repo not found: {model.repo_dir}. "
            "Clone it or update repo_dir in configs/models.yaml."
        )
    return model.repo_dir
