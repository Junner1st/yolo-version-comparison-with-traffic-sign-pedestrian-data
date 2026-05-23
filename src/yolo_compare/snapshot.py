from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yolo_compare.config import ExperimentConfig, ModelConfig, write_yaml
from yolo_compare.dataset import dataset_summary


def write_snapshot(
    run_dir: Path,
    experiment: ExperimentConfig,
    model: ModelConfig,
    prepared_data_yaml: Path,
    command_args: list[str],
    config_paths: list[Path],
    output_name: str = "config.snapshot.yaml",
) -> None:
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for path in config_paths:
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)

    freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=False,
    )

    snapshot: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "command_args": command_args,
        "model": {
            "name": model.name,
            "adapter": model.adapter,
            "weights": model.weights,
            "repo_dir": str(model.repo_dir) if model.repo_dir else None,
            "config_file": model.config_file,
            "runtime": model.runtime,
            "raw": model.raw,
        },
        "dataset": {
            "root": str(experiment.dataset.root),
            "source_yaml": str(experiment.dataset.yaml_path),
            "prepared_yaml": str(prepared_data_yaml),
            "summary": dataset_summary(experiment.dataset.root),
        },
        "train": experiment.train.__dict__,
        "test": experiment.test.__dict__,
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "pip_freeze": freeze.stdout.splitlines(),
        },
    }

    write_yaml(run_dir / output_name, snapshot)
