from __future__ import annotations

from pathlib import Path

from yolo_compare.config import ExperimentConfig
from yolo_compare.version_adapters.base import runtime_python
from yolo_compare.version_adapters.legacy_repo import LegacyRepoAdapter


class YoloV6Adapter(LegacyRepoAdapter):
    train_script = "tools/train.py"
    test_script = "tools/eval.py"
    detect_script = "tools/infer.py"

    def train_command(
        self,
        repo_dir: Path,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
    ) -> list[str]:
        train = experiment.train
        config_file = self.model.config_file or "configs/yolov6n.py"
        return [
            *runtime_python(self.model),
            str(repo_dir / self.train_script),
            "--img-size",
            str(train.imgsz),
            "--batch-size",
            str(train.batch),
            "--epochs",
            str(train.epochs),
            "--data-path",
            str(prepared_data_yaml),
            "--conf-file",
            str(repo_dir / config_file),
            "--output-dir",
            str(run_dir.parent),
            "--name",
            run_dir.name,
            "--workers",
            str(train.workers),
        ]

    def eval_command(
        self,
        repo_dir: Path,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights_path: Path,
    ) -> list[str]:
        train = experiment.train
        return [
            *runtime_python(self.model),
            str(repo_dir / self.test_script),
            "--data",
            str(prepared_data_yaml),
            "--weights",
            str(weights_path),
            "--batch-size",
            str(train.batch),
            "--img-size",
            str(train.imgsz),
            "--task",
            experiment.test.split,
            "--save_dir",
            str(run_dir / f"{experiment.test.split}_eval"),
        ]

    def detect_command(
        self,
        repo_dir: Path,
        experiment: ExperimentConfig,
        run_dir: Path,
        weights_path: Path,
    ) -> list[str]:
        train = experiment.train
        test = experiment.test
        source = experiment.dataset.root / test.split / "images"
        return [
            *runtime_python(self.model),
            str(repo_dir / self.detect_script),
            "--weights",
            str(weights_path),
            "--source",
            str(source),
            "--yaml",
            str(run_dir / "snapshots" / "data.fixed.yaml"),
            "--img-size",
            str(train.imgsz),
            "--conf-thres",
            str(test.conf),
            "--iou-thres",
            str(test.iou),
            "--save-dir",
            str(run_dir / f"{test.split}_predict"),
        ]

