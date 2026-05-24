from __future__ import annotations

from pathlib import Path

from yolo_compare.config import ExperimentConfig
from yolo_compare.dataset import image_dir
from yolo_compare.version_adapters.base import runtime_python
from yolo_compare.version_adapters.legacy_repo import LegacyRepoAdapter


class YoloV7Adapter(LegacyRepoAdapter):
    test_script = "test.py"

    def train_command(
        self,
        repo_dir: Path,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
    ) -> list[str]:
        train = experiment.train
        return [
            *runtime_python(self.model),
            str(repo_dir / self.train_script),
            "--img-size",
            str(train.imgsz),
            "--batch-size",
            str(train.batch),
            "--epochs",
            str(train.epochs),
            "--data",
            str(prepared_data_yaml),
            "--weights",
            self.model.weights,
            "--project",
            str(run_dir),
            "--name",
            "train",
            "--exist-ok",
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
        test = experiment.test
        return [
            *runtime_python(self.model),
            str(repo_dir / self.test_script),
            "--img-size",
            str(train.imgsz),
            "--batch-size",
            str(train.batch),
            "--data",
            str(prepared_data_yaml),
            "--weights",
            str(weights_path),
            "--task",
            test.split,
            "--project",
            str(run_dir),
            "--name",
            test.split,
            "--exist-ok",
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
        source = image_dir(experiment.dataset.root, test.split)
        return [
            *runtime_python(self.model),
            str(repo_dir / self.detect_script),
            "--img-size",
            str(train.imgsz),
            "--conf-thres",
            str(test.conf),
            "--iou-thres",
            str(test.iou),
            "--weights",
            str(weights_path),
            "--source",
            str(source),
            "--project",
            str(run_dir / "predict"),
            "--name",
            test.split,
            "--exist-ok",
        ]
