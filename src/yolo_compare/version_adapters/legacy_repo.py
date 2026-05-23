from __future__ import annotations

from pathlib import Path

from yolo_compare.config import ExperimentConfig
from yolo_compare.subprocesses import run_logged
from yolo_compare.version_adapters.base import VersionAdapter, require_repo, runtime_python


class LegacyRepoAdapter(VersionAdapter):
    train_script = "train.py"
    test_script = "val.py"
    detect_script = "detect.py"

    def train(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        dry_run: bool,
    ) -> None:
        repo_dir = self.model.repo_dir if dry_run and self.model.repo_dir else require_repo(self.model)
        command = self.train_command(repo_dir, experiment, prepared_data_yaml, run_dir)
        if dry_run:
            print("$ " + " ".join(command))
            return
        run_logged(command, repo_dir, run_dir / "logs" / "train.log")

    def test(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights: Path | None,
        dry_run: bool,
    ) -> None:
        repo_dir = self.model.repo_dir if dry_run and self.model.repo_dir else require_repo(self.model)
        weights_path = weights or run_dir / "weights" / "best.pt"
        eval_command = self.eval_command(repo_dir, experiment, prepared_data_yaml, run_dir, weights_path)
        detect_command = self.detect_command(repo_dir, experiment, run_dir, weights_path)

        if dry_run:
            print("$ " + " ".join(eval_command))
            print("$ " + " ".join(detect_command))
            return

        run_logged(eval_command, repo_dir, run_dir / "logs" / "test.log")
        run_logged(detect_command, repo_dir, run_dir / "logs" / "test.log")

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
            "--img",
            str(train.imgsz),
            "--batch",
            str(train.batch),
            "--epochs",
            str(train.epochs),
            "--data",
            str(prepared_data_yaml),
            "--weights",
            self.model.weights,
            "--project",
            str(run_dir.parent),
            "--name",
            run_dir.name,
            "--exist-ok",
            "--workers",
            str(train.workers),
            "--patience",
            str(train.patience),
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
            "--img",
            str(train.imgsz),
            "--batch",
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
            f"{test.split}_eval",
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
        source = experiment.dataset.root / test.split / "images"
        return [
            *runtime_python(self.model),
            str(repo_dir / self.detect_script),
            "--img",
            str(train.imgsz),
            "--conf",
            str(test.conf),
            "--iou-thres",
            str(test.iou),
            "--weights",
            str(weights_path),
            "--source",
            str(source),
            "--project",
            str(run_dir),
            "--name",
            f"{test.split}_predict",
            "--exist-ok",
        ]
