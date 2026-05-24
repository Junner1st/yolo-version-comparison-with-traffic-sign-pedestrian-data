from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

from yolo_compare.config import ExperimentConfig
from yolo_compare.dataset import image_dir, image_paths
from yolo_compare.reports import parse_speed_log, write_legacy_report
from yolo_compare.subprocesses import run_logged
from yolo_compare.version_adapters.base import VersionAdapter, require_repo, runtime_python


LEGACY_ENV = {
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
}


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
        if not dry_run:
            ensure_configured_weights(self.model.weights, self.model.raw.get("weights_url"), repo_dir)
            clear_label_caches(experiment.dataset.root)
        command = self.train_command(repo_dir, experiment, prepared_data_yaml, run_dir)
        if dry_run:
            print("$ " + " ".join(command))
            return
        log_path = run_dir / "logs" / "train.log"
        print(f"Training log: {log_path}")
        run_logged(command, repo_dir, log_path, env=LEGACY_ENV, echo=False)

    def test(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights: Path | None,
        dry_run: bool,
    ) -> None:
        repo_dir = self.model.repo_dir if dry_run and self.model.repo_dir else require_repo(self.model)
        if not dry_run:
            ensure_configured_weights(self.model.weights, self.model.raw.get("weights_url"), repo_dir)
            clear_label_caches(experiment.dataset.root)
        weights_path = self.resolve_weights_path(run_dir, weights)
        eval_command = self.eval_command(repo_dir, experiment, prepared_data_yaml, run_dir, weights_path)
        detect_command = self.detect_command(repo_dir, experiment, run_dir, weights_path)

        if dry_run:
            print("$ " + " ".join(eval_command))
            print("$ " + " ".join(detect_command))
            return

        log_path = run_dir / "logs" / "test.log"
        run_logged(eval_command, repo_dir, log_path, env=LEGACY_ENV)
        detect_seconds = run_logged(detect_command, repo_dir, log_path, env=LEGACY_ENV)

        split_images = image_paths(experiment.dataset.root, experiment.test.split)
        speed = parse_speed_log(log_path)
        if split_images:
            speed["predict_wall_clock"] = detect_seconds * 1000 / len(split_images)
        write_legacy_report(run_dir, experiment.test.split, len(split_images), speed)

    def resolve_weights_path(self, run_dir: Path, weights: Path | None) -> Path:
        weights_path = weights or self.default_weights_path(run_dir)
        if weights_path.exists():
            return weights_path

        default_path = self.default_weights_path(run_dir)
        if weights is None or weights_path.name == "best.pt" or default_path.exists():
            return default_path
        return weights_path

    def default_weights_path(self, run_dir: Path) -> Path:
        train_best = run_dir / "train" / "weights" / "best.pt"
        if train_best.exists() or not (run_dir / "weights" / "best.pt").exists():
            return train_best
        return run_dir / "weights" / "best.pt"

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
            str(run_dir),
            "--name",
            "train",
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
            str(run_dir / "predict"),
            "--name",
            test.split,
            "--exist-ok",
        ]


def ensure_configured_weights(weights: str, weights_url: str | None, repo_dir: Path) -> None:
    if not weights_url:
        return

    weights_path = Path(weights)
    if weights_path.is_absolute():
        return

    target = repo_dir / weights_path
    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {weights_url} to {target}")
    urlretrieve(weights_url, target)


def clear_label_caches(dataset_root: Path) -> None:
    for cache_path in dataset_root.glob("*/labels.cache"):
        cache_path.unlink(missing_ok=True)
