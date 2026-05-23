from __future__ import annotations

from pathlib import Path

from yolo_compare.config import ExperimentConfig
from yolo_compare.dataset import image_paths
from yolo_compare.logging_utils import tee_output
from yolo_compare.reports import write_ultralytics_report
from yolo_compare.version_adapters.base import VersionAdapter


class UltralyticsAdapter(VersionAdapter):
    def train(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        dry_run: bool,
    ) -> None:
        train = experiment.train
        call = {
            "data": str(prepared_data_yaml),
            "epochs": train.epochs,
            "imgsz": train.imgsz,
            "batch": train.batch,
            "workers": train.workers,
            "device": _resolve_device(train.device),
            "patience": train.patience,
            "lr0": train.lr0,
            "lrf": train.lrf,
            "momentum": train.momentum,
            "weight_decay": train.weight_decay,
            "warmup_epochs": train.warmup_epochs,
            "augment": train.augment,
            "amp": train.amp,
            "project": str(run_dir),
            "name": "train",
            "exist_ok": True,
            "plots": True,
            "save": True,
            "save_period": train.save_period,
            "seed": train.seed,
            "verbose": True,
        }

        if dry_run:
            print(f"Ultralytics train: YOLO({self.model.weights!r}).train({call})")
            return

        with tee_output(run_dir / "logs" / "train.log"):
            from ultralytics import YOLO

            model = YOLO(self.model.weights)
            model.train(**call)

            best_model = YOLO(str(run_dir / "train" / "weights" / "best.pt"))
            metrics = best_model.val(
                data=str(prepared_data_yaml),
                imgsz=train.imgsz,
                batch=train.batch,
                device=_resolve_device(train.device),
                split="val",
                plots=True,
                project=str(run_dir),
                name="val",
                exist_ok=True,
                verbose=True,
            )
            write_ultralytics_report(
                metrics,
                run_dir,
                "val",
                image_count=len(image_paths(experiment.dataset.root, "val")),
            )

    def test(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights: Path | None,
        dry_run: bool,
    ) -> None:
        test = experiment.test
        train = experiment.train
        weights_path = weights or best_weights_path(run_dir)
        split_images = image_paths(experiment.dataset.root, test.split)
        source_images = split_images[: test.max_images]

        val_call = {
            "data": str(prepared_data_yaml),
            "imgsz": train.imgsz,
            "batch": train.batch,
            "device": _resolve_device(train.device),
            "split": test.split,
            "plots": True,
            "project": str(run_dir),
            "name": test.split,
            "exist_ok": True,
            "verbose": True,
        }
        predict_call = {
            "source": [str(path) for path in source_images],
            "imgsz": train.imgsz,
            "device": _resolve_device(train.device),
            "conf": test.conf,
            "iou": test.iou,
            "save": True,
            "project": str(run_dir / "predict"),
            "name": test.split,
            "exist_ok": True,
            "verbose": False,
        }

        if dry_run:
            print(f"Ultralytics test: YOLO({str(weights_path)!r}).val({val_call})")
            print(f"Ultralytics predict: YOLO({str(weights_path)!r}).predict({predict_call})")
            return

        with tee_output(run_dir / "logs" / "test.log"):
            from ultralytics import YOLO

            model = YOLO(str(weights_path))
            metrics = model.val(**val_call)
            write_ultralytics_report(metrics, run_dir, test.split, image_count=len(split_images))
            if source_images:
                model.predict(**predict_call)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def best_weights_path(run_dir: Path) -> Path:
    new_path = run_dir / "train" / "weights" / "best.pt"
    if new_path.exists() or not (run_dir / "weights" / "best.pt").exists():
        return new_path
    return run_dir / "weights" / "best.pt"
