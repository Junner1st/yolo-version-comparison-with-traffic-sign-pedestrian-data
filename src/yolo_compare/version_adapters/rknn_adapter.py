from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from yolo_compare.config import ExperimentConfig
from yolo_compare.version_adapters.base import VersionAdapter


class RKNNAdapter(VersionAdapter):
    def train(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        dry_run: bool,
    ) -> None:
        raise SystemExit("The rknn adapter delegates testing to the Rock5B submodule and does not train models.")

    def test(
        self,
        experiment: ExperimentConfig,
        prepared_data_yaml: Path,
        run_dir: Path,
        weights: Path | None,
        dry_run: bool,
    ) -> None:
        del prepared_data_yaml
        if weights is None:
            raise SystemExit("RKNN test needs a trained run with best.pt available.")

        command = build_rock5b_command(
            experiment=experiment,
            model_raw=self.model.raw,
            run_dir=run_dir,
            weights=weights,
        )
        if dry_run:
            print("Rock5B submodule command:")
            print(shlex.join(str(part) for part in command))
            return

        subprocess.run(command, cwd=rock5b_repo_dir(self.model.raw), check=True)


def build_rock5b_command(
    experiment: ExperimentConfig,
    model_raw: dict,
    run_dir: Path,
    weights: Path,
) -> list[str | Path]:
    test = experiment.test
    repo_dir = rock5b_repo_dir(model_raw)
    target = str(model_raw.get("rknn_target", model_raw.get("target", "rk3588")))
    input_size = str(model_raw.get("rknn_input_size", model_raw.get("input_size", experiment.train.imgsz)))
    runner = rock5b_runner(model_raw, repo_dir, weights, target)
    output_split = f"{test.split}_rknn"
    report_dir = run_dir / "reports"

    command: list[str | Path] = [
        *runner,
        repo_dir / "src" / "yolo_ver_comp_bridge.py",
        "test",
        "--weights",
        weights,
        "--target",
        target,
        "--input-size",
        input_size,
        "--input-layout",
        str(model_raw.get("rknn_input_layout", model_raw.get("input_layout", "nhwc"))),
        "--conf",
        str(test.conf),
        "--iou",
        str(test.iou),
        "--split",
        test.split,
        "--data-yaml",
        experiment.dataset.yaml_path,
        "--output-dir",
        run_dir / output_split,
        "--report-yaml",
        report_dir / f"{test.split}_report_rknn.yaml",
        "--report-md",
        report_dir / f"{test.split}_report_rknn.md",
        "--max-images",
        str(test.max_images),
        "--temperature-interval",
        str(model_raw.get("rknn_temperature_interval", model_raw.get("temperature_interval", 1.0))),
    ]

    video = model_raw.get("rock5b_video") or model_raw.get("video")
    if video:
        command.extend(["--video", Path(video)])
    else:
        command.extend(["--dataset-root", experiment.dataset.root])

    if model_raw.get("no_save_video"):
        command.append("--no-save-video")

    if model_raw.get("no_temperature_log"):
        command.append("--no-temperature-log")

    return command


def rock5b_repo_dir(model_raw: dict) -> Path:
    configured = model_raw.get("rock5b_repo_dir")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[3] / "external" / "Rock5B-RKNN-Traffic-Sign-Recognition"


def rock5b_runner(model_raw: dict, repo_dir: Path, weights: Path, target: str) -> list[str]:
    configured_command = model_raw.get("rock5b_command") or os.environ.get("ROCK5B_RKNN_COMMAND")
    if configured_command:
        if isinstance(configured_command, list):
            return [str(part) for part in configured_command]
        return shlex.split(str(configured_command))

    configured_python = model_raw.get("rock5b_python") or os.environ.get("ROCK5B_RKNN_PYTHON")
    if configured_python:
        return [str(configured_python)]

    uv = shutil.which("uv") or "uv"
    command = [uv, "run", "--directory", str(repo_dir)]
    for group in rock5b_uv_groups(model_raw, weights, target):
        command.extend(["--group", group])
    command.append("python")
    return command


def rock5b_uv_groups(model_raw: dict, weights: Path, target: str) -> list[str]:
    raw_groups = model_raw.get("rock5b_uv_groups") or os.environ.get("ROCK5B_RKNN_UV_GROUPS")
    if raw_groups is None:
        groups = ["runtime"]
        if needs_rknn_export(weights, target):
            groups.insert(0, "export")
        return groups
    if isinstance(raw_groups, str):
        return [group.strip() for group in raw_groups.split(",") if group.strip()]
    return [str(group) for group in raw_groups]


def needs_rknn_export(weights: Path, target: str) -> bool:
    if weights.suffix.lower() == ".rknn":
        return False
    expected = weights.with_name(f"{weights.stem}_rknn_model") / f"{weights.stem}-{target}.rknn"
    return not expected.exists()
