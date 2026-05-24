from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_OUTPUT = Path("runs/reports/run_test_comparison.md")
DEFAULT_RUN_NUMBER = "min"


def main() -> None:
    args = parse_args()
    models_config = args.models_config.resolve()
    runs_root = args.runs_root.resolve()
    output = args.output.resolve()
    run_number = parse_run_number(args.run_number)

    models = selected_models(models_config, include_legacy=args.include_legacy)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, adapter in models:
        report_path = test_report(runs_root, model_name, run_number)
        if report_path is None:
            missing.append(model_name)
            continue
        rows.append(row_from_report(model_name, adapter, report_path))

    if missing and run_number is not None:
        missing_runs = ", ".join(f"run_{model}_{run_number}" for model in missing)
        raise SystemExit(f"Missing requested test reports: {missing_runs}")

    markdown = render_markdown(
        rows,
        missing,
        models_config,
        runs_root,
        run_number=run_number,
        include_legacy=args.include_legacy,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare test reports in a Markdown table."
    )
    parser.add_argument("--models-config", type=Path, default=DEFAULT_MODELS_CONFIG)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-number",
        default=DEFAULT_RUN_NUMBER,
        help="Run number to compare, e.g. 2 for run_<model>_2. Defaults to min.",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy adapters such as yolov5/yolov6/yolov7. Defaults to false.",
    )
    return parser.parse_args()


def parse_run_number(value: str) -> int | None:
    if value == "min":
        return None
    try:
        run_number = int(value)
    except ValueError as exc:
        raise SystemExit("--run-number must be 'min' or a positive integer.") from exc
    if run_number < 1:
        raise SystemExit("--run-number must be 'min' or a positive integer.")
    return run_number


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def selected_models(models_config: Path, include_legacy: bool) -> list[tuple[str, str]]:
    raw = read_yaml(models_config)
    return [
        (name, model.get("adapter", ""))
        for name, model in raw.get("models", {}).items()
        if include_legacy or model.get("adapter") == "ultralytics"
    ]


def test_report(runs_root: Path, model_name: str, run_number: int | None) -> Path | None:
    if run_number is not None:
        report_path = runs_root / f"run_{model_name}_{run_number}" / "reports" / "test_report.yaml"
        return report_path if report_path.exists() else None

    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(rf"^run_{re.escape(model_name)}_(\d+)$")

    for run_dir in runs_root.glob(f"run_{model_name}_*"):
        if not run_dir.is_dir():
            continue
        match = pattern.match(run_dir.name)
        if match is None:
            continue
        report_path = run_dir / "reports" / "test_report.yaml"
        if report_path.exists():
            candidates.append((int(match.group(1)), report_path))

    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def row_from_report(model_name: str, adapter: str, report_path: Path) -> dict[str, Any]:
    report = read_yaml(report_path)
    overall = report.get("overall", {})
    ms_per_img = report.get("ms_per_img", {})
    speed = report.get("speed", {})
    total_ms_img = ms_per_img.get("total") or total_speed(speed)

    return {
        "model": model_name,
        "adapter": adapter,
        "run": report_path.parents[1].name,
        "images": report.get("image_count"),
        "precision": overall.get("precision"),
        "recall": overall.get("recall"),
        "map50": overall.get("map50"),
        "map50_95": overall.get("map50_95"),
        "fitness": overall.get("fitness"),
        "ms_img_total": total_ms_img,
        "ms_img_inference": ms_per_img.get("inference") or speed.get("inference"),
        "report": report_path,
    }


def total_speed(speed: dict[str, Any]) -> float | None:
    keys = ("preprocess", "inference", "loss", "postprocess", "nms")
    values = [speed.get(key) for key in keys if speed.get(key) is not None]
    if not values:
        return None
    return sum(float(value) for value in values)


def render_markdown(
    rows: list[dict[str, Any]],
    missing: list[str],
    models_config: Path,
    runs_root: Path,
    run_number: int | None,
    include_legacy: bool,
) -> str:
    run_selection = (
        "lowest run number with `reports/test_report.yaml`"
        if run_number is None
        else f"`run_<model>_{run_number}` only"
    )
    scope = "all configured models" if include_legacy else "models with `adapter: ultralytics`"
    lines = [
        "# Run Test Comparison",
        "",
        f"- Models config: `{models_config}`",
        f"- Runs root: `{runs_root}`",
        f"- Scope: {scope}.",
        f"- Run selection: {run_selection}.",
        "",
        "| Model | Adapter | Run | Images | Precision | Recall | mAP50 | mAP50-95 | Fitness | Total ms/img | Inference ms/img | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {model} | {adapter} | {run} | {images} | {precision} | {recall} | {map50} | {map50_95} | "
            "{fitness} | {ms_img_total} | {ms_img_inference} | [{report_name}]({report}) |".format(
                model=row["model"],
                adapter=row["adapter"],
                run=row["run"],
                images=format_int(row["images"]),
                precision=format_float(row["precision"]),
                recall=format_float(row["recall"]),
                map50=format_float(row["map50"]),
                map50_95=format_float(row["map50_95"]),
                fitness=format_float(row["fitness"]),
                ms_img_total=format_float(row["ms_img_total"]),
                ms_img_inference=format_float(row["ms_img_inference"]),
                report_name=row["report"].name,
                report=row["report"],
            )
        )

    if missing:
        lines.extend(["", "## Missing Test Reports", ""])
        for model_name in missing:
            lines.append(f"- `{model_name}`")

    lines.append("")
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.6f}"


def format_int(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


if __name__ == "__main__":
    main()
