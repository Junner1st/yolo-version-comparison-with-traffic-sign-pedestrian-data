from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")
DEFAULT_RUNS_ROOT = Path("runs")
DEFAULT_OUTPUT = Path("runs/reports/run_test_comparison.md")
DEFAULT_RKNN_OUTPUT = Path("runs/reports/run_test_rknn_comparison.md")
DEFAULT_RUN_NUMBER = "min"
REPORT_BY_BACKEND = {
    "standard": "test_report.yaml",
    "rknn": "test_report_rknn.yaml",
}


def main() -> None:
    args = parse_args()
    models_config = args.models_config.resolve()
    runs_root = args.runs_root.resolve()
    output = (args.output or default_output(args.backend)).resolve()
    run_number = parse_run_number(args.run_number)

    models = selected_models(models_config, include_legacy=args.include_legacy)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for model_name, adapter in models:
        report_path = test_report(runs_root, model_name, run_number, args.backend)
        if report_path is None:
            missing.append(model_name)
            continue
        rows.append(row_from_report(model_name, adapter, report_path))

    if missing and run_number is not None:
        missing_runs = ", ".join(f"run_{model}_{run_number}" for model in missing)
        report_name = REPORT_BY_BACKEND[args.backend]
        raise SystemExit(f"Missing requested {report_name} reports: {missing_runs}")

    thermal_plots = write_thermal_comparison_plots(rows, output)
    markdown = render_markdown(
        rows,
        missing,
        models_config,
        runs_root,
        run_number=run_number,
        include_legacy=args.include_legacy,
        backend=args.backend,
        thermal_plots=thermal_plots,
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
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend",
        choices=tuple(REPORT_BY_BACKEND),
        default="standard",
        help="Report backend to compare. Use rknn for test_report_rknn.yaml.",
    )
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


def default_output(backend: str) -> Path:
    return DEFAULT_RKNN_OUTPUT if backend == "rknn" else DEFAULT_OUTPUT


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


def test_report(runs_root: Path, model_name: str, run_number: int | None, backend: str) -> Path | None:
    report_name = REPORT_BY_BACKEND[backend]
    if run_number is not None:
        report_path = runs_root / f"run_{model_name}_{run_number}" / "reports" / report_name
        return report_path if report_path.exists() else None

    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(rf"^run_{re.escape(model_name)}_(\d+)$")

    for run_dir in runs_root.glob(f"run_{model_name}_*"):
        if not run_dir.is_dir():
            continue
        match = pattern.match(run_dir.name)
        if match is None:
            continue
        report_path = run_dir / "reports" / report_name
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
    temperature = report.get("temperature", {})

    return {
        "model": model_name,
        "adapter": adapter,
        "backend": report.get("backend") or "standard",
        "mode": report.get("mode") or "",
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
        "temperature_samples": temperature.get("samples", []),
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
    backend: str,
    thermal_plots: list[dict[str, str]],
) -> str:
    report_name = REPORT_BY_BACKEND[backend]
    run_selection = (
        f"lowest run number with `reports/{report_name}`"
        if run_number is None
        else f"`run_<model>_{run_number}` only"
    )
    scope = "all configured models" if include_legacy else "models with `adapter: ultralytics`"
    lines = [
        "# Run Test Comparison",
        "",
        f"- Models config: `{models_config}`",
        f"- Runs root: `{runs_root}`",
        f"- Backend: `{backend}` (`reports/{report_name}`).",
        f"- Scope: {scope}.",
        f"- Run selection: {run_selection}.",
        "",
        "| Model | Adapter | Backend | Mode | Run | Images | Precision | Recall | mAP50 | mAP50-95 | Fitness | Total ms/img | Inference ms/img | Report |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            "| {model} | {adapter} | {backend} | {mode} | {run} | {images} | {precision} | {recall} | {map50} | {map50_95} | "
            "{fitness} | {ms_img_total} | {ms_img_inference} | [{report_name}]({report}) |".format(
                model=row["model"],
                adapter=row["adapter"],
                backend=row["backend"],
                mode=row["mode"],
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

    if thermal_plots:
        lines.extend(
            [
                "",
                "## Thermal Samples",
                "",
                "- All available thermal sensors are plotted separately.",
                f"- X axis: time (s).",
                f"- Y axis: temperature (C).",
                "",
            ]
        )
        for plot in thermal_plots:
            lines.extend(
                [
                    f"### {plot['sensor']}",
                    "",
                    f"![{plot['sensor']} thermal comparison]({plot['file']})",
                    "",
                ]
            )

    if missing:
        lines.extend(["", "## Missing Test Reports", ""])
        for model_name in missing:
            lines.append(f"- `{model_name}`")

    lines.append("")
    return "\n".join(lines)


def write_thermal_comparison_plots(rows: list[dict[str, Any]], report_path: Path) -> list[dict[str, str]]:
    remove_stale_thermal_plots(report_path)
    sensors = thermal_sensor_names(rows)
    if not sensors:
        return []

    report_path.parent.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, str]] = []
    for sensor_name in sensors:
        series = thermal_series(rows, sensor_name)
        if not series:
            continue
        output_path = report_path.with_name(f"{report_path.stem}_thermal_{slugify(sensor_name)}.svg")
        output_path.write_text(render_thermal_svg(series, sensor_name), encoding="utf-8")
        plots.append({"sensor": sensor_name, "file": output_path.name})
    return plots


def remove_stale_thermal_plots(report_path: Path) -> None:
    if not report_path.parent.exists():
        return
    for path in report_path.parent.glob(f"{report_path.stem}_thermal*.svg"):
        remove_path(path)


def remove_path(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def thermal_sensor_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        samples = row.get("temperature_samples", [])
        if not isinstance(samples, list):
            continue
        for sample in samples:
            temperatures = sample.get("temperatures_c", {})
            if isinstance(temperatures, dict):
                names.update(str(name) for name in temperatures)
    return sorted(names, key=thermal_sensor_sort_key)


def thermal_sensor_sort_key(sensor_name: str) -> tuple[int, str]:
    order = [
        "soc",
        "bigcore0",
        "bigcore1",
        "littlecore",
        "center",
        "gpu",
        "npu",
    ]
    lower = sensor_name.lower()
    for index, token in enumerate(order):
        if token in lower:
            return index, lower
    return len(order), lower


def thermal_series(rows: list[dict[str, Any]], sensor_name: str) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for row in rows:
        samples = row.get("temperature_samples", [])
        if not isinstance(samples, list):
            continue
        points = sample_points(samples, sensor_name)
        if len(points) < 2:
            continue
        series.append(
            {
                "label": f"{row['model']} ({row['run']})",
                "sensor": sensor_name,
                "points": points,
            }
        )
    return series


def sample_points(samples: list[dict[str, Any]], sensor_name: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for sample in samples:
        temperatures = sample.get("temperatures_c", {})
        if not isinstance(temperatures, dict):
            continue
        if sensor_name not in temperatures:
            continue
        try:
            elapsed = float(sample.get("elapsed_s", 0.0))
            temp = float(temperatures[sensor_name])
        except (TypeError, ValueError):
            continue
        points.append((elapsed, temp))
    return points


def render_thermal_svg(series: list[dict[str, Any]], sensor_name: str) -> str:
    width, height = 980, 520
    left, right, top, bottom = 64, 250, 42, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
    ]
    all_points = [point for item in series for point in item["points"]]
    min_time = 0.0
    max_time = max(point[0] for point in all_points)
    temps = [point[1] for point in all_points]
    min_temp, max_temp = min(temps), max(temps)
    if min_temp == max_temp:
        min_temp -= 0.5
        max_temp += 0.5
    else:
        padding = max(0.5, (max_temp - min_temp) * 0.15)
        min_temp -= padding
        max_temp += padding

    def sx(value: float) -> float:
        if max_time <= min_time:
            return left
        return left + (value - min_time) * plot_width / (max_time - min_time)

    def sy(value: float) -> float:
        return top + (max_temp - value) * plot_height / (max_temp - min_temp)

    y_ticks = tick_values(min(temps), max(temps), 5)
    x_ticks = tick_values(min_time, max_time, 6)
    grid = "\n".join(
        f'  <line x1="{left}" y1="{sy(value):.2f}" x2="{left + plot_width}" y2="{sy(value):.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        f'\n  <text x="{left - 8}" y="{sy(value) + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#475569">{value:.1f}</text>'
        for value in y_ticks
    )
    x_labels = "\n".join(
        f'  <text x="{sx(value):.2f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#475569">{value:.0f}</text>'
        for value in x_ticks
    )

    paths: list[str] = []
    legend: list[str] = []
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        polyline = " ".join(f"{sx(elapsed):.2f},{sy(temp):.2f}" for elapsed, temp in item["points"])
        paths.append(
            f'  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = top + 8 + index * 24
        label = html.escape(str(item["label"]))
        sensor = html.escape(str(item["sensor"]))
        count = len(item["points"])
        legend.append(
            f'  <line x1="{left + plot_width + 28}" y1="{legend_y}" x2="{left + plot_width + 48}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'\n  <text x="{left + plot_width + 56}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#1f2933">{label}</text>'
            f'\n  <text x="{left + plot_width + 56}" y="{legend_y + 18}" font-family="Arial, sans-serif" font-size="10" fill="#64748b">{sensor}, {count} samples</text>'
        )

    title = html.escape(f"Thermal comparison ({sensor_name})")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <title>{title}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="24" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#1f2933">{title}</text>
  <text x="{left + plot_width / 2:.2f}" y="{height - 4}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">time (s)</text>
  <text x="16" y="{top + plot_height / 2:.2f}" transform="rotate(-90 16 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#475569">temp (C)</text>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>
{grid}
{x_labels}
{chr(10).join(paths)}
{chr(10).join(legend)}
</svg>
"""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "sensor"


def tick_values(min_value: float, max_value: float, count: int) -> list[float]:
    if count <= 1 or max_value == min_value:
        return [min_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + step * index for index in range(count)]


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
