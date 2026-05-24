from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from yolo_compare.config import write_yaml


def write_ultralytics_report(
    metrics: Any,
    run_dir: Path,
    split: str,
    image_count: int | None = None,
) -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split": split,
        "image_count": image_count,
        "overall": metric_summary(metrics),
        "per_class": per_class_summary(metrics),
        "speed": normalize_speed(getattr(metrics, "speed", {})),
    }
    report["ms_per_img"] = ms_per_img_summary(report["speed"])

    write_yaml(report_dir / f"{split}_report.yaml", report)
    (report_dir / f"{split}_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )


def write_legacy_report(
    run_dir: Path,
    split: str,
    image_count: int | None,
    speed: dict[str, float],
) -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split": split,
        "image_count": image_count,
        "overall": {},
        "per_class": [],
        "speed": normalize_speed(speed),
    }
    report["ms_per_img"] = ms_per_img_summary(report["speed"])

    write_yaml(report_dir / f"{split}_report.yaml", report)
    (report_dir / f"{split}_report.md").write_text(
        markdown_report(report),
        encoding="utf-8",
    )


def metric_summary(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}

    return {
        "precision": as_float(getattr(box, "mp", None)),
        "recall": as_float(getattr(box, "mr", None)),
        "map50": as_float(getattr(box, "map50", None)),
        "map50_95": as_float(getattr(box, "map", None)),
        "fitness": as_float(getattr(metrics, "fitness", None)),
    }


def per_class_summary(metrics: Any) -> list[dict[str, float | int | str | None]]:
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", {}) or {}
    maps = getattr(box, "maps", []) if box is not None else []
    class_indices = getattr(box, "ap_class_index", range(len(maps))) if box is not None else []

    rows = []
    for class_id in class_indices:
        class_id = int(class_id)
        rows.append(
            {
                "class_id": class_id,
                "name": names.get(class_id, str(class_id)),
                "map50_95": as_float(maps[class_id]),
            }
        )
    return rows


def normalize_speed(speed: Any) -> dict[str, float]:
    if not isinstance(speed, dict):
        return {}
    normalized = {}
    for key, value in speed.items():
        number = as_float(value)
        if number is not None:
            normalized[str(key)] = number
    return normalized


def ms_per_img_summary(speed: dict[str, float]) -> dict[str, float]:
    summary = dict(speed)
    if "total" not in summary:
        total_keys = ("preprocess", "pre_process", "inference", "postprocess", "nms")
        total = sum(summary.get(key, 0.0) for key in total_keys)
        if total > 0:
            summary["total"] = total
    return summary


def parse_speed_text(text: str) -> dict[str, float]:
    speed: dict[str, float] = {}

    yolov5_matches = list(
        re.finditer(
            r"Speed:\s*([\d.]+)ms pre-process,\s*([\d.]+)ms inference,\s*([\d.]+)ms NMS per image",
            text,
        )
    )
    if yolov5_matches:
        match = yolov5_matches[-1]
        speed.update(
            {
                "preprocess": float(match.group(1)),
                "inference": float(match.group(2)),
                "nms": float(match.group(3)),
            }
        )

    yolov7_matches = list(
        re.finditer(
            r"Speed:\s*([\d.]+)/([\d.]+)/([\d.]+)\s*ms inference/NMS/total per",
            text,
        )
    )
    if yolov7_matches:
        match = yolov7_matches[-1]
        speed.update(
            {
                "inference": float(match.group(1)),
                "nms": float(match.group(2)),
                "total": float(match.group(3)),
            }
        )

    yolov6_matches = list(
        re.finditer(
            r"Average (pre-process|inference|NMS) time:\s*([\d.]+)\s*ms",
            text,
        )
    )
    for match in yolov6_matches:
        key = {
            "pre-process": "preprocess",
            "inference": "inference",
            "NMS": "nms",
        }[match.group(1)]
        speed[key] = float(match.group(2))

    return ms_per_img_summary(speed)


def parse_speed_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    return parse_speed_text(log_path.read_text(encoding="utf-8", errors="replace"))


def markdown_report(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        f"# {report['split'].title()} Report",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]

    lines.append(f"| images | {report.get('image_count') or ''} |")
    for key in ("precision", "recall", "map50", "map50_95", "fitness"):
        lines.append(f"| {key} | {format_value(overall.get(key))} |")

    speed = report.get("ms_per_img", {})
    if speed:
        lines.extend(
            [
                "",
                "## Speed",
                "",
                "| Metric | ms/img |",
                "| --- | ---: |",
            ]
        )
        for key in ("preprocess", "inference", "postprocess", "nms", "total", "predict_wall_clock"):
            if key in speed:
                lines.append(f"| {key} | {format_value(speed.get(key))} |")

    lines.extend(
        [
            "",
            "## Per Class",
            "",
            "| Class ID | Name | mAP50-95 |",
            "| ---: | --- | ---: |",
        ]
    )

    for row in report["per_class"]:
        lines.append(
            f"| {row['class_id']} | {row['name']} | {format_value(row['map50_95'])} |"
        )

    lines.append("")
    return "\n".join(lines)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
