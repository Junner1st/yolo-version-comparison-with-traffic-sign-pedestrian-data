from __future__ import annotations

from pathlib import Path
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
        "speed": getattr(metrics, "speed", {}),
    }

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
    return float(value)


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"
