from __future__ import annotations

import re
from pathlib import Path


def sanitize_model_name(model_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", model_name)
    return cleaned.strip("-")


def next_run_number(runs_root: Path, model_name: str) -> int:
    safe_name = sanitize_model_name(model_name)
    pattern = re.compile(rf"^run_{re.escape(safe_name)}_(\d+)$")
    numbers: list[int] = []

    if runs_root.exists():
        for path in runs_root.iterdir():
            if path.is_dir():
                match = pattern.match(path.name)
                if match:
                    numbers.append(int(match.group(1)))

    return max(numbers, default=0) + 1


def create_run_dir(runs_root: Path, model_name: str, run_number: int | None) -> Path:
    number = run_number or next_run_number(runs_root, model_name)
    run_dir = runs_root / f"run_{sanitize_model_name(model_name)}_{number}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "logs").mkdir()
    (run_dir / "snapshots").mkdir()
    return run_dir


def existing_run_dir(runs_root: Path, model_name: str, run_number: int) -> Path:
    return runs_root / f"run_{sanitize_model_name(model_name)}_{run_number}"

