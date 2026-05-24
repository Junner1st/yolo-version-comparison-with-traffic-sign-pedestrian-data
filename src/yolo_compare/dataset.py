from __future__ import annotations

from pathlib import Path

from yolo_compare.config import read_yaml, write_yaml


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def prepare_dataset_yaml(dataset_root: Path, source_yaml: Path, output_yaml: Path) -> Path:
    data = read_yaml(source_yaml)
    root = dataset_root.resolve()

    data["train"] = str(root / "train" / "images")
    data["val"] = str(root / "valid" / "images")
    data["test"] = str(root / "test" / "images")
    data["task"] = "detect"

    write_yaml(output_yaml, data)
    return output_yaml


def image_dir(dataset_root: Path, split: str) -> Path:
    split_dir = "valid" if split == "val" else split
    return dataset_root / split_dir / "images"


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    split_image_dir = image_dir(dataset_root, split)
    paths: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(split_image_dir.glob(pattern))
    return sorted(paths)


def dataset_summary(dataset_root: Path) -> dict[str, int]:
    return {
        split: len(image_paths(dataset_root, split))
        for split in ("train", "valid", "test")
    }
