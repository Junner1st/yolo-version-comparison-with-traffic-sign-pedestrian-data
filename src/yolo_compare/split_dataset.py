from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from yolo_compare.config import read_yaml, write_yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    label_path: Path
    classes: frozenset[int]


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    rng = random.Random(args.seed)

    samples = collect_samples(root)
    class_count = class_image_counts(samples)
    missing = [class_id for class_id, count in class_count.items() if count < 2]
    if missing:
        raise SystemExit(f"Cannot place these classes in both val and test: {missing}")

    train_count, val_count, test_count = target_counts(len(samples), args.ratio)
    val_samples = choose_required_class_samples(samples, set(), val_count, rng)
    test_samples = choose_required_class_samples(samples, val_samples, test_count, rng)

    val_samples = fill_split(samples, val_samples, test_samples, val_count, rng)
    test_samples = fill_split(samples, test_samples, val_samples, test_count, rng)
    assigned = val_samples | test_samples
    train_samples = {sample.stem for sample in samples if sample.stem not in assigned}

    splits = {
        "train": sorted(train_samples),
        "valid": sorted(val_samples),
        "test": sorted(test_samples),
    }
    verify_split(samples, splits, train_count, val_count, test_count)
    rewrite_dataset(root, samples, splits)
    write_manifest(root, samples, splits, args.seed, args.ratio)
    remove_caches(root)

    print_summary(root, samples, splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-split YOLO dataset with class coverage guarantees.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data"))
    parser.add_argument("--ratio", type=int, nargs=3, default=(8, 1, 1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_samples(root: Path) -> list[Sample]:
    samples: dict[str, Sample] = {}
    seen_images: dict[str, Path] = {}

    for split in SPLITS:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image_path.stem in seen_images:
                raise SystemExit(f"Duplicate image stem found: {image_path.stem}")
            seen_images[image_path.stem] = image_path

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise SystemExit(f"Missing label for image: {image_path}")
            samples[image_path.stem] = Sample(
                stem=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                classes=frozenset(read_label_classes(label_path)),
            )

    return list(samples.values())


def read_label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            classes.add(int(float(line.split()[0])))
    return classes


def class_image_counts(samples: list[Sample]) -> dict[int, int]:
    class_ids = sorted({class_id for sample in samples for class_id in sample.classes})
    return {
        class_id: sum(1 for sample in samples if class_id in sample.classes)
        for class_id in class_ids
    }


def target_counts(total: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    ratio_total = sum(ratio)
    val_count = round(total * ratio[1] / ratio_total)
    test_count = round(total * ratio[2] / ratio_total)
    train_count = total - val_count - test_count
    return train_count, val_count, test_count


def choose_required_class_samples(
    samples: list[Sample],
    unavailable: set[str],
    target_count: int,
    rng: random.Random,
) -> set[str]:
    class_ids = sorted({class_id for sample in samples for class_id in sample.classes})
    selected: set[str] = set()
    covered: set[int] = set()
    candidates = [sample for sample in samples if sample.stem not in unavailable]

    while covered != set(class_ids):
        needed = set(class_ids) - covered
        ranked = sorted(
            candidates,
            key=lambda sample: (
                -len(sample.classes & needed),
                min((class_frequency(candidates, class_id) for class_id in sample.classes & needed), default=10**9),
                rng.random(),
            ),
        )
        chosen = next(sample for sample in ranked if sample.classes & needed)
        selected.add(chosen.stem)
        covered.update(chosen.classes)
        candidates = [sample for sample in candidates if sample.stem != chosen.stem]

    if len(selected) > target_count:
        raise SystemExit(f"Class coverage requires {len(selected)} samples, target only has {target_count}.")
    return selected


def fill_split(
    samples: list[Sample],
    split_stems: set[str],
    other_split_stems: set[str],
    target_count: int,
    rng: random.Random,
) -> set[str]:
    selected = set(split_stems)
    all_classes = {class_id for sample in samples for class_id in sample.classes}
    by_stem = {sample.stem: sample for sample in samples}
    candidates = [
        sample
        for sample in samples
        if sample.stem not in selected and sample.stem not in other_split_stems
    ]
    rng.shuffle(candidates)

    for sample in candidates:
        if len(selected) >= target_count:
            break
        train_stems_after = {
            item.stem
            for item in samples
            if item.stem not in selected
            and item.stem not in other_split_stems
            and item.stem != sample.stem
        }
        train_classes_after = {
            class_id
            for stem in train_stems_after
            for class_id in by_stem[stem].classes
        }
        if all_classes <= train_classes_after:
            selected.add(sample.stem)

    if len(selected) != target_count:
        raise SystemExit(f"Could only fill split to {len(selected)} samples; target is {target_count}.")
    return selected


def class_frequency(samples: list[Sample], class_id: int) -> int:
    return sum(1 for sample in samples if class_id in sample.classes)


def verify_split(
    samples: list[Sample],
    splits: dict[str, list[str]],
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    expected = {"train": train_count, "valid": val_count, "test": test_count}
    for split, count in expected.items():
        if len(splits[split]) != count:
            raise SystemExit(f"{split} expected {count}, got {len(splits[split])}")

    all_stems = [stem for split_stems in splits.values() for stem in split_stems]
    if len(all_stems) != len(set(all_stems)):
        raise SystemExit("Split assignment contains duplicate samples.")

    by_stem = {sample.stem: sample for sample in samples}
    all_classes = {class_id for sample in samples for class_id in sample.classes}
    for split in ("train", "valid", "test"):
        present = {class_id for stem in splits[split] for class_id in by_stem[stem].classes}
        missing = sorted(all_classes - present)
        if missing:
            raise SystemExit(f"{split} is missing classes: {missing}")


def rewrite_dataset(root: Path, samples: list[Sample], splits: dict[str, list[str]]) -> None:
    tmp_root = root / "_resplit_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    by_stem = {sample.stem: sample for sample in samples}
    for split, stems in splits.items():
        image_dir = tmp_root / split / "images"
        label_dir = tmp_root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            sample = by_stem[stem]
            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
            shutil.copy2(sample.label_path, label_dir / sample.label_path.name)

    for split in SPLITS:
        for kind in ("images", "labels"):
            target = root / split / kind
            shutil.rmtree(target)
            shutil.move(str(tmp_root / split / kind), str(target))
    shutil.rmtree(tmp_root)


def write_manifest(
    root: Path,
    samples: list[Sample],
    splits: dict[str, list[str]],
    seed: int,
    ratio: tuple[int, int, int],
) -> None:
    by_stem = {sample.stem: sample for sample in samples}
    manifest = {
        "seed": seed,
        "ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
        "counts": {split: len(stems) for split, stems in splits.items()},
        "splits": {
            split: [
                {
                    "stem": stem,
                    "image": by_stem[stem].image_path.name,
                    "label": by_stem[stem].label_path.name,
                    "classes": sorted(by_stem[stem].classes),
                }
                for stem in stems
            ]
            for split, stems in splits.items()
        },
    }
    write_yaml(root / "split_manifest.yaml", manifest)


def remove_caches(root: Path) -> None:
    for cache_path in root.glob("*/labels.cache"):
        cache_path.unlink()


def print_summary(root: Path, samples: list[Sample], splits: dict[str, list[str]]) -> None:
    by_stem = {sample.stem: sample for sample in samples}
    total = sum(len(stems) for stems in splits.values())
    for split, stems in splits.items():
        classes = {class_id for stem in stems for class_id in by_stem[stem].classes}
        print(f"{split:5s}: {len(stems):4d} images ({len(stems) / total:.2%}), classes={len(classes)}")
    print(f"manifest: {root / 'split_manifest.yaml'}")


if __name__ == "__main__":
    main()
