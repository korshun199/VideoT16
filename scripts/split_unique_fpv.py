#!/usr/bin/env python3
"""Делит уникальный пул FPV по исходным видеокаталогам."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path


# Пути проекта, пула и результата разбиения.
PROJECT_DIR = Path(__file__).resolve().parents[1]
POOL_DIR = PROJECT_DIR / "dataset/fpv/unique_training"
SPLIT_DIR = POOL_DIR / "split"
MANIFEST_FILE = POOL_DIR / "manifest.tsv"
VAL_RATIO = 0.20
SEED = 42

# Источники нужны только для восстановления происхождения кадра.
SOURCE_ROOTS = (
    PROJECT_DIR / "dataset/fpv/organized/images",
    PROJECT_DIR / "dataset/fpv/object_photos/images",
    PROJECT_DIR / "dataset/fpv/flash_archive",
    PROJECT_DIR / "dataset/fpv/incoming_flash_20260911/images",
    Path("/media/oleg/FLASH_DRIVE/dataset"),
    Path("/media/oleg/FLASH_DRIVE/dataset_2"),
    Path("/media/oleg/FLASH_DRIVE/dataset_3"),
    Path("/media/oleg/FLASH_DRIVE/dataset_5"),
    Path("/media/oleg/FLASH_DRIVE/dataset1"),
    Path("/media/oleg/FLASH_DRIVE/dataset_4"),
    PROJECT_DIR / "dataset/fpv_windows_import",
)


def image_hash(path: Path) -> str:
    """Возвращает полный SHA-256 изображения."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_group(path: Path) -> str:
    """Возвращает идентификатор исходного видеокаталога."""
    parts = path.parts
    if "images" in parts:
        index = len(parts) - 1 - parts[::-1].index("images")
        if index > 0:
            return parts[index - 1]
    return path.parent.name


def build_source_index() -> dict[str, tuple[str, str]]:
    """Строит индекс hash изображения -> источник и видеокаталог."""
    index: dict[str, tuple[str, str]] = {}
    for root in SOURCE_ROOTS:
        if not root.exists():
            continue
        paths = root.iterdir() if root.name == "images" else root.rglob("*")
        for path in paths:
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                digest = image_hash(path)
                index.setdefault(digest, (str(path), source_group(path)))
    return index


def clean_split() -> None:
    """Удаляет только ранее созданное производное разбиение."""
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)


def main() -> int:
    """Создаёт групповой train/val split, manifest и data.yaml."""
    source_index = build_source_index()
    # Прямой индекс по префиксу имени исключает полный перебор 35 тысяч записей.
    prefix_index = {digest[:12]: (digest, value) for digest, value in source_index.items()}
    samples: list[tuple[Path, Path, str, str, bool]] = []
    conflicts = 0
    for kind in ("objects", "backgrounds"):
        image_dir = POOL_DIR / "images" / kind
        label_dir = POOL_DIR / "labels" / kind
        for image in sorted(image_dir.iterdir()):
            if not image.is_file():
                continue
            prefix = image.stem.rsplit("_", 1)[-1]
            match = prefix_index.get(prefix)
            if match is None:
                raise RuntimeError(f"Источник не найден для {image}")
            digest, (source, group) = match
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise RuntimeError(f"Разметка не найдена для {image}")
            text = label.read_text(encoding="utf-8", errors="replace").strip()
            samples.append((image, label, digest, group, bool(text)))

    groups: dict[str, list[tuple[Path, Path, str, str, bool]]] = defaultdict(list)
    for sample in samples:
        groups[sample[3]].append(sample)
    group_names = sorted(groups)
    random.Random(SEED).shuffle(group_names)
    target_val = max(1, round(len(samples) * VAL_RATIO))
    val_groups: set[str] = set()
    val_count = 0
    for name in group_names:
        if val_count >= target_val:
            break
        val_groups.add(name)
        val_count += len(groups[name])

    clean_split()
    manifest_lines = ["split\tkind\timage\tlabel\tsha256\tsource_group\tsource_file\n"]
    counts = defaultdict(int)
    for image, label, digest, group, is_object in samples:
        split = "val" if group in val_groups else "train"
        kind = "objects" if is_object else "backgrounds"
        destination_image = SPLIT_DIR / split / "images" / kind / image.name
        destination_label = SPLIT_DIR / split / "labels" / kind / label.name
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_image.hardlink_to(image)
        destination_label.hardlink_to(label)
        source_file, _ = source_index[digest]
        manifest_lines.append(f"{split}\t{kind}\t{destination_image}\t{destination_label}\t{digest}\t{group}\t{source_file}\n")
        counts[f"{split}_{kind}"] += 1

    MANIFEST_FILE.write_text("".join(manifest_lines), encoding="utf-8")
    data_file = SPLIT_DIR / "data.yaml"
    data_file.write_text(
        f"path: {SPLIT_DIR}\ntrain: train/images\nval: val/images\nnames:\n  0: quadcopter\n  1: fixed-wing\n",
        encoding="utf-8",
    )
    print(f"Групп источников: {len(groups)}")
    print(f"Train: объекты={counts['train_objects']}, фоны={counts['train_backgrounds']}")
    print(f"Val: объекты={counts['val_objects']}, фоны={counts['val_backgrounds']}")
    print(f"Каталог: {SPLIT_DIR}")
    print(f"Manifest: {MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
