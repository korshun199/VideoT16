#!/usr/bin/env python3
"""Собирает компактный сбалансированный набор для дообучения FPV-модели.

Набор не копирует гигабайты фотографий: в каталоге результата создаются
жёсткие ссылки на локальные исходники. Все входные файлы остаются неизменны.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


# Корень проекта; поэтому скрипт можно запускать из любого каталога.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Итоговый независимый от старых generated-каталогов набор.
OUTPUT_DIR = PROJECT_DIR / "dataset/fpv/compact_training"
# Повторяемый выбор фонов и разбиение на обучение/проверку.
RANDOM_SEED = 42
# Доля роликов, целиком отводимых на проверку.
VALIDATION_RATIO = 0.20
# Поддерживаемые форматы кадров.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Sample:
    """Одна пара изображения и YOLO-разметки с типом и исходным роликом."""

    image: Path
    label: Path
    source_group: str
    kind: str
    digest: str


def file_sha256(path: Path) -> str:
    """Возвращает SHA-256 файла для исключения одинаковых кадров."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_yolo_label(label: Path, *, allow_empty: bool) -> bool:
    """Проверяет YOLO detect-разметку; пустой файл допустим только для фона."""
    try:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError):
        return False
    if not lines:
        return allow_empty
    for line in lines:
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            return False
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return False
        if not all(0.0 <= value <= 1.0 for value in values):
            return False
    return True


def organized_group(name: str) -> str:
    """Извлекает идентификатор ролика из имени канонического кадра."""
    if name.startswith("flash_"):
        return "flash:" + name.split("__", 1)[0][6:]
    if name.startswith("old__"):
        parts = name.split("__", 2)
        return "old:" + (parts[1] if len(parts) > 1 else name)
    return "organized:" + name.split("__", 1)[0]


def incoming_group(name: str) -> str:
    """Извлекает исходный ролик из имени импортированного кадра с флешки."""
    match = re.match(r"(dataset_\d+_[0-9a-f-]{36})_frame_", name)
    return "incoming:" + (match.group(1) if match else name.split("_frame_", 1)[0])


def add_positive_directory(
    image_dir: Path,
    label_dir: Path,
    group_for_name,
    samples: list[Sample],
    seen: set[str],
) -> None:
    """Добавляет уникальные корректно размеченные кадры с квадрокоптером."""
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file() or not valid_yolo_label(label, allow_empty=False):
            continue
        digest = file_sha256(image)
        if digest in seen:
            continue
        seen.add(digest)
        samples.append(Sample(image, label, group_for_name(image.name), "object", digest))


def collect_positives() -> list[Sample]:
    """Собирает старый канон, новый импорт и пять пропущенных ручных кадров."""
    samples: list[Sample] = []
    seen: set[str] = set()
    add_positive_directory(
        PROJECT_DIR / "dataset/fpv/organized/images",
        PROJECT_DIR / "dataset/fpv/organized/labels",
        organized_group,
        samples,
        seen,
    )
    add_positive_directory(
        PROJECT_DIR / "dataset/fpv/incoming_flash_20260911/images",
        PROJECT_DIR / "dataset/fpv/incoming_flash_20260911/labels",
        incoming_group,
        samples,
        seen,
    )
    add_positive_directory(
        PROJECT_DIR / "dataset/fpv/images/annotated",
        PROJECT_DIR / "dataset/fpv/labels/annotated",
        lambda name: "manual:" + name.split("__", 1)[0],
        samples,
        seen,
    )
    return samples


def collect_backgrounds(object_hashes: set[str], needed: int) -> list[Sample]:
    """Выбирает уникальные пустые кадры из старых локальных роликов."""
    candidates_by_group: dict[str, list[Sample]] = defaultdict(list)
    seen: set[str] = set()
    archive = PROJECT_DIR / "dataset/fpv/flash_archive"
    for video_dir in sorted(archive.glob("video_*")):
        image_dir = video_dir / "images"
        label_dir = video_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        group = "flash:" + video_dir.name.removeprefix("video_")
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file() or not valid_yolo_label(label, allow_empty=True):
                continue
            if label.read_text(encoding="utf-8").strip():
                continue
            digest = file_sha256(image)
            if digest in object_hashes or digest in seen:
                continue
            seen.add(digest)
            candidates_by_group[group].append(Sample(image, label, group, "background", digest))

    candidates = [sample for group in sorted(candidates_by_group) for sample in candidates_by_group[group]]
    if len(candidates) < needed:
        raise RuntimeError(f"Недостаточно уникальных фонов: {len(candidates)} < {needed}")
    random.Random(RANDOM_SEED).shuffle(candidates)
    return candidates[:needed]


def split_by_source(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Делит набор по источникам, не разрывая соседние кадры одного ролика."""
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.source_group].append(sample)
    groups = list(by_group)
    random.Random(RANDOM_SEED).shuffle(groups)
    target = round(len(samples) * VALIDATION_RATIO)
    validation_groups: set[str] = set()
    validation_count = 0
    for group in groups:
        if validation_count >= target:
            break
        validation_groups.add(group)
        validation_count += len(by_group[group])
    return {
        "train": [sample for sample in samples if sample.source_group not in validation_groups],
        "val": [sample for sample in samples if sample.source_group in validation_groups],
    }


def link_sample(sample: Sample, split: str, number: int) -> dict[str, str]:
    """Создаёт жёсткие ссылки изображения и метки с безопасным уникальным именем."""
    stem = f"{sample.kind}_{number:05d}_{sample.digest[:12]}"
    image_target = OUTPUT_DIR / split / "images" / f"{stem}{sample.image.suffix.lower()}"
    label_target = OUTPUT_DIR / split / "labels" / f"{stem}.txt"
    image_target.hardlink_to(sample.image)
    label_target.hardlink_to(sample.label)
    return {
        "split": split,
        "kind": sample.kind,
        "source_group": sample.source_group,
        "source_image": str(sample.image.relative_to(PROJECT_DIR)),
        "image": str(image_target.relative_to(OUTPUT_DIR)),
    }


def main() -> int:
    """Создаёт компактный набор и подробный манифест без изменения источников."""
    if OUTPUT_DIR.exists():
        raise SystemExit(f"Каталог уже существует, ничего не перезаписано: {OUTPUT_DIR}")
    positives = collect_positives()
    backgrounds = collect_backgrounds({sample.digest for sample in positives}, len(positives))
    samples = positives + backgrounds
    splits = split_by_source(samples)
    for split in splits:
        (OUTPUT_DIR / split / "images").mkdir(parents=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True)

    manifest: list[dict[str, str]] = []
    for split in ("train", "val"):
        for number, sample in enumerate(sorted(splits[split], key=lambda item: (item.kind, item.digest)), 1):
            manifest.append(link_sample(sample, split, number))

    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR}\ntrain: train/images\nval: val/images\nnames:\n  0: quadcopter\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {split: {kind: sum(item.kind == kind for item in items) for kind in ("object", "background")} for split, items in splits.items()}
    (OUTPUT_DIR / "README.md").write_text(
        "# Компактный сбалансированный датасет FPV\n\n"
        "Собран без копирования исходников: изображения и метки — жёсткие ссылки.\n\n"
        f"Объектов: {len(positives)}. Фонов: {len(backgrounds)}. Всего: {len(samples)}.\n\n"
        f"Train: объектов {counts['train']['object']}, фонов {counts['train']['background']}.\n"
        f"Val: объектов {counts['val']['object']}, фонов {counts['val']['background']}.\n\n"
        "Класс единственный: `0 = quadcopter`. Разбиение сделано целыми исходными роликами.\n",
        encoding="utf-8",
    )
    print(f"Готово: {OUTPUT_DIR}")
    print(f"Объекты: {len(positives)}, фоны: {len(backgrounds)}, всего: {len(samples)}")
    print(f"Train: {len(splits['train'])}; Val: {len(splits['val'])}")
    print(f"Манифест: {OUTPUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
