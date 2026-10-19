#!/usr/bin/env python3
"""Собирает уникальный рабочий набор FPV с контролируемой долей фонов."""

from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


# Корень проекта и каталог результата.
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "dataset/fpv/unique_training"
REVIEW_FILE = OUTPUT_DIR / "review/conflicts.txt"

# Источники перечислены от более доверенного к менее приоритетному.
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

# Доля фонов: на один положительный кадр берём два отрицательных.
BACKGROUND_RATIO = 2
RANDOM_SEED = 42
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Candidate:
    """Кандидат изображения и связанной с ним YOLO-разметки."""

    image: Path
    label: Path | None
    digest: str
    source: str


def label_for(image: Path) -> Path | None:
    """Возвращает соседний YOLO-файл, если он существует."""
    possible = image.parent.parent / "labels" / f"{image.stem}.txt"
    return possible if possible.is_file() else None


def iter_images(root: Path) -> list[Candidate]:
    """Находит изображения в одном источнике и вычисляет SHA-256."""
    if not root.exists():
        return []
    if root.name == "images":
        paths = root.iterdir()
    else:
        paths = root.rglob("*")
    candidates: list[Candidate] = []
    for image in sorted(paths):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        candidates.append(Candidate(image, label_for(image), digest, str(root)))
    return candidates


def label_text(candidate: Candidate) -> str | None:
    """Читает непустую разметку; пустой файл считается фоном."""
    if candidate.label is None:
        return None
    return candidate.label.read_text(encoding="utf-8", errors="replace").strip()


def copy_pair(candidate: Candidate, index: int, kind: str) -> None:
    """Копирует изображение и безопасно переименованную разметку."""
    image_dir = OUTPUT_DIR / "images" / kind
    label_dir = OUTPUT_DIR / "labels" / kind
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:06d}_{candidate.digest[:12]}"
    shutil.copy2(candidate.image, image_dir / f"{stem}{candidate.image.suffix.lower()}")
    label_path = label_dir / f"{stem}.txt"
    if candidate.label is None:
        label_path.write_text("", encoding="utf-8")
    else:
        shutil.copy2(candidate.label, label_path)


def main() -> int:
    """Собирает уникальные объекты, фоны и отчёт конфликтов."""
    groups: dict[str, list[Candidate]] = {}
    total = 0
    for root in SOURCE_ROOTS:
        for candidate in iter_images(root):
            groups.setdefault(candidate.digest, []).append(candidate)
            total += 1

    objects: list[Candidate] = []
    backgrounds: list[Candidate] = []
    conflicts: list[str] = []
    for digest, candidates in sorted(groups.items()):
        labelled = [item for item in candidates if label_text(item)]
        chosen = candidates[0]
        # Предпочитаем размеченный вариант и сохраняем конфликт для проверки.
        if labelled:
            chosen = labelled[0]
        label_variants = {label_text(item) for item in labelled}
        if len(label_variants) > 1:
            conflicts.append(f"{digest}\n" + "\n".join(f"  {item.image}" for item in candidates))
        if labelled:
            objects.append(chosen)
        else:
            backgrounds.append(chosen)

    random.Random(RANDOM_SEED).shuffle(backgrounds)
    background_limit = min(len(backgrounds), len(objects) * BACKGROUND_RATIO)
    selected_backgrounds = backgrounds[:background_limit]

    # Каталог результата создаётся новый; исходники не затрагиваются.
    (OUTPUT_DIR / "review").mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(objects, 1):
        copy_pair(candidate, index, "objects")
    for index, candidate in enumerate(selected_backgrounds, 1):
        copy_pair(candidate, index, "backgrounds")

    REVIEW_FILE.write_text("\n\n".join(conflicts) + ("\n" if conflicts else ""), encoding="utf-8")
    (OUTPUT_DIR / "data.yaml").write_text(
        f"path: {OUTPUT_DIR}\ntrain: images/train\nval: images/val\nnames:\n  0: quadcopter\n  1: fixed-wing\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(
        "# Уникальный рабочий набор FPV\n\n"
        f"Исходных файлов проверено: {total}\n"
        f"Уникальных объектов: {len(objects)}\n"
        f"Выбранных фонов: {len(selected_backgrounds)}\n"
        f"Конфликтных групп разметки: {len(conflicts)}\n"
        "Дубликаты исключены по SHA-256. Исходные каталоги не изменялись.\n",
        encoding="utf-8",
    )
    print(f"Проверено файлов: {total}")
    print(f"Уникальных объектов: {len(objects)}")
    print(f"Выбрано фонов: {len(selected_backgrounds)}")
    print(f"Конфликтных групп: {len(conflicts)}")
    print(f"Результат: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
