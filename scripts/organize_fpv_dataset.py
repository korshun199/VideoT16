#!/usr/bin/env python3
"""Sobiraet chistuyu YOLO-vyborku iz staroy i novoy razmetki FPV."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


# Katalog proekta i istochniki dannyh.
PROJECT_DIR = Path(__file__).resolve().parents[1]
OLD_IMAGES = PROJECT_DIR / "dataset/fpv/images/annotated"
OLD_LABELS = PROJECT_DIR / "dataset/fpv/labels/annotated"
FLASH_ARCHIVE = PROJECT_DIR / "dataset/fpv/flash_archive"
OUTPUT_DIR = PROJECT_DIR / "dataset/fpv/organized"

# Klass 0 v modeli FPV - kvadrokopter. Metka Night iz importa privoditsya k nemu.
TARGET_CLASS = "0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_hash(path: Path) -> str:
    """Vozvrashchaet SHA-256 fajla izobrazheniya dlya udaleniya dubley."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_lines(path: Path) -> list[str]:
    """Chitaet ne pustye stroki razmetki bez povrezhdeniya originala."""
    try:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return []


def bbox_from_polygon(line: str) -> str | None:
    """Preobrazuyet YOLO-polygon v YOLO-bounding-box."""
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        return None
    try:
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)
    return f"{TARGET_CLASS} {(left + right) / 2:.6f} {(top + bottom) / 2:.6f} {right - left:.6f} {bottom - top:.6f}"


def normalize_label(lines: list[str], segmentation: bool = False) -> list[str]:
    """Privodit klass i format razmetki k YOLO detect dlya klassa 0."""
    result: list[str] = []
    for line in lines:
        parts = line.split()
        if segmentation:
            converted = bbox_from_polygon(line)
            if converted:
                result.append(converted)
            continue
        if len(parts) != 5:
            continue
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if all(0 <= value <= 1 for value in values):
            result.append(f"{TARGET_CLASS} " + " ".join(f"{value:.6f}" for value in values))
    return result


def label_for_image(image: Path, labels_dir: Path) -> list[str]:
    """Nakhodit obychnuyu razmetku ili rezervno preobrazuet polygon."""
    ordinary = labels_dir / f"{image.stem}.txt"
    lines = normalize_label(valid_lines(ordinary))
    if lines:
        return lines
    segmentation = labels_dir / f"{image.stem}_seg.txt"
    return normalize_label(valid_lines(segmentation), segmentation=True)


def add_pair(image: Path, labels_dir: Path, prefix: str, seen: set[str], counters: dict[str, int]) -> None:
    """Dobavlyaet odin razmechennyj kadr v canonicalnyj katalog."""
    labels = label_for_image(image, labels_dir)
    if not labels:
        counters["empty"] += 1
        return
    try:
        digest = image_hash(image)
    except OSError:
        counters["bad"] += 1
        return
    if digest in seen:
        counters["duplicate"] += 1
        return
    seen.add(digest)
    safe_name = f"{prefix}__{image.name}"
    shutil.copy2(image, OUTPUT_DIR / "images" / safe_name)
    (OUTPUT_DIR / "labels" / f"{Path(safe_name).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    counters["written"] += 1


def main() -> int:
    """Sobiraet katalog i pokazyvaet ponyatnyj otchet."""
    for directory in (OUTPUT_DIR / "images", OUTPUT_DIR / "labels"):
        directory.mkdir(parents=True, exist_ok=True)
    counters = {"written": 0, "empty": 0, "duplicate": 0, "bad": 0}
    seen: set[str] = set()

    if OLD_IMAGES.is_dir():
        for image in sorted(OLD_IMAGES.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                add_pair(image, OLD_LABELS, "old", seen, counters)

    if FLASH_ARCHIVE.is_dir():
        for images_dir in sorted(FLASH_ARCHIVE.glob("**/images")):
            labels_dir = images_dir.parent / "labels"
            for image in sorted(images_dir.iterdir()):
                if image.suffix.lower() in IMAGE_EXTENSIONS:
                    prefix = images_dir.parent.name.replace("video_", "flash_")
                    add_pair(image, labels_dir, prefix, seen, counters)

    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        "path: " + str(OUTPUT_DIR) + "\ntrain: images\nval: images\nnames:\n  0: quadcopter\n  1: fixed-wing\n",
        encoding="utf-8",
    )
    report = OUTPUT_DIR / "README.txt"
    report.write_text(
        "Canonical dataset VideoT16\n"
        "Class 0: quadcopter\n"
        "Source Night mapped to class 0. Segmentation labels converted to boxes.\n"
        f"Written: {counters['written']}\nEmpty skipped: {counters['empty']}\n"
        f"Duplicates skipped: {counters['duplicate']}\nBad files: {counters['bad']}\n",
        encoding="utf-8",
    )
    print("Organizaciya dataset FPV zavershena")
    print(f"  Zapisano razmechennyh kadrov: {counters['written']}")
    print(f"  Propushcheno pustyh: {counters['empty']}")
    print(f"  Propushcheno dubley: {counters['duplicate']}")
    print(f"  Povrezhdeno: {counters['bad']}")
    print(f"  Rezultat: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
