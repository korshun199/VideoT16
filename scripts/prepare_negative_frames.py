#!/usr/bin/env python3
"""Добавляет кадры комнаты без Фипика как отрицательные примеры."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def main() -> int:
    """Копирует отрицательные кадры в train и val без файлов разметки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/my_cat/images/raw_negative"))
    args = parser.parse_args()
    images = sorted(args.input.glob("*.jpg"))
    if not images:
        raise SystemExit(f"Отрицательные кадры не найдены: {args.input}")
    for index, image in enumerate(images):
        split = "val" if index % 5 == 0 else "train"
        target = Path("dataset/my_cat/images") / split / f"negative_{image.name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
    print(f"Добавлено отрицательных кадров: {len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
