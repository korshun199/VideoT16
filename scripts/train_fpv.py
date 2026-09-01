#!/usr/bin/env python3
"""Дополняет готовую FPV-модель размеченными фотографиями."""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

# Корень проекта нужен при запуске скрипта из любого каталога.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
# Локальный кэш Matplotlib ускоряет импорт и не зависит от домашней среды.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
(PROJECT_DIR / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)

# Исходные размеченные фотографии.
SOURCE_IMAGES = PROJECT_DIR / "dataset/fpv/images/annotated"
# Исходные YOLO-разметки.
SOURCE_LABELS = PROJECT_DIR / "dataset/fpv/labels/annotated"
# Рабочий каталог для разбиения train/val.
GENERATED_DIR = PROJECT_DIR / "dataset/fpv/generated"
# Готовая базовая модель FPV.
BASE_MODEL = PROJECT_DIR / "models/fpv_drone_best.pt"
# Название результата обучения; исходная модель не перезаписывается.
RUN_NAME = "fpv_quadcopter_custom"
# Размер изображения для обучения.
IMAGE_SIZE = 640
# Количество эпох.
EPOCHS = 50
# Размер мини-пакета для CPU ноутбука.
BATCH_SIZE = 4
# Доля проверочных изображений.
VAL_RATIO = 0.2
# Повторяемое разбиение.
RANDOM_SEED = 42


def prepare_dataset() -> Path:
    """Создаёт train/val через символические ссылки без копирования фото."""
    image_files = sorted(
        path for path in SOURCE_IMAGES.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    pairs = [path for path in image_files if (SOURCE_LABELS / f"{path.stem}.txt").is_file()]
    if len(pairs) < 2:
        raise RuntimeError("Нужно минимум две фотографии с YOLO-разметкой")

    # Удаляем только старые производные ссылки train/val, чтобы в обучение
    # случайно не попали кадры от предыдущего состава датасета.
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            generated_path = GENERATED_DIR / kind / split
            if generated_path.exists() or generated_path.is_symlink():
                shutil.rmtree(generated_path)

    random.Random(RANDOM_SEED).shuffle(pairs)
    val_count = max(1, round(len(pairs) * VAL_RATIO))
    splits = {"train": pairs[val_count:], "val": pairs[:val_count]}
    for split, files in splits.items():
        image_dir = GENERATED_DIR / "images" / split
        label_dir = GENERATED_DIR / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in files:
            image_link = image_dir / image_path.name
            label_link = label_dir / f"{image_path.stem}.txt"
            if not image_link.exists():
                image_link.symlink_to(image_path.resolve())
            label_path = SOURCE_LABELS / f"{image_path.stem}.txt"
            if not label_link.exists():
                label_link.symlink_to(label_path.resolve())

    data_path = GENERATED_DIR / "data.yaml"
    data_path.write_text(
        "path: " + str(GENERATED_DIR) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: quadcopter\n"
        "  1: fixed-wing\n",
        encoding="utf-8",
    )
    print(f"Датасет подготовлен: train={len(splits['train'])}, val={len(splits['val'])}")
    return data_path


def main() -> int:
    """Обучает модель и экспортирует лучший результат в ONNX."""
    if not BASE_MODEL.is_file():
        raise RuntimeError(f"Базовая модель не найдена: {BASE_MODEL}")
    data_path = prepare_dataset()
    from ultralytics import YOLO

    print("Обучение FPV-модели на CPU...")
    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(data_path),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device="cpu",
        workers=2,
        cache=False,
        project=str(PROJECT_DIR / "runs/fpv_training"),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    best_path = PROJECT_DIR / "runs/fpv_training" / RUN_NAME / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"Обучение завершилось без best.pt: {best_path}")
    print(f"Лучшая модель: {best_path}")
    print("Экспорт в ONNX...")
    YOLO(str(best_path)).export(format="onnx", imgsz=IMAGE_SIZE, simplify=False)
    print(f"ONNX готов: {best_path.with_suffix('.onnx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
