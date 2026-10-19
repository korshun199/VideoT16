#!/usr/bin/env python3
"""Дообучает одноклассовую модель на всех известных квадрокоптерах."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Пути проекта, данных и исходной модели.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_DIR / "dataset/fpv/unique_training/split/data_quadcopter.yaml"
BASE_MODEL = PROJECT_DIR / "models/fpv_drone_best.pt"
RUNS_DIR = PROJECT_DIR / "runs/fpv_training"
RUN_NAME = "unique_quadcopter_20260914_320"

# Консервативные параметры для дообучения без перезаписи базовой модели.
IMAGE_SIZE = 320
EPOCHS = 30
BATCH_SIZE = 8
PATIENCE = 8
# Фиксированное число ядер для основного прогона обучения.
CPU_CORES = 8


def configure_cpu_limit() -> None:
    """Ограничивает обучение примерно 80 процентами процессора."""
    cpu_count = os.cpu_count() or 1
    allowed_count = min(CPU_CORES, cpu_count)
    try:
        os.sched_setaffinity(0, set(range(allowed_count)))
    except (AttributeError, OSError):
        pass
    os.environ["OMP_NUM_THREADS"] = str(allowed_count)
    os.environ["MKL_NUM_THREADS"] = str(allowed_count)
    os.environ["OPENBLAS_NUM_THREADS"] = str(allowed_count)
    print(f"Ограничение CPU: {allowed_count}/{cpu_count} ядер")


def validate_inputs() -> None:
    """Проверяет наличие модели, YAML и минимальных разделов данных."""
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(f"Базовая модель не найдена: {BASE_MODEL}")
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Конфигурация данных не найдена: {DATA_FILE}")
    for split in ("train", "val"):
        image_dir = DATA_FILE.parent / split / "images"
        if not image_dir.is_dir() or not any(image_dir.rglob("*.jpg")):
            raise RuntimeError(f"Нет изображений в разделе {split}: {image_dir}")


def main() -> int:
    """Запускает осторожное дообучение и экспортирует ONNX-копию."""
    validate_inputs()
    configure_cpu_limit()
    from ultralytics import YOLO

    print(f"База: {BASE_MODEL}")
    print(f"Данные: {DATA_FILE}")
    print("Класс: quadcopter")
    model = YOLO(str(BASE_MODEL))
    if "--check" in sys.argv:
        print(f"Исходные классы модели: {model.names}")
        print("Проверка входов: ОК")
        return 0
    model.train(
        data=str(DATA_FILE),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device="cpu",
        workers=2,
        cache=False,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,
        pretrained=True,
        patience=PATIENCE,
        optimizer="AdamW",
        lr0=0.0005,
        momentum=0.9,
        lrf=0.01,
        warmup_epochs=3.0,
        fliplr=0.5,
        degrees=8.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    best_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"После обучения не найден best.pt: {best_path}")
    print(f"Лучшая модель: {best_path}")
    exported = YOLO(str(best_path)).export(format="onnx", imgsz=IMAGE_SIZE, simplify=False)
    print(f"ONNX экспортирован: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
