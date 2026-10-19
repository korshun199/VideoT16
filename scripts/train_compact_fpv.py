#!/usr/bin/env python3
"""Дообучает FPV YOLO на компактном сбалансированном наборе."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# Корень проекта позволяет запускать скрипт из любого рабочего каталога.
PROJECT_DIR = Path(__file__).resolve().parents[1]
# Базовая общая модель; она сохраняет знание разных типов квадрокоптеров.
BASE_MODEL = PROJECT_DIR / "models/fpv_drone_best.pt"
# Проверенный компактный датасет: объекты и фоны, разделённые по роликам.
DATA_FILE = PROJECT_DIR / "dataset/fpv/compact_training/data.yaml"
# Результаты каждого запуска сохраняются отдельно от исходных моделей.
RUNS_DIR = PROJECT_DIR / "runs/fpv_training"
# Размер соответствует рабочей 320-пиксельной модели Raspberry Pi.
IMAGE_SIZE = 320
# Число логических ядер для математики PyTorch.
CPU_CORES = min(8, os.cpu_count() or 1)


def parse_arguments() -> argparse.Namespace:
    """Читает короткие параметры запуска без изменения исходного набора."""
    parser = argparse.ArgumentParser(description="Дообучение FPV-модели на CPU")
    parser.add_argument("--epochs", type=int, default=15, help="число эпох, по умолчанию 15")
    parser.add_argument("--batch", type=int, default=8, help="размер пакета, по умолчанию 8")
    parser.add_argument("--check", action="store_true", help="только проверить файлы и параметры")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch < 1:
        parser.error("epochs и batch должны быть положительными")
    return arguments


def configure_cpu() -> None:
    """Настраивает число потоков до импорта PyTorch и Ultralytics."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(CPU_CORES)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
    (PROJECT_DIR / ".cache/matplotlib").mkdir(parents=True, exist_ok=True)


def verify_dataset() -> tuple[int, int]:
    """Проверяет пары image/label и возвращает размеры train и val."""
    if not BASE_MODEL.is_file():
        raise RuntimeError(f"Нет базовой модели: {BASE_MODEL}")
    if not DATA_FILE.is_file():
        raise RuntimeError(f"Нет описания датасета: {DATA_FILE}")
    counts: list[int] = []
    for split in ("train", "val"):
        image_dir = DATA_FILE.parent / split / "images"
        label_dir = DATA_FILE.parent / split / "labels"
        images = [path for path in image_dir.iterdir() if path.is_file()]
        missing = [path.name for path in images if not (label_dir / f"{path.stem}.txt").is_file()]
        if missing:
            raise RuntimeError(f"{split}: нет меток для {len(missing)} кадров")
        counts.append(len(images))
    if min(counts) < 1:
        raise RuntimeError("Пустой train или val")
    return counts[0], counts[1]


def main() -> int:
    """Выполняет проверку либо дообучение и экспорт лучшей модели в ONNX."""
    arguments = parse_arguments()
    configure_cpu()
    train_count, val_count = verify_dataset()
    print(f"Датасет: train={train_count}, val={val_count}; размер={IMAGE_SIZE}; CPU={CPU_CORES}")
    if arguments.check:
        print("Проверка: ОК")
        return 0

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(CPU_CORES)
    torch.set_num_interop_threads(1)
    print(f"Обучение: {arguments.epochs} эпох, batch={arguments.batch}. Исходная модель не будет перезаписана.")
    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATA_FILE),
        imgsz=IMAGE_SIZE,
        epochs=arguments.epochs,
        batch=arguments.batch,
        device="cpu",
        workers=2,
        cache=False,
        patience=5,
        pretrained=True,
        fliplr=0.5,
        degrees=8.0,
        translate=0.08,
        scale=0.35,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        project=str(RUNS_DIR),
        name="compact_quadcopter_320",
        exist_ok=False,
    )
    run_dirs = sorted(RUNS_DIR.glob("compact_quadcopter_320*"), key=lambda path: path.stat().st_mtime)
    best_model = run_dirs[-1] / "weights/best.pt" if run_dirs else Path()
    if not best_model.is_file():
        raise RuntimeError("Обучение завершилось без файла best.pt")
    print(f"Лучшая модель: {best_model}")
    YOLO(str(best_model)).export(format="onnx", imgsz=IMAGE_SIZE, simplify=False)
    print(f"ONNX: {best_model.with_suffix('.onnx')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
