#!/usr/bin/env python3
"""Дообучает локальную YOLO на классе «мой кот»."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    """Запускает обучение на CPU и сохраняет лучший локальный файл весов."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--data", type=Path, default=Path("dataset/my_cat/data.yaml"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.base_model.is_file():
        raise SystemExit(f"Не найдена базовая модель: {args.base_model}")
    if not args.data.is_file():
        raise SystemExit(f"Не найден файл датасета: {args.data}")
    if args.epochs < 1:
        raise SystemExit("--epochs должен быть больше нуля")

    from ultralytics import YOLO

    model = YOLO(str(args.base_model))
    model.train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, device="cpu", workers=2, project="runs", name="my_cat")
    best = Path("runs/detect/runs/my_cat/weights/best.pt")
    print(f"Обучение завершено. Лучшие веса: {best}")
    print("Для распознавания запустите: ./run_camera.sh --model runs/my_cat/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
