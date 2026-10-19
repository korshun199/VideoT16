#!/usr/bin/env python3

"""Экспортирует военную YOLO-модель в ONNX для ARM64-проверки."""

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры экспорта модели."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/military_vehicle_best.pt"),
        help="Исходный файл весов YOLO",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arm64-lab/shared/military_vehicle_320.onnx"),
        help="Куда сохранить ONNX-файл",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Размер квадратного входа")
    parser.add_argument("--opset", type=int, default=11, help="Версия набора операторов ONNX")
    parser.add_argument("--simplify", action="store_true", help="Упростить граф ONNX")
    return parser


def main() -> int:
    """Экспортирует модель с фиксированным входом без динамических осей."""
    from ultralytics import YOLO

    args = build_parser().parse_args()
    model = YOLO(str(args.model))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=False,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)
    print(f"ONNX-модель создана: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
