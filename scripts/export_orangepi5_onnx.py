#!/usr/bin/env python3
"""Экспортирует FPV-модель YOLO в ONNX для последующей конвертации в RKNN."""

from pathlib import Path
import shutil

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "fpv_drone_best.pt"
EXPORT_DIR = PROJECT_DIR / "models" / "orangepi5"
TARGET_PATH = EXPORT_DIR / "fpv_drone.onnx"


def main() -> int:
    """Создаёт ONNX-файл с фиксированным размером входа 640x640."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    exported = Path(model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
    ))
    shutil.copy2(exported, TARGET_PATH)
    print(f"ONNX-модель создана: {TARGET_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
