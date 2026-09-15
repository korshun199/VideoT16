#!/usr/bin/env python3
"""Экспортирует облегчённую Anti-UAV и создаёт INT8-вариант для Raspberry."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "models/anti_uav_best.pt"
DEFAULT_CALIBRATION_DIR = PROJECT_DIR / "dataset/fpv/compact_training/train/images"


def arguments() -> argparse.Namespace:
    """Читает безопасные параметры экспорта без изменения исходной модели."""
    parser = argparse.ArgumentParser(description="Ускорение модели Anti-UAV для Raspberry")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-onnx", type=Path, help="готовый FP32 ONNX для квантизации без экспорта")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--skip-export", action="store_true", help="использовать уже готовый FP32 ONNX")
    parser.add_argument("--per-channel", action="store_true", help="точнее квантизовать каждый канал весов")
    parser.add_argument("--force", action="store_true", help="разрешить замену готовых результатов")
    return parser.parse_args()


def calibration_paths(directory: Path, count: int) -> list[Path]:
    """Равномерно выбирает кадры для определения диапазонов INT8."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)
    if not paths:
        raise RuntimeError(f"Нет калибровочных изображений: {directory}")
    if len(paths) <= count:
        return paths
    indexes = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[index] for index in indexes]


def prepare_image(path: Path, size: int) -> np.ndarray:
    """Готовит кадр точно так же, как рабочий ONNX-детектор."""
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Не удалось прочитать кадр: {path}")
    height, width = frame.shape[:2]
    scale = min(size / width, size / height)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized.shape[1]) // 2
    top = (size - resized.shape[0]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0


class ImageCalibrationReader(CalibrationDataReader):
    """Подаёт реальные кадры проекта калибратору ONNX Runtime."""

    def __init__(self, paths: list[Path], input_name: str, size: int) -> None:
        self.paths = iter(paths)
        self.input_name = input_name
        self.size = size

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Возвращает следующий подготовленный кадр или завершает калибровку."""
        try:
            path = next(self.paths)
        except StopIteration:
            return None
        return {self.input_name: prepare_image(path, self.size)}


def main() -> int:
    """Создаёт FP32 256×256 и квантизованную INT8-модель."""
    args = arguments()
    if args.size < 32 or args.size % 32:
        raise SystemExit("--size должен быть не меньше 32 и кратен 32")
    if args.calibration_count < 16:
        raise SystemExit("--calibration-count должен быть не меньше 16")
    if not args.source.is_file():
        raise SystemExit(f"Исходная модель не найдена: {args.source}")

    fp32_path = args.source_onnx or PROJECT_DIR / f"models/anti_uav_fast_{args.size}.onnx"
    int8_suffix = "int8_pc" if args.per_channel else "int8"
    int8_path = PROJECT_DIR / f"models/anti_uav_fast_{args.size}_{int8_suffix}.onnx"
    skip_export = args.skip_export or args.source_onnx is not None
    checked_paths = (int8_path,) if skip_export else (fp32_path, int8_path)
    existing = [path for path in checked_paths if path.exists()]
    if existing and not args.force:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"Результат уже существует: {names}; для замены добавь --force")

    if skip_export:
        if not fp32_path.is_file():
            raise SystemExit(f"Готовый FP32 ONNX не найден: {fp32_path}")
    else:
        with tempfile.TemporaryDirectory(prefix="videot16-anti-uav-") as temporary:
            temporary_dir = Path(temporary)
            temporary_pt = temporary_dir / f"anti_uav_fast_{args.size}.pt"
            shutil.copy2(args.source, temporary_pt)
            exported = Path(
                YOLO(temporary_pt).export(
                    format="onnx",
                    imgsz=args.size,
                    opset=17,
                    simplify=True,
                    dynamic=False,
                    half=False,
                    device="cpu",
                )
            )
            shutil.copy2(exported, fp32_path)

    selected_paths = calibration_paths(DEFAULT_CALIBRATION_DIR, args.calibration_count)
    reader = ImageCalibrationReader(selected_paths, "images", args.size)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # Поканальная шкала точнее, общая требует меньше памяти при сборке.
        per_channel=args.per_channel,
        op_types_to_quantize=["Conv"],
    )
    print(f"FP32: {fp32_path} ({fp32_path.stat().st_size / 1024 / 1024:.1f} МБ)")
    print(f"INT8: {int8_path} ({int8_path.stat().st_size / 1024 / 1024:.1f} МБ)")
    print(f"Калибровочных кадров: {len(selected_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
