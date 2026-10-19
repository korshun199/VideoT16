#!/usr/bin/env python3
"""Запускает переносимый VideoT16 после расшифровки Windows-payload."""

from __future__ import annotations

import sys
import time
from pathlib import Path


def choose_camera_source() -> str:
    """Выбирает USB-capture 1, а при отказе открывает встроенную камеру 0."""
    import cv2

    for index in (1, 0):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not capture.isOpened():
                continue
            time.sleep(0.3)
            ok, frame = capture.read()
            if ok and frame is not None:
                print(f"Camera source: {index}")
                return str(index)
        finally:
            capture.release()
    raise RuntimeError("Не удалось открыть ни USB-capture (1), ни встроенную камеру (0)")


def make_windows_open_capture(original_open_capture, cv2_module):
    """Возвращает открытие числовой Windows-камеры через DirectShow, не V4L2."""
    def open_capture(source, camera_settings=None):
        """Передаёт индекс камеры Windows в DirectShow, остальные источники — штатно."""
        if isinstance(source, int):
            return cv2_module.VideoCapture(source, cv2_module.CAP_DSHOW)
        return original_open_capture(source, camera_settings)
    return open_capture


def main() -> int:
    """Запускает локальное распознавание с камеры Windows."""
    import cv2

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import src.local_object_detection as detection

    source = choose_camera_source()
    detection.open_capture = make_windows_open_capture(detection.open_capture, cv2)
    sys.argv = [
        "videot16-windows",
        "--source", source,
        "--model", str(root / "models" / "fpv_drone_best.onnx"),
        "--confidence-percent", "40",
        "--inference-size", "640",
        "--inference-interval", "1",
        "--clean-window",
    ]
    return detection.main()


if __name__ == "__main__":
    raise SystemExit(main())
