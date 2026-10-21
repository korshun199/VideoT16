#!/usr/bin/env python3
"""Запускает переносимый VideoT16 после расшифровки Windows-payload."""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path


def available_cameras() -> list[int]:
    """Возвращает доступные индексы камер Windows, проверенные через DirectShow."""
    import cv2

    result: list[int] = []
    for index in range(5):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not capture.isOpened():
                continue
            time.sleep(0.3)
            ok, frame = capture.read()
            if ok and frame is not None:
                result.append(index)
        finally:
            capture.release()
    return result


def choose_camera_source(index: int) -> str:
    """Проверяет выбранную пользователем Windows-камеру перед распознаванием."""
    if index not in available_cameras():
        raise RuntimeError(f"Не удалось открыть выбранную камеру Windows: {index}")
    print(f"Camera source: {index}")
    return str(index)


def make_windows_open_capture(original_open_capture, cv2_module):
    """Возвращает открытие числовой Windows-камеры через DirectShow, не V4L2."""
    def open_capture(source, camera_settings=None):
        """Передаёт индекс камеры Windows в DirectShow, остальные источники — штатно."""
        if isinstance(source, int):
            return cv2_module.VideoCapture(source, cv2_module.CAP_DSHOW)
        return original_open_capture(source, camera_settings)
    return open_capture


def main() -> int:
    """Показывает параметры Windows и запускает распознавание с выбранной моделью."""
    import cv2

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    import launcher_ui
    import src.local_object_detection as detection

    configured_root = os.environ.get("VIDEO16_INSTALL_ROOT")
    if not configured_root:
        raise RuntimeError("Не найдена пользовательская папка установки Video16")
    install_root = Path(configured_root).resolve()
    models_dir = install_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    models = launcher_ui.available_models(models_dir)
    selection = launcher_ui.choose_options(install_root, models, available_cameras())
    model_path = models_dir / str(selection["model"])
    source = choose_camera_source(int(selection["camera_index"]))
    detection.open_capture = make_windows_open_capture(detection.open_capture, cv2)
    sys.argv = [
        "videot16-windows",
        "--source", source,
        "--model", str(model_path),
        "--confidence-percent", str(selection["confidence_percent"]),
        "--inference-size", "640",
        "--inference-interval", "1",
    ]
    return detection.main()


if __name__ == "__main__":
    raise SystemExit(main())
