#!/usr/bin/env python3
"""Проверяет задержку чистого видеопотока камеры через композитный J7."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# Добавляем корень проекта, чтобы скрипт одинаково запускался из любого каталога.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def main() -> int:
    """Показывает свежий кадр камеры на выбранном DRM-композитном выходе."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/dev/video0", help="Устройство камеры")
    parser.add_argument("--drm", default="/dev/dri/by-path/platform-1f00144000.vec-card")
    parser.add_argument("--fps", type=float, default=25.0, help="Частота камеры")
    args = parser.parse_args()

    import cv2

    from src.drm_output import DrmOutput
    from src.local_object_detection import configure_low_latency_capture
    from src.realtime import LatestFrameCapture

    capture = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {args.source}")
    configure_low_latency_capture(capture, args.fps)
    frame_capture = LatestFrameCapture(capture)
    output = DrmOutput(args.drm)
    print(
        f"Чистое видео: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS → J7 {output.width}x{output.height}",
        flush=True,
    )
    try:
        while True:
            output.write(frame_capture.latest())
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nПроверка чистого видео остановлена.")
    finally:
        frame_capture.close()
        capture.release()
        output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
