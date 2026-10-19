#!/usr/bin/env python3

"""Проверяет чтение видео и отрисовку OSD на ARM64 без YOLO и камеры."""

import sys
from pathlib import Path

import cv2

# Добавляем корень проекта, чтобы запуск из arm64-lab находил src и config.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inav_msp import InavTelemetry
from src.local_object_detection import draw_detections, draw_telemetry
from src.realtime import Detection


def main() -> None:
    """Обрабатывает весь ролик и сохраняет результат с рамкой и OSD."""
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать результат: {output_path}")

    frames = 0
    telemetry = InavTelemetry(gps_fix=1, gps_satellites=8, ground_speed=5.0, ground_course=90.0)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = (Detection(20 + frames * 15, 130, 100 + frames * 15, 210, "OBJECT", 0.91),)
        annotated = draw_detections(frame, detections)
        draw_telemetry(annotated, telemetry)
        writer.write(annotated)
        frames += 1
    capture.release()
    writer.release()
    if frames != 30:
        raise RuntimeError(f"Ожидалось 30 кадров, получено: {frames}")
    print(f"SMOKE_OK frames={frames} output={output_path}")


if __name__ == "__main__":
    main()
