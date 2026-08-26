#!/usr/bin/env python3

"""Создаёт небольшой тестовый видеопоток без камеры и внешних устройств."""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Записывает движущийся квадрат для проверки чтения кадров."""
    output = Path(__file__).parent / "shared" / "arm64_input.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (640, 360),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео: {output}")
    for index in range(30):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        x = 20 + index * 15
        cv2.rectangle(frame, (x, 130), (x + 80, 210), (0, 255, 0), 3)
        cv2.putText(frame, f"FRAME {index + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    print(output)


if __name__ == "__main__":
    main()

