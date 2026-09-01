#!/usr/bin/env python3
"""Создаёт локальный эталон лица владельца."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face_access import create_models, find_largest_face, make_embedding

# Номер встроенной камеры ноутбука.
CAMERA_SOURCE = 0
# Количество кадров для среднего эталона.
SAMPLES_REQUIRED = 12
# Файл с числовым эталоном лица, не фотография.
REFERENCE_PATH = Path("data/face/oleg.npy")


def main() -> int:
    """Снимает подтверждённые пробелом образцы лица и сохраняет эталон."""
    print("Загрузка моделей лица...")
    detector, recognizer = create_models()
    camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {CAMERA_SOURCE}")

    embeddings: list[np.ndarray] = []
    print("Смотрите в камеру. ПРОБЕЛ — сохранить кадр, Q — выход.")
    try:
        while len(embeddings) < SAMPLES_REQUIRED:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Не удалось получить кадр с камеры")
            face = find_largest_face(detector, frame)
            preview = frame.copy()
            if face is not None:
                x, y, width, height = map(int, face[:4])
                cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(preview, f"Samples: {len(embeddings)}/{SAMPLES_REQUIRED}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Face registration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Регистрация отменена.")
                return 1
            if key == 32:
                if face is None:
                    print("Лицо не найдено, кадр не сохранён.")
                    continue
                embeddings.append(make_embedding(recognizer, frame, face))
                print(f"Кадр принят: {len(embeddings)}/{SAMPLES_REQUIRED}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference = np.mean(np.stack(embeddings), axis=0)
    reference /= max(float(np.linalg.norm(reference)), 1e-12)
    np.save(REFERENCE_PATH, reference.astype(np.float32))
    print(f"Эталон сохранён: {REFERENCE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
