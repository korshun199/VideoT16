#!/usr/bin/env python3
"""Проверяет лицо владельца через встроенную камеру."""

from __future__ import annotations

import sys
import subprocess
import time
from pathlib import Path
from OlegCore import *
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.face_access import create_models, find_largest_face, make_embedding

# Номер встроенной камеры ноутбука.
CAMERA_SOURCE = 0
# Эталон создаётся скриптом face_register.py.
REFERENCE_PATH = Path("data/face/oleg.npy")
# Порог сходства лица: 0.80 соответствует 80% для отображения настройки.
MATCH_THRESHOLD = 0.55
# Сколько секунд состояние должно сохраняться до принятия решения.
DECISION_HOLD_SECONDS = 2.0
# Локальный звук при успешном распознавании.
OPEN_SOUND_PATH = Path("models/open.wav")
CLOSE_SOUND_PATH = Path("models/close.wav")


def play_sound(sound_path: Path) -> None:
    """Воспроизводит переданный WAV через доступный проигрыватель."""
    if not sound_path.is_file():
        print(f"Предупреждение: звуковой файл не найден: {sound_path}")
        return
    players = (
        ["paplay", str(sound_path)],
        ["aplay", "-q", str(sound_path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(sound_path)],
    )
    for command in players:
        if subprocess.call(["which", command[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return


def main() -> int:
    """Проверяет лицо и печатает разрешение при совпадении."""
    if not REFERENCE_PATH.is_file():
        raise RuntimeError("Сначала выполните scripts/face_register.py")
    reference = np.load(REFERENCE_PATH)
    detector, recognizer = create_models()
    camera = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру: {CAMERA_SOURCE}")
    print("Проверка лица. Q — выход.")
    pending_state = None
    pending_started_at = None
    access_state = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Не удалось получить кадр с камеры")
            face = find_largest_face(detector, frame)
            now = time.monotonic()
            if face is not None:
                current = make_embedding(recognizer, frame, face)
                similarity = float(np.dot(reference, current))
                candidate_state = similarity >= MATCH_THRESHOLD
                x, y, width, height = map(int, face[:4])
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"match {similarity:.2f}", (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                candidate_state = False

            if candidate_state != pending_state:
                pending_state = candidate_state
                pending_started_at = now
            elif pending_started_at is not None and now - pending_started_at >= DECISION_HOLD_SECONDS:
                if access_state != pending_state:
                    access_state = pending_state
                    play_sound(OPEN_SOUND_PATH if access_state else CLOSE_SOUND_PATH)
                    message = "Доступ РАЗРЕШЕН!" if access_state else "Доступ ЗАПРЕЩЕН!"
                    (print_green if access_state else print_red)(message)
            cv2.imshow("Face check", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
