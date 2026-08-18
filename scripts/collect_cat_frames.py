#!/usr/bin/env python3
"""Сохраняет кадры Logitech для последующей разметки кота."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2


def main() -> int:
    """Снимает кадры через заданный интервал до нажатия q."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/dev/video4")
    parser.add_argument("--output", type=Path, default=Path("dataset/my_cat/images/raw"))
    parser.add_argument("--interval", type=float, default=2.0, help="Интервал между кадрами в секундах")
    args = parser.parse_args()
    if args.interval <= 0:
        raise SystemExit("--interval должен быть больше нуля")

    args.output.mkdir(parents=True, exist_ok=True)
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source, cv2.CAP_V4L2 if isinstance(source, int) else cv2.CAP_ANY)
    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть камеру: {args.source}")

    saved = 0
    next_capture = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Камера не вернула кадр")
            now = time.monotonic()
            if now >= next_capture:
                target = args.output / f"cat_{saved:05d}.jpg"
                cv2.imwrite(str(target), frame)
                saved += 1
                next_capture = now + args.interval
                print(f"Сохранён кадр: {target}")
            cv2.imshow("Сбор кадров кота — q для выхода", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Всего кадров: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
