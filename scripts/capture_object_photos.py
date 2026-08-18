#!/usr/bin/env python3
"""Сохраняет фотографии предмета по нажатию пробела."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def open_source(value: str):
    """Открывает V4L2-устройство или сетевой поток телефона."""
    source = int(value) if value.isdigit() else value
    if isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/video")):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cv2.VideoCapture(source)


def main() -> int:
    """Показывает поток и сохраняет кадр на Space."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/dev/video4", help="Источник камеры; по умолчанию Logitech Brio /dev/video4")
    parser.add_argument("--name", required=True, help="Имя предмета для каталога")
    parser.add_argument("--output", type=Path, default=Path("dataset/objects"))
    args = parser.parse_args()
    if not args.name.replace("_", "").isalnum():
        raise SystemExit("--name должен содержать только буквы, цифры и _")

    target_dir = args.output / args.name
    target_dir.mkdir(parents=True, exist_ok=True)
    capture = open_source(args.source)
    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть источник: {args.source}")

    saved = len(list(target_dir.glob("*.jpg")))
    window_name = "Съёмка предмета: Space — сохранить, q — выход"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("Источник не вернул кадр")
            preview = frame.copy()
            cv2.putText(preview, f"{args.name}: {saved} фото", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32:
                target = target_dir / f"{args.name}_{saved:05d}.jpg"
                if cv2.imwrite(str(target), frame):
                    saved += 1
                    print(f"Сохранено: {target}", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Всего фотографий: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
