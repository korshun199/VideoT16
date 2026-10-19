#!/usr/bin/env python3
"""Показывает MJPEG-поток Raspberry в отдельном окне OpenCV."""

from __future__ import annotations

import argparse
import urllib.request

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры оконного просмотра."""
    parser = argparse.ArgumentParser(description="Оконный просмотр VideoT16")
    parser.add_argument("url", help="URL MJPEG-потока")
    return parser


def main() -> int:
    """Читает JPEG-кадры из HTTP-потока и обновляет окно."""
    args = build_parser().parse_args()
    request = urllib.request.Request(args.url, headers={"Cache-Control": "no-cache"})
    try:
        response = urllib.request.urlopen(request, timeout=10)
    except OSError as error:
        raise SystemExit(f"Не удалось открыть поток: {error}") from error

    window = "VideoT16 — камера Raspberry"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    buffer = bytearray()
    try:
        while True:
            chunk = response.read(65536)
            if not chunk:
                raise RuntimeError("Поток завершился")
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    if len(buffer) > 2_000_000:
                        del buffer[:-2]
                    break
                jpeg = bytes(buffer[start:end + 2])
                del buffer[:end + 2]
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                cv2.imshow(window, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
    finally:
        response.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
