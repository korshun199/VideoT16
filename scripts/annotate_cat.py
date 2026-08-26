#!/usr/bin/env python3
"""Вручную размечает Фипика прямоугольником и создаёт YOLO-датасет."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2


def yolo_box(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> str:
    """Преобразует пиксельный прямоугольник в формат YOLO."""
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    return f"0 {center_x:.6f} {center_y:.6f} {width / image_width:.6f} {height / image_height:.6f}\n"


def select_box(window_name: str, image):
    """Выделяет объект мышью без проблемного cv2.selectROI."""
    scale = min(1.0, 1280 / image.shape[1], 720 / image.shape[0])
    display_image = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
    state = {"start": None, "end": None, "dragging": False}

    def on_mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["end"] = (x, y)
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state["end"] = (x, y)
            state["dragging"] = False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(image.shape[1], 1280), min(image.shape[0], 720))
    cv2.imshow(window_name, display_image)
    cv2.waitKey(1)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        preview = display_image.copy()
        if state["start"] and state["end"]:
            cv2.rectangle(preview, state["start"], state["end"], (0, 255, 0), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            return None, True
        if key in (ord("c"),):
            cv2.destroyWindow(window_name)
            return None, False
        if key in (ord(" "), 13) and state["start"] and state["end"]:
            x1, y1 = state["start"]
            x2, y2 = state["end"]
            cv2.destroyWindow(window_name)
            box = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            return (box, scale), False


def main() -> int:
    """Последовательно размечает все JPG-кадры."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset/my_cat/images/raw"))
    parser.add_argument("--limit", type=int, default=0, help="Разметить только первые N кадров; 0 — все")
    args = parser.parse_args()
    images = sorted(args.input.glob("*.jpg"))
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"Кадры не найдены: {args.input}")

    for index, image_path in enumerate(images):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Пропуск повреждённого кадра: {image_path}")
            continue
        height, width = image.shape[:2]
        selected, quit_requested = select_box("FIPIK_ANNOTATION", image)
        if quit_requested:
            break
        if selected is None:
            print(f"Пропуск: {image_path.name}")
            continue
        (x, y, box_width, box_height), scale = selected
        x, y = int(x / scale), int(y / scale)
        box_width, box_height = int(box_width / scale), int(box_height / scale)

        split = "val" if index % 5 == 0 else "train"
        image_target = Path("dataset/my_cat/images") / split / image_path.name
        label_target = Path("dataset/my_cat/labels") / split / f"{image_path.stem}.txt"
        image_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_target)
        label_target.write_text(yolo_box(x, y, box_width, box_height, width, height), encoding="utf-8")
        print(f"[{index + 1}/{len(images)}] Сохранено: {split}/{image_path.name}")

    cv2.destroyAllWindows()
    print("Разметка завершена.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
