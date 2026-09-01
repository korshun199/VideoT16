#!/usr/bin/env python3
"""Размечает фотографии FPV-дронов рамками в формате YOLO."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2

# Только новые источники для текущего сеанса разметки.
# Старые ролики здесь намеренно не указаны.
SOURCE_DIRS = [
    Path("dataset/fpv/images/from_videos/8332592220862"),
    Path("dataset/fpv/images/from_videos/additional"),
]
# Каталог, куда будут скопированы фотографии для обучения.
IMAGE_DIR = Path("dataset/fpv/images/annotated")
# Каталог с YOLO-разметкой.
LABEL_DIR = Path("dataset/fpv/labels/annotated")
# Класс текущих фотографий: 0 — квадрокоптер, 1 — самолётный дрон.
CLASS_ID = 0
# Техническое имя окна: ASCII надёжнее работает с OpenCV/Qt.
WINDOW_NAME = "FPV_ANNOTATE"


def yolo_box(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> str:
    """Переводит пиксельную рамку в нормированный формат YOLO."""
    center_x = (x + width / 2) / image_width
    center_y = (y + height / 2) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    return f"{CLASS_ID} {center_x:.6f} {center_y:.6f} {normalized_width:.6f} {normalized_height:.6f}\n"


def select_box(window_name: str, image):
    """Позволяет нарисовать рамку мышью в окне OpenCV."""
    state = {"start": None, "current": None, "finished": False}

    def on_mouse(event, x, y, _flags, _userdata):
        """Запоминает начало, движение и окончание рамки мышью."""
        if event == cv2.EVENT_LBUTTONDOWN:
            state["start"] = (x, y)
            state["current"] = (x, y)
            state["finished"] = False
        elif event == cv2.EVENT_MOUSEMOVE and state["start"] is not None:
            state["current"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and state["start"] is not None:
            state["current"] = (x, y)
            state["finished"] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(100)
    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while True:
            preview = image.copy()
            if state["start"] is not None and state["current"] is not None:
                cv2.rectangle(preview, state["start"], state["current"], (0, 255, 0), 5)
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q")):
                return None, True
            if key in (ord("c"), ord("C")):
                return None, False
            if key == 32 and state["finished"]:
                x1, y1 = state["start"]
                x2, y2 = state["current"]
                x, right = sorted((x1, x2))
                y, bottom = sorted((y1, y2))
                if right > x and bottom > y:
                    return (x, y, right - x, bottom - y), False
    finally:
        cv2.setMouseCallback(window_name, lambda *_args: None)


def main() -> int:
    """Показывает фотографии и сохраняет рамки после подтверждения пробелом."""
    if CLASS_ID not in (0, 1):
        raise SystemExit("CLASS_ID должен быть 0 или 1")
    images = sorted(
        path
        for source_dir in SOURCE_DIRS
        if source_dir.is_dir()
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise SystemExit(f"В новых источниках нет фотографий: {SOURCE_DIRS}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Найдено фотографий: {len(images)}")
    print(f"Класс: {CLASS_ID} ({'quadcopter' if CLASS_ID == 0 else 'fixed-wing'})")
    pending = 0

    try:
        for number, source_path in enumerate(images, 1):
            # Имя результата учитывает папку исходного видео.
            unique_name = f"{source_path.parent.name}__{source_path.stem}"
            target_image = IMAGE_DIR / f"{unique_name}{source_path.suffix.lower()}"
            target_label = LABEL_DIR / f"{unique_name}.txt"
            if target_label.exists():
                continue

            image = cv2.imread(str(source_path))
            if image is None:
                print(f"Пропуск повреждённого файла: {source_path.name}")
                continue
            pending += 1
            print(f"[{number}/{len(images)}] Обведи объект мышью и нажми ПРОБЕЛ")
            selected, quit_requested = select_box(WINDOW_NAME, image)
            if quit_requested:
                print("Разметка остановлена пользователем.")
                return 0
            if selected is None:
                print("Кадр пропущен")
                continue
            x, y, width, height = selected
            shutil.copy2(source_path, target_image)
            target_label.write_text(
                yolo_box(x, y, width, height, image.shape[1], image.shape[0]),
                encoding="utf-8",
            )
            print(f"Сохранено: {target_image.name}")
    finally:
        cv2.destroyAllWindows()
    print(f"Разметка завершена. Новых кадров обработано: {pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
