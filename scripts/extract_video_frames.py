#!/usr/bin/env python3

"""Разбивка всех видео из каталога на кадры для последующей разметки."""

from pathlib import Path

import cv2


# Папка с исходными видеозаписями.
VIDEO_DIR = Path("dataset/fpv/video")
# Для каждого видео создаётся отдельная подпапка с его именем.
OUTPUT_DIR = Path("dataset/fpv/images/from_videos")
# Частота извлечения кадров.
EXTRACT_FPS = 2.0
# Качество JPEG: от 0 до 100.
JPEG_QUALITY = 95
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract_video(video_path: Path) -> int:
    """Извлекает кадры из одного видео и возвращает их количество."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"ОШИБКА: не удалось открыть видео: {video_path}")
        return 0

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0:
        print(f"ОШИБКА: не определён FPS видео: {video_path}")
        capture.release()
        return 0

    step = max(1, round(source_fps / EXTRACT_FPS))
    output_dir = OUTPUT_DIR / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("frame_*.jpg"))
    if existing:
        print(f"ПРОПУСК: папка уже содержит кадры: {output_dir}")
        capture.release()
        return 0

    print(
        f"Видео: {video_path.name} | {source_fps:.2f} FPS | "
        f"кадров: {frame_count} | извлечение: {EXTRACT_FPS:.2f} FPS"
    )

    saved = 0
    source_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % step == 0:
                target = output_dir / f"frame_{saved + 1:06d}.jpg"
                written = cv2.imwrite(
                    str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                if not written:
                    print(f"ОШИБКА: не удалось записать кадр: {target}")
                    return saved
                saved += 1
            source_index += 1
    finally:
        capture.release()

    print(f"Готово: {saved} кадров -> {output_dir}")
    return saved


def main() -> int:
    """Обрабатывает все поддерживаемые видео в заданном каталоге."""
    if EXTRACT_FPS <= 0:
        print("ОШИБКА: EXTRACT_FPS должен быть больше нуля")
        return 2
    if not VIDEO_DIR.is_dir():
        print(f"ОШИБКА: каталог видео не найден: {VIDEO_DIR}")
        return 2

    videos = sorted(
        path
        for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"ОШИБКА: в каталоге нет поддерживаемых видео: {VIDEO_DIR}")
        return 2

    print(f"Найдено видео: {len(videos)}")
    total = sum(extract_video(video_path) for video_path in videos)
    print(f"ВСЕ ГОТОВО: всего кадров извлечено: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
