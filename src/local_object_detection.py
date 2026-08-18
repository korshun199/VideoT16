"""Локальное распознавание объектов с USB-камеры или RTSP-потока."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

RUSSIAN_LABELS = [
    "человек", "велосипед", "автомобиль", "мотоцикл", "самолёт", "автобус", "поезд", "грузовик", "лодка",
    "светофор", "пожарный гидрант", "знак стоп", "паркомат", "скамейка", "птица", "кошка", "собака",
    "лошадь", "овца", "корова", "слон", "медведь", "зебра", "жираф", "рюкзак", "зонт", "сумочка", "галстук",
    "чемодан", "фрисби", "лыжи", "сноуборд", "спортивный мяч", "воздушный змей", "бейсбольная бита",
    "бейсбольная перчатка", "скейтборд", "доска для сёрфинга", "теннисная ракетка", "бутылка", "бокал", "чашка",
    "вилка", "нож", "ложка", "миска", "банан", "яблоко", "сэндвич", "апельсин", "брокколи", "морковь",
    "хот-дог", "пицца", "пончик", "торт", "стул", "диван", "растение в горшке", "кровать", "обеденный стол",
    "унитаз", "телевизор", "ноутбук", "мышь", "пульт", "клавиатура", "мобильный телефон", "микроволновка",
    "духовка", "тостер", "раковина", "холодильник", "книга", "часы", "ваза", "ножницы", "плюшевый мишка",
    "фен", "зубная щётка",
]

def parse_source(value: str) -> int | str:
    """Преобразует номер камеры в число, а URL оставляет строкой."""
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    """Создаёт интерфейс командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="logitech", help="logitech, номер камеры, /dev/videoN или RTSP URL")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Путь к локальным весам YOLO")
    parser.add_argument("--labels", choices=("ru", "en"), default="ru", help="Язык подписей объектов")
    parser.add_argument("--confidence", type=float, default=0.35, help="Минимальная уверенность 0..1")
    parser.add_argument("--output", type=Path, help="Путь записи обработанного видео")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--headless", action="store_true", help="Работать без окна предпросмотра")
    parser.add_argument("--window-width", type=int, default=1280, help="Ширина окна предпросмотра")
    parser.add_argument("--window-height", type=int, default=720, help="Высота окна предпросмотра")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N кадров; 0 — без лимита")
    parser.add_argument("--list-cameras", action="store_true", help="Проверить камеры 0..4")
    return parser


def list_cameras() -> None:
    """Показывает доступные первые десять устройств V4L2."""
    import cv2

    for index in range(10):
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if capture.isOpened():
            print(f"Камера {index}: доступна")
            capture.release()


def create_writer(path: Path, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    """Создаёт MP4-запись с параметрами входного потока."""
    import cv2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть запись: {path}")
    return writer


def open_capture(source: int | str):
    """Открывает USB-камеру через V4L2 или поток через авто-бэкенд."""
    import cv2

    if isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/video")):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cv2.VideoCapture(source)


def resolve_source(source: int | str) -> int | str:
    """Находит внешнюю Logitech по имени V4L2 или стабильному симлинку."""
    if source != "logitech":
        return source
    candidates = sorted(glob.glob("/dev/v4l/by-id/*Logitech*video-index0"))
    if candidates:
        return candidates[0]
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            camera_name = Path(name_file).read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if "logitech" in camera_name or "brio" in camera_name:
            device = Path(name_file).parent.name
            return f"/dev/{device}"
    raise RuntimeError("Logitech Brio не найдена. Проверьте: v4l2-ctl --list-devices")


def configure_local_caches() -> None:
    """Переносит кэши библиотек в каталог проекта."""
    cache_dir = Path(".cache")
    (cache_dir / "ultralytics").mkdir(parents=True, exist_ok=True)
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_dir / "ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))


def run(args: argparse.Namespace) -> int:
    """Запускает захват, локальный инференс и отображение результата."""
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Локальная модель не найдена: {model_path}. Положите веса в этот путь.")
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence должен быть больше 0 и не больше 1")

    try:
        configure_local_caches()
        import cv2
        print("Загрузка локальной модели YOLO на CPU...", flush=True)
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Зависимости не установлены. Выполните: ./scripts/setup_ubuntu.sh") from error

    model = YOLO(str(model_path))
    if args.labels == "ru":
        class_count = len(model.model.names)
        model.model.names = {0: "Фипик"} if class_count == 1 else dict(enumerate(RUSSIAN_LABELS))
    print("Модель загружена. Открываю камеру...", flush=True)
    capture = open_capture(resolve_source(parse_source(args.source)))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть источник камеры: {args.source}")
    if not args.headless:
        cv2.namedWindow("Локальное распознавание объектов", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Локальное распознавание объектов", args.window_width, args.window_height)

    writer: cv2.VideoWriter | None = create_writer(args.output, capture) if args.output else None
    frame_number = 0
    try:
        while args.max_frames <= 0 or frame_number < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Камера не вернула кадр")
            result = model.predict(frame, conf=args.confidence, device="cpu", verbose=False)[0]
            annotated = result.plot()
            if writer:
                writer.write(annotated)
            frame_number += 1

            if not args.headless:
                cv2.imshow("Локальное распознавание объектов", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    target = args.snapshot_dir / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(target), annotated)
                    print(f"Снимок сохранён: {target}")
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Обработано кадров: {frame_number}")
    return 0


def main() -> int:
    """Точка входа приложения."""
    args = build_parser().parse_args()
    if args.list_cameras:
        list_cameras()
        return 0
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Остановлено пользователем.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
