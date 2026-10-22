#!/usr/bin/env python3
"""Показывает живую камеру и отдельно проверяет ONNX-модель на ноутбуке."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.realtime import LatestFrameCapture  # noqa: E402

# Тяжёлые мультимедийные библиотеки загружаются после включения файла журнала.
cv2 = None
draw_overlay = None
infer_full_frame = None
infer_tiles = None
model_input_size = None
OnnxDetector = None

DEFAULT_RUN_OPTIONS = {
    "source": "auto",
    "model": None,
    "confidence": 0.70,
    "min_confidence": 0.05,
    "tile_size": 640,
    "tile_overlap": 0.20,
    "threads": 2,
    "full_frame": False,
}


@dataclass(frozen=True)
class CameraInfo:
    """Проверенный источник камеры и параметры полученного кадра."""

    source: str
    width: int
    height: int
    fps: float


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры изолированного живого испытания и выбор профиля."""
    parser = argparse.ArgumentParser(
        description="Живой тест ONNX-модели на камере; рабочий запуск VideoT16 не меняется."
    )
    parser.add_argument(
        "--profile",
        help="имя сохранённого профиля испытания из config/live_drone_profiles.json",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="камера: auto, номер устройства или /dev/videoN (по умолчанию auto)",
    )
    parser.add_argument("--model", type=Path, default=None, help="путь к локальной ONNX-модели")
    parser.add_argument("--confidence", type=float, default=None, help="порог показа рамки от 0 до 1")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="нижний порог кандидатов модели для диагностического значения MAX",
    )
    parser.add_argument("--tile-size", type=int, default=None, help="масштаб разбиения кадра на фрагменты")
    parser.add_argument("--tile-overlap", type=float, default=None, help="перекрытие фрагментов от 0 до 0.5")
    parser.add_argument("--threads", type=int, default=None, help="потоки ONNX Runtime")
    parser.add_argument(
        "--full-frame",
        action="store_true",
        default=None,
        help="обрабатывать кадр целиком, без фрагментов",
    )
    parser.add_argument("--list-cameras", action="store_true", help="показать доступные камеры и выйти")
    parser.add_argument(
        "--log",
        type=Path,
        help="путь к файлу журнала; по умолчанию создаётся отдельный файл в test_samples/video_trials/logs",
    )
    return parser


def resolve_run_args(
    args: argparse.Namespace,
    profiles_path: Path | None = None,
) -> argparse.Namespace:
    """Собирает параметры запуска из базовых значений, профиля и CLI-переопределений."""
    settings = DEFAULT_RUN_OPTIONS.copy()
    if args.profile:
        path = profiles_path or PROJECT_ROOT / "config" / "live_drone_profiles.json"
        try:
            with Path(path).open("r", encoding="utf-8") as profile_file:
                document = json.load(profile_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Не удалось прочитать профили {path}: {error}") from error
        profiles = document.get("profiles") if isinstance(document, dict) else None
        if not isinstance(profiles, dict) or args.profile not in profiles:
            available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else "нет"
            raise ValueError(
                f"Профиль '{args.profile}' не найден в {path}. Доступны: {available}"
            )
        profile = profiles[args.profile]
        profile_settings = profile.get("settings") if isinstance(profile, dict) else None
        if not isinstance(profile_settings, dict):
            raise ValueError(f"У профиля '{args.profile}' отсутствует объект settings")
        unknown_settings = set(profile_settings) - set(DEFAULT_RUN_OPTIONS)
        if unknown_settings:
            unknown_text = ", ".join(sorted(unknown_settings))
            raise ValueError(f"В профиле '{args.profile}' неизвестные параметры: {unknown_text}")
        settings.update(profile_settings)
        profile_model = settings.get("model")
        if profile_model is not None:
            profile_model = Path(profile_model)
            if not profile_model.is_absolute():
                profile_model = PROJECT_ROOT / profile_model
            settings["model"] = profile_model

    # Явно переданные параметры командной строки имеют приоритет над профилем.
    for option in DEFAULT_RUN_OPTIONS:
        value = getattr(args, option, None)
        if value is not None:
            settings[option] = value

    resolved = vars(args).copy()
    resolved.update(settings)
    return argparse.Namespace(**resolved)


class TeeOutput:
    """Одновременно передаёт текст в исходный поток консоли и файл журнала."""

    def __init__(self, console: TextIO, log_file: TextIO, lock: threading.RLock) -> None:
        """Сохраняет два назначения вывода и общий замок записи."""
        self._console = console
        self._log_file = log_file
        self._lock = lock

    @property
    def encoding(self) -> str:
        """Возвращает кодировку исходного потока консоли."""
        return self._console.encoding or "utf-8"

    def write(self, text: str) -> int:
        """Пишет текст в консоль и журнал, не перемешивая строки потоков."""
        with self._lock:
            self._console.write(text)
            self._log_file.write(text)
        return len(text)

    def flush(self) -> None:
        """Немедленно сбрасывает консольный вывод и файл журнала на диск."""
        with self._lock:
            self._console.flush()
            self._log_file.flush()

    def isatty(self) -> bool:
        """Сообщает, подключён ли исходный поток консоли к терминалу."""
        return self._console.isatty()


@contextmanager
def tee_output(log_path: Path):
    """Дублирует stdout и stderr в журнал и восстанавливает потоки при выходе."""
    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("a", encoding="utf-8", buffering=1) as log_file:
        lock = threading.RLock()
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        saved_stderr_fd = os.dup(2)
        console_stderr = original_stderr
        close_console_stderr = False
        try:
            # Отдельная копия терминала нужна, чтобы C-библиотеки не потеряли
            # свои диагностические сообщения при перенаправлении fd 2 в журнал.
            stderr_fd = original_stderr.fileno()
            console_stderr = os.fdopen(
                os.dup(stderr_fd),
                "w",
                encoding=original_stderr.encoding or "utf-8",
                buffering=1,
            )
            close_console_stderr = True
        except (AttributeError, OSError, ValueError):
            # Тестовые потоки наподобие StringIO не имеют файлового дескриптора.
            pass
        try:
            os.dup2(log_file.fileno(), 2)
        except OSError:
            os.close(saved_stderr_fd)
            if close_console_stderr:
                console_stderr.close()
            raise
        sys.stdout = TeeOutput(original_stdout, log_file, lock)
        sys.stderr = TeeOutput(console_stderr, log_file, lock)
        try:
            yield resolved_path
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stderr_fd)
                if close_console_stderr:
                    console_stderr.close()


def default_log_path() -> Path:
    """Создаёт уникальный путь журнала в локальном каталоге испытаний."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return PROJECT_ROOT / "test_samples" / "video_trials" / "logs" / f"live_drone_trial_{timestamp}.log"


def load_camera_dependency() -> None:
    """Загружает OpenCV после начала журналирования."""
    global cv2
    if cv2 is None:
        import cv2 as opencv

        cv2 = opencv


def load_model_dependencies() -> None:
    """Загружает ONNX Runtime и функции инференса под запись журнала."""
    global OnnxDetector, draw_overlay, infer_full_frame, infer_tiles, model_input_size
    load_camera_dependency()
    if OnnxDetector is None:
        from scripts.compare_video_trials import (
            draw_overlay as overlay_function,
            infer_full_frame as full_frame_function,
            infer_tiles as tiled_function,
            model_input_size as input_size_function,
        )
        from src.onnx_detector import OnnxDetector as detector_class

        draw_overlay = overlay_function
        infer_full_frame = full_frame_function
        infer_tiles = tiled_function
        model_input_size = input_size_function
        OnnxDetector = detector_class


def camera_api() -> int:
    """Выбирает стандартный backend камеры для текущей ОС."""
    load_camera_dependency()
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def camera_nodes() -> list[Path]:
    """На Linux возвращает video-узлы в числовом порядке."""
    nodes = list(Path("/dev").glob("video[0-9]*"))
    return sorted(nodes, key=lambda path: int(path.name[5:]))


def scan_cameras(nodes: list[Path] | None = None) -> list[CameraInfo]:
    """Проверяет, какие video-узлы действительно отдают кадры."""
    load_camera_dependency()
    devices = camera_nodes() if nodes is None else nodes
    available: list[CameraInfo] = []
    for node in devices:
        capture = cv2.VideoCapture(str(node), camera_api())
        try:
            if not capture.isOpened():
                continue
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            available.append(CameraInfo(str(node), width, height, fps))
        finally:
            capture.release()
    return available


def print_cameras(cameras: list[CameraInfo]) -> None:
    """Печатает только источники, с которых удалось прочитать кадр."""
    if not cameras:
        print("Рабочие камеры не найдены. Проверь USB-захват и повтори команду.")
        return
    print("Доступные видеопотоки:")
    for number, camera in enumerate(cameras, start=1):
        fps_text = f"{camera.fps:.1f} FPS" if camera.fps > 0 else "FPS не сообщил драйвер"
        print(f"  {number}. {camera.source} — {camera.width}x{camera.height}, {fps_text}")


def choose_camera(cameras: list[CameraInfo]) -> str:
    """Выбирает единственную камеру автоматически или спрашивает номер."""
    if not cameras:
        raise RuntimeError("Нет доступной камеры. Подключи USB-видеозахват или укажи --source вручную.")
    if len(cameras) == 1:
        print(f"Автоматически выбрана камера {cameras[0].source}.")
        return cameras[0].source
    print_cameras(cameras)
    while True:
        try:
            answer = input("Номер нужного USB-видеозахвата из списка: ").strip()
        except EOFError as error:
            raise RuntimeError("Автовыбор требует номер камеры; укажи её через --source.") from error
        if answer.isdigit() and 1 <= int(answer) <= len(cameras):
            return cameras[int(answer) - 1].source
        print(f"Введи число от 1 до {len(cameras)}.")


def parse_source(value: str) -> int | str:
    """Преобразует строковый номер камеры в индекс OpenCV."""
    return int(value) if value.isdigit() else value


def validate_args(args: argparse.Namespace) -> None:
    """Проверяет диапазоны параметров до открытия камеры и модели."""
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence должен быть больше 0 и не больше 1")
    if not 0.0 < args.min_confidence <= 1.0:
        raise ValueError("--min-confidence должен быть больше 0 и не больше 1")
    if args.min_confidence > args.confidence:
        raise ValueError("--min-confidence не должен быть выше --confidence")
    if not 0.0 <= args.tile_overlap < 0.5:
        raise ValueError("--tile-overlap должен быть от 0 до 0.5")
    if args.tile_size < 1:
        raise ValueError("--tile-size должен быть положительным")
    if args.threads < 1:
        raise ValueError("--threads должен быть не меньше 1")


def draw_status(
    frame,
    args: argparse.Namespace,
    model_size: int,
    detection_enabled: bool,
    raw_confidence: float,
    detection_count: int,
    inference_ms: float | None,
    camera_fps: float,
) -> object:
    """Добавляет к тестовому изображению короткую диагностическую панель."""
    shown = frame.copy()
    height, width = shown.shape[:2]
    cv2.rectangle(shown, (0, 0), (width, min(height, 58)), (8, 18, 24), -1)
    mode = "TILES" if not args.full_frame else "FULL FRAME"
    status = "ON" if detection_enabled else "OFF"
    fps_text = f"{camera_fps:.1f}" if camera_fps > 0 else "--"
    inference_text = f"{inference_ms:.0f} ms" if inference_ms is not None else "-- ms"
    first_line = (
        f"VideoT16 TEST | DETECT {status} | {args.model.name} | input {model_size} | {mode}"
    )
    second_line = (
        f"BOXES {detection_count} (>{args.confidence:.0%}) | MAX {raw_confidence:.0%} | "
        f"INFER {inference_text} | CAMERA {fps_text} FPS | R toggle / Q exit"
    )
    cv2.putText(shown, first_line[:150], (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(shown, second_line[:150], (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 235, 160), 1, cv2.LINE_AA)
    return shown


def fit_window(frame, max_width: int = 1280, max_height: int = 720):
    """Уменьшает только отображаемую копию, сохраняя координаты инференса."""
    height, width = frame.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def console_status(raw_confidence: float, detection_count: int, threshold: float, inference_ms: float) -> str:
    """Формирует консольную диагностику сырой уверенности и видимых рамок."""
    return (
        f"confidence={raw_confidence:.0%} | рамок выше {threshold:.0%}: {detection_count} "
        f"| инференс={inference_ms:.0f} мс"
    )


def run_viewer(args: argparse.Namespace, source: int | str) -> int:
    """Запускает поток камеры, показывает рамки и освобождает устройство при выходе."""
    load_model_dependencies()
    if args.model is None or not args.model.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {args.model or '(не указана)'}")
    model_size = model_input_size(args.model)
    detector = OnnxDetector(
        args.model,
        args.min_confidence,
        False,
        model_size,
        object_label=None,
        threads=args.threads,
    )
    capture_device = cv2.VideoCapture(source, camera_api())
    if not capture_device.isOpened():
        capture_device.release()
        raise RuntimeError(f"Не удалось открыть источник камеры: {source}")
    capture_device.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_fps = float(capture_device.get(cv2.CAP_PROP_FPS))
    camera = LatestFrameCapture(capture_device)
    detection_enabled = True
    raw_confidence = 0.0
    detection_count = 0
    inference_ms: float | None = None
    print(f"Модель: {args.model} | вход: {model_size} | порог рамки: {args.confidence:.0%}")
    print(
        f"Режим: {'кадр целиком' if args.full_frame else f'фрагменты {args.tile_size}px'} | "
        f"источник: {source} | выходной файл не создаётся"
    )
    print(f"Источник камеры открыт | FPS драйвера: {camera_fps:.1f}")
    cv2.namedWindow("VideoT16 — live model trial", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VideoT16 — live model trial", 1280, 800)
    first_frame_reported = False
    try:
        while True:
            frame = camera.latest(timeout=3.0)
            if not first_frame_reported:
                frame_height, frame_width = frame.shape[:2]
                print(f"Первый кадр получен: {frame_width}x{frame_height}")
                first_frame_reported = True
            detections = ()
            if detection_enabled:
                inference_started = time.perf_counter()
                if args.full_frame:
                    detections, _, raw_confidence = infer_full_frame(detector, frame)
                else:
                    detections, _, raw_confidence = infer_tiles(
                        detector, frame, args.tile_size, args.tile_overlap
                    )
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                detection_count = sum(
                    item.confidence >= args.confidence for item in detections
                )
                print(
                    console_status(
                        raw_confidence,
                        detection_count,
                        args.confidence,
                        inference_ms,
                    ),
                    flush=True,
                )
                preview = draw_overlay(frame, detections, args.confidence)
            else:
                preview = frame.copy()
            preview = draw_status(
                preview,
                args,
                model_size,
                detection_enabled,
                raw_confidence,
                detection_count,
                inference_ms,
                camera_fps,
            )
            cv2.imshow("VideoT16 — live model trial", fit_window(preview))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                detection_enabled = not detection_enabled
                if not detection_enabled:
                    raw_confidence = 0.0
                    detection_count = 0
                    inference_ms = None
                print(f"Распознавание {'включено' if detection_enabled else 'выключено'}.")
    finally:
        camera.close()
        capture_device.release()
        cv2.destroyAllWindows()
        print("Источник камеры закрыт.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Обрабатывает параметры, перечисляет камеры или запускает испытание."""
    args = build_parser().parse_args(argv)
    log_path = args.log if args.log is not None else default_log_path()
    try:
        with tee_output(log_path) as resolved_log_path:
            print("=== VideoT16: живое испытание модели ===")
            print(f"Время запуска: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            print(f"Журнал: {resolved_log_path}")
            try:
                args = resolve_run_args(args)
                print(f"Профиль: {args.profile or 'базовые параметры'}")
                print(
                    f"Параметры: source={args.source} | model={args.model} | "
                    f"confidence={args.confidence:.0%} | min-confidence={args.min_confidence:.0%} | "
                    f"tile-size={args.tile_size} | tile-overlap={args.tile_overlap:.2f} | "
                    f"threads={args.threads} | full-frame={args.full_frame}"
                )
                if args.list_cameras:
                    cameras = scan_cameras()
                    print_cameras(cameras)
                    result = 0 if cameras else 1
                else:
                    validate_args(args)
                    if args.model is None:
                        raise ValueError("Укажи локальную ONNX-модель через --model")
                    if not args.model.is_file():
                        raise FileNotFoundError(f"ONNX-модель не найдена: {args.model}")
                    source: int | str
                    if args.source == "auto":
                        source = choose_camera(scan_cameras())
                    else:
                        source = parse_source(args.source)
                    result = run_viewer(args, source)
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
                print(f"Ошибка: {error}", file=sys.stderr)
                result = 2
            except Exception as error:  # noqa: BLE001 — журнал должен сохранить неожиданные ошибки диагностики.
                print(f"Непредвиденная ошибка: {error}", file=sys.stderr)
                traceback.print_exc()
                result = 2
        print(f"Журнал сохранён: {log_path.expanduser().resolve()}")
        return result
    except OSError as error:
        print(f"Не удалось создать файл журнала {log_path}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
