#!/usr/bin/env python3
"""Показывает живую камеру и отдельно проверяет ONNX-модель на ноутбуке."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
import math
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

from src.motion_compensated_detection import MotionAnalyzer, MotionBox, MotionResult  # noqa: E402
from src.realtime import (  # noqa: E402
    Detection,
    LatestFrameCapture,
    LatestInferenceWorker,
    LatestValueWorker,
    SystemStatusMonitor,
)

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

# Имена шрифтов OpenCV, разрешённые для настройки подписей и панели статуса.
OPENCV_FONT_NAMES = frozenset(
    {
        "FONT_HERSHEY_SIMPLEX",
        "FONT_HERSHEY_PLAIN",
        "FONT_HERSHEY_DUPLEX",
        "FONT_HERSHEY_COMPLEX",
        "FONT_HERSHEY_TRIPLEX",
        "FONT_HERSHEY_COMPLEX_SMALL",
        "FONT_HERSHEY_SCRIPT_SIMPLEX",
        "FONT_HERSHEY_SCRIPT_COMPLEX",
    }
)

# Настройки захвата, окна и диагностического оверлея, редактируемые в профиле JSON.
DEFAULT_PROFILE_SECTIONS = {
    "capture": {"width": 640, "height": 480, "fps": 30.0, "buffer_size": 1},
    "display": {
        "window_title": "VideoT16 — live model trial",
        "window_width": 1280,
        "window_height": 800,
        "max_frame_width": 1280,
        "max_frame_height": 720,
        "scale_up": False,
        "poll_delay_ms": 1,
    },
    "diagnostics": {"console_status_interval_ms": 1000},
    "two_stage": {
        "enabled": False,
        "roi_width_percent": 50,
        "roi_height_percent": 50,
        "roi_center_x_percent": 50,
        "roi_center_y_percent": 50,
        "max_candidates": 1,
        "candidate_crop_size": 640,
        "candidate_padding": 2.0,
        "full_scan_interval_ms": 1500,
        "max_detection_age_ms": 300,
        "motion_max_features": 600,
        "motion_feature_quality": 0.01,
        "motion_min_feature_distance": 5,
        "motion_min_background_features": 8,
        "motion_min_background_inliers": 6,
        "motion_ransac_threshold_px": 2.5,
        "motion_max_background_shift_px": 100,
        "motion_luma_threshold": 18,
        "motion_chroma_threshold": 22,
        "motion_min_blob_area": 8,
        "motion_max_blob_area_percent": 30,
        "motion_morphology_radius": 0,
        "motion_suppress_specular": False,
        "motion_specular_value_min": 245,
        "motion_specular_saturation_max": 35,
        "motion_persistence_frames": 1,
        "motion_track_iou_percent": 10,
        "motion_min_local_color_contrast": 0,
    },
    "overlay": {
        "near_threshold_margin": 0.10,
        "near_threshold_color_rgb": [255, 165, 0],
        "box": {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "thickness": 2,
            "color_rgb": [80, 230, 40],
        },
        "label": {
            "font": "FONT_HERSHEY_SIMPLEX",
            "font_scale": 0.55,
            "thickness": 2,
            "offset_y": 7,
            "min_y": 18,
            "color_rgb": [80, 230, 40],
        },
    },
    "status_panel": {
        "position": "bottom",
        "height": 28,
        "background_color_rgb": [35, 35, 35],
        "left": 12,
        "text_y": 19,
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.42,
        "text_color_rgb": [230, 230, 230],
        "thickness": 1,
        "max_text_chars": 96,
    },
}


@dataclass(frozen=True)
class InferenceRequest:
    """Один кадр или вырезанный кандидат, передаваемый фоновому YOLO."""

    frame: object
    origin_x: int
    origin_y: int
    mode: str
    submitted_at: float
    frame_shape: tuple[int, ...]


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
    profile_sections = {
        name: deepcopy(defaults)
        for name, defaults in DEFAULT_PROFILE_SECTIONS.items()
    }
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
        for section_name, section_defaults in DEFAULT_PROFILE_SECTIONS.items():
            configured_section = profile.get(section_name, {})
            if not isinstance(configured_section, dict):
                raise ValueError(
                    f"В профиле '{args.profile}' параметр {section_name} должен быть объектом"
                )
            profile_sections[section_name] = merge_profile_section(
                args.profile,
                section_name,
                configured_section,
                section_defaults,
            )
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
    resolved.update(profile_sections)
    return argparse.Namespace(**resolved)


def merge_profile_section(
    profile_name: str,
    section_name: str,
    configured: dict,
    defaults: dict,
) -> dict:
    """Проверяет имена настроек секции и объединяет профиль со значениями по умолчанию."""
    unknown = set(configured) - set(defaults)
    if unknown:
        raise ValueError(
            f"В секции {section_name} профиля '{profile_name}' неизвестные параметры: "
            f"{', '.join(sorted(unknown))}"
        )
    merged = deepcopy(defaults)
    for key, value in configured.items():
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ValueError(
                    f"Параметр {section_name}.{key} профиля '{profile_name}' должен быть объектом"
                )
            nested_unknown = set(value) - set(merged[key])
            if nested_unknown:
                raise ValueError(
                    f"В секции {section_name}.{key} профиля '{profile_name}' неизвестные параметры: "
                    f"{', '.join(sorted(nested_unknown))}"
                )
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def two_stage_roi_settings(settings: dict) -> dict:
    """Переводит плоские параметры профиля в настройки области поиска движения."""
    return {
        "width_percent": settings["roi_width_percent"],
        "height_percent": settings["roi_height_percent"],
        "center_x_percent": settings["roi_center_x_percent"],
        "center_y_percent": settings["roi_center_y_percent"],
    }


def two_stage_motion_settings(settings: dict) -> dict:
    """Переводит плоские параметры профиля в настройки компенсации движения фона."""
    return {
        "max_features": settings["motion_max_features"],
        "feature_quality": settings["motion_feature_quality"],
        "min_feature_distance": settings["motion_min_feature_distance"],
        "min_background_features": settings["motion_min_background_features"],
        "min_background_inliers": settings["motion_min_background_inliers"],
        "ransac_threshold_px": settings["motion_ransac_threshold_px"],
        "max_background_shift_px": settings["motion_max_background_shift_px"],
        "luma_threshold": settings["motion_luma_threshold"],
        "chroma_threshold": settings["motion_chroma_threshold"],
        "min_blob_area": settings["motion_min_blob_area"],
        "max_blob_area_percent": settings["motion_max_blob_area_percent"],
        "morphology_radius": settings["motion_morphology_radius"],
        "suppress_specular": settings["motion_suppress_specular"],
        "specular_value_min": settings["motion_specular_value_min"],
        "specular_saturation_max": settings["motion_specular_saturation_max"],
        "persistence_frames": settings["motion_persistence_frames"],
        "track_iou_percent": settings["motion_track_iou_percent"],
        "max_candidates": settings["max_candidates"],
        "min_local_color_contrast": settings["motion_min_local_color_contrast"],
    }


def candidate_inference_request(
    frame,
    result: MotionResult,
    candidate: MotionBox,
    settings: dict,
    submitted_at: float,
) -> InferenceRequest:
    """Вырезает вокруг кандидата участок для быстрого YOLO и сохраняет его координаты."""
    frame_height, frame_width = frame.shape[:2]
    roi_x, roi_y, _, _ = result.roi
    center_x = roi_x + (candidate.x1 + candidate.x2) / 2
    center_y = roi_y + (candidate.y1 + candidate.y2) / 2
    padding = float(settings["candidate_padding"])
    requested_side = max(
        int(settings["candidate_crop_size"]),
        round((candidate.x2 - candidate.x1) * padding),
        round((candidate.y2 - candidate.y1) * padding),
    )
    side = max(1, min(requested_side, frame_width, frame_height))
    left = max(0, min(frame_width - side, round(center_x - side / 2)))
    top = max(0, min(frame_height - side, round(center_y - side / 2)))
    crop = frame[top : top + side, left : left + side].copy()
    return InferenceRequest(crop, left, top, "candidate", submitted_at, tuple(frame.shape))


def full_frame_inference_request(frame, submitted_at: float) -> InferenceRequest:
    """Готовит редкий полный обход кадра как страховку от пропущенной цели."""
    return InferenceRequest(frame, 0, 0, "full", submitted_at, tuple(frame.shape))


def move_detections_to_frame(
    detections: tuple[Detection, ...],
    origin_x: int,
    origin_y: int,
    frame_shape: tuple[int, ...],
) -> tuple[Detection, ...]:
    """Переносит рамки из вырезанного кандидата в координаты исходного кадра."""
    frame_height, frame_width = frame_shape[:2]
    moved: list[Detection] = []
    for detection in detections:
        x1 = max(0, min(frame_width, detection.x1 + origin_x))
        y1 = max(0, min(frame_height, detection.y1 + origin_y))
        x2 = max(0, min(frame_width, detection.x2 + origin_x))
        y2 = max(0, min(frame_height, detection.y2 + origin_y))
        if x2 > x1 and y2 > y1:
            moved.append(Detection(x1, y1, x2, y2, detection.name, detection.confidence))
    return tuple(moved)


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

    def write_log_only(self, text: str) -> int:
        """Сохраняет подробную диагностику в журнале, не загромождая консоль."""
        with self._lock:
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


def log_diagnostic(message: str, stream: TextIO | None = None) -> None:
    """Пишет служебную строку только в журнал, если включено журналирование."""
    target = stream if stream is not None else sys.stdout
    write_log_only = getattr(target, "write_log_only", None)
    if callable(write_log_only):
        write_log_only(f"{message}\n")
        target.flush()
    else:
        print(message, file=target)


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
    def is_number(value: object) -> bool:
        """Проверяет, что значение является конечным числом, но не логическим типом."""
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        )

    def is_integer(value: object) -> bool:
        """Проверяет целочисленный JSON-параметр без принятия true/false за 1/0."""
        return isinstance(value, int) and not isinstance(value, bool)

    def valid_rgb(value: object) -> bool:
        """Проверяет массив из трёх целых RGB-компонентов в диапазоне 0–255."""
        return (
            isinstance(value, list)
            and len(value) == 3
            and all(is_integer(channel) and 0 <= channel <= 255 for channel in value)
        )

    if not is_number(args.confidence) or not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence должен быть больше 0 и не больше 1")
    if not is_number(args.min_confidence) or not 0.0 < args.min_confidence <= 1.0:
        raise ValueError("--min-confidence должен быть больше 0 и не больше 1")
    if args.min_confidence > args.confidence:
        raise ValueError("--min-confidence не должен быть выше --confidence")
    if not is_number(args.tile_overlap) or not 0.0 <= args.tile_overlap < 0.5:
        raise ValueError("--tile-overlap должен быть от 0 до 0.5")
    if not is_integer(args.tile_size) or args.tile_size < 1:
        raise ValueError("--tile-size должен быть положительным")
    if not is_integer(args.threads) or args.threads < 1:
        raise ValueError("--threads должен быть не меньше 1")

    capture = getattr(args, "capture", DEFAULT_PROFILE_SECTIONS["capture"])
    width, height = capture["width"], capture["height"]
    if (
        not is_integer(width)
        or not is_integer(height)
        or width < 0
        or height < 0
        or (width == 0) != (height == 0)
    ):
        raise ValueError("capture.width и capture.height должны быть положительными или оба равны 0")
    if not is_number(capture["fps"]) or capture["fps"] < 0:
        raise ValueError("capture.fps должен быть конечным числом не меньше 0")
    if not is_integer(capture["buffer_size"]) or capture["buffer_size"] < 1:
        raise ValueError("capture.buffer_size должен быть не меньше 1")

    display = getattr(args, "display", DEFAULT_PROFILE_SECTIONS["display"])
    for key in ("window_width", "window_height", "max_frame_width", "max_frame_height", "poll_delay_ms"):
        if not is_integer(display[key]) or display[key] < 1:
            raise ValueError(f"display.{key} должен быть положительным целым числом")
    if not isinstance(display["scale_up"], bool):
        raise ValueError("display.scale_up должен быть true или false")
    if not isinstance(display["window_title"], str) or not display["window_title"].strip():
        raise ValueError("display.window_title не должен быть пустым")

    diagnostics = getattr(args, "diagnostics", DEFAULT_PROFILE_SECTIONS["diagnostics"])
    if (
        not is_integer(diagnostics["console_status_interval_ms"])
        or diagnostics["console_status_interval_ms"] < 0
    ):
        raise ValueError("diagnostics.console_status_interval_ms должен быть целым числом от 0")

    two_stage = getattr(args, "two_stage", DEFAULT_PROFILE_SECTIONS["two_stage"])
    if not isinstance(two_stage["enabled"], bool):
        raise ValueError("two_stage.enabled должен быть true или false")
    for key in ("roi_width_percent", "roi_height_percent"):
        if not is_number(two_stage[key]) or not 1 <= two_stage[key] <= 100:
            raise ValueError(f"two_stage.{key} должен быть числом от 1 до 100")
    for key in ("roi_center_x_percent", "roi_center_y_percent"):
        if not is_number(two_stage[key]) or not 0 <= two_stage[key] <= 100:
            raise ValueError(f"two_stage.{key} должен быть числом от 0 до 100")
    integer_ranges = {
        "max_candidates": (1, 20),
        "candidate_crop_size": (32, 4096),
        "full_scan_interval_ms": (100, 60_000),
        "max_detection_age_ms": (20, 10_000),
        "motion_max_features": (8, 4000),
        "motion_min_feature_distance": (1, 200),
        "motion_min_background_features": (4, 1000),
        "motion_min_background_inliers": (4, 1000),
        "motion_luma_threshold": (1, 255),
        "motion_chroma_threshold": (1, 255),
        "motion_min_blob_area": (1, 100_000),
        "motion_morphology_radius": (0, 15),
        "motion_specular_value_min": (0, 255),
        "motion_specular_saturation_max": (0, 255),
        "motion_persistence_frames": (1, 30),
    }
    for key, (minimum, maximum) in integer_ranges.items():
        if not is_integer(two_stage[key]) or not minimum <= two_stage[key] <= maximum:
            raise ValueError(f"two_stage.{key} должен быть целым числом от {minimum} до {maximum}")
    number_ranges = {
        "candidate_padding": (1.0, 20.0),
        "motion_feature_quality": (0.0001, 1.0),
        "motion_ransac_threshold_px": (0.1, 50.0),
        "motion_max_background_shift_px": (1.0, 2000.0),
        "motion_max_blob_area_percent": (0.1, 100.0),
        "motion_track_iou_percent": (0.0, 100.0),
        "motion_min_local_color_contrast": (0.0, 300.0),
    }
    for key, (minimum, maximum) in number_ranges.items():
        if not is_number(two_stage[key]) or not minimum <= two_stage[key] <= maximum:
            raise ValueError(f"two_stage.{key} должен быть числом от {minimum:g} до {maximum:g}")
    if not isinstance(two_stage["motion_suppress_specular"], bool):
        raise ValueError("two_stage.motion_suppress_specular должен быть true или false")
    if two_stage["motion_min_background_features"] > two_stage["motion_max_features"]:
        raise ValueError("two_stage.motion_min_background_features не может быть больше motion_max_features")
    if two_stage["motion_min_background_inliers"] > two_stage["motion_min_background_features"]:
        raise ValueError("two_stage.motion_min_background_inliers не может быть больше числа точек фона")

    overlay = getattr(args, "overlay", DEFAULT_PROFILE_SECTIONS["overlay"])
    box, label = overlay["box"], overlay["label"]
    margin = overlay["near_threshold_margin"]
    if not is_number(margin) or not 0.0 <= margin <= 1.0:
        raise ValueError("overlay.near_threshold_margin должен быть от 0 до 1")
    if not valid_rgb(overlay["near_threshold_color_rgb"]):
        raise ValueError(
            "overlay.near_threshold_color_rgb должен содержать три целых значения RGB от 0 до 255"
        )
    for key in ("scale_x", "scale_y"):
        if not is_number(box[key]) or box[key] <= 0:
            raise ValueError(f"overlay.box.{key} должен быть больше 0")
    if not is_integer(box["thickness"]) or box["thickness"] < 1:
        raise ValueError("overlay.box.thickness должен быть не меньше 1")
    if not isinstance(label["font"], str) or label["font"] not in OPENCV_FONT_NAMES:
        raise ValueError("overlay.label.font должен содержать допустимое имя шрифта OpenCV")
    if not is_number(label["font_scale"]) or label["font_scale"] <= 0:
        raise ValueError("overlay.label.font_scale должен быть больше 0")
    if not is_integer(label["thickness"]) or label["thickness"] < 1:
        raise ValueError("overlay.label.thickness должен быть не меньше 1")
    for key in ("offset_y", "min_y"):
        if not is_integer(label[key]) or label[key] < 0:
            raise ValueError(f"overlay.label.{key} должен быть неотрицательным целым числом")

    for key, color in (
        ("overlay.box.color_rgb", box["color_rgb"]),
        ("overlay.label.color_rgb", label["color_rgb"]),
    ):
        if not valid_rgb(color):
            raise ValueError(f"{key} должен содержать три целых значения RGB от 0 до 255")

    panel = getattr(args, "status_panel", DEFAULT_PROFILE_SECTIONS["status_panel"])
    if not isinstance(panel["position"], str) or panel["position"] not in {"top", "bottom"}:
        raise ValueError("status_panel.position должен быть top или bottom")
    if not is_integer(panel["height"]) or panel["height"] < 1:
        raise ValueError("status_panel.height должен быть положительным целым числом")
    if not isinstance(panel["font"], str) or panel["font"] not in OPENCV_FONT_NAMES:
        raise ValueError("status_panel.font должен содержать допустимое имя шрифта OpenCV")
    if not is_integer(panel["thickness"]) or panel["thickness"] < 1:
        raise ValueError("status_panel.thickness должен быть не меньше 1")
    for key in ("left", "text_y"):
        if not is_integer(panel[key]) or panel[key] < 0:
            raise ValueError(f"status_panel.{key} должен быть неотрицательным целым числом")
    if not is_integer(panel["max_text_chars"]) or panel["max_text_chars"] < 1:
        raise ValueError("status_panel.max_text_chars должен быть положительным целым числом")
    if not is_number(panel["font_scale"]) or panel["font_scale"] <= 0:
        raise ValueError("status_panel.font_scale должен быть больше 0")
    for key in ("background_color_rgb", "text_color_rgb"):
        if not valid_rgb(panel[key]):
            raise ValueError(
                f"status_panel.{key} должен содержать три целых значения RGB от 0 до 255"
            )


def configure_capture_device(capture_device, settings: dict) -> dict[str, float | int]:
    """Запрашивает размер, FPS и буфер из профиля и возвращает параметры драйвера."""
    capture_device.set(cv2.CAP_PROP_BUFFERSIZE, settings["buffer_size"])
    for name, prop, requested in (
        ("ширина", cv2.CAP_PROP_FRAME_WIDTH, settings["width"]),
        ("высота", cv2.CAP_PROP_FRAME_HEIGHT, settings["height"]),
        ("FPS", cv2.CAP_PROP_FPS, settings["fps"]),
    ):
        if requested > 0:
            accepted = capture_device.set(prop, requested)
            log_diagnostic(
                f"Запрос захвата {name}={requested}: "
                f"{'принят драйвером' if accepted else 'драйвер не подтвердил'}"
            )
    actual = {
        "width": int(round(capture_device.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture_device.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture_device.get(cv2.CAP_PROP_FPS)),
    }
    log_diagnostic(
        f"Фактический захват: {actual['width']}x{actual['height']} "
        f"@ {actual['fps']:.1f} FPS"
    )
    return actual


def draw_status(
    frame,
    args: argparse.Namespace,
    system_status_text: str,
    current_confidence: float,
) -> object:
    """Рисует снизу CPU, температуру, текущую уверенность модели и порог рамки."""
    shown = frame.copy()
    height, width = shown.shape[:2]
    panel = args.status_panel
    background = tuple(reversed(panel["background_color_rgb"]))
    text_color = tuple(reversed(panel["text_color_rgb"]))
    panel_height = min(height, panel["height"])
    panel_top = height - panel_height if panel["position"] == "bottom" else 0
    panel_bottom = panel_top + panel_height - 1
    cv2.rectangle(shown, (0, panel_top), (width - 1, panel_bottom), background, -1)

    # Монитор отдаёт стабильный формат TEMP ... CPU ...; при сбое оставляем явные заглушки.
    status_fields = system_status_text.split()
    if len(status_fields) >= 4 and status_fields[0] == "TEMP" and status_fields[2] == "CPU":
        temperature, cpu_load = status_fields[1], status_fields[3]
    else:
        temperature, cpu_load = "--.-C", "--%"
    status_line = (
        f"CPU {cpu_load} | TEMP {temperature} | CONF {current_confidence:.0%} "
        f"| THR {args.confidence:.0%}"
    )
    font = getattr(cv2, panel["font"])
    cv2.putText(
        shown,
        status_line[:panel["max_text_chars"]],
        (panel["left"], min(panel_bottom, panel_top + panel["text_y"])),
        font,
        panel["font_scale"],
        text_color,
        panel["thickness"],
        cv2.LINE_AA,
    )
    return shown


def fit_window(
    frame,
    max_width: int = 1280,
    max_height: int = 720,
    scale_up: bool = False,
):
    """Подгоняет копию кадра к размеру окна, при настройке увеличивая её."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if not scale_up:
        scale = min(1.0, scale)
    if scale == 1.0:
        return frame
    interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=interpolation,
    )


def compose_display_frame(
    frame,
    args: argparse.Namespace,
    system_status_text: str,
    current_confidence: float,
):
    """Центрирует видео без растяжения на широком холсте с общей строкой статуса."""
    display = args.display
    panel = args.status_panel
    canvas_width = display["window_width"]
    canvas_height = display["window_height"]
    panel_height = min(panel["height"], canvas_height - 1)
    video_area_height = canvas_height - panel_height
    fitted = fit_window(
        frame,
        max_width=min(display["max_frame_width"], canvas_width),
        max_height=min(display["max_frame_height"], video_area_height),
        scale_up=display["scale_up"],
    )
    fitted_height, fitted_width = fitted.shape[:2]
    top = (video_area_height - fitted_height) // 2
    bottom = video_area_height - fitted_height - top + panel_height
    left = (canvas_width - fitted_width) // 2
    right = canvas_width - fitted_width - left
    background = tuple(reversed(panel["background_color_rgb"]))
    canvas = cv2.copyMakeBorder(
        fitted,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=background,
    )
    return draw_status(canvas, args, system_status_text, current_confidence)


def console_status(
    raw_confidence: float,
    detection_count: int,
    threshold: float,
    inference_ms: float | None,
) -> str:
    """Формирует консольную диагностику сырой уверенности и видимых рамок."""
    inference_text = "ожидается" if inference_ms is None else f"{inference_ms:.0f} мс"
    return (
        f"confidence={raw_confidence:.0%} | рамок выше {threshold:.0%}: {detection_count} "
        f"| инференс={inference_text}"
    )


def display_poll_delay_ms(configured_delay_ms: int, capture_fps: float) -> int:
    """Ограничивает частоту опроса GUI частотой видеовхода, но не выше 60 Гц."""
    effective_fps = capture_fps if math.isfinite(capture_fps) and capture_fps > 0 else 60.0
    target_fps = min(effective_fps, 60.0)
    return max(configured_delay_ms, math.ceil(1000.0 / target_fps))


def should_emit_console_status(now: float, last_emitted: float | None, interval_ms: int) -> bool:
    """Решает, пора ли записать очередную строку статистики в консоль и журнал."""
    if interval_ms == 0:
        return False
    return last_emitted is None or now - last_emitted >= interval_ms / 1000.0


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
    two_stage_enabled = bool(args.two_stage["enabled"])
    motion_analyzer = (
        MotionAnalyzer(
            two_stage_roi_settings(args.two_stage),
            two_stage_motion_settings(args.two_stage),
        )
        if two_stage_enabled
        else None
    )
    motion_worker = (
        LatestValueWorker(motion_analyzer.process, name="motion-candidate-analysis")
        if motion_analyzer is not None
        else None
    )
    capture_device = cv2.VideoCapture(source, camera_api())
    if not capture_device.isOpened():
        capture_device.release()
        raise RuntimeError(f"Не удалось открыть источник камеры: {source}")
    capture_settings = args.capture
    capture_actual = configure_capture_device(capture_device, capture_settings)
    camera_fps = capture_actual["fps"]
    camera = LatestFrameCapture(capture_device)
    system_status = SystemStatusMonitor()
    detection_enabled = True
    raw_confidence = 0.0
    detection_count = 0
    inference_ms: float | None = None
    last_console_status_at: float | None = None
    log_diagnostic(f"Модель: {args.model} | вход: {model_size} | порог рамки: {args.confidence:.0%}")
    log_diagnostic(
        f"Режим: {'кадр целиком' if args.full_frame else f'фрагменты {args.tile_size}px'} | "
        f"источник: {source} | выходной файл не создаётся"
    )
    log_diagnostic(
        f"Запрос захвата из конфига: {capture_settings['width']}x{capture_settings['height']} "
        f"@ {capture_settings['fps']:.1f} FPS | буфер={capture_settings['buffer_size']}"
    )
    log_diagnostic(f"Источник камеры открыт | FPS драйвера: {camera_fps:.1f}")
    window_title = args.display["window_title"]
    log_diagnostic(
        f"Монитор: окно {args.display['window_width']}x{args.display['window_height']} | "
        f"изображение до {args.display['max_frame_width']}x{args.display['max_frame_height']}"
    )
    log_diagnostic(
        f"Рамка: масштаб X/Y={args.overlay['box']['scale_x']:.2f}/"
        f"{args.overlay['box']['scale_y']:.2f}, толщина={args.overlay['box']['thickness']} px | "
        f"порог={args.confidence:.0%}"
    )
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, args.display["window_width"], args.display["window_height"])
    first_frame_reported = False
    inference_metadata: dict[int, tuple[float, float, int, float, str]] = {}
    inference_metadata_lock = threading.Lock()
    inference_sequence = 0

    def predict_frame(packet: InferenceRequest) -> tuple:
        """Проверяет полный кадр либо быстрый кандидат и сохраняет его метрики."""
        nonlocal inference_sequence
        started_at = time.perf_counter()
        if packet.mode == "candidate":
            detections = move_detections_to_frame(
                tuple(detector(packet.frame)),
                packet.origin_x,
                packet.origin_y,
                packet.frame_shape,
            )
            confidence = float(getattr(detector, "last_best_confidence", 0.0))
        elif args.full_frame:
            detections, _, confidence = infer_full_frame(detector, packet.frame)
        else:
            detections, _, confidence = infer_tiles(
                detector, packet.frame, args.tile_size, args.tile_overlap
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        visible_count = sum(item.confidence >= args.confidence for item in detections)
        with inference_metadata_lock:
            inference_sequence += 1
            inference_metadata[inference_sequence] = (
                confidence,
                elapsed_ms,
                visible_count,
                packet.submitted_at,
                packet.mode,
            )
            # Последние записи нужны, если GUI пропустил несколько опросов результата.
            while len(inference_metadata) > 16:
                inference_metadata.pop(min(inference_metadata))
        return tuple(detections)

    inference_worker = LatestInferenceWorker(predict_frame)
    latest_detections = ()
    last_inference_sequence = 0
    last_submitted_frame = None
    last_motion_frame = None
    last_motion_result: MotionResult | None = None
    last_motion_sequence = 0
    last_full_scan_at = float("-inf")
    latest_detection_submitted_at = 0.0
    latest_inference_mode = "ожидается"
    detection_enabled_since = time.monotonic()
    display_poll_ms = display_poll_delay_ms(args.display["poll_delay_ms"], camera_fps)
    log_diagnostic(
        f"Опрос окна: до {1000 / display_poll_ms:.1f} FPS "
        f"(пауза {display_poll_ms} мс); распознавание работает отдельно"
    )
    if two_stage_enabled:
        log_diagnostic(
            "Двухступенчатый режим: движение фона → кандидат → YOLO | "
            f"полный обход раз в {args.two_stage['full_scan_interval_ms']} мс | "
            f"возраст рамки не больше {args.two_stage['max_detection_age_ms']} мс"
        )
    display_window_started = time.monotonic()
    display_frame_count = 0
    last_console_status_at: float | None = display_window_started
    try:
        while True:
            frame = camera.latest(timeout=3.0)
            if not first_frame_reported:
                frame_height, frame_width = frame.shape[:2]
                log_diagnostic(f"Первый кадр получен: {frame_width}x{frame_height}")
                first_frame_reported = True
            if detection_enabled:
                submitted_at = time.monotonic()
                request: InferenceRequest | None = None
                if two_stage_enabled:
                    if frame is not last_motion_frame:
                        motion_worker.submit(frame)
                        last_motion_frame = frame
                    motion_snapshot = motion_worker.latest()
                    if motion_snapshot.sequence > last_motion_sequence:
                        if not isinstance(motion_snapshot.value, MotionResult):
                            raise RuntimeError("Быстрый анализ вернул недопустимый результат")
                        last_motion_result = motion_snapshot.value
                        last_motion_sequence = motion_snapshot.sequence
                    full_scan_due = (
                        submitted_at - last_full_scan_at
                        >= args.two_stage["full_scan_interval_ms"] / 1000.0
                    )
                    if full_scan_due and (
                        last_motion_result is None or not last_motion_result.boxes
                    ):
                        request = full_frame_inference_request(frame, submitted_at)
                        last_full_scan_at = submitted_at
                    elif last_motion_result is not None and last_motion_result.boxes:
                        request = candidate_inference_request(
                            frame,
                            last_motion_result,
                            last_motion_result.boxes[0],
                            args.two_stage,
                            submitted_at,
                        )
                elif frame is not last_submitted_frame:
                    request = full_frame_inference_request(frame, submitted_at)
                    last_submitted_frame = frame
                if request is not None:
                    inference_worker.submit(request)
                snapshot = inference_worker.latest()
                if snapshot.sequence > last_inference_sequence:
                    with inference_metadata_lock:
                        metadata = inference_metadata.get(snapshot.sequence)
                    last_inference_sequence = snapshot.sequence
                    if metadata is not None and metadata[3] >= detection_enabled_since:
                        raw_confidence, inference_ms, detection_count, latest_detection_submitted_at, latest_inference_mode = metadata
                        latest_detections = snapshot.detections
                if (
                    two_stage_enabled
                    and latest_detection_submitted_at > 0
                    and (time.monotonic() - latest_detection_submitted_at) * 1000.0
                    > args.two_stage["max_detection_age_ms"]
                ):
                    latest_detections = ()
                preview = draw_overlay(frame, latest_detections, args.confidence, args.overlay)
            else:
                latest_detections = ()
                preview = frame.copy()
            preview = compose_display_frame(
                preview,
                args,
                system_status.text(),
                raw_confidence,
            )
            cv2.imshow(window_title, preview)
            key = cv2.waitKey(display_poll_ms) & 0xFF
            display_frame_count += 1
            status_time = time.monotonic()
            if should_emit_console_status(
                status_time,
                last_console_status_at,
                args.diagnostics["console_status_interval_ms"],
            ):
                window_duration = max(status_time - display_window_started, 1e-9)
                display_fps = display_frame_count / window_duration
                detection_age_ms = (
                    (status_time - latest_detection_submitted_at) * 1000.0
                    if latest_detection_submitted_at > 0
                    else None
                )
                stage_status = (
                    f" | путь={latest_inference_mode} | возраст={detection_age_ms:.0f} мс"
                    if detection_age_ms is not None
                    else " | путь=ожидается"
                )
                print(
                    console_status(
                        raw_confidence,
                        detection_count,
                        args.confidence,
                        inference_ms,
                    )
                    + f" | вывод={display_fps:.1f} FPS{stage_status}",
                    flush=True,
                )
                last_console_status_at = status_time
                display_window_started = status_time
                display_frame_count = 0
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                detection_enabled = not detection_enabled
                if not detection_enabled:
                    latest_detections = ()
                    raw_confidence = 0.0
                    detection_count = 0
                    inference_ms = None
                    latest_detection_submitted_at = 0.0
                    latest_inference_mode = "выключено"
                else:
                    detection_enabled_since = time.monotonic()
                    last_submitted_frame = None
                    last_motion_frame = None
                    last_motion_result = None
                    last_motion_sequence = 0
                    last_full_scan_at = float("-inf")
                log_diagnostic(f"Распознавание {'включено' if detection_enabled else 'выключено'}.")
    finally:
        inference_worker.close()
        if motion_worker is not None:
            motion_worker.close()
        camera.close()
        capture_device.release()
        cv2.destroyAllWindows()
        log_diagnostic("Источник камеры закрыт.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Обрабатывает параметры, перечисляет камеры или запускает испытание."""
    args = build_parser().parse_args(argv)
    log_path = args.log if args.log is not None else default_log_path()
    try:
        with tee_output(log_path) as resolved_log_path:
            log_diagnostic("=== VideoT16: живое испытание модели ===")
            log_diagnostic(f"Время запуска: {datetime.now().astimezone().isoformat(timespec='seconds')}")
            log_diagnostic(f"Журнал: {resolved_log_path}")
            try:
                args = resolve_run_args(args)
                validate_args(args)
                log_diagnostic(f"Профиль: {args.profile or 'базовые параметры'}")
                log_diagnostic(
                    f"Параметры: source={args.source} | model={args.model} | "
                    f"confidence={args.confidence:.0%} | min-confidence={args.min_confidence:.0%} | "
                    f"tile-size={args.tile_size} | tile-overlap={args.tile_overlap:.2f} | "
                    f"threads={args.threads} | full-frame={args.full_frame} | "
                    f"capture={args.capture['width']}x{args.capture['height']}@{args.capture['fps']:.1f} | "
                    f"window={args.display['window_width']}x{args.display['window_height']} | "
                    f"frame={args.overlay['box']['thickness']}px"
                )
                if args.list_cameras:
                    cameras = scan_cameras()
                    print_cameras(cameras)
                    result = 0 if cameras else 1
                else:
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
        resolved_log_path = log_path.expanduser().resolve()
        try:
            shown_log_path = resolved_log_path.relative_to(PROJECT_ROOT)
        except ValueError:
            shown_log_path = resolved_log_path
        print(f"Журнал: {shown_log_path}")
        return result
    except OSError as error:
        print(f"Не удалось создать файл журнала {log_path}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
