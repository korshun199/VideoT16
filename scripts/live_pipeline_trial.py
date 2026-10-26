#!/usr/bin/env python3
"""V2: независимый захват и плавный задержанный вывод с сопровождением рамки.

Стенд запускается через `./lcam --v2`, не открывает UART, J7 или OSD и не
меняет штатный `./lcam`. Камера складывает кадры в ограниченный кольцевой
буфер; окно проигрывает их с конфигурируемой задержкой, пока YOLO и оптический
трекер работают параллельно.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from scripts.compare_video_trials import (  # noqa: E402
    draw_overlay,
    infer_full_frame,
    infer_tiles,
    model_input_size,
)
from src.live_pipeline import (  # noqa: E402
    DetectionSnapshot,
    FrameHub,
    InferenceRequest,
    LatestDetectionWorker,
)
from src.realtime import Detection, SystemStatusMonitor  # noqa: E402


DEFAULT_PIPELINE = {
    "threads": 2,
    "submission_interval_frames": 1,
    "display_fps": 60,
    "output_delay_ms": 2200,
    "output_buffer_max_mb": 512,
    "box_hold_ms": 1000,
    "max_match_distance_fraction": 0.25,
}

DEFAULT_CAPTURE = {"width": 640, "height": 480, "fps": 30.0, "buffer_size": 1}
DEFAULT_DISPLAY = {
    "window_title": "VideoT16 — V2 YOLO",
    "window_width": 1280,
    "window_height": 800,
    "max_frame_width": 1280,
    "max_frame_height": 720,
    "scale_up": False,
    "poll_delay_ms": 1,
}
DEFAULT_STATUS_PANEL = {
    "position": "bottom",
    "height": 28,
    "background_color_rgb": [35, 35, 35],
    "left": 12,
    "text_y": 19,
    "font": "FONT_HERSHEY_SIMPLEX",
    "font_scale": 0.42,
    "text_color_rgb": [230, 230, 230],
    "thickness": 1,
    "max_text_chars": 120,
}
DEFAULT_OVERLAY = {
    "near_threshold_margin": 0.10,
    "near_threshold_color_rgb": [255, 165, 0],
    "box": {"scale_x": 1.0, "scale_y": 1.0, "thickness": 1, "color_rgb": [80, 230, 40]},
    "label": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.55,
        "thickness": 1,
        "offset_y": 7,
        "min_y": 18,
        "color_rgb": [80, 230, 40],
    },
}


@dataclass(frozen=True)
class V2Profile:
    """Полный набор проверенных параметров отдельного V2-запуска."""

    name: str
    source: int | str
    model: Path
    confidence: float
    min_confidence: float
    tile_size: int
    tile_overlap: float
    threads: int
    full_frame: bool
    capture: dict[str, Any]
    display: dict[str, Any]
    status_panel: dict[str, Any]
    overlay: dict[str, Any]
    pipeline: dict[str, Any]
    console_status_interval_ms: int


@dataclass(frozen=True)
class DisplayOverlaySelection:
    """Рамка для конкретного выходного кадра и ID её кадра-источника."""

    detections: tuple[Any, ...]
    source_sequence: int | None
    age_ms: float | None
    mode: str
    latest_result: DetectionSnapshot | None
    source_sequence_end: int | None = None


class DelayedOverlaySelector:
    """Сводит рамку к ID выводимого кадра по двум соседним ответам YOLO."""

    def __init__(self, hold_ms: int) -> None:
        """Создаёт ограниченное по времени состояние рамки отображения."""
        if hold_ms < 0:
            raise ValueError("box_hold_ms не может быть отрицательным")
        self._hold_ns = hold_ms * 1_000_000
        self._latest_seen_sequence = 0
        self._latest_result: DetectionSnapshot | None = None
        self._overlay_result: DetectionSnapshot | None = None

    def clear(self, after_sequence: int = 0) -> None:
        """Сбрасывает последнюю рамку после отключения распознавания."""
        self._latest_seen_sequence = max(0, after_sequence)
        self._latest_result = None
        self._overlay_result = None

    def select(
        self,
        frame: Any,
        before: DetectionSnapshot | None,
        after: DetectionSnapshot | None,
        confidence_threshold: float,
        near_threshold_margin: float,
        max_match_distance_fraction: float = 0.25,
    ) -> DisplayOverlaySelection:
        """Интерполирует рамки между соседними результатами, иначе кратко держит старую."""
        result = before
        if result is not None and self._latest_seen_sequence < result.source_sequence <= frame.sequence:
            self._latest_seen_sequence = result.source_sequence
            self._latest_result = result
            minimum_confidence = max(0.0, confidence_threshold - near_threshold_margin)
            if any(item.confidence >= minimum_confidence for item in result.detections):
                self._overlay_result = result
            else:
                # Новый ответ без допустимых объектов отменяет прежнюю рамку:
                # иначе старые координаты переживают пропажу цели до box_hold_ms.
                self._overlay_result = None

        if before is not None and after is not None and before.source_sequence < frame.sequence < after.source_sequence:
            interpolated = self._interpolate(
                before,
                after,
                frame.sequence,
                frame.bgr.shape[1],
                frame.bgr.shape[0],
                max(0.0, max_match_distance_fraction),
                max(0.0, confidence_threshold - near_threshold_margin),
            )
            if interpolated:
                return DisplayOverlaySelection(
                    interpolated,
                    frame.sequence,
                    0.0,
                    "INTERPOLATED",
                    self._latest_result,
                    after.source_sequence,
                )

        if before is not None and before.source_sequence == frame.sequence:
            minimum_confidence = max(0.0, confidence_threshold - near_threshold_margin)
            exact = tuple(item for item in before.detections if item.confidence >= minimum_confidence)
            if exact:
                self._overlay_result = before
                return DisplayOverlaySelection(exact, frame.sequence, 0.0, "YOLO", self._latest_result)

        overlay = self._overlay_result
        if overlay is None or frame.captured_at_ns < overlay.source_captured_at_ns:
            return DisplayOverlaySelection((), None, None, "NONE", self._latest_result)
        age_ms = (frame.captured_at_ns - overlay.source_captured_at_ns) / 1_000_000
        if age_ms * 1_000_000 > self._hold_ns:
            self._overlay_result = None
            return DisplayOverlaySelection((), overlay.source_sequence, age_ms, "EXPIRED", self._latest_result)
        mode = "YOLO" if overlay.source_sequence == frame.sequence else "HOLD"
        return DisplayOverlaySelection(overlay.detections, overlay.source_sequence, age_ms, mode, self._latest_result)

    @staticmethod
    def _interpolate(
        before: DetectionSnapshot,
        after: DetectionSnapshot,
        sequence: int,
        frame_width: int,
        frame_height: int,
        max_distance_fraction: float,
        minimum_confidence: float,
    ) -> tuple[Detection, ...]:
        """Сопоставляет объекты одного класса по близости и линейно двигает рамку."""
        left = [item for item in before.detections if item.confidence >= minimum_confidence]
        right = [item for item in after.detections if item.confidence >= minimum_confidence]
        if not left or not right:
            return ()
        ratio = (sequence - before.source_sequence) / (after.source_sequence - before.source_sequence)
        maximum_distance = math.hypot(frame_width, frame_height) * max_distance_fraction
        candidates = []
        for left_index, first in enumerate(left):
            first_center = ((first.x1 + first.x2) / 2, (first.y1 + first.y2) / 2)
            for right_index, second in enumerate(right):
                if first.name != second.name:
                    continue
                second_center = ((second.x1 + second.x2) / 2, (second.y1 + second.y2) / 2)
                distance = math.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])
                if distance <= maximum_distance:
                    candidates.append((distance, left_index, right_index))
        matched_left: set[int] = set()
        matched_right: set[int] = set()
        interpolated = []
        for _, left_index, right_index in sorted(candidates):
            if left_index in matched_left or right_index in matched_right:
                continue
            matched_left.add(left_index)
            matched_right.add(right_index)
            first = left[left_index]
            second = right[right_index]
            interpolated.append(Detection(
                x1=round(first.x1 + (second.x1 - first.x1) * ratio),
                y1=round(first.y1 + (second.y1 - first.y1) * ratio),
                x2=round(first.x2 + (second.x2 - first.x2) * ratio),
                y2=round(first.y2 + (second.y2 - first.y2) * ratio),
                name=first.name,
                confidence=first.confidence + (second.confidence - first.confidence) * ratio,
            ))
        return tuple(interpolated)


class CameraCaptureWorker:
    """Считывает кадры и публикует их сразу в видеобуфер и mailbox YOLO."""

    def __init__(
        self,
        capture: Any,
        hub: FrameHub,
        worker: LatestDetectionWorker,
        submission_interval_frames: int,
        enabled: threading.Event,
    ) -> None:
        """Создаёт первый этап и отправляет каждый разрешённый кадр в YOLO-mailbox."""
        self._capture = capture
        self._hub = hub
        self._worker = worker
        self._submission_interval_frames = submission_interval_frames
        self._enabled = enabled
        self._stop_requested = threading.Event()
        self._stats_lock = threading.Lock()
        self._submitted_sequence = 0
        self._submitted_count = 0
        self._thread = threading.Thread(
            target=self._run,
            name="v2-camera-capture",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        """Непрерывно считывает камеру и отправляет только свежие кадры в hub."""
        try:
            while not self._stop_requested.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    if self._stop_requested.is_set():
                        break
                    raise RuntimeError("Камера не вернула кадр")
                captured_at_ns = time.monotonic_ns()
                packet = self._hub.publish(frame, None, captured_at_ns=captured_at_ns)
                if (
                    self._enabled.is_set()
                    and packet.sequence % self._submission_interval_frames == 0
                ):
                    self._worker.submit(
                        InferenceRequest(
                            source=packet,
                            mode="full_scan",
                            submitted_at_ns=time.monotonic_ns(),
                            image=packet.bgr,
                        )
                    )
                    with self._stats_lock:
                        self._submitted_sequence = packet.sequence
                        self._submitted_count += 1
        except Exception as error:  # noqa: BLE001 — ошибка должна дойти до главного цикла.
            self._hub.fail(error)

    def close(self) -> None:
        """Останавливает чтение и освобождает VideoCapture, чтобы снять блокировку read()."""
        self._stop_requested.set()
        self._capture.release()
        self._thread.join(timeout=2.0)

    def statistics(self) -> dict[str, int]:
        """Возвращает ID и число кадров, помещённых в буфер распознавания."""
        with self._stats_lock:
            return {
                "submitted_sequence": self._submitted_sequence,
                "submitted_count": self._submitted_count,
            }


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры изолированного запуска V2."""
    parser = argparse.ArgumentParser(
        description="VideoT16 V2: свежий вход для YOLO и отдельное плавное окно с задержкой."
    )
    parser.add_argument("--profile", default="sky", help="имя профиля из config/live_drone_profiles.json")
    parser.add_argument("--log", type=Path, help="необязательный путь журнала")
    return parser


def _merge(defaults: dict[str, Any], values: object) -> dict[str, Any]:
    """Рекурсивно объединяет JSON-раздел с безопасными значениями V2."""
    merged = deepcopy(defaults)
    if not isinstance(values, dict):
        return merged
    for key, value in values.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_source(value: object) -> int | str:
    """Сохраняет путь устройства либо переводит строковый номер в индекс OpenCV."""
    if not isinstance(value, str) or not value:
        raise ValueError("settings.source должен быть непустой строкой")
    return int(value) if value.isdigit() else value


def load_profile(profile_name: str, profiles_path: Path | None = None) -> V2Profile:
    """Читает один профиль, не используя и не включая `two_stage` старого стенда."""
    path = profiles_path or PROJECT_ROOT / "config" / "live_drone_profiles.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Не удалось прочитать профиль {path}: {error}") from error
    profiles = document.get("profiles") if isinstance(document, dict) else None
    profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        available = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else "нет"
        raise ValueError(f"Профиль '{profile_name}' не найден. Доступны: {available}")
    settings = profile.get("settings")
    if not isinstance(settings, dict):
        raise ValueError(f"У профиля '{profile_name}' отсутствует раздел settings")
    model_value = settings.get("model")
    if not isinstance(model_value, str) or not model_value:
        raise ValueError("settings.model должен быть непустым путём")
    model = Path(model_value)
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    diagnostics = profile.get("diagnostics") if isinstance(profile.get("diagnostics"), dict) else {}
    profile_v2 = _merge(DEFAULT_PIPELINE, profile.get("pipeline_v2"))
    return V2Profile(
        name=profile_name,
        source=_parse_source(settings.get("source")),
        model=model,
        confidence=float(settings.get("confidence", 0.70)),
        min_confidence=float(settings.get("min_confidence", 0.05)),
        tile_size=int(settings.get("tile_size", 640)),
        tile_overlap=float(settings.get("tile_overlap", 0.20)),
        threads=int(profile_v2.get("threads", settings.get("threads", 2))),
        full_frame=bool(settings.get("full_frame", False)),
        capture=_merge(DEFAULT_CAPTURE, profile.get("capture")),
        display=_merge(DEFAULT_DISPLAY, profile.get("display")),
        status_panel=_merge(DEFAULT_STATUS_PANEL, profile.get("status_panel")),
        overlay=_merge(DEFAULT_OVERLAY, profile.get("overlay")),
        pipeline=profile_v2,
        console_status_interval_ms=int(diagnostics.get("console_status_interval_ms", 1000)),
    )


def validate_profile(profile: V2Profile) -> None:
    """Проверяет V2-настройки до открытия модели и USB-устройства."""
    if not 0.0 < profile.confidence <= 1.0:
        raise ValueError("settings.confidence должен быть больше 0 и не больше 1")
    if not 0.0 < profile.min_confidence <= profile.confidence:
        raise ValueError("settings.min_confidence должен быть больше 0 и не выше confidence")
    if profile.tile_size < 32:
        raise ValueError("settings.tile_size должен быть не меньше 32")
    if not 0.0 <= profile.tile_overlap < 0.5:
        raise ValueError("settings.tile_overlap должен быть от 0 до 0.5")
    if profile.threads < 1:
        raise ValueError("pipeline_v2.threads должен быть не меньше 1")
    for key in ("width", "height"):
        if not isinstance(profile.capture[key], int) or profile.capture[key] < 32:
            raise ValueError(f"capture.{key} должен быть целым числом не меньше 32")
    if not isinstance(profile.capture["fps"], (int, float)) or profile.capture["fps"] <= 0:
        raise ValueError("capture.fps должен быть больше 0")
    if not isinstance(profile.capture["buffer_size"], int) or profile.capture["buffer_size"] < 1:
        raise ValueError("capture.buffer_size должен быть не меньше 1")
    for key in ("window_width", "window_height", "max_frame_width", "max_frame_height"):
        if not isinstance(profile.display[key], int) or profile.display[key] < 1:
            raise ValueError(f"display.{key} должен быть положительным целым числом")
    pipeline = profile.pipeline
    if not isinstance(pipeline["submission_interval_frames"], int) or not 1 <= pipeline["submission_interval_frames"] <= 120:
        raise ValueError("pipeline_v2.submission_interval_frames должен быть от 1 до 120")
    if not isinstance(pipeline["display_fps"], (int, float)) or not 1 <= pipeline["display_fps"] <= 120:
        raise ValueError("pipeline_v2.display_fps должен быть от 1 до 120")
    if not isinstance(pipeline["output_delay_ms"], int) or not 100 <= pipeline["output_delay_ms"] <= 5000:
        raise ValueError("pipeline_v2.output_delay_ms должен быть от 100 до 5000")
    if not isinstance(pipeline["output_buffer_max_mb"], int) or not 64 <= pipeline["output_buffer_max_mb"] <= 2048:
        raise ValueError("pipeline_v2.output_buffer_max_mb должен быть от 64 до 2048")
    if not isinstance(pipeline["box_hold_ms"], int) or not 0 <= pipeline["box_hold_ms"] <= 5000:
        raise ValueError("pipeline_v2.box_hold_ms должен быть от 0 до 5000")
    max_match_distance = profile.pipeline["max_match_distance_fraction"]
    if not isinstance(max_match_distance, (int, float)) or not 0.0 < max_match_distance <= 1.0:
        raise ValueError("pipeline_v2.max_match_distance_fraction должен быть в диапазоне (0, 1]")
    requested_frame_bytes = int(profile.capture["width"]) * int(profile.capture["height"]) * 3
    required_buffer_bytes = math.ceil(
        requested_frame_bytes * float(profile.capture["fps"]) * int(pipeline["output_delay_ms"]) / 1000
    )
    configured_buffer_bytes = int(pipeline["output_buffer_max_mb"]) * 1024 * 1024
    if required_buffer_bytes > configured_buffer_bytes:
        raise ValueError(
            "pipeline_v2.output_buffer_max_mb недостаточен для заданных capture и output_delay_ms "
            f"(нужно не менее {math.ceil(required_buffer_bytes / 1024 / 1024)} МиБ)"
        )
    panel = profile.status_panel
    if not isinstance(panel["height"], int) or panel["height"] < 1:
        raise ValueError("status_panel.height должен быть положительным целым числом")
    if not isinstance(panel["font_scale"], (int, float)) or panel["font_scale"] <= 0:
        raise ValueError("status_panel.font_scale должен быть больше 0")
    if not isinstance(panel["thickness"], int) or panel["thickness"] < 1:
        raise ValueError("status_panel.thickness должен быть положительным целым числом")
    if not isinstance(panel["max_text_chars"], int) or panel["max_text_chars"] < 1:
        raise ValueError("status_panel.max_text_chars должен быть положительным целым числом")
    _font(panel["font"])
    _bgr(panel["background_color_rgb"])
    _bgr(panel["text_color_rgb"])
    overlay = profile.overlay
    _font(overlay["label"]["font"])
    for color in (
        overlay["near_threshold_color_rgb"],
        overlay["box"]["color_rgb"],
        overlay["label"]["color_rgb"],
    ):
        _bgr(color)
    if not isinstance(overlay["box"]["thickness"], int) or overlay["box"]["thickness"] < 1:
        raise ValueError("overlay.box.thickness должен быть положительным целым числом")
    if not isinstance(overlay["label"]["font_scale"], (int, float)) or overlay["label"]["font_scale"] <= 0:
        raise ValueError("overlay.label.font_scale должен быть больше 0")
    if not isinstance(overlay["label"]["thickness"], int) or overlay["label"]["thickness"] < 1:
        raise ValueError("overlay.label.thickness должен быть положительным целым числом")
    if profile.console_status_interval_ms < 0:
        raise ValueError("diagnostics.console_status_interval_ms должен быть неотрицательным")


def default_log_path() -> Path:
    """Создаёт новый путь журнала V2, не смешивая его со старыми запусками."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return PROJECT_ROOT / "test_samples" / "video_trials" / "logs" / f"live_pipeline_v2_{stamp}.log"


def make_logger(log_path: Path) -> logging.Logger:
    """Настраивает короткий консольный и полный файловый журнал одного запуска."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"videot16.v2.{log_path.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def close_logger(logger: logging.Logger) -> None:
    """Закрывает файловый журнал, чтобы его можно было сразу открыть после выхода."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def configure_capture(capture: Any, settings: dict[str, Any], logger: logging.Logger) -> dict[str, float | int]:
    """Запрашивает параметры V4L2 и записывает фактически принятый режим."""
    capture.set(cv2.CAP_PROP_BUFFERSIZE, settings["buffer_size"])
    for title, property_id, requested in (
        ("ширина", cv2.CAP_PROP_FRAME_WIDTH, settings["width"]),
        ("высота", cv2.CAP_PROP_FRAME_HEIGHT, settings["height"]),
        ("FPS", cv2.CAP_PROP_FPS, settings["fps"]),
    ):
        accepted = capture.set(property_id, requested)
        logger.info("Запрос захвата %s=%s: %s", title, requested, "принят" if accepted else "не подтверждён")
    actual = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    logger.info("Фактический захват: %(width)sx%(height)s @ %(fps).1f FPS", actual)
    return actual


def _font(name: object) -> int:
    """Возвращает шрифт OpenCV или понятную ошибку конфигурации."""
    if not isinstance(name, str) or not hasattr(cv2, name):
        raise ValueError(f"Неизвестный шрифт OpenCV: {name}")
    return int(getattr(cv2, name))


def _bgr(rgb: object) -> tuple[int, int, int]:
    """Преобразует RGB-конфигурацию в порядок цветов OpenCV BGR."""
    if not isinstance(rgb, list) or len(rgb) != 3 or any(not isinstance(value, int) for value in rgb):
        raise ValueError("Цвет должен быть массивом из трёх целых RGB-значений")
    if any(not 0 <= value <= 255 for value in rgb):
        raise ValueError("Компоненты RGB должны быть в диапазоне 0..255")
    return rgb[2], rgb[1], rgb[0]


def _fit(frame: Any, max_width: int, max_height: int, scale_up: bool) -> Any:
    """Масштабирует только копию для экрана, сохраняя пропорции видеокадра."""
    height, width = frame.shape[:2]
    factor = min(max_width / width, max_height / height)
    if not scale_up:
        factor = min(1.0, factor)
    if math.isclose(factor, 1.0):
        return frame
    interpolation = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(frame, (max(1, round(width * factor)), max(1, round(height * factor))), interpolation=interpolation)


def _status_fields(system_status: str) -> tuple[str, str]:
    """Разбирает устойчивую строку `TEMP ... CPU ...` без зависимости от платформы."""
    values = system_status.split()
    if len(values) >= 4 and values[0] == "TEMP" and values[2] == "CPU":
        return values[1], values[3]
    return "--.-C", "--%"


def compose_frame(
    packet: Any,
    profile: V2Profile,
    system_text: str,
    result: Any | None,
    detections: tuple[Any, ...] | None = None,
    output_delay_ms: int = 0,
    overlay_source_sequence: int | None = None,
    overlay_mode: str = "NONE",
    overlay_age_ms: float | None = None,
) -> Any:
    """Показывает цветной выходной кадр и явно маркирует источник рамки."""
    display = profile.display
    panel = profile.status_panel
    panel_height = min(int(panel["height"]), int(display["window_height"]) - 1)
    video_limit_height = min(int(display["max_frame_height"]), int(display["window_height"]) - panel_height)
    visible_detections = detections if detections is not None else (result.detections if result is not None else ())
    display_source = draw_overlay(packet.bgr, visible_detections, profile.confidence, profile.overlay)
    fitted = _fit(
        display_source,
        min(int(display["max_frame_width"]), int(display["window_width"])),
        video_limit_height,
        bool(display["scale_up"]),
    )
    canvas = cv2.copyMakeBorder(
        fitted,
        (video_limit_height - fitted.shape[0]) // 2,
        video_limit_height - fitted.shape[0] - (video_limit_height - fitted.shape[0]) // 2 + panel_height,
        (int(display["window_width"]) - fitted.shape[1]) // 2,
        int(display["window_width"]) - fitted.shape[1] - (int(display["window_width"]) - fitted.shape[1]) // 2,
        cv2.BORDER_CONSTANT,
        value=_bgr(panel["background_color_rgb"]),
    )
    temperature, cpu = _status_fields(system_text)
    confidence = result.raw_confidence if result is not None else 0.0
    age_ms = (
        max(0, time.monotonic_ns() - result.source_captured_at_ns) / 1_000_000
        if result is not None
        else None
    )
    result_text = (
        f"SEQ {packet.sequence} | YOLO {result.source_sequence} D{len(result.detections)} | "
        f"BOX {overlay_source_sequence if overlay_source_sequence is not None else '-'}:{overlay_mode} | "
        f"OUT {output_delay_ms}ms | CPU {cpu} | TEMP {temperature} | "
        f"CONF {confidence:.0%} THR {profile.confidence:.0%} | "
        f"YOLO {result.inference_ms:.0f}ms | AGE {age_ms:.0f}ms"
        if result is not None
        else f"SEQ {packet.sequence} | OUT {output_delay_ms}ms | CPU {cpu} | TEMP {temperature} | WAITING YOLO"
    )
    if overlay_age_ms is not None:
        result_text += f" | BAGE {overlay_age_ms:.0f}ms"
    status_top = canvas.shape[0] - panel_height
    cv2.rectangle(canvas, (0, status_top), (canvas.shape[1] - 1, canvas.shape[0] - 1), _bgr(panel["background_color_rgb"]), -1)
    cv2.putText(
        canvas,
        result_text[: int(panel["max_text_chars"])],
        (int(panel["left"]), min(canvas.shape[0] - 3, status_top + int(panel["text_y"]))),
        _font(panel["font"]),
        float(panel["font_scale"]),
        _bgr(panel["text_color_rgb"]),
        int(panel["thickness"]),
        cv2.LINE_AA,
    )
    return canvas


def _camera_api() -> int:
    """Выбирает основной backend камеры для Linux, Windows или иной ОС."""
    if sys.platform.startswith("linux"):
        return cv2.CAP_V4L2
    if sys.platform == "win32":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _camera_source(source: int | str) -> int | str:
    """Преобразует Linux-путь by-id в обычный video-узел для OpenCV V4L2."""
    if isinstance(source, str) and sys.platform.startswith("linux"):
        path = Path(source)
        if path.is_symlink():
            return str(path.resolve())
    return source


def run(profile: V2Profile, logger: logging.Logger) -> int:
    """Запускает три независимых этапа: захват, YOLO и отложенный вывод."""
    if not profile.model.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {profile.model}")
    from src.onnx_detector import OnnxDetector

    model_size = model_input_size(profile.model)
    detector = OnnxDetector(
        profile.model,
        profile.min_confidence,
        False,
        model_size,
        threads=profile.threads,
    )
    source = _camera_source(profile.source)
    capture = cv2.VideoCapture(source, _camera_api())
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть источник камеры: {profile.source}")
    try:
        capture_actual = configure_capture(capture, profile.capture, logger)
    except Exception:
        capture.release()
        raise
    output_delay_ms = int(profile.pipeline["output_delay_ms"])
    output_buffer_max_bytes = int(profile.pipeline["output_buffer_max_mb"]) * 1024 * 1024
    actual_frame_bytes = capture_actual["width"] * capture_actual["height"] * 3
    actual_required_bytes = math.ceil(actual_frame_bytes * capture_actual["fps"] * output_delay_ms / 1000)
    if actual_required_bytes > output_buffer_max_bytes:
        capture.release()
        raise ValueError(
            "Лимита буфера недостаточно для фактического режима камеры: "
            f"нужно не менее {math.ceil(actual_required_bytes / 1024 / 1024)} МиБ"
        )
    history_size = max(96, math.ceil((output_delay_ms + 2000) * capture_actual["fps"] / 1000) + 4)
    hub = FrameHub(history_size, output_buffer_max_bytes)
    status_monitor = SystemStatusMonitor(interval=1.0)

    def predict(request: InferenceRequest):
        """Запускает полный или плиточный YOLO для кадра, связанного с запросом."""
        if profile.full_frame:
            detections, _, confidence = infer_full_frame(detector, request.image)
        else:
            detections, _, confidence = infer_tiles(
                detector, request.image, profile.tile_size, profile.tile_overlap
            )
        return detections, confidence

    worker = LatestDetectionWorker(predict, name="v2-yolo")
    overlay_selector = DelayedOverlaySelector(int(profile.pipeline["box_hold_ms"]))
    detection_enabled = threading.Event()
    detection_enabled.set()
    camera = CameraCaptureWorker(
        capture,
        hub,
        worker,
        int(profile.pipeline["submission_interval_frames"]),
        detection_enabled,
    )
    started = time.monotonic()
    try:
        cv2.namedWindow(profile.display["window_title"], cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(profile.display["window_title"], profile.display["window_width"], profile.display["window_height"])
        logger.info(
            "V2 single YOLO window: profile=%s | model=%s | input=%s | source=%s | capture=%sx%s@%.1f | ORT threads=%s",
            profile.name,
            profile.model,
            model_size,
            source,
            capture_actual["width"],
            capture_actual["height"],
            capture_actual["fps"],
            profile.threads,
        )
        logger.info(
            "Три этапа: capture-buffer → YOLO-result-buffer → delayed-display | задержка=%dms | буфер до %dMiB | FPS окна=%s | box-hold=%dms",
            output_delay_ms,
            profile.pipeline["output_buffer_max_mb"],
            profile.pipeline["display_fps"],
            profile.pipeline["box_hold_ms"],
        )
        logger.info("Этап 1 сохраняет кадры; этап 2 распознаёт свежие ID; этап 3 привязывает рамку к ID выводимого кадра.")
        last_display_sequence = 0
        display_period = 1.0 / float(profile.pipeline["display_fps"])
        next_display_at = time.perf_counter()
        last_log_at = time.monotonic()
        last_logged_capture_sequence = 0
        rendered = 0
        started = last_log_at
        while True:
            packet = hub.delayed_next(last_display_sequence, output_delay_ms)
            if packet is None:
                key = cv2.waitKey(max(1, int(profile.display.get("poll_delay_ms", 1)))) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue
            display_gap = max(0, packet.sequence - last_display_sequence - 1)
            last_display_sequence = packet.sequence
            before, after = worker.snapshots_around(packet.sequence) if detection_enabled.is_set() else (None, None)
            result = before
            if detection_enabled.is_set():
                selection = overlay_selector.select(
                    packet,
                    before,
                    after,
                    profile.confidence,
                    float(profile.overlay["near_threshold_margin"]),
                    float(profile.pipeline["max_match_distance_fraction"]),
                )
            else:
                overlay_selector.clear(worker.statistics()["latest_result_sequence"] or 0)
                selection = DisplayOverlaySelection((), None, None, "OFF", None)
            display_frame = compose_frame(
                packet,
                profile,
                status_monitor.text(),
                result,
                selection.detections,
                output_delay_ms,
                selection.source_sequence,
                selection.mode if detection_enabled.is_set() else "OFF",
                selection.age_ms,
            )
            cv2.imshow(profile.display["window_title"], display_frame)
            rendered += 1
            delay = max(1, int(profile.display.get("poll_delay_ms", 1)))
            key = cv2.waitKey(delay) & 0xFF
            now = time.monotonic()
            if profile.console_status_interval_ms and now - last_log_at >= profile.console_status_interval_ms / 1000:
                elapsed = max(0.001, now - last_log_at)
                fps = rendered / elapsed
                frame_buffers = hub.buffer_status()
                inference_input = camera.statistics()
                inference_output = worker.statistics()
                captured_sequence = int(frame_buffers["latest_sequence"] or 0)
                capture_fps = max(0, captured_sequence - last_logged_capture_sequence) / elapsed
                last_logged_capture_sequence = captured_sequence
                detection_count = len(result.detections) if result is not None else 0
                eligible_count = (
                    sum(
                        item.confidence >= profile.confidence - float(profile.overlay["near_threshold_margin"])
                        for item in result.detections
                    )
                    if result is not None
                    else 0
                )
                output_span = (
                    f"{frame_buffers['oldest_output_sequence']}..{frame_buffers['newest_output_sequence']}"
                )
                visible_box_count = sum(
                    item.confidence >= profile.confidence - float(profile.overlay["near_threshold_margin"])
                    for item in selection.detections
                )
                if result is None:
                    logger.info(
                        "STAGES cap=%s@%.1f framebuf=%s(%d,%dMiB) | infer_in=%s/%d infer_out=WAITING | display=%d skip=%d overlay=-:NONE | fps=%.1f",
                        frame_buffers["latest_sequence"], capture_fps, output_span, frame_buffers["output_frames"],
                        frame_buffers["output_bytes"] // (1024 * 1024), inference_input["submitted_sequence"],
                        inference_input["submitted_count"], packet.sequence, display_gap, fps,
                    )
                else:
                    overlay_id = (
                        f"{selection.source_sequence}-{selection.source_sequence_end}"
                        if selection.source_sequence_end is not None
                        else selection.source_sequence if selection.source_sequence is not None else "-"
                    )
                    overlay_age = selection.age_ms if selection.age_ms is not None else -1
                    logger.info(
                        "STAGES cap=%s@%.1f framebuf=%s(%d,%dMiB) | infer_in=%s/%d overwrite=%d | infer_out=%d det=%d pass=%d raw=%.0f%% q=%.0fms yolo=%.0fms | display=%d skip=%d overlay=%s:%s boxes=%d bage=%.0fms | fps=%.1f",
                        frame_buffers["latest_sequence"], capture_fps, output_span, frame_buffers["output_frames"],
                        frame_buffers["output_bytes"] // (1024 * 1024), inference_input["submitted_sequence"],
                        inference_input["submitted_count"], inference_output["replaced"], result.source_sequence,
                        detection_count, eligible_count, result.raw_confidence * 100, result.queue_ms,
                        result.inference_ms, packet.sequence, display_gap, overlay_id, selection.mode,
                        visible_box_count, overlay_age, fps,
                    )
                last_log_at = now
                rendered = 0
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                if detection_enabled.is_set():
                    detection_enabled.clear()
                else:
                    detection_enabled.set()
                logger.info("Распознавание V2 %s.", "включено" if detection_enabled.is_set() else "выключено")
            next_display_at += display_period
            remaining_display_time = next_display_at - time.perf_counter()
            if remaining_display_time > 0:
                time.sleep(remaining_display_time)
            else:
                next_display_at = time.perf_counter()
    finally:
        camera.close()
        worker.close()
        cv2.destroyAllWindows()
        logger.info("V2 остановлен; длительность %.1f с", time.monotonic() - started)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Читает профиль, создаёт журнал и запускает базовый стенд V2."""
    args = build_parser().parse_args(argv)
    log_path = (args.log or default_log_path()).expanduser().resolve()
    logger = make_logger(log_path)
    try:
        profile = load_profile(args.profile)
        validate_profile(profile)
        logger.info("=== VideoT16 V2 live YOLO ===")
        logger.info("Журнал: %s", log_path)
        return run(profile, logger)
    except Exception as error:
        logger.exception("Ошибка запуска V2: %s", error)
        return 2
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем.")
        return 130
    finally:
        close_logger(logger)


if __name__ == "__main__":
    raise SystemExit(main())
