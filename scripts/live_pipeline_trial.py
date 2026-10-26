#!/usr/bin/env python3
"""Отдельный V2-стенд живого распознавания с точной привязкой рамки к кадру.

Стенд запускается через `./lcam --v2`, не открывает UART, J7 или OSD и не
меняет штатный `./lcam`. Единственное окно показывает кадр, который обработала
YOLO, чтобы рамка совпадала с изображением, а задержку можно было измерять.
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
    FrameHub,
    InferenceRequest,
    LatestDetectionWorker,
)
from src.onnx_detector import OnnxDetector  # noqa: E402
from src.realtime import SystemStatusMonitor  # noqa: E402


DEFAULT_PIPELINE = {
    "threads": 2,
    "submission_interval_frames": 1,
    "display_fps": 60,
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


class CameraCaptureWorker:
    """Читает VideoCapture в отдельном потоке и публикует пронумерованные кадры."""

    def __init__(self, capture: Any, hub: FrameHub) -> None:
        """Создаёт поток захвата без неиспользуемой подготовки кадров трекера."""
        self._capture = capture
        self._hub = hub
        self._stop_requested = threading.Event()
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
                    raise RuntimeError("Камера не вернула кадр")
                self._hub.publish(frame, None)
        except Exception as error:  # noqa: BLE001 — ошибка должна дойти до главного цикла.
            self._hub.fail(error)

    def close(self) -> None:
        """Останавливает чтение и освобождает VideoCapture, чтобы снять блокировку read()."""
        self._stop_requested.set()
        self._capture.release()
        self._thread.join(timeout=2.0)


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры изолированного запуска V2."""
    parser = argparse.ArgumentParser(
        description="VideoT16 V2: одно окно показывает кадр и рамку одного результата YOLO."
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
    analysis_frame: Any | None = None,
) -> Any:
    """Строит одно окно из кадра YOLO или живого кадра до первого ответа."""
    display = profile.display
    panel = profile.status_panel
    panel_height = min(int(panel["height"]), int(display["window_height"]) - 1)
    video_limit_height = min(int(display["max_frame_height"]), int(display["window_height"]) - panel_height)
    display_source = analysis_frame if analysis_frame is not None else packet.bgr
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
    frame_sequence = result.source_sequence if result is not None else packet.sequence
    result_text = (
        f"SEQ {frame_sequence} | CPU {cpu} | TEMP {temperature} | "
        f"CONF {confidence:.0%} | THR {profile.confidence:.0%} | "
        f"YOLO {result.inference_ms:.0f}ms | AGE {age_ms:.0f}ms"
        if result is not None
        else f"SEQ {packet.sequence} | CPU {cpu} | TEMP {temperature} | WAITING YOLO"
    )
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


def prepare_analysis_frame(result: Any, profile: V2Profile) -> Any:
    """Рисует реальные рамки YOLO на исходном кадре для единственного окна.

    Этот кадр намеренно не подменяется текущим: координаты детекций относятся
    только к `result.source`, а не к более свежему изображению с камеры.
    """
    return draw_overlay(
        result.source.bgr,
        result.detections,
        profile.confidence,
        profile.overlay,
    )


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
    """Запускает одно окно с исходными кадрами результатов YOLO."""
    if not profile.model.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {profile.model}")
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
    hub = FrameHub(2)
    camera = CameraCaptureWorker(capture, hub)
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
        logger.info("V2 не использует profiles.%s.two_stage; окно показывает исходный кадр каждого результата YOLO.", profile.name)
        last_display_sequence = 0
        last_submitted_sequence = 0
        detection_enabled = True
        cached_result_sequence = 0
        cached_analysis_frame = None
        display_period = 1.0 / float(profile.pipeline["display_fps"])
        next_display_at = time.perf_counter()
        last_log_at = time.monotonic()
        rendered = 0
        started = last_log_at
        while True:
            packet = hub.latest(last_display_sequence, timeout=1.0)
            if packet is None:
                continue
            last_display_sequence = packet.sequence
            if detection_enabled and packet.sequence - last_submitted_sequence >= int(profile.pipeline["submission_interval_frames"]):
                worker.submit(
                    InferenceRequest(
                        source=packet,
                        mode="full_scan",
                        submitted_at_ns=time.monotonic_ns(),
                        image=packet.bgr,
                    )
                )
                last_submitted_sequence = packet.sequence
            result = worker.latest() if detection_enabled else None
            if result is not None and result.source_sequence != cached_result_sequence:
                cached_result_sequence = result.source_sequence
                cached_analysis_frame = prepare_analysis_frame(result, profile)
            display_frame = compose_frame(
                packet,
                profile,
                status_monitor.text(),
                result,
                cached_analysis_frame,
            )
            cv2.imshow(profile.display["window_title"], display_frame)
            rendered += 1
            delay = max(1, int(profile.display.get("poll_delay_ms", 1)))
            key = cv2.waitKey(delay) & 0xFF
            now = time.monotonic()
            if profile.console_status_interval_ms and now - last_log_at >= profile.console_status_interval_ms / 1000:
                elapsed = max(0.001, now - last_log_at)
                fps = rendered / elapsed
                if result is None:
                    logger.info("display=%.1f FPS | current_frame=%d | YOLO=waiting", fps, packet.sequence)
                else:
                    age_ms = max(0, time.monotonic_ns() - result.source_captured_at_ns) / 1_000_000
                    logger.info(
                        "display=%.1f FPS | current=%d | shown=%d | raw=%.0f%% | boxes=%d | queue=%.0fms | yolo=%.0fms | age=%.0fms",
                        fps,
                        packet.sequence,
                        result.source_sequence,
                        result.raw_confidence * 100,
                        len(result.detections),
                        result.queue_ms,
                        result.inference_ms,
                        age_ms,
                    )
                last_log_at = now
                rendered = 0
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                detection_enabled = not detection_enabled
                logger.info("Распознавание V2 %s.", "включено" if detection_enabled else "выключено")
                if detection_enabled:
                    last_submitted_sequence = 0
            next_display_at += display_period
            remaining_display_time = next_display_at - time.perf_counter()
            if remaining_display_time > 0:
                time.sleep(remaining_display_time)
            else:
                next_display_at = time.perf_counter()
    finally:
        worker.close()
        camera.close()
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
