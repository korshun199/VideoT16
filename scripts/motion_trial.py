#!/usr/bin/env python3
"""Живой прототип поиска движущихся объектов после компенсации движения камеры."""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "motion_trial.json"

if str(PROJECT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_DIR))

from src.motion_compensated_detection import MotionAnalyzer, MotionResult
from src.onnx_detector import OnnxDetector
from src.realtime import InferenceSnapshot, LatestInferenceWorker, SystemStatusMonitor


class LatestCameraFrames:
    """Захватывает видео отдельно и хранит только самый свежий кадр."""

    def __init__(self, capture: cv2.VideoCapture) -> None:
        """Запускает один поток чтения для заданного видеоустройства."""
        self.capture = capture
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stop_requested = threading.Event()
        self.sequence = 0
        self.frame: np.ndarray | None = None
        self.error: RuntimeError | None = None
        self.thread = threading.Thread(target=self._read_loop, name="motion-trial-camera", daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        """Читает видеоустройство и перезаписывает кадр вместо создания очереди."""
        try:
            while not self.stop_requested.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Видеоустройство перестало отдавать кадры")
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.ready.set()
        except Exception as error:  # noqa: BLE001 — передаём причину главному циклу.
            with self.lock:
                self.error = error if isinstance(error, RuntimeError) else RuntimeError(str(error))
                self.ready.set()

    def read_new(self, after_sequence: int, timeout: float = 0.05) -> tuple[int, np.ndarray] | None:
        """Возвращает новый кадр или None по тайм-ауту, не дублируя старый."""
        deadline = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.error is not None:
                    raise self.error
                if self.frame is not None and self.sequence > after_sequence:
                    return self.sequence, self.frame.copy()
                self.ready.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.ready.wait(remaining):
                return None

    def close(self) -> None:
        """Останавливает поток чтения и затем освобождает камеру."""
        self.stop_requested.set()
        self.capture.release()
        self.thread.join(timeout=2.0)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Загружает JSON прототипа и проверяет обязательные группы настроек."""
    settings = json.loads(path.read_text(encoding="utf-8"))
    required = {"capture", "roi", "motion", "detector", "display", "diagnostics"}
    if not isinstance(settings, dict) or not required.issubset(settings):
        raise ValueError("В motion_trial.json отсутствует одна или несколько обязательных групп.")
    validate_config(settings)
    return settings


def validate_config(settings: dict[str, Any]) -> None:
    """Проверяет диапазоны параметров до открытия камеры или загрузки модели."""
    def number(section: str, key: str, minimum: float, maximum: float, integer: bool = False) -> float:
        """Проверяет числовое поле и возвращает его как число для связанных ограничений."""
        value = settings[section][key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} должно быть числом")
        numeric = float(value)
        if not minimum <= numeric <= maximum or (integer and not numeric.is_integer()):
            kind = "целым числом" if integer else "числом"
            raise ValueError(f"{section}.{key} должно быть {kind} от {minimum:g} до {maximum:g}")
        return numeric

    capture = settings["capture"]
    if not isinstance(capture["camera_device"], str) or not capture["camera_device"]:
        raise ValueError("capture.camera_device должен быть непустой строкой")
    if capture["pixel_format"] not in {"MJPG", "YUYV"}:
        raise ValueError("capture.pixel_format: доступны MJPG и YUYV")
    number("capture", "width", 320, 7680, integer=True)
    number("capture", "height", 240, 4320, integer=True)
    number("capture", "fps", 1, 120)
    number("capture", "buffer_size", 1, 16, integer=True)

    for key in ("width_percent", "height_percent"):
        number("roi", key, 1, 100)
    for key in ("center_x_percent", "center_y_percent"):
        number("roi", key, 0, 100)

    integer_ranges = {
        "max_features": (8, 4000),
        "min_background_features": (4, 1000),
        "min_background_inliers": (4, 1000),
        "min_feature_distance": (1, 200),
        "luma_threshold": (1, 255),
        "chroma_threshold": (1, 255),
        "min_blob_area": (1, 100000),
        "morphology_radius": (0, 15),
        "specular_value_min": (0, 255),
        "specular_saturation_max": (0, 255),
        "persistence_frames": (1, 30),
        "max_candidates": (1, 1000),
        "opencv_threads": (1, 64),
    }
    for key, limits in integer_ranges.items():
        number("motion", key, *limits, integer=True)
    float_ranges = {
        "feature_quality": (0.0001, 1.0),
        "ransac_threshold_px": (0.1, 50),
        "max_background_shift_px": (1, 2000),
        "max_blob_area_percent": (0.1, 100),
        "track_iou_percent": (0, 100),
        "min_local_color_contrast": (0, 300),
    }
    for key, limits in float_ranges.items():
        number("motion", key, *limits)
    if settings["motion"]["min_background_features"] > settings["motion"]["max_features"]:
        raise ValueError("motion.min_background_features не может быть больше motion.max_features")
    if settings["motion"]["min_background_inliers"] > settings["motion"]["min_background_features"]:
        raise ValueError("motion.min_background_inliers не может быть больше числа точек фона")
    if not isinstance(settings["motion"]["suppress_specular"], bool):
        raise ValueError("motion.suppress_specular должно быть true или false")

    detector = settings["detector"]
    if not isinstance(detector["enabled"], bool):
        raise ValueError("detector.enabled должно быть true или false")
    if not isinstance(detector["model_path"], str) or not detector["model_path"]:
        raise ValueError("detector.model_path должен быть непустой строкой")
    number("detector", "confidence_percent", 0, 100)
    number("detector", "inference_size", 32, 2048, integer=True)
    number("detector", "threads", 0, 64, integer=True)
    number("detector", "max_detection_age_ms", 50, 10000, integer=True)

    display = settings["display"]
    if display["output_device"] not in {"monitor", "none"}:
        raise ValueError("display.output_device должен быть monitor или none")
    if not isinstance(display["show_motion_mask"], bool) or not isinstance(display["show_roi"], bool):
        raise ValueError("Параметры display.show_motion_mask и display.show_roi должны быть true/false")
    number("display", "window_width", 320, 7680, integer=True)
    number("display", "window_height", 240, 4320, integer=True)
    number("display", "mask_opacity_percent", 0, 100)
    for key in ("roi_thickness", "motion_box_thickness", "detector_box_thickness"):
        number("display", key, 1, 10, integer=True)
    number("display", "font_scale", 0.1, 5)
    number("display", "status_height", 20, 200, integer=True)
    for key in (
        "motion_color_rgb", "detector_color_rgb", "roi_color_rgb", "mask_color_rgb",
        "status_background_rgb", "status_text_rgb",
    ):
        color = display[key]
        if not isinstance(color, list) or len(color) != 3 or any(
            isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255
            for value in color
        ):
            raise ValueError(f"display.{key} должен содержать три целых RGB-значения 0..255")

    if not isinstance(settings["diagnostics"]["log_dir"], str) or not settings["diagnostics"]["log_dir"]:
        raise ValueError("diagnostics.log_dir должен быть непустой строкой")
    number("diagnostics", "interval_ms", 100, 60000, integer=True)


def configure_logging(settings: dict[str, Any]) -> Path:
    """Создаёт отдельный журнал прототипа и одновременно выводит сообщения в консоль."""
    log_dir = PROJECT_DIR / str(settings["diagnostics"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"motion_trial_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def resolve_camera(device: str) -> str | int:
    """Разрешает явный путь камеры либо однозначно выбирает внешний USB видеовход."""
    def as_capture_source(value: str) -> str | int:
        """Преобразует путь V4L2 в номер устройства для сборок OpenCV без захвата по имени."""
        if value.isdigit():
            return int(value)
        candidate = Path(value)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return value
        match = re.fullmatch(r"video(\d+)", resolved.name)
        if resolved.parent == Path("/dev") and match:
            return int(match.group(1))
        return value

    if device != "auto":
        return as_capture_source(device)

    if platform.system() == "Windows":
        for index in range(10):
            probe = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            is_open = probe.isOpened()
            probe.release()
            if is_open:
                return index
        raise RuntimeError("Не найдена камера. Выбери её номер в config/motion_trial.json.")

    aliases = sorted((Path("/dev/v4l/by-id")).glob("*video-index0"))
    external = [
        alias for alias in aliases
        if not any(word in alias.name.lower() for word in ("integrated", "internal", "built-in"))
    ]
    if len(external) == 1:
        return as_capture_source(str(external[0]))
    if not external:
        raise RuntimeError(
            "Автовыбор не нашёл внешний USB-видеозахват. Выбери camera_device в меню настроек."
        )
    names = ", ".join(alias.name for alias in external)
    raise RuntimeError(f"Найдено несколько внешних видеоустройств ({names}); выбери одно в меню.")


def open_camera(settings: dict[str, Any]) -> tuple[cv2.VideoCapture, str | int]:
    """Открывает выбранную камеру и запрашивает формат, размер, FPS и короткий буфер."""
    capture_settings = settings["capture"]
    source = resolve_camera(str(capture_settings["camera_device"]))
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть видеоустройство: {source}")
    try:
        pixel_format = str(capture_settings["pixel_format"])
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(capture_settings["width"]))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(capture_settings["height"]))
        capture.set(cv2.CAP_PROP_FPS, float(capture_settings["fps"]))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, int(capture_settings["buffer_size"]))
    except Exception:
        capture.release()
        raise
    return capture, source


def make_detector(settings: dict[str, Any]) -> tuple[OnnxDetector | None, list[float], threading.Lock]:
    """Создаёт YOLO-детектор, если он включён, и выбирает число потоков CPU."""
    detector_settings = settings["detector"]
    confidence_state = [0.0]
    confidence_lock = threading.Lock()
    if not bool(detector_settings["enabled"]):
        return None, confidence_state, confidence_lock

    model_path = (PROJECT_DIR / str(detector_settings["model_path"])).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    requested_threads = int(detector_settings["threads"])
    cpu_threads = max(1, os.cpu_count() or 1) if requested_threads == 0 else requested_threads
    detector = OnnxDetector(
        model_path,
        confidence=float(detector_settings["confidence_percent"]) / 100,
        generic_label=False,
        size=int(detector_settings["inference_size"]),
        threads=cpu_threads,
    )
    shape = detector.session.get_inputs()[0].shape
    fixed_size = [dimension for dimension in shape[-2:] if isinstance(dimension, int)]
    if len(fixed_size) == 2 and fixed_size[0] != fixed_size[1]:
        raise ValueError(f"У модели прямоугольный вход {fixed_size}; этот прототип пока ожидает квадратный.")
    if fixed_size and any(dimension != int(detector_settings["inference_size"]) for dimension in fixed_size):
        raise ValueError(
            f"Размер модели {fixed_size} не совпадает с detector.inference_size="
            f"{detector_settings['inference_size']}. Исправь значение через меню."
        )
    logging.info("Модель YOLO загружена; потоки ONNX Runtime: %s", cpu_threads)
    return detector, confidence_state, confidence_lock


def _rgb_to_bgr(rgb: list[int]) -> tuple[int, int, int]:
    """Преобразует заданный в JSON цвет RGB в порядок каналов OpenCV."""
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _draw_text(image: np.ndarray, text: str, point: tuple[int, int], color: tuple[int, int, int], scale: float) -> None:
    """Рисует читаемую латинскую подпись с тонкой тёмной подложкой."""
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    result: MotionResult,
    snapshot: InferenceSnapshot,
    settings: dict[str, Any],
    system_status: str,
    best_confidence: float,
) -> np.ndarray:
    """Рисует ROI, кандидатов движения и отдельно свежие рамки YOLO."""
    display = settings["display"]
    detector_settings = settings["detector"]
    image = frame.copy()
    roi_x, roi_y, roi_width, roi_height = result.roi
    motion_color = _rgb_to_bgr(display["motion_color_rgb"])
    detector_color = _rgb_to_bgr(display["detector_color_rgb"])
    roi_color = _rgb_to_bgr(display["roi_color_rgb"])
    font_scale = float(display["font_scale"])

    if bool(display["show_roi"]):
        cv2.rectangle(
            image,
            (roi_x, roi_y),
            (roi_x + roi_width - 1, roi_y + roi_height - 1),
            roi_color,
            int(display["roi_thickness"]),
        )

    for box in result.boxes:
        left, top = roi_x + box.x1, roi_y + box.y1
        right, bottom = roi_x + box.x2, roi_y + box.y2
        cv2.rectangle(image, (left, top), (right, bottom), motion_color, int(display["motion_box_thickness"]))
        _draw_text(image, f"MOVE? {box.changed_pixels}px", (left, max(18, top - 5)), motion_color, font_scale)

    detection_age_ms = (time.monotonic() - snapshot.completed_at) * 1000 if snapshot.completed_at else float("inf")
    show_detections = detection_age_ms <= int(detector_settings["max_detection_age_ms"])
    if show_detections:
        for detection in snapshot.detections:
            left, top = roi_x + detection.x1, roi_y + detection.y1
            right, bottom = roi_x + detection.x2, roi_y + detection.y2
            cv2.rectangle(image, (left, top), (right, bottom), detector_color, int(display["detector_box_thickness"]))
            _draw_text(
                image,
                f"YOLO {detection.confidence * 100:.0f}%",
                (left, max(18, top - 5)),
                detector_color,
                font_scale,
            )

    status_height = int(display["status_height"])
    status_top = max(0, image.shape[0] - status_height)
    image[status_top:, :] = _rgb_to_bgr(display["status_background_rgb"])
    confidence_threshold = int(detector_settings["confidence_percent"])
    yolo_status = (
        f"YOLO {best_confidence * 100:.0f}% / {confidence_threshold}%"
        if bool(detector_settings["enabled"])
        else "YOLO OFF"
    )
    metrics = f"MOVE {len(result.boxes)} | {yolo_status} | {system_status}"
    _draw_text(image, metrics, (12, min(image.shape[0] - 8, status_top + 24)), _rgb_to_bgr(display["status_text_rgb"]), font_scale)
    shift_text = "BG=OK" if result.compensated else "BG=RAW"
    if result.background_shift is not None:
        shift_text += f" dx={result.background_shift[0]:+.1f} dy={result.background_shift[1]:+.1f}"
    shift_text += f" inliers={result.background_inliers}"
    _draw_text(
        image,
        shift_text,
        (12, min(image.shape[0] - 3, status_top + 48)),
        _rgb_to_bgr(display["status_text_rgb"]),
        max(0.35, font_scale * 0.75),
    )

    if bool(display["show_motion_mask"]):
        roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
        mask = result.mask > 0
        tint = np.empty_like(roi)
        tint[:] = _rgb_to_bgr(display["mask_color_rgb"])
        opacity = max(0.0, min(1.0, float(display["mask_opacity_percent"]) / 100))
        tinted = cv2.addWeighted(roi, 1 - opacity, tint, opacity, 0)
        roi[mask] = tinted[mask]
        max_width = min(360, image.shape[1] // 3)
        max_height = min(270, image.shape[0] // 3)
        scale = min(max_width / roi_width, max_height / roi_height)
        inset = cv2.resize(roi, (max(1, round(roi_width * scale)), max(1, round(roi_height * scale))))
        inset_height, inset_width = inset.shape[:2]
        destination_x = image.shape[1] - inset_width - 12
        destination_y = max(4, status_top - inset_height - 12)
        image[destination_y : destination_y + inset_height, destination_x : destination_x + inset_width] = inset
        cv2.rectangle(
            image,
            (destination_x, destination_y),
            (destination_x + inset_width - 1, destination_y + inset_height - 1),
            _rgb_to_bgr(display["mask_color_rgb"]),
            1,
        )

    return image


def _format_candidates(result: MotionResult, limit: int = 5) -> str:
    """Готовит короткую диагностику координат и цвето-яркостного контраста."""
    return ";".join(
        f"{box.x1},{box.y1},{box.x2},{box.y2}:area={box.changed_pixels}:"
        f"dL={box.mean_luma_delta:.1f}:dC={box.mean_chroma_delta:.1f}:"
        f"local={box.local_color_contrast:.1f}:n={box.frames_seen}"
        for box in result.boxes[:limit]
    ) or "нет"


def run(settings: dict[str, Any]) -> int:
    """Запускает захват, немедленный поиск движения и независимую проверку YOLO."""
    cv2.setNumThreads(int(settings["motion"]["opencv_threads"]))
    capture, source = open_camera(settings)
    camera_reader: LatestCameraFrames | None = None
    inference_worker: LatestInferenceWorker | None = None
    show_window = settings["display"]["output_device"] == "monitor"
    window_created = False
    try:
        detector, confidence_state, confidence_lock = make_detector(settings)
        if detector is not None:
            def predict(crop: np.ndarray) -> tuple[Any, ...]:
                """Выполняет YOLO на центральном кропе и сохраняет сырую уверенность."""
                detections = detector(crop)
                with confidence_lock:
                    confidence_state[0] = detector.last_best_confidence
                return detections

            inference_worker = LatestInferenceWorker(predict)

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc_text = "".join(chr((actual_fourcc >> (8 * index)) & 0xFF) for index in range(4))
        logging.info(
        "Источник=%s | фактически=%sx%s@%.2f | формат=%s | запрошено=%sx%s@%s %s",
        f"/dev/video{source}" if platform.system() != "Windows" and isinstance(source, int) else source,
            actual_width,
            actual_height,
            actual_fps,
            fourcc_text,
            settings["capture"]["width"],
            settings["capture"]["height"],
            settings["capture"]["fps"],
            settings["capture"]["pixel_format"],
        )
        camera_reader = LatestCameraFrames(capture)
        analyzer = MotionAnalyzer(settings["roi"], settings["motion"])
        system_status = SystemStatusMonitor(interval=1.0)
        if show_window:
            window_name = "VideoT16 — motion trial"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(window_name, int(settings["display"]["window_width"]), int(settings["display"]["window_height"]))
            cv2.moveWindow(window_name, 0, 0)

        logging.info("Клавиши в окне: Esc или Q — выход. YOLO=%s", "включена" if detector else "выключена")
        last_sequence = 0
        last_result: MotionResult | None = None
        last_frame: np.ndarray | None = None
        interval_seconds = max(0.1, int(settings["diagnostics"]["interval_ms"]) / 1000)
        last_log_time = time.monotonic()
        interval_frames = 0
        start_time = last_log_time

        while True:
            newest = camera_reader.read_new(last_sequence)
            if newest is not None:
                last_sequence, last_frame = newest
                last_result = analyzer.process(last_frame)
                interval_frames += 1
                if inference_worker is not None:
                    roi_x, roi_y, roi_width, roi_height = last_result.roi
                    inference_worker.submit(
                        last_frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width].copy()
                    )

            if last_frame is not None and last_result is not None and show_window:
                snapshot = inference_worker.latest() if inference_worker is not None else InferenceSnapshot()
                with confidence_lock:
                    best_confidence = confidence_state[0]
                image = draw_frame(last_frame, last_result, snapshot, settings, system_status.text(), best_confidence)
                max_width = int(settings["display"]["window_width"])
                max_height = int(settings["display"]["window_height"])
                display_scale = min(max_width / image.shape[1], max_height / image.shape[0])
                output_size = (max(1, round(image.shape[1] * display_scale)), max(1, round(image.shape[0] * display_scale)))
                interpolation = cv2.INTER_AREA if display_scale < 1 else cv2.INTER_LINEAR
                cv2.imshow(window_name, cv2.resize(image, output_size, interpolation=interpolation))
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break

            now = time.monotonic()
            if now - last_log_time >= interval_seconds and last_result is not None:
                elapsed = max(0.001, now - last_log_time)
                with confidence_lock:
                    best_confidence = confidence_state[0]
                snapshot = inference_worker.latest() if inference_worker is not None else InferenceSnapshot()
                inference_age = (now - snapshot.completed_at) * 1000 if snapshot.completed_at else -1
                shift = last_result.background_shift or (0.0, 0.0)
                logging.info(
                    "кадр=%d fps_loop=%.1f YOLO=%.0f%% boxes=%d age=%.0fms | "
                    "MOVE=%d luma_px=%d chroma_px=%d bg=%s shift=(%.1f,%.1f) inliers=%d | %s | %s",
                    last_sequence,
                    interval_frames / elapsed,
                    best_confidence * 100,
                    len(snapshot.detections),
                    inference_age,
                    len(last_result.boxes),
                    last_result.luma_pixels,
                    last_result.chroma_pixels,
                    "compensated" if last_result.compensated else "raw",
                    shift[0],
                    shift[1],
                    last_result.background_inliers,
                    last_result.status,
                    _format_candidates(last_result),
                )
                last_log_time = now
                interval_frames = 0
            elif last_frame is None:
                time.sleep(0.01)
    except KeyboardInterrupt:
        logging.info("Остановка по Ctrl+C")
    except Exception:
        logging.exception("Ошибка внутри видеоконвейера")
        return 1
    finally:
        if inference_worker is not None:
            inference_worker.close()
        if camera_reader is not None:
            camera_reader.close()
        elif capture is not None:
            capture.release()
        if window_created:
            cv2.destroyAllWindows()
    if "start_time" in locals():
        logging.info("Завершено; длительность %.1f с", time.monotonic() - start_time)
    return 0


def main() -> int:
    """Читает конфигурацию, настраивает журнал и возвращает понятный код ошибки."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        settings = load_config()
        log_path = configure_logging(settings)
        logging.info("Журнал: %s", log_path)
        return run(settings)
    except Exception as error:  # noqa: BLE001 — причина запуска нужна и в файле, и оператору.
        logging.exception("Ошибка живого испытания движения: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
