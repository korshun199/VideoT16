"""Параллельный USB-захват и покадровое распознавание с диагностическими журналами."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import cv2

from detector import Detection, OnnxRuntimeDetector, OpenCVDnnDetector


PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PROGRAM_DIR.parent
LOG_DIR = PROJECT_DIR / "tmp" / "video_cap_buf"


def now_text() -> str:
    """Возвращает местное время с часовым поясом для журнала."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Frame:
    """Хранит исходное изображение и временные отметки одного кадра."""

    number: int
    image: Any
    captured_at: str
    read_started_ns: int
    captured_ns: int

    @property
    def read_ms(self) -> float:
        """Возвращает длительность чтения кадра в миллисекундах."""
        return (self.captured_ns - self.read_started_ns) / 1_000_000


@dataclass(frozen=True)
class DisplayResult:
    """Хранит сигнал присутствия и рамки для того же исходного кадра."""

    number: int
    captured_ns: int
    recognized_ns: int
    inference_ms: float
    presence_candidates: tuple[Detection, ...]
    detections: tuple[Detection, ...]


class FrameBuffer:
    """Хранит исходные кадры FIFO до того, как будущий вывод их использует."""

    def __init__(self, max_bytes: int) -> None:
        """Создаёт пустой буфер исходных кадров с пределом памяти."""
        if max_bytes <= 0:
            raise ValueError("Предел памяти буфера должен быть положительным")
        self.max_bytes = max_bytes
        self.frames: deque[Frame] = deque()
        self.bytes_used = 0
        self._condition = threading.Condition()
        self._finished = False

    def append(self, frame: Frame) -> bool:
        """Добавляет копию исходного кадра и будит ожидающий поток распознавания."""
        with self._condition:
            frame_bytes = int(frame.image.nbytes)
            if frame_bytes <= 0:
                raise ValueError("Получен пустой кадр")
            if self.bytes_used + frame_bytes > self.max_bytes:
                return False
            stored = Frame(
                frame.number,
                frame.image.copy(),
                frame.captured_at,
                frame.read_started_ns,
                frame.captured_ns,
            )
            self.frames.append(stored)
            self.bytes_used += frame_bytes
            self._condition.notify_all()
            return True

    def wait_for(self, number: int, stop_requested: threading.Event) -> Frame | None:
        """Ждёт указанный кадр, не удаляя его из буфера исходных изображений."""
        with self._condition:
            while not stop_requested.is_set():
                for frame in self.frames:
                    if frame.number == number:
                        return frame
                    if frame.number > number:
                        raise RuntimeError(
                            f"Пропуск в буфере кадров: ожидался {number}, найден {frame.number}"
                        )
                if self._finished:
                    return None
                self._condition.wait(timeout=0.1)
            return None

    def take_latest_after(self, number: int) -> tuple[Frame | None, tuple[int, ...]]:
        """В режиме малой задержки берёт самый свежий кадр и возвращает отброшенные номера."""
        with self._condition:
            eligible = [frame for frame in self.frames if frame.number > number]
            if not eligible:
                return None, ()

            latest = eligible[-1]
            skipped_numbers = tuple(
                frame.number for frame in eligible if frame.number < latest.number
            )
            if skipped_numbers:
                skipped = set(skipped_numbers)
                retained: deque[Frame] = deque()
                for frame in self.frames:
                    if frame.number in skipped:
                        self.bytes_used -= int(frame.image.nbytes)
                    else:
                        retained.append(frame)
                self.frames = retained
                self._condition.notify_all()
            return latest, skipped_numbers

    def is_finished(self) -> bool:
        """Сообщает, остановлен ли источник и больше ли не будет новых кадров."""
        with self._condition:
            return self._finished

    def get(self, number: int) -> Frame | None:
        """Возвращает исходный кадр по номеру, не удаляя его из FIFO."""
        with self._condition:
            return next((frame for frame in self.frames if frame.number == number), None)

    def remove(self, number: int) -> Frame | None:
        """Удаляет только указанный кадр и освобождает учтённую память после показа."""
        with self._condition:
            for index, frame in enumerate(self.frames):
                if frame.number == number:
                    del self.frames[index]
                    self.bytes_used -= int(frame.image.nbytes)
                    self._condition.notify_all()
                    return frame
            return None

    def snapshot(self) -> tuple[tuple[Frame, ...], int]:
        """Возвращает снимок исходного FIFO и занятый объём пикселей."""
        with self._condition:
            return tuple(self.frames), self.bytes_used

    def stats(self) -> tuple[int, int]:
        """Возвращает число исходных кадров и занятый объём без копии очереди."""
        with self._condition:
            return len(self.frames), self.bytes_used

    def finish(self) -> None:
        """Сообщает ожидающим потокам, что новых кадров больше не поступит."""
        with self._condition:
            self._finished = True
            self._condition.notify_all()

    def clear(self) -> None:
        """Освобождает исходные изображения только при явном завершении сеанса."""
        with self._condition:
            self.frames.clear()
            self.bytes_used = 0
            self._finished = True
            self._condition.notify_all()


class DisplayBuffer:
    """Хранит компактные результаты распознавания по номеру исходного кадра."""

    def __init__(self, max_items: int) -> None:
        """Создаёт пустой FIFO результатов с ограничением числа кадров."""
        if max_items <= 0:
            raise ValueError("Предел результатов должен быть положительным")
        self.max_items = max_items
        self.results: deque[DisplayResult] = deque()
        self.bytes_estimated = 0
        self._condition = threading.Condition()

    @staticmethod
    def estimate_bytes(result: DisplayResult) -> int:
        """Оценивает размер записи и её объектов Python для диагностического лимита."""
        objects = result.presence_candidates + result.detections
        return 176 + sum(112 + len(item.label.encode("utf-8")) for item in objects)

    def append(self, result: DisplayResult) -> bool:
        """Добавляет результат в FIFO либо сообщает, что число записей исчерпано."""
        with self._condition:
            if len(self.results) >= self.max_items:
                return False
            self.results.append(result)
            self.bytes_estimated += self.estimate_bytes(result)
            self._condition.notify_all()
            return True

    def get(self, number: int) -> DisplayResult | None:
        """Возвращает результат распознавания по номеру, не удаляя его."""
        with self._condition:
            return next((result for result in self.results if result.number == number), None)

    def remove(self, number: int) -> DisplayResult | None:
        """Удаляет только указанный результат после подтверждения показа кадра."""
        with self._condition:
            for index, result in enumerate(self.results):
                if result.number == number:
                    del self.results[index]
                    self.bytes_estimated -= self.estimate_bytes(result)
                    self._condition.notify_all()
                    return result
            return None

    def snapshot(self) -> tuple[tuple[DisplayResult, ...], int]:
        """Возвращает снимок FIFO результатов и оценку занятой памяти."""
        with self._condition:
            return tuple(self.results), self.bytes_estimated

    def latest_after(self, number: int) -> tuple[DisplayResult, ...]:
        """Возвращает результаты новее уже показанного кадра."""
        with self._condition:
            return tuple(result for result in self.results if result.number > number)

    def stats(self) -> tuple[int, int]:
        """Возвращает число результатов и оценку памяти без копии FIFO."""
        with self._condition:
            return len(self.results), self.bytes_estimated

    def clear(self) -> None:
        """Освобождает результаты при явном завершении сеанса."""
        with self._condition:
            self.results.clear()
            self.bytes_estimated = 0
            self._condition.notify_all()


class JsonlLog:
    """Потокобезопасно записывает события одного потока в JSONL-файл."""

    def __init__(self, path: Path) -> None:
        """Создаёт новый журнал и запрещает случайную перезапись прежнего."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("x", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        """Записывает событие с календарным и монотонным временем."""
        record = {
            "event": event,
            "time": now_text(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self._file.write(encoded + "\n")
            self._file.flush()

    def close(self) -> None:
        """Закрывает журнал после остановки записывающего потока."""
        with self._lock:
            self._file.close()


def fourcc_text(value: float) -> str:
    """Преобразует код видеорежима OpenCV в читаемую строку."""
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")


def frame_signal_stats(image: Any) -> dict[str, float | int]:
    """Оценивает яркость редкой сетки пикселей без сохранения видеокадра."""
    height, width = image.shape[:2]
    sample = cv2.resize(
        image,
        (min(64, width), min(48, height)),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    minimum, maximum, _, _ = cv2.minMaxLoc(gray)
    dark_pixels = cv2.countNonZero(cv2.inRange(gray, 0, 8))
    return {
        "sample_mean_gray": round(cv2.mean(gray)[0], 2),
        "sample_min_gray": int(minimum),
        "sample_max_gray": int(maximum),
        "sample_near_black_percent": round(dark_pixels * 100 / gray.size, 2),
    }


def validate_settings(settings: dict[str, Any]) -> None:
    """Проверяет настройки конвейера до запуска камеры или модели."""
    if not isinstance(settings, dict):
        raise ValueError("Настройки должны быть объектом JSON")

    source = settings.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source: укажите путь к USB-видеоустройству")
    for key in ("width", "height", "fps", "buffer_max_mb", "inference_size",
                "recognition_workers", "inference_threads_per_worker",
                "result_buffer_max_items", "box_thickness"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key}: требуется положительное целое число")
    pixel_format = settings.get("pixel_format")
    if not isinstance(pixel_format, str) or len(pixel_format) != 4 or not pixel_format.isascii():
        raise ValueError("pixel_format: требуется четырёхсимвольный код, например YUYV")
    if not isinstance(settings.get("model_path"), str) or not settings["model_path"].strip():
        raise ValueError("model_path: укажите путь к ONNX-модели")
    if settings.get("inference_backend") not in ("onnxruntime", "opencv"):
        raise ValueError("inference_backend: допустимы onnxruntime или opencv")
    confidence_settings = (
        ("confidence_percent", 0, 100),
        ("presence_confidence_percent", 0, 100),
        ("nms_iou_threshold", 0, 1),
        ("font_scale", 0, 10),
    )
    for key, low, high in confidence_settings:
        default = settings.get("confidence_percent") if key == "presence_confidence_percent" else None
        value = settings.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not low <= value <= high
                or (key == "font_scale" and value == 0)):
            raise ValueError(f"{key}: значение вне допустимого диапазона {low}..{high}")
    if settings.get("presence_confidence_percent", settings["confidence_percent"]) > settings[
        "confidence_percent"
    ]:
        raise ValueError("presence_confidence_percent не может быть выше confidence_percent")
    if settings.get("recognition_frame_policy", "all") not in ("all", "latest"):
        raise ValueError("recognition_frame_policy: допустимы all или latest")
    class_names = settings.get("class_names")
    if not isinstance(class_names, list) or not class_names or any(
            not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("class_names: требуется список названий классов модели")
    for key in ("box_color_bgr", "text_color_bgr", "weak_candidate_color_bgr"):
        color = settings.get(
            key, [255, 0, 255] if key == "weak_candidate_color_bgr" else None
        )
        if (not isinstance(color, list) or len(color) != 3
                or any(isinstance(channel, bool) or not isinstance(channel, int)
                       or not 0 <= channel <= 255 for channel in color)):
            raise ValueError(f"{key}: укажите три целых компонента BGR от 0 до 255")


def load_settings(path: Path) -> dict[str, Any]:
    """Читает JSON и применяет общую проверку для GUI и обычного запуска."""
    with path.open("r", encoding="utf-8") as file:
        settings = json.load(file)
    validate_settings(settings)
    return settings


def resolve_model_path(settings: dict[str, Any], settings_path: Path) -> Path:
    """Разрешает путь модели относительно файла настроек, если путь не абсолютный."""
    path = Path(settings["model_path"]).expanduser()
    if not path.is_absolute():
        path = settings_path.resolve().parent / path
    return path.resolve()


class PipelineSession:
    """Связывает параллельные потоки и хранит кадры до подтверждения их показа."""

    def __init__(
        self,
        settings: dict[str, Any],
        capture_log: JsonlLog,
        recognition_log: JsonlLog,
        detectors: list[Any] | Any,
        capture_factory: Callable[..., Any] = cv2.VideoCapture,
        debug_frames: int | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        """Создаёт буферы и сигналы; длительность задаётся только CLI-параметром."""
        self.settings = settings
        self.recognition_frame_policy = settings.get("recognition_frame_policy", "all")
        self.capture_log = capture_log
        self.recognition_log = recognition_log
        # Поддерживаем один распознаватель для старых диагностических вызовов
        # и список экземпляров для параллельной обработки кадров.
        if isinstance(detectors, (list, tuple)):
            self.detectors = tuple(detectors)
        else:
            self.detectors = (detectors,)
        if not self.detectors:
            raise ValueError("Нужен хотя бы один экземпляр распознавателя")
        self.detector = self.detectors[0]
        self.capture_factory = capture_factory
        self.debug_frames = debug_frames
        self.duration_seconds = duration_seconds
        self.frame_buffer = FrameBuffer(settings["buffer_max_mb"] * 1024 * 1024)
        self.display_buffer = DisplayBuffer(settings["result_buffer_max_items"])
        self.stop_requested = threading.Event()
        self.processing_finished = threading.Event()
        self.pair_condition = threading.Condition()
        self.reason = ""
        self.capture_error: str | None = None
        self.recognition_error: str | None = None
        self.display_error: str | None = None
        self.frame_count = 0
        self.recognized_count = 0
        self.displayed_count = 0
        self.presence_frame_count = 0
        self.localized_frame_count = 0
        self.last_presence_frame = 0
        self.last_presence_summary = "модель ещё не выдала результат"
        self.recognition_skipped_count = 0
        self.stale_result_count = 0
        self.last_displayed_number = 0
        self.capture_started_ns: int | None = None
        self.capture_finished_ns: int | None = None
        self.recognition_finished_ns: int | None = None
        self._state_lock = threading.Lock()
        self._dispatch_lock = threading.Lock()
        self._next_frame_number = 1
        self.capture_thread = threading.Thread(target=self._capture, name="video-capture")
        self.recognition_threads = [
            threading.Thread(
                target=self._recognize,
                args=(index, detector),
                name=f"video-recognition-{index + 1}",
            )
            for index, detector in enumerate(self.detectors)
        ]

    def start(self) -> None:
        """Сначала запускает потребителя, затем начинает получать новые кадры."""
        self.recognition_log.write(
            "recognition_started",
            started_ns=time.monotonic_ns(),
            workers=len(self.recognition_threads),
            frame_policy=self.recognition_frame_policy,
        )
        for thread in self.recognition_threads:
            thread.start()
        self.capture_thread.start()

    def request_stop(self, reason: str) -> None:
        """Просит конвейер завершиться и сохраняет первую причину остановки."""
        with self._state_lock:
            if not self.reason:
                self.reason = reason
        self.stop_requested.set()
        self.frame_buffer.finish()
        with self.pair_condition:
            self.pair_condition.notify_all()

    def join(self) -> None:
        """Дожидается захвата и распознавания, затем будит поток отображения."""
        self.capture_thread.join()
        for thread in self.recognition_threads:
            thread.join()
        self.recognition_finished_ns = time.monotonic_ns()
        _, estimated_bytes = self.display_buffer.stats()
        self.recognition_log.write(
            "recognition_finished",
            reason=self.reason,
            workers=len(self.recognition_threads),
            frames_recognized=self.recognized_count,
            frames_with_presence_candidate=self.presence_frame_count,
            frames_with_box=self.localized_frame_count,
            frames_skipped=self.recognition_skipped_count,
            stale_results=self.stale_result_count,
            result_buffer_estimated_bytes=estimated_bytes,
            recognition_finished_ns=self.recognition_finished_ns,
        )
        self.processing_finished.set()
        with self.pair_condition:
            self.pair_condition.notify_all()

    def remove_displayed_pair(
        self, number: int
    ) -> tuple[int, int, int, int] | None:
        """Удаляет показанную пару и устаревшие пары в режиме приоритета свежести."""
        with self._state_lock:
            if number <= self.last_displayed_number:
                return None
            with self.frame_buffer._condition:
                with self.display_buffer._condition:
                    frame_index = next(
                        (index for index, frame in enumerate(self.frame_buffer.frames)
                         if frame.number == number),
                        None,
                    )
                    result_index = next(
                        (index for index, result in enumerate(self.display_buffer.results)
                         if result.number == number),
                        None,
                    )
                    if frame_index is None or result_index is None:
                        return None
                    if self.recognition_frame_policy == "latest":
                        removed_frames = [
                            frame for frame in self.frame_buffer.frames if frame.number <= number
                        ]
                        removed_results = [
                            result for result in self.display_buffer.results
                            if result.number <= number
                        ]
                        self.frame_buffer.frames = deque(
                            frame for frame in self.frame_buffer.frames if frame.number > number
                        )
                        self.display_buffer.results = deque(
                            result for result in self.display_buffer.results
                            if result.number > number
                        )
                    else:
                        removed_frames = [self.frame_buffer.frames[frame_index]]
                        removed_results = [self.display_buffer.results[result_index]]
                        del self.frame_buffer.frames[frame_index]
                        del self.display_buffer.results[result_index]
                    self.frame_buffer.bytes_used -= sum(
                        int(frame.image.nbytes) for frame in removed_frames
                    )
                    self.display_buffer.bytes_estimated -= sum(
                        self.display_buffer.estimate_bytes(result) for result in removed_results
                    )
                    self.last_displayed_number = number
                    self.frame_buffer._condition.notify_all()
                    self.display_buffer._condition.notify_all()
                    return (
                        len(self.frame_buffer.frames),
                        self.frame_buffer.bytes_used,
                        len(self.display_buffer.results),
                        self.display_buffer.bytes_estimated,
                    )

    def _capture(self) -> None:
        """Читает USB-кадры, ставит им номера и сохраняет их в первом FIFO."""
        device = None
        local_reason = ""
        signal_interval = 1
        last_signal_frame = 0
        last_signal_image = None
        try:
            source = str(Path(self.settings["source"]).resolve())
            device = self.capture_factory(source, cv2.CAP_V4L2)
            if not device.isOpened():
                raise RuntimeError(f"Не удалось открыть источник видео: {self.settings['source']}")

            pixel_format = self.settings["pixel_format"]
            requested = (
                ("pixel_format", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format)),
                ("width", cv2.CAP_PROP_FRAME_WIDTH, self.settings["width"]),
                ("height", cv2.CAP_PROP_FRAME_HEIGHT, self.settings["height"]),
                ("fps", cv2.CAP_PROP_FPS, self.settings["fps"]),
            )
            accepted = {name: bool(device.set(prop, value)) for name, prop, value in requested}
            actual = {
                "width": device.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": device.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "fps": device.get(cv2.CAP_PROP_FPS),
                "pixel_format": fourcc_text(device.get(cv2.CAP_PROP_FOURCC)),
            }
            measured_fps = actual["fps"] if actual["fps"] > 0 else self.settings["fps"]
            signal_interval = max(1, int(round(measured_fps)))
            self.capture_log.write("capture_started", requested=self.settings,
                                   accepted=accepted, actual=actual,
                                   debug_frames=self.debug_frames,
                                   duration_seconds=self.duration_seconds)
            print(f"Захват: {actual['width']:.0f}×{actual['height']:.0f}, "
                  f"{actual['fps']:.1f} FPS, {actual['pixel_format']}", flush=True)

            number = 0
            deadline_ns: int | None = None
            while not self.stop_requested.is_set():
                read_started_ns = time.monotonic_ns()
                if self.capture_started_ns is None:
                    self.capture_started_ns = read_started_ns
                    if self.duration_seconds is not None:
                        deadline_ns = read_started_ns + int(
                            self.duration_seconds * 1_000_000_000
                        )
                elif self.debug_frames is None and deadline_ns is not None \
                        and read_started_ns >= deadline_ns:
                    local_reason = "duration"
                    break

                ok, image = device.read()
                captured_ns = time.monotonic_ns()
                if not ok or image is None:
                    raise RuntimeError("Оцифровщик не вернул кадр; проверьте входной видеосигнал")
                number += 1
                captured_at = now_text()
                frame = Frame(number, image, captured_at, read_started_ns, captured_ns)
                if not self.frame_buffer.append(frame):
                    local_reason = "capture_buffer_full"
                    frame_count, bytes_used = self.frame_buffer.stats()
                    self.capture_log.write("capture_buffer_full", rejected_frame=number,
                                           buffer_frames=frame_count, buffer_bytes=bytes_used,
                                           frame_bytes=int(image.nbytes))
                    break
                self.frame_count = number
                last_signal_image = image
                with self.pair_condition:
                    self.pair_condition.notify_all()
                frame_count, bytes_used = self.frame_buffer.stats()
                self.capture_log.write(
                    "frame_captured",
                    frame=frame.number,
                    captured_at=captured_at,
                    read_started_ns=read_started_ns,
                    captured_ns=captured_ns,
                    read_ms=round((captured_ns - read_started_ns) / 1_000_000, 3),
                    buffer_frames=frame_count,
                    buffer_bytes=bytes_used,
                    frame_bytes=int(image.nbytes),
                )
                if number == 1 or number % signal_interval == 0:
                    self.capture_log.write(
                        "capture_signal_sample",
                        frame=number,
                        **frame_signal_stats(image),
                    )
                    last_signal_frame = number
                if self.debug_frames is not None and number >= self.debug_frames:
                    local_reason = "debug_frames"
                    break
            if self.stop_requested.is_set():
                local_reason = self.reason or "stop_requested"
        except Exception as error:
            self.capture_error = str(error)
            local_reason = "capture_error"
            try:
                self.capture_log.write("capture_error", message=self.capture_error)
            except OSError:
                pass
        finally:
            if device is not None:
                device.release()
            if (self.frame_count and last_signal_image is not None
                    and last_signal_frame != self.frame_count):
                try:
                    self.capture_log.write(
                        "capture_signal_sample",
                        frame=self.frame_count,
                        final=True,
                        **frame_signal_stats(last_signal_image),
                    )
                except OSError:
                    pass
            self.capture_finished_ns = time.monotonic_ns()
            with self._state_lock:
                if not self.reason:
                    self.reason = local_reason or "capture_stopped"
            self.frame_buffer.finish()
            with self.pair_condition:
                self.pair_condition.notify_all()
            try:
                frame_count, bytes_used = self.frame_buffer.stats()
                self.capture_log.write(
                    "capture_finished",
                    reason=self.reason,
                    frames_captured=self.frame_count,
                    buffer_frames=frame_count,
                    buffer_bytes=bytes_used,
                    capture_started_ns=self.capture_started_ns,
                    capture_finished_ns=self.capture_finished_ns,
                )
            except OSError:
                pass
            if self.reason == "duration":
                print(
                    f"\nЗахват {self.duration_seconds:g} с завершён: "
                    f"{self.frame_count} кадров. Дожидаюсь распознавания и показа…",
                    flush=True,
                )

    def _next_frame_for_recognition(self) -> Frame | None:
        """Выдаёт FIFO-кадр либо самый новый кадр, если важнее свежесть ответа."""
        if self.recognition_frame_policy == "all":
            with self._dispatch_lock:
                number = self._next_frame_number
                self._next_frame_number += 1
            return self.frame_buffer.wait_for(number, self.stop_requested)

        while not self.stop_requested.is_set():
            with self._dispatch_lock:
                last_dispatched = self._next_frame_number - 1
                frame, skipped_numbers = self.frame_buffer.take_latest_after(last_dispatched)
                if frame is not None:
                    self._next_frame_number = frame.number + 1
                    self.recognition_skipped_count += len(skipped_numbers)
                finished = self.frame_buffer.is_finished()

            if skipped_numbers:
                self.recognition_log.write(
                    "recognition_frames_skipped",
                    after_frame=last_dispatched,
                    before_frame=frame.number if frame is not None else None,
                    frames=list(skipped_numbers),
                    count=len(skipped_numbers),
                    reason="newer_frame_available",
                )
            if frame is not None:
                return frame
            if finished:
                return None
            with self.pair_condition:
                self.pair_condition.wait(timeout=0.02)
        return None

    def _recognize(self, worker_index: int, detector: Any) -> None:
        """Берёт уникальные кадры из общего FIFO и журналирует результат своей модели."""
        worker_number = worker_index + 1
        worker_started_ns = time.monotonic_ns()
        frames_recognized = 0
        self.recognition_log.write(
            "recognition_worker_started",
            worker=worker_number,
            thread=threading.current_thread().name,
            started_ns=worker_started_ns,
        )
        try:
            while not self.stop_requested.is_set():
                frame = self._next_frame_for_recognition()
                if frame is None:
                    break
                inference_started_ns = time.monotonic_ns()
                detect_with_presence = getattr(detector, "detect_with_presence", None)
                if callable(detect_with_presence):
                    presence_candidates, detections, inference_ms = detect_with_presence(frame.image)
                else:
                    detections, inference_ms = detector.detect(frame.image)
                    presence_candidates = detections
                recognized_ns = time.monotonic_ns()
                result = DisplayResult(
                    number=frame.number,
                    captured_ns=frame.captured_ns,
                    recognized_ns=recognized_ns,
                    inference_ms=inference_ms,
                    presence_candidates=presence_candidates,
                    detections=detections,
                )
                with self._state_lock:
                    stale = frame.number <= self.last_displayed_number
                    if stale:
                        self.stale_result_count += 1
                    elif not self.display_buffer.append(result):
                        raise RuntimeError(
                            "Буфер результатов заполнен; исходный кадр оставлен в буфере захвата"
                        )
                    else:
                        self.recognized_count += 1
                        if presence_candidates:
                            self.presence_frame_count += 1
                        if detections:
                            self.localized_frame_count += 1
                        if frame.number >= self.last_presence_frame:
                            max_presence = max(
                                (item.confidence for item in presence_candidates), default=0.0
                            )
                            if detections:
                                max_box = max(item.confidence for item in detections)
                                self.last_presence_summary = (
                                    f"локализован объект {max_box * 100:.0f}%"
                                )
                            elif presence_candidates:
                                self.last_presence_summary = (
                                    f"кандидат присутствия {max_presence * 100:.0f}%, "
                                    "рамка ниже порога"
                                )
                            else:
                                self.last_presence_summary = "модель не видит кандидата"
                            self.last_presence_frame = frame.number
                if stale:
                    self.recognition_log.write(
                        "recognition_result_stale",
                        frame=frame.number,
                        last_displayed_frame=self.last_displayed_number,
                        capture_to_result_ms=round(
                            (recognized_ns - frame.captured_ns) / 1_000_000, 3
                        ),
                    )
                    del frame, result, detections, presence_candidates
                    continue
                with self.pair_condition:
                    self.pair_condition.notify_all()
                result_count, estimated_bytes = self.display_buffer.stats()
                frame_count, frame_bytes = self.frame_buffer.stats()
                self.recognition_log.write(
                    "frame_recognized",
                    frame=frame.number,
                    worker=worker_number,
                    capture_to_result_ms=round(
                        (recognized_ns - frame.captured_ns) / 1_000_000, 3
                    ),
                    queue_wait_ms=round(
                        (inference_started_ns - frame.captured_ns) / 1_000_000, 3
                    ),
                    inference_ms=round(inference_ms, 3),
                    model_presence_candidate=bool(presence_candidates),
                    presence_max_confidence=round(
                        max((item.confidence for item in presence_candidates), default=0.0), 6
                    ),
                    presence_candidates=[
                        {
                            "class_id": item.class_id,
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                            "box": [item.left, item.top, item.right, item.bottom],
                        }
                        for item in presence_candidates
                    ],
                    detections=[
                        {
                            "class_id": item.class_id,
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                            "box": [item.left, item.top, item.right, item.bottom],
                        }
                        for item in detections
                    ],
                    result_buffer_items=result_count,
                    result_buffer_estimated_bytes=estimated_bytes,
                    source_buffer_frames=frame_count,
                    source_buffer_bytes=frame_bytes,
                )
                frames_recognized += 1
                # Не удерживаем ссылку рабочего потока на полный кадр,
                # когда поток отображения уже может удалить его из FIFO.
                del frame, result, detections, presence_candidates
        except Exception as error:
            with self._state_lock:
                if self.recognition_error is None:
                    self.recognition_error = str(error)
            self.recognition_log.write(
                "recognition_error",
                worker=worker_number,
                message=str(error),
            )
            self.request_stop("recognition_error")
        finally:
            self.recognition_log.write(
                "recognition_worker_finished",
                worker=worker_number,
                frames_recognized=frames_recognized,
                worker_started_ns=worker_started_ns,
                worker_finished_ns=time.monotonic_ns(),
            )

    def timing_summary(self) -> dict[str, float | int | None]:
        """Считает длительность захвата, дренирования и готовности всех результатов."""
        capture_started = self.capture_started_ns
        capture_finished = self.capture_finished_ns
        recognition_finished = self.recognition_finished_ns
        capture_ms = None
        drain_ms = None
        total_ms = None
        if capture_started is not None and capture_finished is not None:
            capture_ms = (capture_finished - capture_started) / 1_000_000
        if capture_finished is not None and recognition_finished is not None:
            drain_ms = (recognition_finished - capture_finished) / 1_000_000
        if capture_started is not None and recognition_finished is not None:
            total_ms = (recognition_finished - capture_started) / 1_000_000
        result_count, result_bytes = self.display_buffer.stats()
        frame_count, frame_bytes = self.frame_buffer.stats()
        return {
            "frames_captured": self.frame_count,
            "frames_recognized": self.recognized_count,
            "frames_displayed": self.displayed_count,
            "frames_with_presence_candidate": self.presence_frame_count,
            "frames_with_box": self.localized_frame_count,
            "frames_skipped": self.recognition_skipped_count,
            "stale_results": self.stale_result_count,
            "capture_ms": capture_ms,
            "post_capture_drain_ms": drain_ms,
            "capture_to_results_ready_ms": total_ms,
            "source_buffer_frames": frame_count,
            "source_buffer_bytes": frame_bytes,
            "result_buffer_items": result_count,
            "result_buffer_estimated_bytes": result_bytes,
        }

    def close(self) -> None:
        """Останавливает потоки и очищает оставшиеся данные при выходе из программы."""
        self.request_stop("session_closed")
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        for thread in self.recognition_threads:
            if thread.is_alive():
                thread.join()
        self.frame_buffer.clear()
        self.display_buffer.clear()


def _format_ms(value: float | int | None) -> str:
    """Форматирует измерение, помечая отсутствие кадров как «нет данных»."""
    return "нет данных" if value is None else f"{value / 1000:.3f} с"


def _make_detectors(settings: dict[str, Any], settings_path: Path) -> list[Any]:
    """Создаёт нужное число независимых распознавателей из единого JSON-файла."""
    backend = settings["inference_backend"]
    detector_type = OnnxRuntimeDetector if backend == "onnxruntime" else OpenCVDnnDetector
    model_path = resolve_model_path(settings, settings_path)
    detectors: list[Any] = []
    for _ in range(settings["recognition_workers"]):
        detectors.append(detector_type(
            model_path=model_path,
            input_size=settings["inference_size"],
            confidence_percent=settings["confidence_percent"],
            nms_iou=settings["nms_iou_threshold"],
            class_names=settings["class_names"],
            box_color_bgr=tuple(settings["box_color_bgr"]),
            text_color_bgr=tuple(settings["text_color_bgr"]),
            weak_candidate_color_bgr=tuple(
                settings.get("weak_candidate_color_bgr", [255, 0, 255])
            ),
            box_thickness=settings["box_thickness"],
            font_scale=settings["font_scale"],
            threads=settings["inference_threads_per_worker"],
            presence_confidence_percent=settings.get(
                "presence_confidence_percent", settings["confidence_percent"]
            ),
        ))
    return detectors


def main(argv: list[str] | None = None) -> int:
    """Запускает три параллельных контура и показывает кадры в оконном режиме."""
    parser = argparse.ArgumentParser(description="VideoT16: захват и параллельное распознавание")
    parser.add_argument("--settings", type=Path, default=PROGRAM_DIR / "settings.json",
                        help="путь к JSON-настройкам")
    parser.add_argument("--debug-frames", type=int, metavar="N",
                        help="отладка: захватить N кадров вместо ограничения по времени")
    parser.add_argument("--duration", type=float, metavar="СЕК",
                        help="длительность этого запуска в секундах; настройки файла не меняются")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="число распознавателей только для этого запуска; JSON не меняется")
    args = parser.parse_args(argv)
    if args.debug_frames is not None and args.debug_frames <= 0:
        parser.error("--debug-frames: количество кадров должно быть больше нуля")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration: укажите положительное число секунд")
    if args.workers is not None and args.workers <= 0:
        parser.error("--workers: укажите положительное целое число")

    capture_log: JsonlLog | None = None
    recognition_log: JsonlLog | None = None
    display_log: JsonlLog | None = None
    try:
        settings = load_settings(args.settings)
        if args.workers is not None:
            settings["recognition_workers"] = args.workers
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        capture_log = JsonlLog(LOG_DIR / f"capture_{run_id}.jsonl")
        recognition_log = JsonlLog(LOG_DIR / f"recognition_{run_id}.jsonl")
        display_log = JsonlLog(LOG_DIR / f"display_{run_id}.jsonl")
        model_path = resolve_model_path(settings, args.settings)
        load_started_ns = time.monotonic_ns()
        recognition_log.write("model_loading", path=str(model_path))
        detectors = _make_detectors(settings, args.settings)
        model_load_ms = (time.monotonic_ns() - load_started_ns) / 1_000_000
        recognition_log.write(
            "model_ready",
            model_load_ms=round(model_load_ms, 3),
            backend=settings["inference_backend"],
            workers=len(detectors),
            inference_threads_per_worker=settings["inference_threads_per_worker"],
            providers=getattr(detectors[0], "providers", ["OpenCV CPU"]),
            input_size=settings["inference_size"],
            classes=settings["class_names"],
        )
    except Exception as error:
        print(f"Ошибка подготовки: {error}", file=sys.stderr)
        if recognition_log is not None:
            recognition_log.write("preparation_error", message=str(error))
            recognition_log.close()
        if capture_log is not None:
            capture_log.close()
        if display_log is not None:
            display_log.close()
        return 1

    session = PipelineSession(
        settings,
        capture_log,
        recognition_log,
        detectors,
        debug_frames=args.debug_frames,
        duration_seconds=args.duration,
    )
    print(f"Журнал захвата: {capture_log.path}", flush=True)
    print(f"Журнал распознавания: {recognition_log.path}", flush=True)
    print(f"Журнал отображения: {display_log.path}", flush=True)
    print(
        f"Модель загружена за {model_load_ms:.1f} мс: {model_path}; "
        f"движок={settings['inference_backend']}, "
        f"распознавателей={len(detectors)}, "
        f"потоков на распознаватель={settings['inference_threads_per_worker']}",
        flush=True,
    )
    print(
        f"Режим: {settings.get('recognition_frame_policy', 'all')}; "
        f"кандидат присутствия от "
        f"{settings.get('presence_confidence_percent', settings['confidence_percent']):g}%, "
        f"рамка от {settings['confidence_percent']:g}%.",
        flush=True,
    )
    if args.duration is None:
        print("Длительность захвата: без ограничения (остановка Esc или Ctrl+C)", flush=True)
    else:
        print(
            f"Длительность захвата: {args.duration:g} с "
            "(разовый параметр; settings.json не изменён)",
            flush=True,
        )
    if args.workers is not None:
        print(
            f"Число распознавателей: {settings['recognition_workers']} "
            "(разовый параметр; settings.json не изменён)",
            flush=True,
        )
    exit_code = 0
    try:
        from display import run_display

        exit_code = run_display(session, display_log)
        summary = session.timing_summary()
        print(f"Захвачено: {summary['frames_captured']}; "
              f"проверено моделью: {summary['frames_recognized']}; "
              f"кандидат присутствия: {summary['frames_with_presence_candidate']}; "
              f"рамка: {summary['frames_with_box']}; "
              f"пропущено ради свежести: {summary['frames_skipped']}; "
              f"показано: {summary['frames_displayed']}; "
              f"устаревших результатов отброшено: {summary['stale_results']}", flush=True)
        print(f"Время захвата: {_format_ms(summary['capture_ms'])}; "
              f"после захвата до готовности результатов: "
              f"{_format_ms(summary['post_capture_drain_ms'])}; "
              f"всего до готовности: {_format_ms(summary['capture_to_results_ready_ms'])}",
              flush=True)
        print(f"Перед очисткой осталось в буферах: кадров={summary['source_buffer_frames']} "
              f"({summary['source_buffer_bytes'] / 1048576:.1f} МиБ), "
              f"результатов={summary['result_buffer_items']} "
              f"({summary['result_buffer_estimated_bytes']} байт, оценка).", flush=True)
        if session.capture_error:
            print(f"Ошибка захвата: {session.capture_error}", file=sys.stderr)
            capture_log.write("session_failed", stage="capture", message=session.capture_error)
            exit_code = 1
        if session.recognition_error:
            print(f"Ошибка распознавания: {session.recognition_error}", file=sys.stderr)
            exit_code = 1
        if session.display_error:
            print(f"Ошибка отображения: {session.display_error}", file=sys.stderr)
            exit_code = 1
        capture_log.write("session_finished", **summary, reason=session.reason)
    except KeyboardInterrupt:
        print("\nОстановка по Ctrl+C; дожидаюсь потоков и освобождаю оставшиеся кадры.", flush=True)
        exit_code = 130
    except Exception as error:
        print(f"Ошибка запуска отображения: {error}", file=sys.stderr)
        display_log.write("display_start_error", message=str(error))
        exit_code = 1
    finally:
        session.close()
        try:
            capture_log.write("session_closed", source_frames=0, result_items=0)
            recognition_log.write("session_closed", source_frames=0, result_items=0)
            display_log.write("session_closed", source_frames=0, result_items=0)
        finally:
            capture_log.close()
            recognition_log.close()
            display_log.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
