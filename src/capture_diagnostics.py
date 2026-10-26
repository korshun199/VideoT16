"""Лёгкие измерения фактического чтения кадров без записи видео на диск.

Модуль не открывает устройство сам и не меняет его режим. Его используют
живой конвейер V2 и отдельная утилита проверки формата оцифровщика.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import zlib
from typing import Any


@dataclass(frozen=True)
class CaptureTelemetry:
    """Компактный снимок недавних измерений захвата."""

    samples: int
    observed_fps: float
    read_p50_ms: float
    read_p90_ms: float
    interval_p50_ms: float
    interval_p90_ms: float
    coarse_same_percent: float


def coarse_frame_signature(frame: Any) -> int:
    """Возвращает дешёвую сигнатуру редкой сетки кадра для поиска повторов.

    Это не проверка идентичности всего изображения: маленький объект может не
    попасть в сетку. Показатель нужен только как сигнал о возможном повторении
    кадров при номинальных 60 FPS.
    """
    shape = getattr(frame, "shape", ())
    if len(shape) < 2:
        raise ValueError("Для сигнатуры нужен кадр с высотой и шириной")
    height, width = int(shape[0]), int(shape[1])
    if height < 1 or width < 1:
        raise ValueError("Размер кадра для сигнатуры должен быть положительным")
    step_y = max(1, height // 12)
    step_x = max(1, width // 16)
    sampled = frame[::step_y, ::step_x]
    return zlib.crc32(sampled.tobytes())


class RollingCaptureMetrics:
    """Хранит ограниченное окно длительностей чтения и интервалов кадров."""

    def __init__(self, window_size: int = 240) -> None:
        """Создаёт окно недавних кадров, не накапливая историю видео."""
        if window_size < 2:
            raise ValueError("window_size должен быть не меньше 2")
        self._window_size = window_size
        self._lock = threading.Lock()
        self._read_ns: deque[int] = deque(maxlen=window_size)
        self._interval_ns: deque[int] = deque(maxlen=window_size - 1)
        self._same_signature: deque[bool] = deque(maxlen=window_size - 1)
        self._first_finished_ns: int | None = None
        self._last_finished_ns: int | None = None
        self._last_signature: int | None = None
        self._samples = 0

    def add(self, read_started_ns: int, read_finished_ns: int, signature: int) -> None:
        """Добавляет одно измерение чтения и соседний интервал между кадрами."""
        with self._lock:
            if read_started_ns < 0 or read_finished_ns < 0:
                raise ValueError("Временные метки чтения не могут быть отрицательными")
            if read_finished_ns < read_started_ns:
                raise ValueError("Чтение кадра не может закончиться раньше начала")
            if self._last_finished_ns is not None and read_finished_ns < self._last_finished_ns:
                raise ValueError("Временные метки кадров должны идти по порядку")
            self._read_ns.append(read_finished_ns - read_started_ns)
            if self._last_finished_ns is not None:
                self._interval_ns.append(read_finished_ns - self._last_finished_ns)
                self._same_signature.append(signature == self._last_signature)
            else:
                self._first_finished_ns = read_finished_ns
            self._last_finished_ns = read_finished_ns
            self._last_signature = signature
            self._samples += 1

    @staticmethod
    def _percentile_ms(values: deque[int], fraction: float) -> float:
        """Возвращает ближайший ранг процентиля в миллисекундах."""
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index] / 1_000_000

    def snapshot(self) -> CaptureTelemetry:
        """Возвращает статистику недавнего окна без доступа к кадрам."""
        with self._lock:
            observed_fps = 0.0
            if (
                self._samples > 1
                and self._first_finished_ns is not None
                and self._last_finished_ns is not None
                and self._last_finished_ns > self._first_finished_ns
            ):
                observed_fps = (self._samples - 1) * 1_000_000_000 / (
                    self._last_finished_ns - self._first_finished_ns
                )
            same_percent = 0.0
            if self._same_signature:
                same_percent = statistics.fmean(self._same_signature) * 100
            return CaptureTelemetry(
                samples=self._samples,
                observed_fps=observed_fps,
                read_p50_ms=self._percentile_ms(self._read_ns, 0.50),
                read_p90_ms=self._percentile_ms(self._read_ns, 0.90),
                interval_p50_ms=self._percentile_ms(self._interval_ns, 0.50),
                interval_p90_ms=self._percentile_ms(self._interval_ns, 0.90),
                coarse_same_percent=same_percent,
            )
