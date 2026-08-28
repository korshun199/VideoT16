"""Фоновые работники для конвейера реального времени."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Detection:
    """Одна готовая рамка объекта без зависимости от Ultralytics."""

    x1: int
    y1: int
    x2: int
    y2: int
    name: str
    confidence: float


@dataclass(frozen=True)
class InferenceSnapshot:
    """Последний полностью обработанный результат нейросети."""

    sequence: int = 0
    submitted_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    detections: tuple[Detection, ...] = ()


class SystemStatusMonitor:
    """Показывает температуру и фактическую загрузку CPU без внешних библиотек."""

    def __init__(self, interval: float = 1.0) -> None:
        """Создаёт монитор с периодом обновления показателей."""
        self._interval = interval
        self._last_update = 0.0
        self._last_total = 0
        self._last_idle = 0
        self._text = "TEMP --.-C CPU --%"

    @staticmethod
    def _temperature() -> str:
        """Читает температуру CPU из штатного интерфейса Linux."""
        try:
            raw_value = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
            return f"{int(raw_value) / 1000:.1f}C"
        except (OSError, ValueError):
            return "--.-C"

    @staticmethod
    def _cpu_counters() -> tuple[int, int] | None:
        """Читает суммарные счётчики времени CPU из /proc/stat."""
        try:
            fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            values = [int(value) for value in fields]
        except (OSError, IndexError, ValueError):
            return None
        if len(values) < 4:
            return None
        return sum(values), values[3] + (values[4] if len(values) > 4 else 0)

    def text(self) -> str:
        """Возвращает кэшированную строку системного состояния."""
        now = time.monotonic()
        if now - self._last_update < self._interval:
            return self._text
        self._last_update = now
        counters = self._cpu_counters()
        cpu_percent = "--"
        if counters is not None:
            total, idle = counters
            total_delta = total - self._last_total
            idle_delta = idle - self._last_idle
            if total_delta > 0:
                cpu_percent = str(round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100))))
            self._last_total, self._last_idle = total, idle
        self._text = f"TEMP {self._temperature()} CPU {cpu_percent}%"
        return self._text


class FrameRateMonitor:
    """Считает фактическую частоту кадров основного видеовыхода."""

    def __init__(self, interval: float = 1.0) -> None:
        """Создаёт счётчик кадров с периодом обновления показателя."""
        self._interval = interval
        self._started_at = time.monotonic()
        self._last_update = self._started_at
        self._frames = 0
        self._text = "FPS --.-"

    def text(self, count_frame: bool = True) -> str:
        """Возвращает средний FPS за последний измерительный интервал."""
        if count_frame:
            self._frames += 1
        now = time.monotonic()
        elapsed = now - self._last_update
        if elapsed >= self._interval:
            self._text = f"TS {self._frames / elapsed:.1f}"
            self._frames = 0
            self._last_update = now
        return self._text


class LatestFrameCapture:
    """Читает камеру в фоне и хранит только последний кадр."""

    def __init__(self, capture: Any) -> None:
        """Запускает поток чтения переданного устройства камеры."""
        self._capture = capture
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._frame: Any | None = None
        self._sequence = 0
        self._error: RuntimeError | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="latest-camera-frame",
            daemon=True,
        )
        self._thread.start()

    def latest(self, timeout: float = 2.0) -> Any:
        """Возвращает самый новый кадр, не отдавая устаревшую очередь."""
        return self.latest_with_sequence(timeout)[0]

    def latest_with_sequence(self, timeout: float = 2.0) -> tuple[Any, int]:
        """Возвращает кадр и его номер, чтобы не обрабатывать дубликаты."""
        if not self._frame_ready.wait(timeout):
            raise RuntimeError("Камера не вернула кадр за отведённое время")
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._frame is None:
                raise RuntimeError("Камера не вернула кадр")
            return self._frame, self._sequence

    def _run(self) -> None:
        """Непрерывно читает камеру и заменяет старый кадр новым."""
        try:
            while not self._stop_requested.is_set():
                ok, frame = self._capture.read()
                if not ok:
                    raise RuntimeError("Камера не вернула кадр")
                with self._lock:
                    self._frame = frame
                    self._sequence += 1
                    self._frame_ready.set()
        except Exception as error:  # noqa: BLE001 — передаём ошибку главному потоку.
            with self._lock:
                self._error = error if isinstance(error, RuntimeError) else RuntimeError(str(error))
                self._frame_ready.set()

    def close(self) -> None:
        """Останавливает поток чтения камеры перед освобождением устройства."""
        self._stop_requested.set()
        self._thread.join(timeout=2.0)


class LatestInferenceWorker:
    """Обрабатывает только самый свежий кадр, не задерживая видеовывод."""

    def __init__(self, predict: Callable[[Any], tuple[Detection, ...]]) -> None:
        """Запускает фоновый инференс переданной функцией."""
        self._predict = predict
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._pending_frame: Any | None = None
        self._pending_submitted_at = 0.0
        self._snapshot = InferenceSnapshot()
        self._error: RuntimeError | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="video-inference",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: Any) -> None:
        """Заменяет ожидающий кадр новым, чтобы не копить видеозадержку."""
        with self._lock:
            self._pending_frame = frame
            self._pending_submitted_at = time.monotonic()
            self._frame_ready.set()

    def latest(self) -> InferenceSnapshot:
        """Возвращает последний результат или сообщает об ошибке потока."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._snapshot

    def _take_pending_frame(self) -> tuple[Any | None, float]:
        """Забирает единственный свежий кадр для обработки."""
        with self._lock:
            frame = self._pending_frame
            submitted_at = self._pending_submitted_at
            self._pending_frame = None
            self._pending_submitted_at = 0.0
            if frame is None:
                self._frame_ready.clear()
            return frame, submitted_at

    def _run(self) -> None:
        """Выполняет инференс последовательно в отдельном потоке."""
        while not self._stop_requested.is_set():
            if not self._frame_ready.wait(0.1):
                continue
            frame, submitted_at = self._take_pending_frame()
            if frame is None:
                continue
            started_at = time.monotonic()
            try:
                detections = self._predict(frame)
            except Exception as error:  # noqa: BLE001 — ошибка должна дойти до главного потока.
                with self._lock:
                    self._error = RuntimeError(f"Ошибка фонового распознавания: {error}")
                self._stop_requested.set()
                return
            with self._lock:
                self._snapshot = InferenceSnapshot(
                    sequence=self._snapshot.sequence + 1,
                    submitted_at=submitted_at,
                    started_at=started_at,
                    completed_at=time.monotonic(),
                    detections=tuple(detections),
                )

    def close(self) -> None:
        """Просит поток завершиться и кратко ожидает окончания инференса."""
        self._stop_requested.set()
        self._frame_ready.set()
        self._thread.join(timeout=2.0)
