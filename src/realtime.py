"""Фоновые работники для конвейера реального времени."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
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
    completed_at: float = 0.0
    detections: tuple[Detection, ...] = ()


class LatestInferenceWorker:
    """Обрабатывает только самый свежий кадр, не задерживая видеовывод."""

    def __init__(self, predict: Callable[[Any], tuple[Detection, ...]]) -> None:
        """Запускает фоновый инференс переданной функцией."""
        self._predict = predict
        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._pending_frame: Any | None = None
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
            self._frame_ready.set()

    def latest(self) -> InferenceSnapshot:
        """Возвращает последний результат или сообщает об ошибке потока."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._snapshot

    def _take_pending_frame(self) -> Any | None:
        """Забирает единственный свежий кадр для обработки."""
        with self._lock:
            frame = self._pending_frame
            self._pending_frame = None
            if frame is None:
                self._frame_ready.clear()
            return frame

    def _run(self) -> None:
        """Выполняет инференс последовательно в отдельном потоке."""
        while not self._stop_requested.is_set():
            if not self._frame_ready.wait(0.1):
                continue
            frame = self._take_pending_frame()
            if frame is None:
                continue
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
                    completed_at=time.monotonic(),
                    detections=tuple(detections),
                )

    def close(self) -> None:
        """Просит поток завершиться и кратко ожидает окончания инференса."""
        self._stop_requested.set()
        self._frame_ready.set()
        self._thread.join(timeout=2.0)
