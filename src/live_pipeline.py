"""Чистые примитивы одно-процессного конвейера живого распознавания.

Модуль не открывает камеру, окно, UART или модель. Его задача — сохранить
принадлежность результата нейросети конкретному кадру, чтобы рамка никогда не
рисовалась на другом кадре видеопотока.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable

from src.realtime import Detection


@dataclass(frozen=True)
class TrackingFrame:
    """Уменьшенное представление кадра для быстрого сопровождения.

    Полный цветной BGR-кадр занимает много памяти. В кольцевой истории нужны
    только номер, время и подготовленное небольшое серое изображение.
    """

    sequence: int
    captured_at_ns: int
    image: Any


@dataclass(frozen=True)
class FramePacket:
    """Один кадр видеопотока с неизменяемой идентичностью.

    `bgr` никто из работников не изменяет. Для рисования всегда создаётся
    отдельная копия, а в истории сопровождения хранится `tracking.image`.
    """

    sequence: int
    captured_at_ns: int
    bgr: Any
    tracking: TrackingFrame


@dataclass(frozen=True)
class InferenceRequest:
    """Задание YOLO, жёстко связанное с исходным `FramePacket`."""

    source: FramePacket
    mode: str
    submitted_at_ns: int
    image: Any
    origin_x: int = 0
    origin_y: int = 0


@dataclass(frozen=True)
class DetectionResult:
    """Результат YOLO с номером и временем именно обработанного кадра."""

    source: FramePacket
    source_sequence: int
    source_captured_at_ns: int
    submitted_at_ns: int
    started_at_ns: int
    finished_at_ns: int
    mode: str
    detections: tuple[Detection, ...]
    raw_confidence: float

    @property
    def inference_ms(self) -> float:
        """Возвращает длительность инференса, не включая ожидание в mailbox."""
        return max(0, self.finished_at_ns - self.started_at_ns) / 1_000_000

    @property
    def queue_ms(self) -> float:
        """Возвращает ожидание задания перед началом инференса."""
        return max(0, self.started_at_ns - self.submitted_at_ns) / 1_000_000


@dataclass(frozen=True)
class OverlayPacket:
    """Готовая рамка, которую разрешено рисовать только на одном кадре."""

    frame_sequence: int
    source_sequence: int
    detections: tuple[Detection, ...]
    status: str


class FrameHub:
    """Публикует свежий кадр и хранит компактную историю для трекера.

    Камерный поток вызывает `publish`. Поток вывода читает `latest`, а будущий
    трекер запрашивает `tracking_after`. Полный кадр в истории не дублируется.
    """

    def __init__(self, tracking_history_size: int = 96) -> None:
        """Создаёт концентратор с ограниченной историей уменьшенных кадров."""
        if tracking_history_size < 2:
            raise ValueError("tracking_history_size должен быть не меньше 2")
        self._history: deque[TrackingFrame] = deque(maxlen=tracking_history_size)
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._latest: FramePacket | None = None
        self._next_sequence = 1
        self._error: RuntimeError | None = None

    def publish(
        self,
        bgr: Any,
        tracking_image: Any,
        captured_at_ns: int | None = None,
    ) -> FramePacket:
        """Присваивает кадру номер и заменяет только свежую версию.

        Никакая очередь полных кадров не создаётся: медленный потребитель
        получает только актуальный `latest`, а трекер — ограниченную историю
        облегчённых кадров.
        """
        captured = time.monotonic_ns() if captured_at_ns is None else captured_at_ns
        if captured < 0:
            raise ValueError("captured_at_ns не может быть отрицательным")
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            tracking = TrackingFrame(sequence, captured, tracking_image)
            packet = FramePacket(sequence, captured, bgr, tracking)
            self._latest = packet
            self._history.append(tracking)
            self._new_frame.set()
            return packet

    def latest(self, after_sequence: int = 0, timeout: float = 0.0) -> FramePacket | None:
        """Возвращает кадр новее `after_sequence` либо `None` по тайм-ауту."""
        if timeout < 0:
            raise ValueError("timeout не может быть отрицательным")
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._error is not None:
                    raise self._error
                packet = self._latest
                if packet is not None and packet.sequence > after_sequence:
                    return packet
                self._new_frame.clear()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._new_frame.wait(remaining)

    def tracking_after(self, sequence: int) -> tuple[TrackingFrame, ...]:
        """Возвращает сохранённые облегчённые кадры строго после номера кадра."""
        with self._lock:
            return tuple(item for item in self._history if item.sequence > sequence)

    def fail(self, error: Exception) -> None:
        """Передаёт ошибку потока захвата всем ожидающим потребителям."""
        with self._lock:
            self._error = error if isinstance(error, RuntimeError) else RuntimeError(str(error))
            self._new_frame.set()


class LatestDetectionWorker:
    """Выполняет YOLO в фоне без накопления старых кадров.

    Один worker намеренно владеет одной ONNX-сессией. Параллельность самой
    модели задаётся её настройками ONNX Runtime и измеряется отдельно; несколько
    сессий не создаются без подтверждённого ускорения.
    """

    def __init__(
        self,
        predict: Callable[[InferenceRequest], tuple[tuple[Detection, ...], float]],
        name: str = "live-yolo",
    ) -> None:
        """Запускает фоновый worker с функцией, возвращающей рамки и confidence."""
        self._predict = predict
        self._lock = threading.Lock()
        self._request_ready = threading.Event()
        self._stop_requested = threading.Event()
        self._pending: InferenceRequest | None = None
        self._latest: DetectionResult | None = None
        self._error: RuntimeError | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, request: InferenceRequest) -> None:
        """Заменяет ожидающее задание свежим и отбрасывает устаревшее.

        Уже начатый инференс не прерывается: большинство ONNX-вызовов нельзя
        безопасно отменить. Но после него worker возьмёт только последнее
        поступившее задание.
        """
        with self._lock:
            if self._error is not None:
                raise self._error
            self._pending = request
            self._request_ready.set()

    def latest(self) -> DetectionResult | None:
        """Возвращает готовый результат с исходным номером кадра."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._latest

    def _take_pending(self) -> InferenceRequest | None:
        """Забирает единственное актуальное задание из mailbox."""
        with self._lock:
            request = self._pending
            self._pending = None
            if request is None:
                self._request_ready.clear()
            return request

    def _run(self) -> None:
        """Последовательно вызывает YOLO, сохраняя привязку результата к кадру."""
        while not self._stop_requested.is_set():
            if not self._request_ready.wait(0.1):
                continue
            request = self._take_pending()
            if request is None:
                continue
            started_at_ns = time.monotonic_ns()
            try:
                detections, raw_confidence = self._predict(request)
            except Exception as error:  # noqa: BLE001 — ошибка должна дойти до UI.
                with self._lock:
                    self._error = RuntimeError(f"Ошибка фонового распознавания: {error}")
                self._stop_requested.set()
                return
            result = DetectionResult(
                source=request.source,
                source_sequence=request.source.sequence,
                source_captured_at_ns=request.source.captured_at_ns,
                submitted_at_ns=request.submitted_at_ns,
                started_at_ns=started_at_ns,
                finished_at_ns=time.monotonic_ns(),
                mode=request.mode,
                detections=tuple(detections),
                raw_confidence=float(raw_confidence),
            )
            with self._lock:
                self._latest = result

    def close(self) -> None:
        """Завершает worker и кратко ожидает текущий вызов нейросети."""
        self._stop_requested.set()
        self._request_ready.set()
        self._thread.join(timeout=2.0)


def overlay_for_frame(
    frame: FramePacket,
    overlay: OverlayPacket | None,
) -> OverlayPacket | None:
    """Возвращает оверлей только при точном совпадении номера кадра.

    Это последний барьер перед отрисовкой. Даже ошибка в будущем трекере не
    позволит случайно нарисовать старую рамку на свежем изображении.
    """
    if overlay is None or overlay.frame_sequence != frame.sequence:
        return None
    return overlay
