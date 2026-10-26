"""Кадровый буфер, привязка YOLO и синхронизация live-конвейера.

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
class DetectionSnapshot:
    """Компактная история YOLO без ссылки на полноразмерный исходный кадр."""

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
        """Возвращает длительность инференса этого результата."""
        return max(0, self.finished_at_ns - self.started_at_ns) / 1_000_000

    @property
    def queue_ms(self) -> float:
        """Возвращает время ожидания задания перед инференсом."""
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

    Камера вызывает `publish`; экран читает живой или задержанный кадр.
    Полные кадры хранятся только в ограниченном буфере вывода, трекеру доступна
    отдельная компактная история серых кадров.
    """

    def __init__(
        self,
        tracking_history_size: int = 96,
        output_buffer_max_bytes: int = 0,
    ) -> None:
        """Создаёт ограниченные буферы сопровождения и отложенного вывода."""
        if tracking_history_size < 2:
            raise ValueError("tracking_history_size должен быть не меньше 2")
        if output_buffer_max_bytes < 0:
            raise ValueError("output_buffer_max_bytes не может быть отрицательным")
        self._history: deque[TrackingFrame] = deque(maxlen=tracking_history_size)
        self._output: deque[FramePacket] = deque()
        self._output_buffer_max_bytes = output_buffer_max_bytes
        self._output_buffer_bytes = 0
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._latest: FramePacket | None = None
        self._next_sequence = 1
        self._error: RuntimeError | None = None

    def buffer_status(self) -> dict[str, int | None]:
        """Возвращает номера крайних кадров и объём ограниченного видеобуфера."""
        with self._lock:
            return {
                "latest_sequence": self._latest.sequence if self._latest else None,
                "oldest_output_sequence": self._output[0].sequence if self._output else None,
                "newest_output_sequence": self._output[-1].sequence if self._output else None,
                "output_frames": len(self._output),
                "output_bytes": self._output_buffer_bytes,
            }

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
        frame_bytes = max(1, int(getattr(bgr, "nbytes", 0)))
        if self._output_buffer_max_bytes and frame_bytes > self._output_buffer_max_bytes:
            raise ValueError("Один видеокадр превышает лимит буфера вывода")
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
            tracking = TrackingFrame(sequence, captured, tracking_image)
            packet = FramePacket(sequence, captured, bgr, tracking)
            self._latest = packet
            self._history.append(tracking)
            if self._output_buffer_max_bytes:
                frame_bytes = max(1, frame_bytes)
                self._output.append(packet)
                self._output_buffer_bytes += frame_bytes
                while self._output and self._output_buffer_bytes > self._output_buffer_max_bytes:
                    removed = self._output.popleft()
                    self._output_buffer_bytes -= max(1, int(getattr(removed.bgr, "nbytes", 0)))
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

    def tracking_frame(self, sequence: int) -> TrackingFrame | None:
        """Возвращает один серый кадр для привязки результата YOLO."""
        with self._lock:
            return next((item for item in self._history if item.sequence == sequence), None)

    def delayed_latest(
        self,
        after_sequence: int,
        delay_ms: int,
        now_ns: int | None = None,
    ) -> FramePacket | None:
        """Выбирает самый свежий накопленный кадр, уже созревший для вывода."""
        if delay_ms < 0:
            raise ValueError("delay_ms не может быть отрицательным")
        now = time.monotonic_ns() if now_ns is None else now_ns
        cutoff = now - delay_ms * 1_000_000
        with self._lock:
            if self._error is not None:
                raise self._error
            return next(
                (
                    packet
                    for packet in reversed(self._output)
                    if packet.sequence > after_sequence and packet.captured_at_ns <= cutoff
                ),
                None,
            )

    def delayed_next(
        self,
        after_sequence: int,
        delay_ms: int,
        now_ns: int | None = None,
    ) -> FramePacket | None:
        """Возвращает следующий по порядку созревший кадр для плавного playback."""
        if delay_ms < 0:
            raise ValueError("delay_ms не может быть отрицательным")
        now = time.monotonic_ns() if now_ns is None else now_ns
        cutoff = now - delay_ms * 1_000_000
        with self._lock:
            if self._error is not None:
                raise self._error
            return next(
                (
                    packet
                    for packet in self._output
                    if packet.sequence > after_sequence and packet.captured_at_ns <= cutoff
                ),
                None,
            )

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
        self._history: deque[DetectionSnapshot] = deque(maxlen=256)
        self._error: RuntimeError | None = None
        self._submitted_count = 0
        self._replaced_count = 0
        self._completed_count = 0
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
            self._submitted_count += 1
            if self._pending is not None:
                self._replaced_count += 1
            self._pending = request
            self._request_ready.set()

    def statistics(self) -> dict[str, int | None]:
        """Снимок счётчиков mailbox, чтобы видеть потерянные устаревшие задания."""
        with self._lock:
            return {
                "submitted": self._submitted_count,
                "replaced": self._replaced_count,
                "completed": self._completed_count,
                "latest_result_sequence": self._latest.source_sequence if self._latest else None,
            }

    def latest(self) -> DetectionResult | None:
        """Возвращает готовый результат с исходным номером кадра."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return self._latest

    def latest_at_or_before(self, sequence: int) -> DetectionSnapshot | None:
        """Находит свежий результат YOLO, не относящийся к будущему кадру."""
        with self._lock:
            if self._error is not None:
                raise self._error
            return next(
                (item for item in reversed(self._history) if item.source_sequence <= sequence),
                None,
            )

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
                self._completed_count += 1
                self._history.append(
                    DetectionSnapshot(
                        source_sequence=result.source_sequence,
                        source_captured_at_ns=result.source_captured_at_ns,
                        submitted_at_ns=result.submitted_at_ns,
                        started_at_ns=result.started_at_ns,
                        finished_at_ns=result.finished_at_ns,
                        mode=result.mode,
                        detections=result.detections,
                        raw_confidence=result.raw_confidence,
                    )
                )

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
