"""Параллельное сопоставление, разметка и оконный показ кадров."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
import signal
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeyEvent, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

if TYPE_CHECKING:
    from capture import JsonlLog, PipelineSession


@dataclass
class DisplayPacket:
    """Передаёт готовый QImage оконному потоку и получает подтверждение показа."""

    number: int
    image: Any
    prepared_ns: int
    acknowledged: threading.Event = field(default_factory=threading.Event)
    displayed: bool = False
    presented_ns: int | None = None
    error: str | None = None


class DisplayWorker(threading.Thread):
    """Показывает кадры по порядку, не связываясь с Qt-виджетами из фонового потока."""

    def __init__(self, session: PipelineSession, display_log: JsonlLog) -> None:
        """Создаёт поток вывода и ограниченную очередь не более одного кадра."""
        super().__init__(name="video-display", daemon=False)
        self.session = session
        self.display_log = display_log
        self.output_queue: Queue[DisplayPacket] = Queue(maxsize=1)
        self.stop_requested = threading.Event()
        self.finished = threading.Event()
        self.next_number = 0
        self.started_ns: int | None = None
        self.finished_ns: int | None = None

    def request_stop(self, reason: str) -> None:
        """Останавливает вывод и будит ожидающий пару поток."""
        self.stop_requested.set()
        with self.session.pair_condition:
            self.session.pair_condition.notify_all()
        self.display_log.write("display_stop_requested", reason=reason)

    def _fail(self, message: str, frame: int | None = None) -> None:
        """Записывает неисправность сопоставления/рисования и останавливает конвейер."""
        with self.session._state_lock:
            if self.session.display_error is None:
                self.session.display_error = message
        self.display_log.write("display_error", frame=frame, message=message)
        self.session.request_stop("display_error")
        self.stop_requested.set()

    def _wait_for_pair(self) -> tuple[Any, Any] | None:
        """Выбирает точную пару; в режиме latest берёт самую свежую готовую пару."""
        missing: dict[str, Any] | None = None
        with self.session.pair_condition:
            while not self.stop_requested.is_set():
                if self.session.recognition_frame_policy == "latest":
                    candidates = sorted(
                        self.session.display_buffer.latest_after(self.next_number),
                        key=lambda result: result.number,
                        reverse=True,
                    )
                else:
                    expected = self.next_number + 1
                    result = self.session.display_buffer.get(expected)
                    candidates = [] if result is None else [result]

                for result in candidates:
                    frame = self.session.frame_buffer.get(result.number)
                    if frame is not None:
                        return frame, result

                if self.session.processing_finished.is_set():
                    if self.session.recognition_frame_policy == "latest":
                        return None
                    if expected > self.session.frame_count:
                        return None
                    missing_frame = self.session.frame_buffer.get(expected)
                    missing_result = self.session.display_buffer.get(expected)
                    missing = {
                        "frame": expected,
                        "source_frame_present": missing_frame is not None,
                        "recognition_result_present": missing_result is not None,
                    }
                    break
                self.session.pair_condition.wait(timeout=0.1)
        if missing is not None:
            self._fail(
                "После завершения распознавания не найдена полная пара буферов",
                frame=int(missing["frame"]),
            )
            self.display_log.write("display_pair_missing", **missing)
        return None

    @staticmethod
    def _to_qimage(bgr_image: Any) -> Any:
        """Создаёт независимую копию QImage из размеченного кадра BGR."""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]
        return QImage(
            rgb_image.data,
            width,
            height,
            int(rgb_image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()

    def _offer_to_window(self, packet: DisplayPacket) -> bool:
        """Передаёт изображение UI и ждёт подтверждения его отрисовки."""
        while not self.stop_requested.is_set():
            try:
                self.output_queue.put(packet, timeout=0.1)
                break
            except Full:
                continue
        else:
            return False

        while not self.stop_requested.is_set():
            if packet.acknowledged.wait(timeout=0.1):
                return packet.displayed
        return packet.acknowledged.is_set() and packet.displayed

    def run(self) -> None:
        """Готовит кадры по порядку и освобождает пару только после показа окном."""
        self.started_ns = time.monotonic_ns()
        shown = 0
        self.display_log.write("display_started", started_ns=self.started_ns)
        try:
            while not self.stop_requested.is_set():
                pair = self._wait_for_pair()
                if pair is None:
                    break
                frame, result = pair
                render_started_ns = time.monotonic_ns()
                self.display_log.write(
                    "display_pair_selected",
                    frame=frame.number,
                    captured_ns=frame.captured_ns,
                    recognized_ns=result.recognized_ns,
                    detections=len(result.detections),
                    model_presence_candidate=bool(result.presence_candidates),
                    presence_max_confidence=max(
                        (item.confidence for item in result.presence_candidates), default=0.0
                    ),
                    source_buffer_frames=self.session.frame_buffer.stats()[0],
                    result_buffer_items=self.session.display_buffer.stats()[0],
                )
                try:
                    annotated = self.session.detector.annotate(frame.image, result.detections)
                    qimage = self._to_qimage(annotated)
                except Exception as error:
                    self._fail(f"Не удалось подготовить кадр для окна: {error}", frame.number)
                    break

                prepared_ns = time.monotonic_ns()
                packet = DisplayPacket(frame.number, qimage, prepared_ns)
                self.display_log.write(
                    "display_frame_ready",
                    frame=frame.number,
                    draw_and_convert_ms=round((prepared_ns - render_started_ns) / 1_000_000, 3),
                    capture_to_ready_ms=round((prepared_ns - frame.captured_ns) / 1_000_000, 3),
                    image_width=qimage.width(),
                    image_height=qimage.height(),
                )
                if not self._offer_to_window(packet):
                    if packet.error and not self.stop_requested.is_set():
                        self._fail(f"Оконный вывод не показал кадр: {packet.error}", frame.number)
                    self.display_log.write(
                        "display_frame_not_shown",
                        frame=frame.number,
                        window_closed=self.stop_requested.is_set(),
                    )
                    break

                presented_ns = packet.presented_ns or time.monotonic_ns()
                removed = self.session.remove_displayed_pair(frame.number)
                if removed is None:
                    self._fail("Показанный кадр потерял пару в буферах до удаления", frame.number)
                    break
                source_count, source_bytes, result_count, result_bytes = removed
                shown += 1
                previous_number = self.next_number
                self.next_number = frame.number
                with self.session._state_lock:
                    self.session.displayed_count = shown
                if frame.number > previous_number + 1:
                    self.display_log.write(
                        "display_frames_skipped",
                        after_frame=previous_number,
                        before_frame=frame.number,
                        count=frame.number - previous_number - 1,
                        reason="freshest_ready_result",
                    )
                self.display_log.write(
                    "frame_displayed",
                    frame=frame.number,
                    captured_ns=frame.captured_ns,
                    recognized_ns=result.recognized_ns,
                    display_prepared_ns=prepared_ns,
                    display_presented_ns=presented_ns,
                    capture_to_present_ms=round((presented_ns - frame.captured_ns) / 1_000_000, 3),
                    recognition_to_present_ms=round((presented_ns - result.recognized_ns) / 1_000_000, 3),
                    render_ms=round((prepared_ns - render_started_ns) / 1_000_000, 3),
                    ui_wait_ms=round((presented_ns - prepared_ns) / 1_000_000, 3),
                    model_presence_candidate=bool(result.presence_candidates),
                    presence_max_confidence=max(
                        (item.confidence for item in result.presence_candidates), default=0.0
                    ),
                    detections=[
                        {
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                            "box": [item.left, item.top, item.right, item.bottom],
                        }
                        for item in result.detections
                    ],
                    remaining_source_frames=source_count,
                    remaining_source_bytes=source_bytes,
                    remaining_result_items=result_count,
                    remaining_result_estimated_bytes=result_bytes,
                )
                del frame, result, annotated, qimage, packet, pair
        except Exception as error:
            self._fail(f"Неожиданная ошибка потока отображения: {error}")
        finally:
            self.finished_ns = time.monotonic_ns()
            self.display_log.write(
                "display_finished",
                frames_displayed=shown,
                last_displayed_frame=self.next_number,
                started_ns=self.started_ns,
                finished_ns=self.finished_ns,
                reason="stop_requested" if self.stop_requested.is_set() else "all_frames_done",
            )
            self.finished.set()


class VideoView(QWidget):
    """Рисует текущий кадр в главном Qt-потоке с сохранением пропорций."""

    def __init__(self) -> None:
        """Создаёт чёрную область просмотра."""
        super().__init__()
        self.setMinimumSize(640, 360)
        self._image: Any = None
        self._pending_packet: DisplayPacket | None = None

    @property
    def has_pending_packet(self) -> bool:
        """Сообщает, ждёт ли уже переданный кадр фактической отрисовки окна."""
        return self._pending_packet is not None

    def set_packet(self, packet: DisplayPacket) -> None:
        """Устанавливает один кадр и запускает отрисовку в основном UI-потоке."""
        self._image = packet.image
        self._pending_packet = packet
        self.update()

    def abort_pending(self) -> None:
        """Отвечает фоновой нити, если окно закрывается до показа кадра."""
        packet = self._pending_packet
        if packet is not None and not packet.acknowledged.is_set():
            packet.displayed = False
            packet.error = "окно закрыто до отрисовки"
            packet.acknowledged.set()
        self._pending_packet = None

    def paintEvent(self, event: Any) -> None:
        """Рисует вписанный по пропорциям кадр и подтверждает его оконный показ."""
        del event
        painter = QPainter(self)
        packet = self._pending_packet
        painted = False
        try:
            if not painter.isActive():
                raise RuntimeError("Qt не смог начать отрисовку виджета")
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            if self._image is not None and not self.rect().isEmpty():
                target = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
                x = (self.width() - target.width()) // 2
                y = (self.height() - target.height()) // 2
                painter.drawImage(QRect(x, y, target.width(), target.height()), self._image)
                painted = self.isVisible() and target.width() > 0 and target.height() > 0
        except Exception as error:
            if packet is not None:
                packet.error = str(error)
        finally:
            painter.end()

        if packet is not None and not packet.acknowledged.is_set():
            packet.displayed = painted
            packet.presented_ns = time.monotonic_ns() if painted else None
            packet.acknowledged.set()
            self._pending_packet = None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Закрывает окно по Esc, не перехватывая остальные клавиши Qt."""
        if event.key() == Qt.Key.Key_Escape:
            self.window().close()
            return
        super().keyPressEvent(event)

class VideoWindow(QMainWindow):
    """Отвечает за максимизированное окно, приём готовых изображений и Esc."""

    def __init__(
        self,
        session: PipelineSession,
        worker: DisplayWorker,
        interrupt_requested: threading.Event,
    ) -> None:
        """Создаёт окно просмотра и периодически опрашивает ограниченную очередь."""
        super().__init__()
        self.session = session
        self.worker = worker
        self.interrupt_requested = interrupt_requested
        self.setWindowTitle("VideoT16 — параллельный захват, распознавание и показ")
        self.viewer = VideoView()
        self.setCentralWidget(self.viewer)
        self._escape_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._escape_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._escape_shortcut.setAutoRepeat(False)
        self._escape_shortcut.activated.connect(self.close)
        self.statusBar().showMessage("Ожидание первого кадра…  Esc в активном окне или крестик — выход")
        self._last_status_message = "Ожидание первого кадра…  Esc в активном окне или крестик — выход"
        self._timer = QTimer(self)
        self._timer.setInterval(10)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _stop_pipeline(self) -> None:
        """Останавливает конвейер при закрытии окна пользователем."""
        reason = "ctrl_c" if self.interrupt_requested.is_set() else "window_closed"
        self.worker.request_stop(reason)
        self.session.request_stop(reason)

    def _poll(self) -> None:
        """Передаёт UI кадр и закрывает окно после завершения опыта."""
        if self.interrupt_requested.is_set():
            self.close()
            return

        if not self.viewer.has_pending_packet:
            try:
                packet = self.worker.output_queue.get_nowait()
            except Empty:
                packet = None
            if packet is not None:
                self.viewer.set_packet(packet)

        shown = self.session.displayed_count
        captured = self.session.frame_count
        with self.session._state_lock:
            presence_summary = self.session.last_presence_summary
            presence_frame = self.session.last_presence_frame
        presence_status = (
            f"Модель: {presence_summary} (кадр {presence_frame})"
            if presence_frame
            else f"Модель: {presence_summary}"
        )
        exit_hint = f"{presence_status}  |  Esc в активном окне или крестик — выход"
        if self.session.display_error:
            message = f"Ошибка вывода: {self.session.display_error}  |  {exit_hint}"
        elif self.session.capture_error:
            message = f"Ошибка захвата: {self.session.capture_error}  |  {exit_hint}"
        elif self.session.recognition_error:
            message = f"Ошибка распознавания: {self.session.recognition_error}  |  {exit_hint}"
        elif self.session.processing_finished.is_set() and self.worker.finished.is_set():
            self.worker.display_log.write(
                "display_run_completed",
                frames_displayed=shown,
                frames_captured=captured,
                reason=self.session.reason,
            )
            self.close()
            return
        else:
            message = f"Показано {shown}  |  захвачено {captured}  |  {exit_hint}"
        if message != self._last_status_message:
            self.statusBar().showMessage(message)
            self._last_status_message = message

    def closeEvent(self, event: Any) -> None:
        """Останавливает конвейер и подтверждает закрытие окна просмотра."""
        self._timer.stop()
        self.viewer.abort_pending()
        self._stop_pipeline()
        super().closeEvent(event)


def run_display(session: PipelineSession, display_log: JsonlLog) -> int:
    """Параллельно запускает конвейер и показывает готовые совпавшие кадры."""
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("VideoT16")
    application.setQuitOnLastWindowClosed(True)

    worker = DisplayWorker(session, display_log)
    interrupt_requested = threading.Event()
    window = VideoWindow(session, worker, interrupt_requested)
    pipeline_started = False
    coordinator: threading.Thread | None = None
    exit_code = 0
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def record_sigint(signum: int, frame: Any) -> None:
        """Отмечает Ctrl+C без исключения внутри Qt-обработчика кадра."""
        del signum, frame
        interrupt_requested.set()

    signal.signal(signal.SIGINT, record_sigint)
    try:
        worker.start()
        session.start()
        pipeline_started = True

        def wait_for_pipeline() -> None:
            """Дожидается рабочих потоков вне UI, чтобы окно не зависало."""
            try:
                session.join()
            except Exception as error:
                display_log.write("pipeline_join_error", message=str(error))
                session.request_stop("pipeline_join_error")
            finally:
                session.processing_finished.set()
                with session.pair_condition:
                    session.pair_condition.notify_all()
                worker.finished.wait()

        coordinator = threading.Thread(
            target=wait_for_pipeline,
            name="pipeline-coordinator",
            daemon=False,
        )
        coordinator.start()
        display_log.write(
            "display_window_opened",
            maximum=True,
            source=session.settings["source"],
            requested_fps=session.settings["fps"],
        )
        window.showMaximized()
        exit_code = application.exec()
        if interrupt_requested.is_set():
            exit_code = 130
    except KeyboardInterrupt:
        print("\nОстанавливаю конвейер по Ctrl+C…", flush=True)
        exit_code = 130
    except Exception as error:
        display_log.write("display_start_error", message=str(error))
        session.request_stop("display_start_error")
        print(f"Ошибка оконного вывода: {error}", flush=True)
        exit_code = 1
    finally:
        # Оставляем безопасный обработчик активным, пока фоновые потоки
        # завершаются: повторный Ctrl+C не должен оборвать очистку посередине.
        try:
            worker.request_stop("display_loop_finished")
            session.request_stop("display_loop_finished")
            if coordinator is not None:
                coordinator.join()
            elif pipeline_started:
                session.join()
            worker.join()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)

    summary = session.timing_summary()
    display_log.write(
        "display_session_finished",
        **summary,
        reason=session.reason,
        capture_error=session.capture_error,
        recognition_error=session.recognition_error,
        display_error=session.display_error,
    )
    if session.capture_error or session.recognition_error or session.display_error:
        exit_code = max(exit_code, 1)
    return exit_code
