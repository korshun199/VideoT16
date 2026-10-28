"""Параллельное сопоставление, разметка и оконный показ кадров."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from pathlib import Path
import signal
import threading
import time
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QKeyEvent, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

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
    """Показывает свежие кадры, не ожидая очередной инференс для каждого из них."""

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
        self.last_result_number = 0
        self.overlay_result: Any = None

    def request_stop(self, reason: str) -> None:
        """Останавливает вывод и будит ожидающий свежий кадр поток."""
        self.stop_requested.set()
        with self.session.pair_condition:
            self.session.pair_condition.notify_all()
        self.display_log.write("display_stop_requested", reason=reason)

    def _fail(self, message: str, frame: int | None = None) -> None:
        """Записывает неисправность потока показа и останавливает конвейер."""
        with self.session._state_lock:
            if self.session.display_error is None:
                self.session.display_error = message
        self.display_log.write("display_error", frame=frame, message=message)
        self.session.request_stop("display_error")
        self.stop_requested.set()

    def _wait_for_frame(self) -> tuple[Any, int, bool] | None:
        """Ждёт свежий видеокадр и после конца потока позволяет обновить финальный оверлей."""
        while not self.stop_requested.is_set():
            frame, skipped = self.session.preview_buffer.wait_after(
                self.next_number, self.stop_requested
            )
            if frame is not None:
                return frame, skipped, False
            if self.stop_requested.is_set():
                return None
            if self.session.processing_finished.is_set():
                final_frame = self.session.preview_buffer.latest()
                if final_frame is None:
                    return None
                pending = self.session.display_buffer.latest_after(self.last_result_number)
                if any(result.number <= final_frame.number for result in pending):
                    return final_frame, 0, True
                return None
            # Распознавание может завершиться позже захвата; координатор
            # разбудит нас после результата и после окончательного завершения.
            with self.session.pair_condition:
                self.session.pair_condition.wait(timeout=0.1)
        return None

    @staticmethod
    def _to_qimage(bgr_image: Any) -> Any:
        """Копирует BGR-кадр в QImage без лишнего преобразования цветового порядка."""
        height, width = bgr_image.shape[:2]
        return QImage(
            bgr_image.data,
            width,
            height,
            int(bgr_image.strides[0]),
            QImage.Format.Format_BGR888,
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
        """Показывает каждый доступный свежий кадр с последним ещё актуальным результатом."""
        self.started_ns = time.monotonic_ns()
        shown = 0
        self.display_log.write("display_started", started_ns=self.started_ns)
        try:
            while not self.stop_requested.is_set():
                selected = self._wait_for_frame()
                if selected is None:
                    break
                frame, skipped_frames, final_refresh = selected
                result_updates = self.session.display_buffer.take_through(frame.number)
                for result in result_updates:
                    if result.number > self.last_result_number:
                        self.last_result_number = result.number
                    # Пустой результат не обрывает рамку резко: последняя
                    # положительная находка живёт только заданное число мс.
                    if result.detections or result.presence_candidates:
                        self.overlay_result = result
                confidence_threshold = (
                    float(self.session.settings["confidence_percent"]) / 100.0
                )
                hold_ns = int(self.session.settings.get("display_box_hold_ms", 250)) * 1_000_000
                overlay_age_ns = (
                    frame.captured_ns - self.overlay_result.captured_ns
                    if self.overlay_result is not None
                    else None
                )
                overlay_is_fresh = (
                    overlay_age_ns is not None and 0 <= overlay_age_ns <= hold_ns
                )
                overlay_result = self.overlay_result if overlay_is_fresh else None
                # Кандидаты ниже рабочего порога показываются отдельно для диагностики.
                weak_candidates = tuple(
                    item for item in (overlay_result.presence_candidates if overlay_result else ())
                    if item.confidence < confidence_threshold
                )
                detections = overlay_result.detections if overlay_result else ()
                render_started_ns = time.monotonic_ns()
                self.display_log.write(
                    "display_frame_selected",
                    frame=frame.number,
                    captured_ns=frame.captured_ns,
                    recognition_frame=(overlay_result.number if overlay_result else None),
                    recognition_age_ms=(
                        round(overlay_age_ns / 1_000_000, 3)
                        if overlay_age_ns is not None else None
                    ),
                    result_updates=len(result_updates),
                    final_refresh=final_refresh,
                    detections=len(detections),
                    weak_candidates=len(weak_candidates),
                    model_presence_candidate=bool(
                        overlay_result and overlay_result.presence_candidates
                    ),
                    presence_max_confidence=max(
                        (
                            item.confidence
                            for item in (overlay_result.presence_candidates if overlay_result else ())
                        ),
                        default=0.0,
                    ),
                    source_buffer_frames=self.session.frame_buffer.stats()[0],
                    result_buffer_items=self.session.display_buffer.stats()[0],
                )
                try:
                    annotated = self.session.detector.annotate(
                        frame.image, detections, weak_candidates
                    )
                    qimage = self._to_qimage(annotated)
                except Exception as error:
                    self._fail(f"Не удалось подготовить кадр для окна: {error}", frame.number)
                    break

                prepared_ns = time.monotonic_ns()
                packet = DisplayPacket(frame.number, qimage, prepared_ns)
                self.display_log.write(
                    "display_frame_ready",
                    frame=frame.number,
                    recognition_frame=(overlay_result.number if overlay_result else None),
                    recognition_age_ms=(
                        round(overlay_age_ns / 1_000_000, 3)
                        if overlay_age_ns is not None else None
                    ),
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
                shown += 1
                previous_number = self.next_number
                self.next_number = max(self.next_number, frame.number)
                with self.session._state_lock:
                    self.session.displayed_count = shown
                    self.session.last_displayed_number = max(
                        self.session.last_displayed_number, frame.number
                    )
                if frame.number > previous_number + 1:
                    self.display_log.write(
                        "display_frames_skipped",
                        after_frame=previous_number,
                        before_frame=frame.number,
                        count=skipped_frames,
                        reason="freshest_capture_frame",
                    )
                self.display_log.write(
                    "frame_displayed",
                    frame=frame.number,
                    captured_ns=frame.captured_ns,
                    recognition_frame=(overlay_result.number if overlay_result else None),
                    recognized_ns=(overlay_result.recognized_ns if overlay_result else None),
                    recognition_age_ms=(
                        round(overlay_age_ns / 1_000_000, 3)
                        if overlay_age_ns is not None else None
                    ),
                    box_hold_active=overlay_is_fresh,
                    display_prepared_ns=prepared_ns,
                    display_presented_ns=presented_ns,
                    capture_to_present_ms=round((presented_ns - frame.captured_ns) / 1_000_000, 3),
                    recognition_to_present_ms=(
                        round((presented_ns - overlay_result.recognized_ns) / 1_000_000, 3)
                        if overlay_result is not None else None
                    ),
                    render_ms=round((prepared_ns - render_started_ns) / 1_000_000, 3),
                    ui_wait_ms=round((presented_ns - prepared_ns) / 1_000_000, 3),
                    model_presence_candidate=bool(
                        overlay_result and overlay_result.presence_candidates
                    ),
                    presence_max_confidence=max(
                        (
                            item.confidence
                            for item in (overlay_result.presence_candidates if overlay_result else ())
                        ),
                        default=0.0,
                    ),
                    detections=[
                        {
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                            "box": [item.left, item.top, item.right, item.bottom],
                        }
                        for item in detections
                    ],
                    weak_candidates=[
                        {
                            "label": item.label,
                            "confidence": round(item.confidence, 6),
                            "box": [item.left, item.top, item.right, item.bottom],
                        }
                        for item in weak_candidates
                    ],
                    skipped_capture_frames=skipped_frames,
                    pending_source_frames=self.session.frame_buffer.stats()[0],
                    pending_result_items=self.session.display_buffer.stats()[0],
                    final_refresh=final_refresh,
                )
                del frame, result_updates, annotated, qimage, packet, selected
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
        """Показывает кадр с выбранным масштабированием и сглаженной интерполяцией."""
        del event
        painter = QPainter(self)
        packet = self._pending_packet
        painted = False
        try:
            if not painter.isActive():
                raise RuntimeError("Qt не смог начать отрисовку виджета")
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            if self._image is not None and not self.rect().isEmpty():
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                if self.window().property("scaleMode") == "stretch":
                    target = self.rect()
                else:
                    fitted = self._image.size().scaled(
                        self.size(), Qt.AspectRatioMode.KeepAspectRatio
                    )
                    target = QRect(
                        (self.width() - fitted.width()) // 2,
                        (self.height() - fitted.height()) // 2,
                        fitted.width(),
                        fitted.height(),
                    )
                painter.drawImage(target, self._image)
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


class SystemMetricsMonitor:
    """Собирает общую загрузку CPU и температуру процессора без внешних библиотек."""

    def __init__(self) -> None:
        """Запоминает начальные счётчики CPU для расчёта загрузки между опросами."""
        self._previous_cpu: tuple[int, int] | None = None

    @staticmethod
    def _read_cpu_counters() -> tuple[int, int] | None:
        """Читает суммарные счётчики всех ядер из Linux `/proc/stat`."""
        try:
            first_line = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0]
            fields = first_line.split()
            if not fields or fields[0] != "cpu":
                return None
            counters = [int(value) for value in fields[1:]]
        except (OSError, IndexError, ValueError):
            return None
        if len(counters) < 5:
            return None
        total = sum(counters)
        idle = counters[3] + counters[4]
        return total, idle

    @staticmethod
    def _temperature_value(raw_value: str) -> float | None:
        """Преобразует sysfs температуру из тысячных долей градуса в Celsius."""
        try:
            value = int(raw_value.strip()) / 1000.0
        except ValueError:
            return None
        return value if -20.0 <= value <= 150.0 else None

    @classmethod
    def _read_cpu_temperature(cls) -> float | None:
        """Находит датчик CPU, предпочитая пакетный датчик и CPU-зоны."""
        thermal_root = Path("/sys/class/thermal")
        preferred_types = ("x86_pkg_temp", "TCPU", "cpu-thermal", "cpu_thermal")
        zones: list[tuple[int, Path]] = []
        try:
            for type_path in thermal_root.glob("thermal_zone*/type"):
                try:
                    sensor_type = type_path.read_text(encoding="ascii").strip()
                except OSError:
                    continue
                if sensor_type in preferred_types:
                    zones.append((preferred_types.index(sensor_type), type_path.parent / "temp"))
        except OSError:
            zones = []

        for _, temperature_path in sorted(zones, key=lambda item: item[0]):
            try:
                value = cls._temperature_value(temperature_path.read_text(encoding="ascii"))
            except OSError:
                continue
            if value is not None:
                return value

        # На системах без thermal_zone датчик CPU часто доступен через hwmon.
        hwmon_root = Path("/sys/class/hwmon")
        try:
            for name_path in hwmon_root.glob("hwmon*/name"):
                try:
                    sensor_name = name_path.read_text(encoding="ascii").strip().lower()
                except OSError:
                    continue
                if sensor_name not in {"coretemp", "k10temp", "cpu_thermal", "cpu-thermal"}:
                    continue
                readings = sorted(name_path.parent.glob("temp*_input"))
                for reading_path in readings:
                    try:
                        value = cls._temperature_value(reading_path.read_text(encoding="ascii"))
                    except OSError:
                        continue
                    if value is not None:
                        return value
        except OSError:
            pass
        return None

    def snapshot(self) -> tuple[str, str]:
        """Возвращает текущие CPU и температуру в коротком виде для панели."""
        current_cpu = self._read_cpu_counters()
        cpu_text = "--.-%"
        if current_cpu is not None:
            if self._previous_cpu is not None:
                total_delta = current_cpu[0] - self._previous_cpu[0]
                idle_delta = current_cpu[1] - self._previous_cpu[1]
                if total_delta > 0:
                    busy_percent = max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))
                    cpu_text = f"{busy_percent:.1f}%"
            self._previous_cpu = current_cpu

        temperature = self._read_cpu_temperature()
        temperature_text = "--.-°C" if temperature is None else f"{temperature:.1f}°C"
        return cpu_text, temperature_text


class SystemInfoPanel(QFrame):
    """Показывает заметную нижнюю полосу с телеметрией и счётчиками кадров."""

    def __init__(self) -> None:
        """Создаёт компактные поля CPU, температуры, захвата и показа."""
        super().__init__()
        self.setObjectName("systemInfoPanel")
        self.setFixedHeight(48)
        self.setStyleSheet(
            "QFrame#systemInfoPanel { background: #252a30; border-top: 1px solid #48515b; }"
            "QLabel { color: #e3e8ee; background: transparent; border: 0;"
            " font-family: 'DejaVu Sans Mono', monospace; font-size: 13px;"
            " font-weight: 600; padding: 0 8px; }"
            "QLabel#cpuMetric { color: #69e58b; font-size: 15px; }"
            "QLabel#temperatureMetric { color: #ffd166; font-size: 15px; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(18)

        self.cpu_label = QLabel("CPU --.-%")
        self.cpu_label.setObjectName("cpuMetric")
        self.temperature_label = QLabel("ТЕМП --.-°C")
        self.temperature_label.setObjectName("temperatureMetric")
        self.captured_label = QLabel("ЗАХВАЧЕНО 0")
        self.displayed_label = QLabel("ПОКАЗАНО 0")

        layout.addWidget(self.cpu_label)
        layout.addWidget(self.temperature_label)
        layout.addStretch(1)
        layout.addWidget(self.captured_label)
        layout.addWidget(self.displayed_label)

    def update_values(
        self,
        cpu_text: str,
        temperature_text: str,
        captured: int,
        displayed: int,
    ) -> None:
        """Обновляет системные показатели и счётчики кадров в панели."""
        self.cpu_label.setText(f"CPU {cpu_text}")
        self.temperature_label.setText(f"ТЕМП {temperature_text}")
        self.captured_label.setText(f"ЗАХВАЧЕНО {captured}")
        self.displayed_label.setText(f"ПОКАЗАНО {displayed}")


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
        self.setProperty("scaleMode", session.settings.get("display_scale_mode", "fit"))
        self.viewer = VideoView()
        self.system_metrics = SystemMetricsMonitor()
        self.info_panel = SystemInfoPanel()
        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.viewer, 1)
        central_layout.addWidget(self.info_panel, 0)
        self.setCentralWidget(central_widget)
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
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setInterval(1000)
        self._metrics_timer.timeout.connect(self._update_info_panel)
        self._metrics_timer.start()
        self._update_info_panel()

    def _update_info_panel(self) -> None:
        """Обновляет CPU и температуру раз в секунду, не трогая видеопотоки."""
        cpu_text, temperature_text = self.system_metrics.snapshot()
        self.info_panel.update_values(
            cpu_text,
            temperature_text,
            self.session.frame_count,
            self.session.displayed_count,
        )

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
            message = exit_hint
        if message != self._last_status_message:
            self.statusBar().showMessage(message)
            self._last_status_message = message

    def closeEvent(self, event: Any) -> None:
        """Останавливает конвейер и подтверждает закрытие окна просмотра."""
        self._timer.stop()
        self._metrics_timer.stop()
        self.viewer.abort_pending()
        self._stop_pipeline()
        super().closeEvent(event)


def run_display(session: PipelineSession, display_log: JsonlLog) -> int:
    """Параллельно запускает конвейер и показывает свежие кадры с оверлеем."""
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
