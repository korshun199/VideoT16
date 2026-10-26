#!/usr/bin/env python3
"""Простой GTK/GStreamer-плеер для точной пошаговой проверки видеозаписей."""

from __future__ import annotations

import math
import sys
from pathlib import Path


# Наносекунд в секунде по шкале времени GStreamer.
NANOSECONDS_PER_SECOND = 1_000_000_000


def stepped_position_ns(
    current_ns: int,
    duration_ns: int | None,
    direction: int,
) -> int:
    """Возвращает позицию после шага на секунду с ограничением границ видео."""
    if direction not in (-1, 1):
        raise ValueError("Направление шага должно быть -1 или 1")
    target_ns = max(0, current_ns + direction * NANOSECONDS_PER_SECOND)
    if duration_ns is not None and duration_ns >= 0:
        target_ns = min(target_ns, duration_ns)
    return target_ns


def format_time_ns(position_ns: int | None) -> str:
    """Форматирует время дорожки как ММ:СС либо ЧЧ:ММ:СС."""
    if position_ns is None or position_ns < 0:
        return "--:--"
    total_seconds = position_ns // NANOSECONDS_PER_SECOND
    seconds = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def timeline_position_ns(fraction: float, duration_ns: int | None) -> int | None:
    """Преобразует положение шкалы от 0 до 1 во время внутри видео."""
    if duration_ns is None or duration_ns <= 0:
        return None
    if not math.isfinite(fraction):
        raise ValueError("Положение шкалы должно быть конечным числом")
    bounded_fraction = min(1.0, max(0.0, fraction))
    return round(bounded_fraction * duration_ns)


def main(argv: list[str] | None = None) -> int:
    """Запускает оконный плеер для одного видеофайла."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("Использование: step_video_player.py ФАЙЛ_ВИДЕО", file=sys.stderr)
        return 2

    video_path = Path(arguments[0]).expanduser().resolve()
    if not video_path.is_file():
        print(f"Ошибка: видеофайл не найден: {video_path}", file=sys.stderr)
        return 2

    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Gst", "1.0")
        from gi.repository import Gio, GLib, Gst, Gtk
    except (ImportError, ValueError) as error:
        print(f"Ошибка: не удалось загрузить GTK4/GStreamer: {error}", file=sys.stderr)
        return 3

    Gst.init(None)
    if Gst.ElementFactory.find("playbin") is None:
        print("Ошибка: в GStreamer не найден элемент playbin", file=sys.stderr)
        return 3
    if Gst.ElementFactory.find("gtk4paintablesink") is None:
        print("Ошибка: в GStreamer не найден видеосинк gtk4paintablesink", file=sys.stderr)
        return 3

    class StepPlayerWindow:
        """Создаёт окно, кнопки управления и GStreamer-конвейер воспроизведения."""

        def __init__(self, application) -> None:
            """Готовит окно для выбранного файла и ставит видео на паузу."""
            self._window = Gtk.ApplicationWindow(application=application)
            self._window.set_title("VideoT16 — видеоплеер")
            self._window.set_default_size(1600, 900)
            # Окно сразу занимает максимум места под видеокадр; его можно свернуть вручную.
            self._window.maximize()
            # Запоминаем активную перемотку и отличаем её от обновления шкалы плеером.
            self._seek_timeout_id = None
            self._seek_pending = False
            self._updating_seek_scale = False

            # Создаём видеовыход GTK до конвейера, чтобы сразу показать первый кадр.
            self._video_sink = Gst.ElementFactory.make("gtk4paintablesink", "video-output")
            self._playbin = Gst.ElementFactory.make("playbin", "video-player")
            if self._video_sink is None or self._playbin is None:
                raise RuntimeError("GStreamer не смог создать элементы видеоплеера")
            self._playbin.set_property("video-sink", self._video_sink)
            self._playbin.set_property("uri", video_path.as_uri())

            # Оставляем почти всю высоту окна видео, а управление собираем в одну строку.
            root_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            root_box.set_margin_top(6)
            root_box.set_margin_bottom(6)
            root_box.set_margin_start(8)
            root_box.set_margin_end(8)
            self._window.set_child(root_box)

            video_picture = Gtk.Picture.new_for_paintable(
                self._video_sink.get_property("paintable")
            )
            video_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
            video_picture.set_can_shrink(True)
            video_picture.set_hexpand(True)
            video_picture.set_vexpand(True)
            root_box.append(video_picture)

            control_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            control_row.set_valign(Gtk.Align.CENTER)
            root_box.append(control_row)

            # Подписываем четыре кнопки русскими короткими командами.
            controls = (
                ("Пуск", self._play),
                ("Стоп", self._pause),
                ("Назад −1 с", lambda _button: self._step(-1)),
                ("Вперёд +1 с", lambda _button: self._step(1)),
            )
            for label, callback in controls:
                button = Gtk.Button(label=label)
                button.set_size_request(112, 44)
                button.connect("clicked", callback)
                control_row.append(button)

            # Текущее время, шкала перемотки и длительность стоят рядом с кнопками.
            self._position_label = Gtk.Label(label="00:00")
            self._position_label.set_width_chars(8)
            self._position_label.set_xalign(0.5)
            control_row.append(self._position_label)

            self._seek_scale = Gtk.Scale.new_with_range(
                Gtk.Orientation.HORIZONTAL,
                0.0,
                1.0,
                0.001,
            )
            self._seek_scale.set_draw_value(False)
            self._seek_scale.set_hexpand(True)
            self._seek_scale.set_size_request(240, -1)
            self._seek_scale.set_sensitive(False)
            self._seek_scale.set_tooltip_text("Нажми или перетащи ползунок для перехода по видео")
            self._seek_scale.connect("value-changed", self._on_seek_value_changed)
            control_row.append(self._seek_scale)

            self._duration_label = Gtk.Label(label="--:--")
            self._duration_label.set_width_chars(8)
            self._duration_label.set_xalign(0.5)
            control_row.append(self._duration_label)

            # Подключаем события GStreamer и обновление часов.
            self._bus = self._playbin.get_bus()
            self._bus.add_signal_watch()
            self._bus.connect("message", self._on_bus_message)
            self._window.connect("close-request", self._on_close_request)
            GLib.timeout_add(200, self._refresh_position)

            # Открываем файл на паузе, чтобы пилот сам выбрал момент старта.
            state_result = self._playbin.set_state(Gst.State.PAUSED)
            if state_result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer не смог открыть этот видеофайл")

        def _play(self, _button) -> None:
            """Запускает непрерывное воспроизведение или начинает ролик сначала."""
            self._window.set_title("VideoT16 — видеоплеер")
            position_ok, position = self._playbin.query_position(Gst.Format.TIME)
            duration_ok, duration = self._playbin.query_duration(Gst.Format.TIME)
            if position_ok and duration_ok and duration > 0 and position >= duration:
                self._playbin.seek_simple(
                    Gst.Format.TIME,
                    Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE,
                    0,
                )
            result = self._playbin.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                self._report_error("Не удалось запустить воспроизведение")

        def _pause(self, _button) -> None:
            """Ставит ролик на паузу, сохраняя текущий кадр и позицию."""
            self._playbin.set_state(Gst.State.PAUSED)

        def _step(self, direction: int) -> None:
            """Ставит видео на паузу и точно перемещается на одну секунду."""
            self._cancel_pending_timeline_seek()
            self._playbin.set_state(Gst.State.PAUSED)
            position_ok, position = self._playbin.query_position(Gst.Format.TIME)
            if not position_ok:
                self._report_error("Плеер ещё определяет позицию видео")
                return
            duration_ok, duration = self._playbin.query_duration(Gst.Format.TIME)
            target = stepped_position_ns(
                position,
                duration if duration_ok else None,
                direction,
            )
            seek_flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE
            if not self._playbin.seek_simple(Gst.Format.TIME, seek_flags, target):
                self._report_error("Формат не поддержал точный переход")

        def _on_seek_value_changed(self, _scale) -> None:
            """Отправляет запрос перемотки после короткой паузы в движении шкалы."""
            if self._updating_seek_scale:
                return
            self._seek_pending = True
            if self._seek_timeout_id is not None:
                GLib.source_remove(self._seek_timeout_id)
            self._seek_timeout_id = GLib.timeout_add(120, self._commit_timeline_seek)

        def _commit_timeline_seek(self) -> bool:
            """Перемещает декодер в выбранную точку и сохраняет режим паузы/пуска."""
            self._seek_timeout_id = None
            self._seek_pending = False
            duration_ok, duration = self._playbin.query_duration(Gst.Format.TIME)
            target = timeline_position_ns(
                self._seek_scale.get_value(),
                duration if duration_ok else None,
            )
            if target is None:
                self._report_error("Длительность видео пока неизвестна")
                return False
            seek_flags = Gst.SeekFlags.FLUSH | Gst.SeekFlags.ACCURATE
            if not self._playbin.seek_simple(Gst.Format.TIME, seek_flags, target):
                self._report_error("Не удалось перейти к выбранному времени")

        def _cancel_pending_timeline_seek(self) -> None:
            """Отменяет отложенную перемотку, чтобы она не перебила шаг кнопкой."""
            if self._seek_timeout_id is not None:
                GLib.source_remove(self._seek_timeout_id)
                self._seek_timeout_id = None
            self._seek_pending = False
            return False

        def _report_error(self, message: str) -> None:
            """Показывает состояние в заголовке окна и пишет подробность в журнал."""
            self._window.set_title("VideoT16 — ошибка воспроизведения")
            print(f"Ошибка плеера: {message}", file=sys.stderr, flush=True)

        def _refresh_position(self) -> bool:
            """Обновляет отображаемую позицию и длительность ролика."""
            position_ok, position = self._playbin.query_position(Gst.Format.TIME)
            duration_ok, duration = self._playbin.query_duration(Gst.Format.TIME)
            current = format_time_ns(position if position_ok else None)
            total = format_time_ns(duration if duration_ok else None)
            self._position_label.set_text(current)
            self._duration_label.set_text(total)
            self._seek_scale.set_sensitive(bool(duration_ok and duration > 0))
            if duration_ok and duration > 0 and position_ok and not self._seek_pending:
                self._updating_seek_scale = True
                self._seek_scale.set_value(min(1.0, max(0.0, position / duration)))
                self._updating_seek_scale = False
            return True

        def _on_bus_message(self, _bus, message) -> None:
            """Обрабатывает ошибки декодирования и завершение воспроизведения."""
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                self._playbin.set_state(Gst.State.PAUSED)
                self._report_error(f"Видео: {error.message}")
                if debug:
                    print(f"GStreamer: {debug}", file=sys.stderr, flush=True)
            elif message.type == Gst.MessageType.EOS:
                self._playbin.set_state(Gst.State.PAUSED)
                self._window.set_title("VideoT16 — видео завершено")

        def _on_close_request(self, *_arguments) -> bool:
            """Останавливает декодер и освобождает файл при закрытии окна."""
            self._cancel_pending_timeline_seek()
            self._playbin.set_state(Gst.State.NULL)
            self._bus.remove_signal_watch()
            return False

        def show(self) -> None:
            """Показывает окно плеера на рабочем столе."""
            self._window.present()

    # Создаём отдельный экземпляр приложения, не перехватывая чужие GTK-окна.
    app = Gtk.Application(
        application_id="ru.videot16.StepPlayer",
        flags=Gio.ApplicationFlags.NON_UNIQUE,
    )
    player_windows = []

    def activate(application) -> None:
        """Создаёт и показывает единственное окно выбранного видео."""
        try:
            player = StepPlayerWindow(application)
            player_windows.append(player)
            player.show()
        except Exception as error:
            print(f"Ошибка создания окна плеера: {error}", file=sys.stderr, flush=True)
            application.quit()

    app.connect("activate", activate)
    try:
        return int(app.run(None))
    except Exception as error:
        print(f"Ошибка запуска плеера: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
