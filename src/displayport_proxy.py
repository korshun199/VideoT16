"""Прозрачный двухканальный прокси цифрового OSD по MSP DisplayPort."""

from __future__ import annotations

import signal
import threading
from dataclasses import dataclass


MSP_DISPLAYPORT = 182
MSP_DP_HEARTBEAT = 0
MSP_DP_RELEASE = 1
MSP_DP_CLEAR_SCREEN = 2
MSP_DP_WRITE_STRING = 3
MSP_DP_DRAW_SCREEN = 4
MSP_DP_OPTIONS = 5


@dataclass(frozen=True)
class MspFrame:
    """Проверенный кадр MSPv1 без изменения полезной нагрузки."""

    command: int
    payload: bytes


def build_msp_frame(command: int, payload: bytes = b"") -> bytes:
    """Формирует кадр MSPv1 с XOR-контрольной суммой."""
    if not 0 <= command <= 255 or len(payload) > 255:
        raise ValueError("Команда MSP и размер payload должны быть в диапазоне 0..255")
    checksum = len(payload) ^ command
    for byte in payload:
        checksum ^= byte
    # Команды от полётника к внешнему DisplayPort-устройству идут как $M>.
    return b"$M>" + bytes((len(payload), command)) + payload + bytes((checksum,))


class MspV1Parser:
    """Накапливает произвольные порции UART и выдаёт только проверенные кадры."""

    def __init__(self) -> None:
        """Создаёт пустой буфер протокола."""
        self._buffer = bytearray()

    def feed(self, data: bytes) -> tuple[MspFrame, ...]:
        """Разбирает входные байты и отбрасывает повреждённые кадры."""
        self._buffer.extend(data)
        frames: list[MspFrame] = []
        while True:
            marker = self._buffer.find(b"$M")
            if marker < 0:
                self._buffer.clear()
                break
            if marker:
                del self._buffer[:marker]
            if len(self._buffer) < 6:
                break
            if self._buffer[2] not in (ord("<"), ord(">"), ord("!")):
                del self._buffer[:2]
                continue
            payload_size = self._buffer[3]
            frame_size = payload_size + 6
            if len(self._buffer) < frame_size:
                break
            raw = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            checksum = payload_size ^ raw[4]
            for byte in raw[5:-1]:
                checksum ^= byte
            if checksum == raw[-1] and raw[2] != ord("!"):
                frames.append(MspFrame(raw[4], raw[5:-1]))
        return tuple(frames)


def displayport_write_string(column: int, row: int, text: str, attribute: int = 0) -> bytes:
    """Создаёт цифровую команду DisplayPort для строки до 30 байт."""
    if not 0 <= column <= 255 or not 0 <= row <= 255 or not 0 <= attribute <= 255:
        raise ValueError("Координаты и атрибут DisplayPort должны быть 0..255")
    encoded = text.encode("ascii", "replace")[:30]
    return build_msp_frame(
        MSP_DISPLAYPORT,
        bytes((MSP_DP_WRITE_STRING, row, column, attribute)) + encoded,
    )


def displayport_clear_screen() -> bytes:
    """Создаёт команду очистки цифрового OSD."""
    return build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_CLEAR_SCREEN,)))


def displayport_draw_screen() -> bytes:
    """Создаёт команду вывода накопленного экрана."""
    return build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_DRAW_SCREEN,)))


def displayport_heartbeat() -> bytes:
    """Создаёт heartbeat для поддержания цифрового OSD-сеанса."""
    return build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_HEARTBEAT,)))


def displayport_options(canvas: int = 1) -> bytes:
    """Выбирает сетку DisplayPort: 1 означает HD 50x18."""
    if not 0 <= canvas <= 3:
        raise ValueError("Режим сетки DisplayPort должен быть от 0 до 3")
    return build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_OPTIONS, canvas)))


class DisplayPortProxy:
    """Пересылает MSP между полётником и цифровым видеопередатчиком."""

    def __init__(self, flight_controller_port, video_port) -> None:
        """Принимает два уже открытых pyserial-порта и запускает мост."""
        self._fc = flight_controller_port
        self._video = video_port
        self._write_lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = (
            threading.Thread(target=self._forward, args=(self._fc, self._video), name="osd-fc-to-video", daemon=True),
            threading.Thread(target=self._forward, args=(self._video, self._fc), name="osd-video-to-fc", daemon=True),
        )
        for thread in self._threads:
            thread.start()

    def _forward(self, source, target) -> None:
        """Прозрачно пересылает байты без разбора и изменения OSD."""
        while not self._stop.is_set():
            data = source.read(source.in_waiting or 1)
            if data:
                with self._write_lock:
                    target.write(data)
                    target.flush()

    def send_displayport(self, packet: bytes) -> None:
        """Добавляет проверенный пакет OSD в сторону цифрового видеопередатчика."""
        if not packet.startswith(b"$M>"):
            raise ValueError("Разрешено добавлять только исходящий MSPv1-пакет")
        with self._write_lock:
            self._video.write(packet)
            self._video.flush()

    def close(self) -> None:
        """Останавливает оба направления прокси."""
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)


class DisplayPortOverlay:
    """Добавляет рамку и подпись в цифровую OSD-сетку."""

    def __init__(self, proxy: DisplayPortProxy, columns: int = 50, rows: int = 18) -> None:
        """Создаёт наложение для HD-canvas DisplayPort."""
        if columns < 10 or rows < 6:
            raise ValueError("Canvas DisplayPort слишком мал")
        self._proxy = proxy
        self._columns = columns
        self._rows = rows
        self._last_lines: tuple[tuple[int, int, str], ...] = ()
        self._last_status = ""
        # Инициализируем VTX как HD DisplayPort-устройство до первой рамки.
        self._proxy.send_displayport(displayport_heartbeat())
        self._proxy.send_displayport(displayport_options(1))
        self._proxy.send_displayport(displayport_clear_screen())
        self._proxy.send_displayport(displayport_draw_screen())

    def _send(self, column: int, row: int, text: str) -> None:
        """Отправляет строку в сторону цифрового видеопередатчика."""
        self._proxy.send_displayport(displayport_write_string(column, row, text))

    def clear(self) -> None:
        """Затирает только строки, добавленные нашей рамкой."""
        for column, row, text in self._last_lines:
            self._send(column, row, " " * len(text))
        if self._last_lines:
            self._proxy.send_displayport(displayport_draw_screen())
        self._last_lines = ()

    def update_status(self, text: str) -> None:
        """Выводит системную температуру и нагрузку CPU в нижней строке OSD."""
        clean_text = text.encode("ascii", "replace").decode("ascii")[: self._columns]
        if clean_text == self._last_status:
            return
        if self._last_status:
            self._send(0, self._rows - 1, " " * len(self._last_status))
        if clean_text:
            self._send(0, self._rows - 1, clean_text)
        self._last_status = clean_text
        self._proxy.send_displayport(displayport_draw_screen())

    def update(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> None:
        """Преобразует пиксельную рамку в ASCII-команды DisplayPort."""
        self.clear()
        left = max(0, min(self._columns - 8, round(x1 * self._columns / max(1, width))))
        right = max(left + 6, min(self._columns - 1, round(x2 * self._columns / max(1, width))))
        top = max(1, min(self._rows - 3, round(y1 * self._rows / max(1, height))))
        # Нижняя строка зарезервирована под TEMP/CPU и не пересекается рамкой.
        bottom = max(top + 2, min(self._rows - 2, round(y2 * self._rows / max(1, height))))
        inner = max(2, right - left - 1)
        lines = [(top, "+" + "-" * inner + "+"), (bottom, "+" + "-" * inner + "+")]
        lines.extend((row, "|" + " " * inner + "|") for row in range(top + 1, bottom))
        # Фиксированная подпись цифрового OSD для обнаруженного объекта.
        text = "FPV-DRON"[:inner]
        lines.append((max(0, top - 1), text))
        for row, value in sorted(lines):
            self._send(left, row, value)
        self._last_lines = tuple((left, row, value) for row, value in lines)
        self._proxy.send_displayport(displayport_draw_screen())


def main() -> int:
    """Запускает прозрачный мост между портом полётника и цифровым VTX."""
    import argparse
    import serial

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight-controller", required=True, help="UART стороны полётника")
    parser.add_argument("--video-transmitter", required=True, help="UART стороны цифрового VTX")
    parser.add_argument("--baudrate", type=int, default=115200)
    args = parser.parse_args()
    fc = serial.Serial(args.flight_controller, args.baudrate, timeout=0.05)
    video = serial.Serial(args.video_transmitter, args.baudrate, timeout=0.05)
    proxy = DisplayPortProxy(fc, video)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    print(
        f"Цифровой OSD-прокси: FC={args.flight_controller} VTX={args.video_transmitter}",
        flush=True,
    )
    try:
        while not stop.wait(1.0):
            pass
    finally:
        proxy.close()
        fc.close()
        video.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
