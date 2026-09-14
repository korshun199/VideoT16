#!/usr/bin/env python3
"""Проверяет вывод Raspberry в цифровой VTX без камеры и полётника."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

# При запуске из /dev/shm/scripts добавляем корень расшифрованного проекта.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.displayport_proxy import DisplayPortOverlay, displayport_draw_screen
from src.realtime import SystemStatusMonitor


class VtxOnlyProxy:
    """Адаптер одного UART Ascent для автономного теста OSD."""

    def __init__(self, port: serial.Serial) -> None:
        """Сохраняет открытый последовательный порт VTX."""
        self._port = port

    def send_displayport(self, packet: bytes) -> None:
        """Отправляет готовую команду DisplayPort только в сторону VTX."""
        self._port.write(packet)
        self._port.flush()


def draw_test_screen(overlay: DisplayPortOverlay) -> None:
    """Повторно отправляет тестовую рамку для восстановления после контакта."""
    # Повторная передача всех строк нужна: однопроводной TX не сообщает об обрыве.
    overlay._send(17, 7, "+----------------+")
    for row in range(8, 11):
        overlay._send(17, row, "|                |")
    overlay._send(17, 11, "+----------------+")
    overlay._send(18, 6, "V16 OSD TEST")
    overlay._proxy.send_displayport(displayport_draw_screen())


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры автономного OSD-теста."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtx-port", default="/dev/ttyAMA2", help="UART TX Raspberry к Ascent RX")
    parser.add_argument("--baudrate", type=int, default=115200, help="Скорость UART")
    return parser


def main() -> int:
    """Запускает тестовую рамку и TEMP CPU до нажатия Ctrl+C."""
    args = build_parser().parse_args()
    monitor = SystemStatusMonitor()
    serial_port = serial.Serial(args.vtx_port, args.baudrate, timeout=0.05)
    proxy = VtxOnlyProxy(serial_port)
    overlay = DisplayPortOverlay(proxy)
    try:
        # Фиксированная рамка нужна только для проверки канала Raspberry -> VTX.
        draw_test_screen(overlay)
        print(f"OSD-тест: VTX={args.vtx_port}, 115200. Остановка: Ctrl+C", flush=True)
        last_screen_refresh = 0.0
        while True:
            # Каждую секунду повторяем экран: после переподключения TX Ascent
            # снова получает рамку даже при неизменном статусе CPU.
            now = time.monotonic()
            if now - last_screen_refresh >= 1.0:
                draw_test_screen(overlay)
                last_screen_refresh = now
            overlay.update_status(monitor.text())
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nOSD-тест остановлен", flush=True)
        return 0
    finally:
        serial_port.close()


if __name__ == "__main__":
    raise SystemExit(main())
