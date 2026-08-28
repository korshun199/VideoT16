#!/usr/bin/env python3
"""Устанавливает положение штатного прицела через CLI Betaflight."""

from __future__ import annotations

import argparse
import time

from src.betaflight_osd import encode_osd_position


def main() -> int:
    """Входит в CLI, меняет координаты и выходит без команды save."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB-порт полётника")
    parser.add_argument("--x", type=int, default=5, help="Колонка OSD: 0..30")
    parser.add_argument("--y", type=int, default=3, help="Строка OSD: 0..15")
    parser.add_argument("--save", action="store_true", help="Сохранить настройки и перезапустить Betaflight")
    args = parser.parse_args()

    try:
        import serial

        position = encode_osd_position(args.x, args.y)
        with serial.Serial(args.port, baudrate=115200, timeout=0.2) as serial_port:
            serial_port.write(b"#\r\n")
            serial_port.flush()
            time.sleep(0.4)
            serial_port.read(4096)
            serial_port.write(f"set osd_crosshairs_pos = {position}\r\n".encode())
            serial_port.flush()
            time.sleep(0.4)
            response = serial_port.read(4096).decode(errors="replace")
            serial_port.write(("save\r\n" if args.save else "exit\r\n").encode())
            serial_port.flush()
            if args.save:
                time.sleep(1.0)
    except (ModuleNotFoundError, OSError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    print(f"Betaflight CLI: X={args.x}, Y={args.y}, значение={position}")
    if "Invalid name" in response or "ERROR" in response:
        print(f"Ответ полётника: {response.strip()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
