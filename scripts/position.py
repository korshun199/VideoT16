#!/usr/bin/env python3
"""Устанавливает позицию штатного прицела Betaflight через USB MSP."""

from __future__ import annotations

import argparse


def load_serial():
    """Загружает pyserial с понятной ошибкой для оператора."""
    try:
        import serial
    except ModuleNotFoundError as error:
        raise RuntimeError("Не установлен pyserial: установите пакет в .venv") from error
    return serial


def main() -> int:
    """Читает координаты, отправляет команду и завершает работу."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB-порт полётника")
    parser.add_argument("--x", type=int, default=14, help="Колонка OSD: 0..30")
    parser.add_argument("--y", type=int, default=8, help="Строка OSD: 0..15")
    parser.add_argument("--hide", action="store_true", help="Скрыть прицел")
    args = parser.parse_args()

    try:
        serial = load_serial()
        from src.osd import set_crosshairs_position

        with serial.Serial(args.port, baudrate=115200, timeout=0.2) as serial_port:
            set_crosshairs_position(serial_port, args.x, args.y, not args.hide)
    except (RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1
    except serial.SerialException as error:
        print(f"Ошибка порта {args.port}: {error}")
        return 1

    state = "видим" if not args.hide else "скрыт"
    print(f"Прицел установлен: X={args.x}, Y={args.y}, {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
