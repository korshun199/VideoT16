#!/usr/bin/env python3
"""Загружает тёмный квадратный указатель в шрифт аналогового OSD."""

from __future__ import annotations

from src.osd import upload_box_glyphs


PORT = "/dev/ttyACM0"  # USB-порт полётника


def main() -> int:
    """Открывает MSP-порт и записывает символы в память MAX7456."""
    try:
        import serial

        with serial.Serial(PORT, 115200, timeout=0.2) as serial_port:
            upload_box_glyphs(serial_port)
    except (ModuleNotFoundError, OSError) as error:
        print(f"Ошибка: {error}")
        return 1

    print("Загружены символы тёмной квадратной рамки OSD: 0x72, 0x73, 0x74")
    print("Символы записаны в NVM MAX7456; стандартный шрифт можно вернуть через Configurator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
