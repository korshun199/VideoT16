#!/usr/bin/env python3
"""Загружает пользовательский указатель после появления полётника."""

from __future__ import annotations

import time
from pathlib import Path

from src.osd import upload_box_glyphs


# ================= НАСТРОЙКИ =================
PORT = "/dev/ttyACM0"  # Порт полётника по USB
WAIT_SECONDS = 30  # Максимальное ожидание подключения
POLL_SECONDS = 1  # Интервал проверки порта
# =============================================


def wait_for_port(port_name: str) -> None:
    """Ожидает появления последовательного порта полётника."""
    deadline = time.monotonic() + WAIT_SECONDS
    while not Path(port_name).exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Порт не появился за {WAIT_SECONDS} секунд: {port_name}")
        print(f"Ожидание полётника: {port_name}", flush=True)
        time.sleep(POLL_SECONDS)


def main() -> int:
    """Ждёт полётник и один раз загружает квадратный указатель."""
    try:
        import serial

        wait_for_port(PORT)
        with serial.Serial(PORT, 115200, timeout=0.2) as serial_port:
            upload_box_glyphs(serial_port)
    except (ModuleNotFoundError, OSError, TimeoutError) as error:
        print(f"Ошибка автозагрузки указателя: {error}")
        return 1

    print("Квадратный указатель загружен после подключения полётника")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
