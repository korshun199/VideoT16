#!/usr/bin/env python3
"""Краткий тест четырёх моторов по шкале 0–10 через CLI Betaflight."""

from __future__ import annotations

import argparse
import time

from src.motor import run_motor, stop_motors


# USB-порт полётного контроллера.
PORT = "/dev/ttyACM0"
# Максимальная длительность теста моторов.
DURATION_SECONDS = 2.0
# Реальный запуск разрешается только после ручного изменения на True.
RUN_ENABLED = True
# Подтверждение: все пропеллеры сняты, аппарат закреплён.
NO_PROPS_CONFIRMED = True
# Показывать ответы CLI для диагностики обмена.
SHOW_RESPONSES = True


def read_prompt(port):
    """Читает ответ Betaflight до приглашения CLI."""
    response = port.read_until(b"# ", size=512)
    if SHOW_RESPONSES:
        print(f"CLI: {response!r}")
    if b"###ERROR" in response:
        raise RuntimeError(f"Betaflight CLI отклонил команду: {response.decode(errors='replace').strip()}")
    if b"#" not in response:
        raise RuntimeError("Betaflight CLI не ответил приглашением")
    return response


def parse_args() -> argparse.Namespace:
    """Разбирает только четыре уровня моторов 0–10."""
    parser = argparse.ArgumentParser(description="Тест четырёх моторов Betaflight по шкале 0–10")
    parser.add_argument("levels", nargs=4, type=int, metavar="MOTOR", help="Уровни четырёх моторов")
    return parser.parse_args()


def main() -> int:
    """Показывает команду или выполняет ограниченный тест с автоостановкой."""
    args = parse_args()
    levels = tuple(args.levels)
    if not 0 < DURATION_SECONDS <= 3:
        raise SystemExit("DURATION_SECONDS должна быть больше 0 и не больше 3 секунд")
    print(f"Моторы: {levels}; длительность: {DURATION_SECONDS:.1f} с")
    if not RUN_ENABLED:
        print("DRY-RUN: измените RUN_ENABLED = True в начале файла для запуска")
        return 0
    if not NO_PROPS_CONFIRMED:
        raise SystemExit("Сначала измените NO_PROPS_CONFIRMED = True после снятия пропеллеров")

    import serial

    with serial.Serial(PORT, baudrate=115200, timeout=0.2) as port:
        try:
            mt =1
            # Ждём настоящего приглашения CLI, а не угадываем задержку.
            port.reset_input_buffer()
            port.write(b"#\r\n")
            read_prompt(port)
            command = run_motor(port, mt , 20)
            print(f"Отправка: {command!r}")
            read_prompt(port)
            deadline = time.monotonic() + DURATION_SECONDS
            while time.monotonic() < deadline:
                time.sleep(0.1)
            port.write(b"exit\r\n")
            port.flush()
        finally:
            stop_motors(port)
    print("Тест завершён: отправлена команда остановки всех моторов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
