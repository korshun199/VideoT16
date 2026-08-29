#!/usr/bin/env python3
"""Краткий тест четырёх моторов по шкале 0–10 через CLI Betaflight."""

from __future__ import annotations

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
# Уровни четырёх моторов: индексы Betaflight 0, 1, 2, 3.
MOTOR_LEVELS = (1, 1, 3, 1)


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


def main() -> int:
    """Показывает команду или выполняет ограниченный тест с автоостановкой."""
    levels = MOTOR_LEVELS
    if len(levels) != 4 or any(not 0 <= level <= 10 for level in levels):
        raise SystemExit("MOTOR_LEVELS должен содержать четыре уровня от 0 до 10")
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
            # Ждём настоящего приглашения CLI, а не угадываем задержку.
            port.reset_input_buffer()
            port.write(b"#\r\n")
            read_prompt(port)

            mt = 0
            while mt in range(0, 4):
                command = run_motor(port, mt, levels[mt])
                print(f"Отправка: {command!r}")
                read_prompt(port)
                deadline = time.monotonic() + DURATION_SECONDS
                while time.monotonic() < deadline:
                    time.sleep(0.1)
                stop_motors(port)
                # Забираем ответы на четыре команды остановки перед следующим мотором.
                for _ in range(4):
                    read_prompt(port)
                mt += 1
            port.write(b"exit\r\n")
            port.flush()
        finally:
            stop_motors(port)
    print("Тест завершён: моторы проверены по очереди, все остановлены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
