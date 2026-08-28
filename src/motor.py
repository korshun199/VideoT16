"""Формирование безопасных CLI-команд теста моторов Betaflight."""

from __future__ import annotations

MOTOR_COUNT = 4
PWM_MIN = 1000
PWM_MAX = 2000


def level_to_pwm(level: int) -> int:
    """Переводит уровень 0–10 в тестовый импульс 1000–2000 мкс."""
    if not 0 <= level <= 10:
        raise ValueError("Уровень мотора должен быть от 0 до 10")
    return PWM_MIN + level * 100


def build_motor_commands(levels: tuple[int, int, int, int]) -> tuple[bytes, ...]:
    """Формирует CLI-команды motor для четырёх моторов."""
    if len(levels) != MOTOR_COUNT:
        raise ValueError("Нужно указать ровно четыре уровня моторов")
    values = tuple(level_to_pwm(level) for level in levels)
    return tuple(f"motor {index} {value}\r\n".encode() for index, value in enumerate(values))


def run_motor(serial_port, motor_number: int, level: int) -> bytes:
    """Запускает один мотор по индексу Betaflight 0–3 и уровню 0–10."""
    if not 0 <= motor_number < MOTOR_COUNT:
        raise ValueError(f"Номер мотора должен быть от 0 до {MOTOR_COUNT - 1}")
    command = f"motor {motor_number} {level_to_pwm(level)}\r\n".encode()
    serial_port.write(command)
    serial_port.flush()
    return command


def stop_motors(serial_port) -> None:
    """Отправляет CLI-команды остановки всех моторов."""
    for command in build_motor_commands((0, 0, 0, 0)):
        serial_port.write(command)
    serial_port.flush()
