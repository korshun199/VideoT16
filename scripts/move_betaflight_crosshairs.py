#!/usr/bin/env python3
"""Перемещает штатный указатель Betaflight по заданным координатам."""

from __future__ import annotations

import argparse
import time


def parse_points(value: str) -> list[tuple[int, int]]:
    """Преобразует строку X,Y;X,Y в список координат OSD."""
    points = []
    for item in value.split(";"):
        try:
            x_text, y_text = item.split(",", 1)
            points.append((int(x_text), int(y_text)))
        except ValueError as error:
            raise ValueError(f"Неверная точка '{item}', нужен формат X,Y") from error
    if not points:
        raise ValueError("Список координат пуст")
    return points


def main() -> int:
    """Открывает MSP-порт и циклически отправляет координаты указателя."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB-порт полётника")
    parser.add_argument(
        "--points",
        default="14,8;5,3;25,3;25,12;5,12;14,8",
        help="Точки X,Y через точку с запятой",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Пауза между точками")
    parser.add_argument("--loops", type=int, default=0, help="Число циклов, 0 — бесконечно")
    args = parser.parse_args()

    if args.interval <= 0 or args.loops < 0:
        print("Ошибка: интервал должен быть больше нуля, циклы — неотрицательными")
        return 1

    try:
        import serial
        from src.betaflight_osd import read_msp_response, set_crosshairs_position

        points = parse_points(args.points)
        with serial.Serial(args.port, baudrate=115200, timeout=0.2) as serial_port:
            cycle = 0
            while args.loops == 0 or cycle < args.loops:
                for x, y in points:
                    set_crosshairs_position(serial_port, x, y)
                    response = read_msp_response(serial_port)
                    if response and response.startswith(b"$M!"):
                        print(f"Ответ Betaflight: ошибка MSP ({response.hex()})", flush=True)
                    elif not response:
                        print("Ответ Betaflight: нет ответа", flush=True)
                    else:
                        print(f"Ответ Betaflight: {response.hex()}", flush=True)
                    print(f"Указатель: X={x}, Y={y}", flush=True)
                    time.sleep(args.interval)
                cycle += 1
    except KeyboardInterrupt:
        print("\nПеремещение остановлено")
    except (ModuleNotFoundError, OSError, ValueError) as error:
        print(f"Ошибка: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
