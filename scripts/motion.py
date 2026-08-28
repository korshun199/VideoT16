#!/usr/bin/env python3
from __future__ import annotations

import time,sys

from src.osd import set_crosshairs_position


# ================= НАСТРОЙКИ ПРОГУЛКИ =================
PORT = "/dev/ttyACM0"  # USB-порт полётника
INTERVAL_SECONDS = 1  # Пауза между точками в секундах
LOOPS = 0  # Число повторов: 0 — бесконечно
VISIBLE = True  
# =====================================================
POINTS = [[sys.argv[1], sys.argv[2],]]

def validate_config() -> list[tuple[int, int]]:
    """Проверяет настройки и возвращает координаты в удобном формате."""
    if INTERVAL_SECONDS <= 0 or LOOPS < 0 or not POINTS:
        raise ValueError("Нужны interval > 0, loops >= 0 и непустой список points")
    return [(int(point[0]), int(point[1])) for point in POINTS]


def move_crosshairs(
    serial_port,
    points: list[tuple[int, int]],
    interval_seconds: float,
    loops: int,
    visible: bool = True,
) -> None:
    """Перемещает указатель по точкам; функцию можно вызвать из детектора."""
    cycle = 0
    while loops == 0 or cycle < loops:
        for x, y in points:
            set_crosshairs_position(serial_port, x, y, visible)
            print(f"Указатель: X={x}, Y={y}", flush=True)
            time.sleep(interval_seconds)
        cycle += 1


def main() -> int:
    """Открывает порт и циклически перемещает указатель без сохранения во flash."""
    try:
        import serial

        points = validate_config()
        with serial.Serial(PORT, 115200, timeout=0.2) as port:
            move_crosshairs(port, points, INTERVAL_SECONDS, LOOPS, VISIBLE)
    except KeyboardInterrupt:
        print("\nПрогулка остановлена")
    except (ModuleNotFoundError, OSError, ValueError) as error:
        print(f"Ошибка настроек или порта: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
