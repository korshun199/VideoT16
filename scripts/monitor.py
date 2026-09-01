#!/usr/bin/env python3
"""Цветной монитор температуры, CPU и памяти Linux."""

from __future__ import annotations

import os
import shutil
import subprocess
import time


# Интервал обновления показателей в секундах.
INTERVAL_SECONDS = 5
# Порог температуры, после которого цвет становится жёлтым.
TEMP_WARNING = 65.0
# Порог температуры, после которого цвет становится красным.
TEMP_CRITICAL = 80.0
# Порог загрузки CPU для жёлтого цвета.
CPU_WARNING = 75.0
# Порог загрузки CPU для красного цвета.
CPU_CRITICAL = 95.0

RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def cpu_times() -> tuple[int, int]:
    """Возвращает суммарное и простоявшее время CPU из /proc/stat."""
    with open("/proc/stat", encoding="ascii") as proc_stat:
        fields = proc_stat.readline().split()
    values = [int(value) for value in fields[1:]
              if value.isdigit()]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def cpu_percent(previous: tuple[int, int]) -> tuple[float, tuple[int, int]]:
    """Считает загрузку CPU между двумя измерениями."""
    current = cpu_times()
    total_delta = current[0] - previous[0]
    idle_delta = current[1] - previous[1]
    if total_delta <= 0:
        return 0.0, current
    return (1 - idle_delta / total_delta) * 100, current


def temperature() -> float | None:
    """Читает максимальную температуру из thermal_zone Linux."""
    values = []
    for index in range(20):
        path = f"/sys/class/thermal/thermal_zone{index}/temp"
        try:
            with open(path, encoding="ascii") as sensor:
                values.append(int(sensor.read().strip()) / 1000)
        except (FileNotFoundError, ValueError, PermissionError, OSError):
            continue
    if values:
        return max(values)
    if shutil.which("vcgencmd"):
        result = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True, text=True, check=False
        )
        try:
            return float(result.stdout.split("=")[1].split("'")[0])
        except (IndexError, ValueError):
            pass
    return None


def memory_percent() -> float:
    """Возвращает процент занятой оперативной памяти."""
    memory = {}
    with open("/proc/meminfo", encoding="ascii") as proc_mem:
        for line in proc_mem:
            name, value, *_ = line.split()
            memory[name.rstrip(":")] = int(value)
    total = memory.get("MemTotal", 0)
    available = memory.get("MemAvailable", memory.get("MemFree", 0))
    return 0.0 if total == 0 else (1 - available / total) * 100


def color(value: float, warning: float, critical: float) -> str:
    """Выбирает цвет показателя по двум порогам."""
    if value >= critical:
        return RED
    if value >= warning:
        return YELLOW
    return GREEN


def main() -> None:
    """Показывает показатели до нажатия Ctrl+C."""
    previous = cpu_times()
    print(f"{CYAN}Монитор системы VideoT16. Ctrl+C — остановка{RESET}")
    try:
        while True:
            time.sleep(INTERVAL_SECONDS)
            load, previous = cpu_percent(previous)
            temp = temperature()
            memory = memory_percent()
            temp_text = "нет данных" if temp is None else f"{temp:.1f}°C"
            temp_color = GREEN if temp is None else color(temp, TEMP_WARNING, TEMP_CRITICAL)
            print(
                f"{time.strftime('%d.%m.%Y %H:%M:%S')} | "
                f"CPU: {color(load, CPU_WARNING, CPU_CRITICAL)}{load:5.1f}%{RESET} | "
                f"Температура: {temp_color}{temp_text}{RESET} | "
                f"RAM: {memory:5.1f}%"
            )
    except KeyboardInterrupt:
        print(f"\n{CYAN}Монитор остановлен.{RESET}")


if __name__ == "__main__":
    main()
