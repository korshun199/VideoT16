"""Портативное чтение загрузки CPU и доступной температуры процессора."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass(frozen=True)
class SystemMetrics:
    """Содержит загрузку CPU и температуру; отсутствующий датчик равен None."""

    cpu_percent: float
    temperature_c: float | None


def _cpu_temperature() -> float | None:
    """Ищет CPU-датчик там, где ОС и драйверы разрешают чтение температуры."""
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        # В Windows набор сенсоров зависит от модели и драйверов; ошибка датчика
        # не должна останавливать распознавание или окно.
        return None
    preferred: list[float] = []
    for entries in sensors.values():
        for entry in entries:
            current = entry.current
            if current is None or not -20.0 <= current <= 150.0:
                continue
            label = f"{entry.label} {entry.__class__.__name__}".lower()
            if any(token in label for token in ("cpu", "core", "package", "tdie", "tctl")):
                preferred.append(float(current))
    return max(preferred) if preferred else None


def read_system_metrics() -> SystemMetrics:
    """Считывает мгновенную оценку загрузки CPU и доступную температуру."""
    try:
        cpu = float(psutil.cpu_percent(interval=None))
    except (OSError, RuntimeError):
        cpu = 0.0
    return SystemMetrics(cpu_percent=cpu, temperature_c=_cpu_temperature())
