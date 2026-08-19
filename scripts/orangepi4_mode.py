#!/usr/bin/env python3
"""Запускает любую команду в самостоятельном режиме ограниченных ресурсов Orange Pi 4."""

from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Разбирает параметры режима и команду, которую нужно запустить."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-mb", type=int, default=1024, help="Лимит памяти в МБ")
    parser.add_argument("--cores", type=int, default=2, help="Количество медленных ядер")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Команда после разделителя --")
    return parser.parse_args()


def core_frequency(cpu: int) -> int:
    """Возвращает максимальную частоту ядра из данных Linux."""
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq")
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return sys.maxsize


def select_cores(count: int) -> list[int]:
    """Выбирает доступные ядра с наименьшей частотой."""
    available = list(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("Нет доступных процессорных ядер")
    return sorted(available, key=core_frequency)[:max(1, min(count, len(available)))]


def main() -> int:
    """Устанавливает ограничения и запускает дочерний процесс."""
    args = parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command.pop(0)
    if not command:
        raise SystemExit("Укажите команду после --")
    if args.memory_mb < 256:
        print("Предупреждение: 128 МБ недостаточно для запуска Python и YOLO.", flush=True)
    if args.cores < 1 or args.memory_mb < 64:
        raise SystemExit("Укажите положительные значения --cores и --memory-mb")

    cores = select_cores(args.cores)
    os.sched_setaffinity(0, set(cores))
    memory_limit = args.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    print(f"Режим Orange Pi 4: ядра {cores}, память {args.memory_mb} МБ", flush=True)
    os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
