"""Определение каталога приложения для исходного и упакованного запуска."""

from __future__ import annotations

from pathlib import Path
import os
import sys


def application_root() -> Path:
    """Возвращает каталог Run.exe либо корень WinT16 при запуске из Python."""
    data_root = os.environ.get("VIDEOT16_APP_ROOT")
    if data_root:
        return Path(data_root).resolve()
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        # При прямом запуске Engine.exe находим корень установки выше каталога версии.
        for candidate in (executable_dir, *executable_dir.parents):
            if (candidate / "current.txt").is_file() and (candidate / "settings.json").is_file():
                return candidate
        return executable_dir
    return Path(__file__).resolve().parents[1]
