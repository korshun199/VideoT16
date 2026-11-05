"""Запускает выбранную версию приложения с общими пользовательскими файлами."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox


def main() -> int:
    """Проверяет установленную версию и запускает её без окна консоли."""
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    try:
        version = (root / "current.txt").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", version):
            raise ValueError("Некорректная запись установленной версии")
        executable = root / "versions" / version / "Engine.exe"
        if not executable.is_file():
            raise FileNotFoundError(f"Файл версии не найден: {executable}")
        environment = os.environ.copy()
        environment["VIDEOT16_APP_ROOT"] = str(root)
        environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
        return subprocess.call([str(executable)], cwd=str(root), env=environment)
    except (OSError, ValueError) as error:
        window = tk.Tk()
        window.withdraw()
        messagebox.showerror("VideoT16", f"Не удалось запустить программу:\n{error}", parent=window)
        window.destroy()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
