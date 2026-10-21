#!/usr/bin/env python3
"""Устанавливает Video16 в папку, выбранную пользователем.

Пользователь скачивает этот файл вручную. Скрипт получает ``start.py`` из
ветки выпуска, запускает его в режиме установки и пишет весь ход в
``install.log`` рядом со скриптом. Уже подтверждённые payload и зависимости
повторно не скачиваются и не устанавливаются.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


GITHUB_RAW_BASE = "https://github.com/korshun199/VideoT16/raw/refs/heads/win16/win16"
DEFAULT_START_URL = f"{GITHUB_RAW_BASE}/start.py"


class InstallError(RuntimeError):
    """Сообщает ошибку установки без технической трассировки пользователю."""


def configure_console() -> None:
    """Не даёт локальной кодировке консоли Windows оборвать установку."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


class Journal:
    """Печатает этапы установки в консоль и общий install.log."""

    def __init__(self, path: Path) -> None:
        """Создаёт папку программы и открывает UTF-8 журнал."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        """Закрывает журнал после всех этапов."""
        self._file.close()

    def line(self, message: str, state: str = "INFO") -> None:
        """Добавляет датированную строку одновременно в два вывода."""
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{stamp}] [{state}] {message}"
        print(text, flush=True)
        self._file.write(text + "\n")


def install_root() -> Path:
    """Возвращает созданную пользователем папку, где лежит install.py."""
    return Path(__file__).resolve().parent


def sync_start(url: str, destination: Path, journal: Journal) -> None:
    """Скачивает start.py только при изменении файла на GitHub."""
    metadata_path = destination.with_suffix(".http.json")
    headers: dict[str, str] = {}
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("etag"):
                headers["If-None-Match"] = str(metadata["etag"])
            if metadata.get("last_modified"):
                headers["If-Modified-Since"] = str(metadata["last_modified"])
        except (OSError, json.JSONDecodeError):
            pass
    request = urllib.request.Request(url, headers=headers)
    temporary = destination.with_suffix(".py.part")
    temporary.unlink(missing_ok=True)
    journal.line(f"Проверяю start.py в GitHub: {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
            metadata_path.write_text(
                json.dumps(
                    {
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
    except urllib.error.HTTPError as error:
        if error.code == 304 and destination.is_file():
            journal.line("start.py не изменился, использую установленную копию", "SKIP")
            return
        raise InstallError(f"не удалось получить start.py: HTTP {error.code}") from error
    except OSError as error:
        if destination.is_file():
            journal.line("Сеть недоступна, использую ранее сохранённый start.py", "WARN")
            return
        raise InstallError(f"не удалось скачать start.py: {error}") from error
    temporary.replace(destination)
    journal.line("start.py скачан из GitHub", "OK")


def main() -> int:
    """Ставит загрузчик и передаёт ему подготовку зависимостей и payload."""
    configure_console()
    print("=== Video16: установка ===", flush=True)
    if platform.system() != "Windows":
        print("[ОШИБКА] install.py предназначен для Windows", flush=True)
        return 2
    root = install_root()
    journal = Journal(root / "install.log")
    try:
        journal.line("=== Video16: начало установки из GitHub ===")
        journal.line(f"Каталог программы: {root}")
        start_path = root / "start.py"
        sync_start(DEFAULT_START_URL, start_path, journal)
        journal.line("Подготавливаю зависимости и зашифрованный выпуск")
        process = subprocess.run([sys.executable, str(start_path), "--install-only", "--no-pause"])
        if process.returncode != 0:
            raise InstallError(f"подготовка Video16 завершилась с кодом {process.returncode}")
        journal.line("Установка завершена", "OK")
        journal.line(f"Поместите activation.dat в {root} и запустите start.py", "NEXT")
        return 0
    except InstallError as error:
        journal.line(str(error), "ERROR")
        return 1
    except Exception as error:  # noqa: BLE001 -- сохраняем причину для владельца.
        journal.line(f"непредвиденная ошибка: {error}", "ERROR")
        return 1
    finally:
        journal.line("=== Video16: завершение установки ===")
        journal.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if platform.system() == "Windows":
            input("Нажмите Enter, чтобы закрыть окно...")
