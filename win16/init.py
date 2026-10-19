#!/usr/bin/env python3
"""Создаёт одноразовый паспорт Windows-компьютера для Video16.

Пользователь запускает этот файл один раз и передаёт владельцу только
получившийся ``init.dat``. После выпуска ``activation.dat`` файл init.dat
можно удалить: при каждом запуске Video16 считывает оборудование заново.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path


class InitError(RuntimeError):
    """Сообщает понятную пользователю ошибку создания паспорта."""


def configure_console() -> None:
    """Не даёт локальной кодировке консоли Windows оборвать работу скрипта."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, OSError):
            pass


def read_identity() -> dict[str, object]:
    """Считывает UUID, BIOS и наличие TPM штатным PowerShell Windows."""
    script = r"""
$computer = Get-CimInstance Win32_ComputerSystemProduct | Select-Object -First 1 UUID,Name
$bios = Get-CimInstance Win32_BIOS | Select-Object -First 1 SerialNumber
$tpm = Get-Tpm | Select-Object -First 1 TpmPresent,ManagedAuthLevel
[pscustomobject]@{ computer = $computer; bios = $bios; tpm = $tpm } |
    ConvertTo-Json -Compress -Depth 4
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def device_key(identity: dict[str, object]) -> str:
    """Проверяет обязательные поля и строит стабильный отпечаток железа."""
    computer = identity.get("computer") or {}
    bios = identity.get("bios") or {}
    tpm = identity.get("tpm") or {}
    values = (
        str(computer.get("UUID", "")).strip().upper(),
        str(bios.get("SerialNumber", "")).strip().upper(),
        str(tpm.get("TpmPresent", "")).strip().lower(),
    )
    if not values[0] or not values[1]:
        raise InitError("Windows не вернула UUID или серийный номер BIOS")
    return "|".join(values)


def create_init(destination: Path) -> Path:
    """Записывает неперсональный для Git паспорт в явно заданный файл."""
    identity = read_identity()
    device_key(identity)
    document = {"format": "Video16 Windows init v2", "identity": identity}
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    """Создаёт init.dat рядом со скачанным init.py и показывает следующий шаг."""
    configure_console()
    print("=== Video16: паспорт устройства ===", flush=True)
    if platform.system() != "Windows":
        print("[ОШИБКА] init.py предназначен для Windows", flush=True)
        return 2
    destination = Path(__file__).resolve().with_name("init.dat")
    try:
        create_init(destination)
    except Exception as error:  # noqa: BLE001 -- пользователю нужен текст реальной причины.
        print(f"[ОШИБКА] Не удалось создать init.dat: {error}", flush=True)
        return 1
    print(f"[OK] Создан файл: {destination}", flush=True)
    print("Передайте владельцу только init.dat для выдачи activation.dat.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if platform.system() == "Windows":
            input("Нажмите Enter, чтобы закрыть окно...")
