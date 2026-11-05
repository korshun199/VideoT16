"""Офлайн-установка WinT16 в отдельный пользовательский каталог."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, messagebox


def validate_payload(payload: Path) -> None:
    """Проверяет состав комплекта и не допускает персональные ключи и открытые модели."""
    allowed = {"Run.exe", "Update.exe", "logo.png", "settings.json", "current.txt", "models", "versions"}
    if not payload.is_dir() or {item.name for item in payload.iterdir()} != allowed:
        raise ValueError("Комплект установки имеет неожиданный состав")
    for name in ("Run.exe", "Update.exe", "logo.png", "settings.json", "current.txt"):
        if not (payload / name).is_file():
            raise ValueError(f"В комплекте отсутствует файл {name}")
    version = (payload / "current.txt").read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", version):
        raise ValueError("Неверная версия приложения")
    versions = payload / "versions"
    if not versions.is_dir() or {item.name for item in versions.iterdir()} != {version}:
        raise ValueError("Комплект должен содержать только одну версию")
    if not (versions / version / "Engine.exe").is_file():
        raise ValueError("В комплекте отсутствует программа распознавания")
    models = payload / "models"
    if not models.is_dir() or not any(item.is_file() and item.suffix.lower() == ".v16m" for item in models.iterdir()):
        raise ValueError("В комплекте отсутствует зашифрованная модель")
    try:
        settings = json.loads((payload / "settings.json").read_text(encoding="utf-8"))
        model_name = settings["model_path"]
        if not isinstance(model_name, str):
            raise ValueError("Неверный путь модели")
        model_path = (payload / model_name).resolve()
        if (payload.resolve() not in model_path.parents or model_path.suffix.casefold() != ".v16m"
                or not model_path.is_file()):
            raise ValueError("Настройки указывают на отсутствующую зашифрованную модель")
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Не удалось проверить настройки модели в комплекте") from error
    forbidden = ("auth.dat", "auth.previous.", "auth.incoming.", "init.dat", "init.previous.",
                 "device.key", "owner_keys", "model_aes.key", "activation_signing_private")
    for path in payload.rglob("*"):
        name = path.name.casefold()
        if (path.is_symlink() or any(name == item or name.startswith(item) for item in forbidden)
                or path.suffix.casefold() in (".onnx", ".pt")):
            raise ValueError(f"Запрещённый файл в комплекте: {path.name}")


def install_payload(payload: Path, destination: Path) -> None:
    """Копирует проверенный комплект в новый каталог, не заменяя существующие файлы."""
    validate_payload(payload)
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Папка уже существует: {destination}. Выберите другую папку.")
    if destination == payload.resolve() or payload.resolve() in destination.parents:
        raise ValueError("Нельзя устанавливать программу внутрь установочного комплекта")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(payload, destination, dirs_exist_ok=False)
    validate_payload(destination)


def create_shortcut(destination: Path) -> Path:
    """Создаёт ярлык рабочего стола через штатный Windows COM, не заменяя имеющийся."""
    if os.name != "nt":
        raise OSError("Ярлык рабочего стола поддерживается только в Windows")
    target = str(destination / "Run.exe").replace("'", "''")
    workdir = str(destination).replace("'", "''")
    script = (
        "$ErrorActionPreference='Stop';"
        "$desktop=[Environment]::GetFolderPath('Desktop');"
        "$path=Join-Path $desktop 'VideoT16.lnk';"
        "if(Test-Path -LiteralPath $path){$path=Join-Path $desktop ('VideoT16-' + "
        "[DateTime]::Now.ToString('yyyyMMdd-HHmmss-ffff') + '.lnk')};"
        "$shell=New-Object -ComObject WScript.Shell;"
        "$link=$shell.CreateShortcut($path);"
        f"$link.TargetPath='{target}';$link.WorkingDirectory='{workdir}';"
        "$link.Description='VideoT16';$link.Save();"
        "[Console]::Out.Write($path)"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                            capture_output=True, text=True, timeout=30,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode != 0 or not result.stdout.strip():
        raise OSError("Не удалось создать ярлык на рабочем столе")
    return Path(result.stdout.strip())


def main() -> int:
    """Показывает окно установки либо запускает проверку с явно заданной папкой."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-to", type=Path)
    parser.add_argument("--no-shortcut", action="store_true")
    arguments = parser.parse_args()
    payload = (Path(getattr(sys, "_MEIPASS")) / "payload" if getattr(sys, "frozen", False)
               else Path(__file__).resolve().parent / "payload")
    if arguments.install_to:
        install_payload(payload, arguments.install_to)
        if not arguments.no_shortcut:
            create_shortcut(arguments.install_to.resolve())
        print(json.dumps({"installed": str(arguments.install_to.resolve())}, ensure_ascii=True))
        return 0

    window = tk.Tk()
    window.title("Установка VideoT16")
    window.resizable(False, False)
    location = tk.StringVar(value=str(Path.home() / "VideoT16"))
    status = tk.StringVar(value="Выберите папку и нажмите «Установить».")
    tk.Label(window, text="Установка VideoT16", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, padx=18, pady=(16, 8), sticky="w")
    tk.Label(window, text="Папка программы:").grid(row=1, column=0, columnspan=2, padx=18, sticky="w")
    tk.Entry(window, textvariable=location, width=55).grid(row=2, column=0, padx=(18, 6), pady=6)

    def choose() -> None:
        """Предлагает выбрать родительскую папку для нового каталога VideoT16."""
        folder = filedialog.askdirectory(parent=window, title="Где установить VideoT16?")
        if folder:
            location.set(str(Path(folder) / "VideoT16"))

    def perform() -> None:
        """Копирует комплект, создаёт ярлык и предлагает открыть приложение."""
        try:
            value = location.get().strip()
            if not value:
                raise ValueError("Укажите папку установки")
            destination = Path(value).expanduser().resolve()
            status.set("Идёт установка. Пожалуйста, подождите…")
            install_button.config(state="disabled")
            window.update_idletasks()
            install_payload(payload, destination)
        except (OSError, ValueError) as error:
            status.set("Установка не завершена.")
            messagebox.showerror("VideoT16", str(error), parent=window)
            install_button.config(state="normal")
            return
        try:
            create_shortcut(destination)
        except (OSError, subprocess.SubprocessError) as error:
            messagebox.showwarning("VideoT16", f"Программа установлена, но ярлык не создан: {error}\nЗапустите Run.exe из папки {destination}.", parent=window)
        status.set("Установка завершена.")
        if messagebox.askyesno("VideoT16", "Программа установлена. Запустить её сейчас?", parent=window):
            environment = os.environ.copy()
            environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            try:
                subprocess.Popen([str(destination / "Run.exe")], cwd=str(destination), env=environment)
            except OSError as error:
                messagebox.showerror("VideoT16", f"Не удалось запустить программу: {error}", parent=window)
        window.destroy()

    tk.Button(window, text="Обзор…", command=choose).grid(row=2, column=1, padx=(0, 18))
    tk.Label(window, textvariable=status, anchor="w").grid(row=3, column=0, columnspan=2, padx=18, pady=8, sticky="w")
    install_button = tk.Button(window, text="Установить", command=perform, width=18)
    install_button.grid(row=4, column=0, padx=18, pady=(2, 18), sticky="e")
    tk.Button(window, text="Отмена", command=window.destroy, width=12).grid(row=4, column=1, padx=(0, 18), pady=(2, 18))
    tk.Label(window, text="Author: Sinelnikov Oleg", font=("Arial", 8), fg="#777777").grid(
        row=5, column=0, columnspan=2, padx=18, pady=(0, 8), sticky="e")
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
