#!/usr/bin/env python3
"""Окно выбора модели, камеры и порога перед запуском Video16 в Windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LauncherError(RuntimeError):
    """Сообщает понятную причину, по которой запуск из окна невозможен."""


def available_models(models_dir: Path) -> list[str]:
    """Возвращает имена ONNX-моделей из каталога пользователя рядом с программой."""
    return sorted(path.name for path in models_dir.glob("*.onnx") if path.is_file())


def read_settings(path: Path) -> dict[str, Any]:
    """Читает предыдущий выбор пользователя, не ломая запуск повреждённым JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(path: Path, selection: dict[str, Any]) -> None:
    """Сохраняет выбранные параметры только в пользовательской папке Video16."""
    path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_selection(
    model: str,
    camera_index: int,
    confidence_percent: int,
    models: list[str],
    cameras: list[int],
) -> dict[str, Any]:
    """Проверяет значения окна перед сохранением и запуском распознавания."""
    if model not in models:
        raise LauncherError("Выберите доступную ONNX-модель")
    if camera_index not in cameras:
        raise LauncherError("Выберите доступную камеру")
    if not 1 <= confidence_percent <= 100:
        raise LauncherError("Порог уверенности должен быть от 1 до 100")
    return {
        "format": "Video16 Windows settings v1",
        "model": model,
        "camera_index": camera_index,
        "confidence_percent": confidence_percent,
    }


def choose_options(
    install_root: Path,
    models: list[str],
    cameras: list[int],
) -> dict[str, Any]:
    """Показывает окно выбора и возвращает сохранённые пользователем параметры."""
    if not models:
        raise LauncherError(
            f"В каталоге {install_root / 'models'} нет ONNX-моделей. "
            "Поместите туда подготовленную модель с расширением .onnx и запустите снова."
        )
    if not cameras:
        raise LauncherError("Windows не обнаружила доступную камеру")

    import tkinter as tk
    from tkinter import messagebox, ttk

    settings_path = install_root / "settings.json"
    previous = read_settings(settings_path)
    default_model = str(previous.get("model", models[0]))
    if default_model not in models:
        default_model = models[0]
    default_camera = int(previous.get("camera_index", cameras[0]))
    if default_camera not in cameras:
        default_camera = cameras[0]
    default_confidence = int(previous.get("confidence_percent", 40))
    default_confidence = min(100, max(1, default_confidence))

    window = tk.Tk()
    window.title("Video16 — параметры распознавания")
    window.resizable(False, False)
    frame = ttk.Frame(window, padding=16)
    frame.grid(sticky="nsew")

    model_value = tk.StringVar(value=default_model)
    camera_value = tk.StringVar(value=str(default_camera))
    confidence_value = tk.IntVar(value=default_confidence)
    result: dict[str, Any] = {}

    ttk.Label(frame, text="Модель распознавания:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(
        frame, textvariable=model_value, values=models, state="readonly", width=42
    ).grid(row=1, column=0, sticky="ew", pady=(2, 12))

    ttk.Label(frame, text="Устройство захвата:").grid(row=2, column=0, sticky="w")
    camera_labels = [f"Камера {index}" for index in cameras]
    camera_box = ttk.Combobox(frame, values=camera_labels, state="readonly", width=42)
    camera_box.current(cameras.index(default_camera))
    camera_box.grid(row=3, column=0, sticky="ew", pady=(2, 12))

    ttk.Label(frame, text="Уверенность распознавания, %:").grid(
        row=4, column=0, sticky="w"
    )
    confidence_label = ttk.Label(frame, text=str(default_confidence))
    ttk.Scale(
        frame,
        from_=1,
        to=100,
        variable=confidence_value,
        orient="horizontal",
        command=lambda _value: confidence_label.configure(
            text=str(int(confidence_value.get()))
        ),
    ).grid(row=5, column=0, sticky="ew", pady=(2, 2))
    confidence_label.grid(row=6, column=0, sticky="e", pady=(0, 12))

    def launch() -> None:
        """Проверяет выбор, сохраняет его и закрывает окно для запуска."""
        try:
            chosen_camera = cameras[camera_box.current()]
            selection = validate_selection(
                model_value.get(),
                chosen_camera,
                int(confidence_value.get()),
                models,
                cameras,
            )
            save_settings(settings_path, selection)
        except (LauncherError, OSError, ValueError) as error:
            messagebox.showerror("Video16", str(error), parent=window)
            return
        result.update(selection)
        window.destroy()

    def cancel() -> None:
        """Завершает запуск без изменения прежних параметров."""
        window.destroy()

    buttons = ttk.Frame(frame)
    buttons.grid(row=7, column=0, sticky="e")
    ttk.Button(buttons, text="Отмена", command=cancel).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(buttons, text="Запустить", command=launch).grid(row=0, column=1)
    window.protocol("WM_DELETE_WINDOW", cancel)
    window.mainloop()
    if not result:
        raise LauncherError("Запуск отменён пользователем")
    return result
