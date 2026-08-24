"""Загрузка и применение настроек, изменяемых оператором через веб-панель."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path("config/runtime_settings.json")

DEFAULT_SETTINGS: dict[str, Any] = {
    "detection": {
        "confidence_percent": 60,
        "inference_size": 256,
        "inference_interval": 2,
        "generic_label": True,
    },
    "object": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.7,
        "text_thickness": 2,
        "box_thickness": 2,
        "color_rgb": [0, 255, 0],
    },
    "osd_text": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.5,
        "thickness": 1,
        "shadow_thickness": 2,
        "color_rgb": [255, 255, 0],
        "shadow_color_rgb": [0, 0, 0],
        "left": 20,
        "top": 30,
        "line_spacing": 24,
    },
    "horizon": {"thickness": 1, "color_rgb": [255, 255, 0]},
    "axis": {"thickness": 1, "color_rgb": [255, 255, 255]},
    "flight_status": {"bar_thickness": 1, "bar_width": 18, "bar_height": 120},
    "arm_banner": {"font_scale": 1.0, "thickness": 2, "blink_period": 1.0},
}


def _merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно объединяет настройки с безопасными значениями по умолчанию."""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Читает настройки или возвращает значения по умолчанию."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return deepcopy(DEFAULT_SETTINGS)
    return _merge(DEFAULT_SETTINGS, data if isinstance(data, dict) else {})


def save_settings(settings: dict[str, Any], path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Проверяет структуру и атомарно сохраняет настройки оператора."""
    normalized = _merge(DEFAULT_SETTINGS, settings)
    detection = normalized["detection"]
    if not 1 <= float(detection["confidence_percent"]) <= 100:
        raise ValueError("confidence_percent должен быть от 1 до 100")
    if not 32 <= int(detection["inference_size"]) <= 1280:
        raise ValueError("inference_size должен быть от 32 до 1280")
    if not 1 <= int(detection["inference_interval"]) <= 30:
        raise ValueError("inference_interval должен быть от 1 до 30")
    for section in ("object", "osd_text", "horizon", "axis"):
        for key in ("color_rgb", "shadow_color_rgb"):
            if key in normalized[section]:
                color = normalized[section][key]
                if len(color) != 3 or any(not 0 <= int(value) <= 255 for value in color):
                    raise ValueError(f"{section}.{key} должен содержать три значения 0..255")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="runtime_settings.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(normalized, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return normalized


def _bgr(value: list[int]) -> tuple[int, int, int]:
    """Преобразует привычный оператору RGB в формат BGR OpenCV."""
    red, green, blue = (max(0, min(255, int(item))) for item in value[:3])
    return blue, green, red


def apply_osd_settings(settings: dict[str, Any], styles: dict[str, dict[str, Any]]) -> None:
    """Применяет веб-настройки к уже импортированным словарям OSD."""
    object_values = settings["object"]
    styles["object"].update(object_values)
    styles["object"]["color"] = _bgr(object_values["color_rgb"])

    text_values = settings["osd_text"]
    styles["text"].update(text_values)
    styles["text"]["color"] = _bgr(text_values["color_rgb"])
    styles["text"]["shadow_color"] = _bgr(text_values["shadow_color_rgb"])

    horizon_values = settings["horizon"]
    styles["horizon"].update(horizon_values)
    styles["horizon"]["color"] = _bgr(horizon_values["color_rgb"])

    axis_values = settings["axis"]
    styles["axis"].update(axis_values)
    styles["axis"]["color"] = _bgr(axis_values["color_rgb"])

    styles["flight_status"].update(settings["flight_status"])
    styles["arm_banner"].update(settings["arm_banner"])
