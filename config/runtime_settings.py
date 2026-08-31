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
        "text_min_y": 20,
        "text_offset_y": 8,
    },
    "osd_text": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.5,
        "thickness": 1,
        "shadow_thickness": 2,
        "color_rgb": [255, 255, 0],
        "shadow_color_rgb": [0, 0, 0],
        "shadow_offset_x": 1,
        "shadow_offset_y": 1,
        "left": 20,
        "top": 30,
        "line_spacing": 24,
    },
    "system_status": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 0.5,
        "color_rgb": [0, 0, 255],
        "shadow_color_rgb": [0, 0, 0],
        "thickness": 1,
        "shadow_thickness": 2,
        "shadow_offset_x": 1,
        "shadow_offset_y": 1,
        "left": 20,
        "bottom": 18,
    },
    "horizon": {
        "pitch_scale": 4.0,
        "pitch_limit": 45.0,
        "min_length": 240,
        "length_ratio": 0.5,
        "thickness": 1,
        "color_rgb": [255, 255, 0],
        "shadow_color_rgb": [0, 0, 0],
        "shadow_thickness": 3,
    },
    "axis": {
        "vertical_half_length": 50,
        "horizontal_half_length": 64,
        "thickness": 1,
        "color_rgb": [255, 255, 255],
        "shadow_color_rgb": [0, 0, 0],
        "shadow_thickness": 3,
    },
    "flight_status": {
        "right": 40,
        "top": 30,
        "line_spacing": 24,
        "bar_top_offset": 36,
        "bar_width": 18,
        "bar_height": 120,
        "bar_color_rgb": [255, 255, 0],
        "bar_thickness": 1,
        "fill_color_rgb": [255, 255, 0],
        "shadow_color_rgb": [0, 0, 0],
        "shadow_thickness": 3,
        "marker_color_rgb": [255, 255, 255],
        "marker_shadow_color_rgb": [0, 0, 0],
        "marker_thickness": 1,
        "marker_shadow_thickness": 4,
        "marker_overhang": 4,
    },
    "arm_banner": {
        "font": "FONT_HERSHEY_SIMPLEX",
        "font_scale": 1.0,
        "disarmed_color_rgb": [255, 0, 0],
        "armed_color_rgb": [0, 255, 0],
        "shadow_color_rgb": [0, 0, 0],
        "thickness": 2,
        "shadow_thickness": 6,
        "shadow_offset_x": 2,
        "shadow_offset_y": 2,
        "bottom_margin": 30,
        "blink_period": 1.0,
    },
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
    for section in ("object", "osd_text", "system_status", "horizon", "axis", "flight_status", "arm_banner"):
        for key in (
            "color_rgb", "shadow_color_rgb", "bar_color_rgb", "fill_color_rgb",
            "marker_color_rgb", "marker_shadow_color_rgb", "disarmed_color_rgb",
            "armed_color_rgb",
        ):
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
    styles["text"]["shadow_offset"] = (
        int(text_values["shadow_offset_x"]), int(text_values["shadow_offset_y"])
    )

    if "system_status" in styles:
        system_values = settings["system_status"]
        styles["system_status"].update(system_values)
        styles["system_status"]["color"] = _bgr(system_values["color_rgb"])
        styles["system_status"]["shadow_color"] = _bgr(system_values["shadow_color_rgb"])
        styles["system_status"]["shadow_offset"] = (
            int(system_values["shadow_offset_x"]), int(system_values["shadow_offset_y"])
        )

    horizon_values = settings["horizon"]
    styles["horizon"].update(horizon_values)
    styles["horizon"]["color"] = _bgr(horizon_values["color_rgb"])
    styles["horizon"]["shadow_color"] = _bgr(horizon_values["shadow_color_rgb"])

    axis_values = settings["axis"]
    styles["axis"].update(axis_values)
    styles["axis"]["color"] = _bgr(axis_values["color_rgb"])
    styles["axis"]["shadow_color"] = _bgr(axis_values["shadow_color_rgb"])

    flight_values = settings["flight_status"]
    styles["flight_status"].update(flight_values)
    for key in ("bar_color", "fill_color", "shadow_color", "marker_color", "marker_shadow_color"):
        styles["flight_status"][key] = _bgr(flight_values[f"{key}_rgb"])

    arm_values = settings["arm_banner"]
    styles["arm_banner"].update(arm_values)
    for key in ("disarmed_color", "armed_color", "shadow_color"):
        styles["arm_banner"][key] = _bgr(arm_values[f"{key}_rgb"])
    styles["arm_banner"]["shadow_offset"] = (
        int(arm_values["shadow_offset_x"]), int(arm_values["shadow_offset_y"])
    )
