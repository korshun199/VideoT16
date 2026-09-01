#!/usr/bin/env bash

set -Eeuo pipefail

# Настройки безопасного запуска VideoT16 на Raspberry Pi 5.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
CAMERA_SOURCE="/dev/video0"
CAMERA_INPUT="easycap"
DIGITAL_CAMERA_DEVICE=""
MODEL_PATH="models/fpv_drone_custom_320.onnx"
CONFIDENCE_PERCENT="20"
INFERENCE_SIZE="320"
INFERENCE_INTERVAL="2"
CAMERA_FPS="25"
GENERIC_LABEL="1"
HEADLESS="1"
# Два выхода настраиваются отдельно в config/runtime_settings.json.
J7_OUTPUT_ENABLED="1"
FLIGHT_CONTROLLER_OSD_ENABLED="1"
DRM_DEVICE="/dev/dri/by-path/platform-1f00144000.vec-card"
FRAMEBUFFER_DEVICE=""
INAV_PORT="/dev/ttyACM0"
TEMPERATURE_INTERVAL="5"
TEMPERATURE_WARNING="70"
TEMPERATURE_CRITICAL="80"

cd "$PROJECT_DIR"

# Источник видео выбирается в runtime_settings.json.
configured_camera="$($PYTHON_BIN -c 'import json
try:
    print(json.load(open("config/runtime_settings.json", encoding="utf-8"))["detection"].get("camera_source", "easycap"))
except (KeyError, OSError, TypeError, ValueError):
    print("easycap")
')"
configured_digital_device="$($PYTHON_BIN -c 'import json
try:
    print(json.load(open("config/runtime_settings.json", encoding="utf-8"))["detection"].get("digital_camera_device", ""))
except (KeyError, OSError, TypeError, ValueError):
    print("")
')"
configured_j7="$($PYTHON_BIN -c 'import json
try: print("1" if json.load(open("config/runtime_settings.json", encoding="utf-8"))["video_output"].get("j7_enabled", True) else "0")
except (KeyError, OSError, TypeError, ValueError): print("1")
')"
configured_fc_osd="$($PYTHON_BIN -c 'import json
try: print("1" if json.load(open("config/runtime_settings.json", encoding="utf-8"))["video_output"].get("flight_controller_osd_enabled", True) else "0")
except (KeyError, OSError, TypeError, ValueError): print("1")
')"
configured_drm="$($PYTHON_BIN -c 'import json
try: print(json.load(open("config/runtime_settings.json", encoding="utf-8"))["video_output"].get("drm_device", ""))
except (KeyError, OSError, TypeError, ValueError): print("")
')"
configured_fc_port="$($PYTHON_BIN -c 'import json
try: print(json.load(open("config/runtime_settings.json", encoding="utf-8"))["video_output"].get("flight_controller_port", ""))
except (KeyError, OSError, TypeError, ValueError): print("")
')"
if [[ "$configured_camera" == "easycap" ]]; then
    CAMERA_INPUT="easycap"
    # EasyCap может получить любой номер /dev/videoN; используем стабильное имя udev.
    CAMERA_SOURCE="$(find /dev/v4l/by-id -maxdepth 1 -type l -name '*video-index0' 2>/dev/null | head -n 1)"
    if [[ -z "$CAMERA_SOURCE" ]]; then
        CAMERA_SOURCE="/dev/video0"
    fi
elif [[ "$configured_camera" == "digital" && "$configured_digital_device" == /dev/video* ]]; then
    CAMERA_INPUT="digital"
    # CSI-камера Raspberry Pi должна открываться через Picamera2, а не V4L2.
    CAMERA_SOURCE="picamera"
else
    printf '[ОШИБКА] Источник камеры не настроен: %s %s\n' "$configured_camera" "$configured_digital_device" >&2
    exit 2
fi

if [[ "$configured_j7" == "0" ]]; then
    J7_OUTPUT_ENABLED="0"
    DRM_DEVICE=""
elif [[ "$configured_j7" == "1" && -n "$configured_drm" ]]; then
    DRM_DEVICE="$configured_drm"
fi
if [[ "$configured_fc_osd" == "0" ]]; then
    FLIGHT_CONTROLLER_OSD_ENABLED="0"
    INAV_PORT=""
elif [[ "$configured_fc_osd" == "1" && -n "$configured_fc_port" ]]; then
    INAV_PORT="$configured_fc_port"
fi

# Веб-панель может выбрать только модель из каталога models.
if [[ -f "config/runtime_settings.json" ]]; then
    configured_model="$($PYTHON_BIN -c '
import json, sys
try:
    value = json.load(open("config/runtime_settings.json", encoding="utf-8"))["detection"]["model_path"]
except (KeyError, OSError, TypeError, ValueError):
    value = "models/fpv_drone_custom_320.onnx"
if value.startswith("models/") and value.endswith(".onnx") and "/" not in value[7:]:
    print(value)
else:
    print("models/fpv_drone_custom_320.onnx")
')"
    if [[ -f "$configured_model" ]]; then
        MODEL_PATH="$configured_model"
    fi
fi

RUN_ARGS=(
    --source "$CAMERA_SOURCE"
    --model "$MODEL_PATH"
    --confidence-percent "$CONFIDENCE_PERCENT"
    --inference-size "$INFERENCE_SIZE"
    --inference-interval "$INFERENCE_INTERVAL"
    --camera-fps "$CAMERA_FPS"
)

if [[ "$GENERIC_LABEL" == "1" ]]; then
    RUN_ARGS+=(--generic-label)
fi
if [[ "$HEADLESS" == "1" ]]; then
    RUN_ARGS+=(--headless)
fi
if [[ -n "$DRM_DEVICE" ]]; then
    RUN_ARGS+=(--drm "$DRM_DEVICE")
fi
if [[ -n "$FRAMEBUFFER_DEVICE" ]]; then
    RUN_ARGS+=(--framebuffer "$FRAMEBUFFER_DEVICE")
fi
if [[ -n "$INAV_PORT" ]]; then
    RUN_ARGS+=(--inav-port "$INAV_PORT")
fi

printf 'Запуск Raspberry Pi: камера=%s, FPS=%s, модель=%s, порог=%s%%, размер=%s, каждый %d-й кадр\n' \
    "$CAMERA_SOURCE" "$CAMERA_FPS" "$MODEL_PATH" "$CONFIDENCE_PERCENT" "$INFERENCE_SIZE" "$INFERENCE_INTERVAL"
printf 'Выходы: J7=%s, штатный Betaflight OSD=%s\n' \
    "$([[ "$J7_OUTPUT_ENABLED" == "1" ]] && echo ON || echo OFF)" \
    "$([[ "$FLIGHT_CONTROLLER_OSD_ENABLED" == "1" ]] && echo ON || echo OFF)"

# Следит за температурой и останавливает распознавание при опасном нагреве.
monitor_temperature() {
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        temperature="$(vcgencmd measure_temp 2>/dev/null | tr -cd '0-9.')"
        throttle="$(vcgencmd get_throttled 2>/dev/null | tr -d '\r')"
        if [[ -n "$temperature" ]]; then
            if awk "BEGIN { exit !($temperature >= $TEMPERATURE_CRITICAL) }"; then
                kill "$MAIN_PID" 2>/dev/null || true
                return
            elif awk "BEGIN { exit !($temperature >= $TEMPERATURE_WARNING) }"; then
                : # Контроль работает, штатный журнал остаётся только для OSD.
            fi
        fi
        sleep "$TEMPERATURE_INTERVAL"
    done
}

"$PYTHON_BIN" -m src.local_object_detection "${RUN_ARGS[@]}" &
MAIN_PID=$!
monitor_temperature &
MONITOR_PID=$!
set +e
wait "$MAIN_PID"
EXIT_CODE=$?
set -e
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
exit "$EXIT_CODE"
