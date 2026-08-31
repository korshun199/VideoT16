#!/usr/bin/env bash

set -Eeuo pipefail

# Настройки безопасного запуска VideoT16 на Raspberry Pi 5.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
CAMERA_SOURCE="/dev/video0"
MODEL_PATH="models/fpv_drone_custom.onnx"
CONFIDENCE_PERCENT="20"
INFERENCE_SIZE="640"
INFERENCE_INTERVAL="2"
CAMERA_FPS="25"
GENERIC_LABEL="1"
HEADLESS="1"
DRM_DEVICE="/dev/dri/by-path/platform-1f00144000.vec-card"
FRAMEBUFFER_DEVICE=""
INAV_PORT="/dev/ttyACM0"
TEMPERATURE_INTERVAL="5"
TEMPERATURE_WARNING="70"
TEMPERATURE_CRITICAL="80"

cd "$PROJECT_DIR"

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

# Следит за температурой и останавливает распознавание при опасном нагреве.
monitor_temperature() {
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        temperature="$(vcgencmd measure_temp 2>/dev/null | tr -cd '0-9.')"
        throttle="$(vcgencmd get_throttled 2>/dev/null | tr -d '\r')"
        if [[ -n "$temperature" ]]; then
            printf '[%s] Температура: %s°C | Питание: %s\n' \
                "$(date '+%d.%m.%Y %H:%M:%S')" "$temperature" "$throttle" >&2
            if awk "BEGIN { exit !($temperature >= $TEMPERATURE_CRITICAL) }"; then
                printf 'КРИТИЧЕСКАЯ температура — останавливаю VideoT16.\n' >&2
                kill "$MAIN_PID" 2>/dev/null || true
                return
            elif awk "BEGIN { exit !($temperature >= $TEMPERATURE_WARNING) }"; then
                printf 'Предупреждение: Raspberry Pi нагрелась.\n' >&2
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
