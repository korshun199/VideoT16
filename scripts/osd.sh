#!/usr/bin/env bash

set -Eeuo pipefail

# Корень проекта.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Основные параметры локального видеотеста.
CAMERA_DEVICE="/dev/video0"  # Внешняя USB-камера Brio 90.

# Подпись и объединённая модель FPV для текущего теста.
OBJECT_LABEL="FPV-DRON"
MODEL_PATH="$PROJECT_DIR/runs/fpv_training/fpv_quadcopter_merged/weights/best.onnx"
INFERENCE_SIZE="640"


CONFIDENCE_PERCENT="50"
INFERENCE_INTERVAL="2"
CAMERA_FPS="30"

# Проверяет обязательные файлы и устройства до запуска.
check_requirements() {
    if [[ ! -e "$CAMERA_DEVICE" ]]; then
        printf '[ОШИБКА] USB-камера не найдена: %s\n' "$CAMERA_DEVICE" >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_PATH" ]]; then
        printf '[ОШИБКА] Модель не найдена: %s\n' "$MODEL_PATH" >&2
        exit 1
    fi
    if [[ ! -x "$PROJECT_DIR/.venv/bin/python3" ]]; then
        printf '[ОШИБКА] Не найден Python виртуального окружения\n' >&2
        exit 1
    fi
}

# Находит подключённый HDMI/DP-монитор в X11.
find_external_monitor() {
    if ! command -v xrandr >/dev/null 2>&1; then
        return 0
    fi
    xrandr --query 2>/dev/null \
        | awk '$1 ~ /^(HDMI|DP)/ && $2 == "connected" { print $1; exit }'
}

check_requirements
MONITOR_NAME="$(find_external_monitor)"

ARGS=(
    --source "$CAMERA_DEVICE"
    --model "$MODEL_PATH"
    --object-label "$OBJECT_LABEL"
    --percent "$CONFIDENCE_PERCENT"
    --inference-size "$INFERENCE_SIZE"
    --inference-interval "$INFERENCE_INTERVAL"
    --camera-fps "$CAMERA_FPS"
    --fullscreen
)

if [[ -n "$MONITOR_NAME" ]]; then
    ARGS+=(--monitor "$MONITOR_NAME")
    printf 'HDMI-монитор: %s\n' "$MONITOR_NAME"
else
    printf 'HDMI-монитор автоматически не найден; используется экран по умолчанию\n'
fi

printf 'USB-камера: %s\n' "$CAMERA_DEVICE"
printf 'Наша графика: рамка + температура + CPU\n'
printf 'Штатный OSD полётника: не используется\n'
printf 'Распознавание: порог %s%%, размер %s, каждый %s-й кадр\n' \
    "$CONFIDENCE_PERCENT" "$INFERENCE_SIZE" "$INFERENCE_INTERVAL"

exec "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection "${ARGS[@]}"
