#!/usr/bin/env bash
set -euo pipefail

# Запускает FPV-модель, экспортированную в RKNN, на Orange Pi 5.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${FPV_RKNN_MODEL:-$PROJECT_DIR/models/orangepi5/fpv_drone.rknn}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ALERT_ARGS=()
if [[ -f "$PROJECT_DIR/sounds/fpv_detected.wav" ]]; then
  ALERT_ARGS=(--alert-wav "$PROJECT_DIR/sounds/fpv_detected.wav")
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Ошибка: RKNN-модель не найдена: $MODEL_PATH" >&2
  echo "Сначала экспортируйте её на x86-компьютере по инструкции README.md." >&2
  exit 1
fi

exec "$PYTHON_BIN" -m src.local_object_detection \
  --source "${CAMERA_SOURCE:-/dev/video0}" \
  --model "$MODEL_PATH" \
  --device auto \
  --confidence "${FPV_CONFIDENCE:-0.35}" \
  "${ALERT_ARGS[@]}" \
  "$@"
