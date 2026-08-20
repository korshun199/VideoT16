#!/usr/bin/env bash
set -euo pipefail

# Запускает моделирование: камера, INAV и HDMI на ноутбуке.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection \
  --source "${CAMERA_SOURCE:-logitech}" \
  --model "${MODEL_PATH:-$PROJECT_DIR/models/yolov8n.pt}" \
  --inav-port "${INAV_PORT:-/dev/ttyACM0}" \
  "$@"
