#!/usr/bin/env bash
set -euo pipefail

# Запускает базовое распознавание с ограничениями Orange Pi 4.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMORY_MB="${ORANGEPI4_MEMORY_MB:-4096}"
CORES="${ORANGEPI4_CORES:-2}"
exec "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/scripts/orangepi4_mode.py" \
  --memory-mb "$MEMORY_MB" \
  --cores "$CORES" \
  -- "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection \
  --source "${CAMERA_SOURCE:-/dev/video0}" \
  --model "${MODEL_PATH:-$PROJECT_DIR/models/yolov8n.pt}" \
  "$@"
