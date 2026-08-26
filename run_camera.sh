#!/usr/bin/env bash
set -euo pipefail

# Запускает локальное распознавание с внешней Logitech Brio.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$PROJECT_DIR/models/yolov8n.pt"
CUSTOM_MODEL="$PROJECT_DIR/runs/detect/runs/my_cat/weights/best.pt"
if [[ -f "$CUSTOM_MODEL" ]]; then
  MODEL_PATH="$CUSTOM_MODEL"
fi
exec "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection \
  --source /dev/video4 \
  --model "$MODEL_PATH" \
  "$@"
