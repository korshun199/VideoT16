#!/usr/bin/env bash
set -euo pipefail

# Запускает базовую модель YOLO с внешней Logitech Brio.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection \
  --source /dev/video4 \
  --model "$PROJECT_DIR/models/yolov8n.pt" \
  "$@"
