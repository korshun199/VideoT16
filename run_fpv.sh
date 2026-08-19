#!/usr/bin/env bash
set -euo pipefail

# Запускает готовую модель обнаружения дронов с Logitech Brio.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection \
  --source /dev/video0 \
  --model "$PROJECT_DIR/models/fpv_drone_best.pt" \
  --confidence 0.35 \
  "$@"
