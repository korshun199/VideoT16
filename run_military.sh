#!/usr/bin/env bash
set -euo pipefail

# Запускает модель обнаружения военной техники.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALERT_ARGS=()
if [[ -f "$PROJECT_DIR/sounds/fpv_detected.wav" ]]; then
  ALERT_ARGS=(--alert-wav "$PROJECT_DIR/sounds/fpv_detected.wav")
fi

MODE="full"
ARGS=()
while (($#)); do
  case "$1" in
    -o|--orange)
      MODE="orange"
      shift
      ;;
    -f|--full)
      MODE="full"
      shift
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

COMMAND=(
  "$PROJECT_DIR/.venv/bin/python3" -m src.local_object_detection
  --source "${CAMERA_SOURCE:-/dev/video0}"
  --model "${MODEL_PATH:-$PROJECT_DIR/models/military_vehicle_best.pt}"
  --confidence "${CONFIDENCE:-0.65}"
  --generic-label
  "${ALERT_ARGS[@]}"
  "${ARGS[@]}"
)

if [[ "$MODE" == "orange" ]]; then
  exec "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/scripts/orangepi4_mode.py" \
    --memory-mb "${ORANGEPI4_MEMORY_MB:-4096}" \
    --cores "${ORANGEPI4_CORES:-2}" \
    -- "${COMMAND[@]}"
fi

exec "${COMMAND[@]}"
