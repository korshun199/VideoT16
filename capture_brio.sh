#!/usr/bin/env bash
set -euo pipefail

# Запускает съёмку предмета с Logitech Brio.
if [[ $# -lt 1 ]]; then
  echo "Использование: ./capture_brio.sh имя_предмета" >&2
  exit 2
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$PROJECT_DIR/.venv/bin/python3" "$PROJECT_DIR/scripts/capture_object_photos.py" \
  --source /dev/video4 \
  --name "$1" \
  "${@:2}"
