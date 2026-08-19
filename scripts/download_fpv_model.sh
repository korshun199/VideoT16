#!/usr/bin/env bash
set -euo pipefail

# Восстанавливает FPV-модель из Hugging Face и проверяет целостность файла.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="$PROJECT_DIR/models/fpv_drone_best.pt"
MODEL_URL="https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"
EXPECTED_SHA256="bf24a20e69b28896a0c7e4855c72146149fa5c25845fa5e54beda2b93cf79824"

mkdir -p "$PROJECT_DIR/models"
curl -L --fail --retry 3 -o "$MODEL_PATH" "$MODEL_URL"
ACTUAL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Ошибка: SHA-256 модели не совпадает." >&2
  exit 1
fi
echo "FPV-модель восстановлена и проверена: $MODEL_PATH"
