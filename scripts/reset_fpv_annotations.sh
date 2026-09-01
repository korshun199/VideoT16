#!/usr/bin/env bash
set -euo pipefail

# Удаляет только старые результаты разметки FPV.
# Исходные видео и кадры из from_videos не затрагиваются.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_DIR="$PROJECT_DIR/dataset/fpv/images/annotated"
LABEL_DIR="$PROJECT_DIR/dataset/fpv/labels/annotated"

printf 'Будут удалены только старые размеченные копии и YOLO .txt:\n'
printf '  %s\n  %s\n' "$IMAGE_DIR" "$LABEL_DIR"
printf 'Видео и кадры из dataset/fpv/images/from_videos НЕ удаляются.\n'
read -r -p 'Введите RESET_FPV_ANNOTATIONS для продолжения: ' confirmation
if [[ "$confirmation" != "RESET_FPV_ANNOTATIONS" ]]; then
    printf 'Отмена: ничего не удалено.\n'
    exit 0
fi

mkdir -p "$IMAGE_DIR" "$LABEL_DIR"
find "$IMAGE_DIR" -mindepth 1 -maxdepth 1 -type f -delete
find "$LABEL_DIR" -mindepth 1 -maxdepth 1 -type f -name '*.txt' -delete
printf 'Старые результаты разметки очищены. Кадров для новой разметки: '
find "$PROJECT_DIR/dataset/fpv/images/from_videos" -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l
