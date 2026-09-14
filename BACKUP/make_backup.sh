#!/usr/bin/env bash
# Собирает переносимую резервную копию только разработки VideoT16.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="$PROJECT_DIR/BACKUP"
STAMP="$(date '+%Y%m%d-%H%M%S')"
STAGING_DIR="$(mktemp -d -t videot16-backup.XXXXXX)"
ARCHIVE_PATH="$BACKUP_DIR/VideoT16-dev-$STAMP.tar.gz"

# Удаляет временный каталог даже при отмене сборки.
cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

# Копирует файл или каталог, если он существует.
include_path() {
    local relative_path="$1"
    if [[ -e "$PROJECT_DIR/$relative_path" ]]; then
        mkdir -p "$STAGING_DIR/$(dirname "$relative_path")"
        cp -a "$PROJECT_DIR/$relative_path" "$STAGING_DIR/$relative_path"
    else
        printf '[ПРОПУСК] отсутствует: %s\n' "$relative_path"
    fi
}

# Добавляет основные исходники, конфигурацию, тесты и документацию.
include_path AGENTS.md
include_path README.md
include_path README_OLEG_MASHA.md
include_path RASPBERRY_TRANSFER.md
include_path SESSION_HANDOFF.md
include_path WORKING_SCHEME.md
include_path RESTORE_RASPBERRY.md
include_path SECURITY.md
include_path requirements-raspberry.txt
include_path run_raspberry.sh
include_path videot16.service
include_path deploy/videot16.service
include_path deploy/videot16-restart
include_path deploy/videot16-web.service
include_path deploy/videot16-web.sudoers
include_path config
include_path src
include_path tests
include_path docs
include_path web_config
include_path scripts/conf
include_path scripts/vtest
include_path scripts/osd_test.py
include_path scripts/train_fpv.py
include_path scripts/install_raspberry_service.sh
include_path scripts/setup_ubuntu.sh
include_path models/fpv_drone_best.onnx
include_path models/FPV_MODEL.md

# Исходники безопасности полезны для разработки, но production-файлы и auto.py
# с локальной фразой намеренно не копируются.
for security_file in security/pack.py security/enc.py security/dec.c security/gen.py security/unpack.py security/view security/videot16.service; do
    include_path "$security_file"
done

# Удаляет случайно попавшие секретные и временные имена из staging.
find "$STAGING_DIR" -type f \( \
    -name 'production_payload.v16e' -o \
    -name 'production_activation.json' -o \
    -name 'launcher.py' -o \
    -name 'auto.py' -o \
    -name 'confidence.conf' -o \
    -name '*.before_confidence.*' \
\) -delete
find "$STAGING_DIR" -type d \( -name '.git' -o -name '.venv' -o -name '__pycache__' -o -name '.cache' \) -prune -exec rm -rf {} +

# Создаёт список файлов и контрольные SHA-256 внутри будущего архива.
(
    cd "$STAGING_DIR"
    find . -type f -printf '%P\n' | sort > BACKUP_FILE_LIST.txt
    sha256sum $(cat BACKUP_FILE_LIST.txt) > BACKUP_SHA256SUMS.txt
)

printf '\nСостав резервной копии:\n'
du -sh "$STAGING_DIR"
cat "$STAGING_DIR/BACKUP_FILE_LIST.txt"
printf '\nАрхив: %s\n' "$ARCHIVE_PATH"
read -r -p 'Введите СОХРАНИТЬ для создания архива: ' confirmation
if [[ "$confirmation" != "СОХРАНИТЬ" ]]; then
    printf '[ОТМЕНА] Архив не создан.\n'
    exit 0
fi

# Финальная проверка запрещённых имён перед упаковкой.
if find "$STAGING_DIR" -type f | grep -E '(production_payload|production_activation|/launcher\.py$|/auto\.py$|confidence\.conf)' >/dev/null; then
    printf '[ОШИБКА] В staging обнаружен запрещённый файл.\n' >&2
    exit 1
fi

tar -czf "$ARCHIVE_PATH" -C "$STAGING_DIR" .
chmod 600 "$ARCHIVE_PATH"
printf '[OK] Архив создан: %s\n' "$ARCHIVE_PATH"
printf '[OK] Размер: %s\n' "$(du -h "$ARCHIVE_PATH" | cut -f1)"
