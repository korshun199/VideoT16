#!/usr/bin/env bash
set -Eeuo pipefail

# Архив системы VideoT16 с Raspberry на внешний диск ноутбука.
# Запускается на ноутбуке, Raspberry только отдаёт файлы по SSH.

PI_USER="oleg"
PI_HOST="192.168.20.109"
ARCHIVE_DIR="/media/oleg/VideoT16_Backups"
ARCHIVE_NAME="videot16-system-$(date '+%Y-%m-%d_%H-%M-%S').tar.gz"
PI_SERVICE="videot16.service"

RED='\033[1;31m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; RESET='\033[0m'
info() { printf '%b\n' "${CYAN}[INFO]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf '%b\n' "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

ARCHIVE_PATH="$ARCHIVE_DIR/$ARCHIVE_NAME"
TEMP_PATH="$ARCHIVE_PATH.part"
MANIFEST="/tmp/videot16-manifest-$$.txt"
SERVICE_WAS_ACTIVE=0

cleanup() {
    rm -f "$TEMP_PATH"
    ssh "$PI_USER@$PI_HOST" "sudo rm -f '$MANIFEST'" >/dev/null 2>&1 || true
    if [[ "$SERVICE_WAS_ACTIVE" == "1" ]]; then
        info "Запускаю VideoT16 обратно..."
        ssh "$PI_USER@$PI_HOST" "sudo systemctl start '$PI_SERVICE'" || warn "VideoT16 не удалось запустить автоматически"
    fi
}
trap cleanup EXIT INT TERM

[[ "$EUID" -ne 0 ]] || fail "Запускайте без sudo: скрипт сам запросит права."
[[ -d "$ARCHIVE_DIR" ]] || fail "Каталог архива не найден: $ARCHIVE_DIR"
mountpoint -q "$ARCHIVE_DIR" || fail "Внешний диск не смонтирован в $ARCHIVE_DIR"
command -v ssh >/dev/null || fail "Не найдена команда ssh"
command -v tar >/dev/null || fail "Не найдена команда tar"
command -v gzip >/dev/null || fail "Не найдена команда gzip"

printf '%b\n' "${CYAN}=== VideoT16: компактный архив системы ===${RESET}"
info "Источник: Raspberry Pi $PI_USER@$PI_HOST"
info "Назначение: $ARCHIVE_PATH"

ssh -o ConnectTimeout=8 "$PI_USER@$PI_HOST" 'test -d /home/oleg/VideoT16' \
    || fail "Проект VideoT16 не найден на Raspberry"

info "Подтвердите sudo на Raspberry..."
ssh -tt "$PI_USER@$PI_HOST" 'sudo -v'

ssh "$PI_USER@$PI_HOST" "sudo sh -c 'printf \"VideoT16 system manifest\\n\" > $MANIFEST; uname -a >> $MANIFEST; cat /etc/os-release >> $MANIFEST; dpkg-query -W >> $MANIFEST'"

if ssh "$PI_USER@$PI_HOST" "systemctl is-active --quiet '$PI_SERVICE'"; then
    SERVICE_WAS_ACTIVE=1
    info "Останавливаю VideoT16 на время архивирования..."
    ssh "$PI_USER@$PI_HOST" "sudo systemctl stop '$PI_SERVICE'"
fi
ssh "$PI_USER@$PI_HOST" 'sudo sync'

warn "Архивируются файлы всей системы, но не пустые виртуальные разделы ядра."
ssh "$PI_USER@$PI_HOST" "sudo tar -C / --numeric-owner --xattrs --acls -czf - \
    --exclude=proc --exclude=sys --exclude=dev --exclude=run \
    --exclude=tmp --exclude=mnt --exclude=media --exclude=lost+found \
    --exclude=home/oleg/VideoT16/runs \
    --exclude=home/oleg/VideoT16/reports \
    --exclude=home/oleg/VideoT16/dataset \
    --exclude=home/oleg/.cache \
    . '$MANIFEST'" > "$TEMP_PATH"

gzip -t "$TEMP_PATH"
mv "$TEMP_PATH" "$ARCHIVE_PATH"
sha256sum "$ARCHIVE_PATH" > "$ARCHIVE_PATH.sha256"
sync

ok "Компактный архив создан и проверен."
ls -lh "$ARCHIVE_PATH" "$ARCHIVE_PATH.sha256"
