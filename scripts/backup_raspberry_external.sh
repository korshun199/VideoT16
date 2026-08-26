#!/usr/bin/env bash
set -Eeuo pipefail

# Сжатый слепок рабочей SD-карты Raspberry на внешний диск.

SOURCE_DEVICE="/dev/mmcblk0"
BACKUP_DEVICE="/dev/sda"
SERVICE_NAME="videot16.service"
BACKUP_DIR=""
IMAGE_NAME="raspberrypi-videot16-$(date '+%Y-%m-%d_%H-%M-%S').img.gz"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() { printf '%b\n' "${CYAN}[INFO]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf '%b\n' "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

SERVICE_WAS_ACTIVE=0

cleanup() {
    if [[ "$SERVICE_WAS_ACTIVE" == "1" ]]; then
        info "Возвращаю VideoT16 после создания слепка..."
        sudo systemctl start "$SERVICE_NAME" || warn "VideoT16 не удалось запустить автоматически"
    fi
}
trap cleanup EXIT INT TERM

[[ "$EUID" -ne 0 ]] || fail "Запустите без sudo: скрипт сам запросит права."
command -v lsblk >/dev/null || fail "Не найдена команда lsblk"
command -v findmnt >/dev/null || fail "Не найдена команда findmnt"
command -v gzip >/dev/null || fail "Не найдена команда gzip"
command -v sha256sum >/dev/null || fail "Не найдена команда sha256sum"
[[ -b "$SOURCE_DEVICE" ]] || fail "Рабочая SD не найдена: $SOURCE_DEVICE"
[[ -b "$BACKUP_DEVICE" ]] || fail "Внешний диск не найден: $BACKUP_DEVICE"
[[ "$SOURCE_DEVICE" != "$BACKUP_DEVICE" ]] || fail "Источник и внешний диск совпали"

SOURCE_SIZE_BYTES="$(lsblk -bndo SIZE "$SOURCE_DEVICE")"
BACKUP_SIZE_BYTES="$(lsblk -bndo SIZE "$BACKUP_DEVICE")"
[[ "$SOURCE_SIZE_BYTES" -gt 0 ]] || fail "Размер рабочей SD не определён"
[[ "$BACKUP_SIZE_BYTES" -ge "$SOURCE_SIZE_BYTES" ]] || fail "Внешний диск меньше рабочей SD"

if [[ -z "$BACKUP_DIR" ]]; then
    BACKUP_DIR="$(findmnt -rn -S "${BACKUP_DEVICE}1" -o TARGET | head -n1)"
fi
[[ -n "$BACKUP_DIR" && -d "$BACKUP_DIR" ]] || fail "Раздел внешнего диска не смонтирован: ${BACKUP_DEVICE}1"

AVAILABLE_BYTES="$(df -B1 --output=avail "$BACKUP_DIR" | tail -n1 | tr -d '[:space:]')"
[[ "$AVAILABLE_BYTES" -ge "$SOURCE_SIZE_BYTES" ]] || fail "На внешнем диске недостаточно места"

IMAGE_PATH="$BACKUP_DIR/$IMAGE_NAME"
TEMP_PATH="$IMAGE_PATH.part"

printf '%b\n' "${CYAN}=== VideoT16: слепок рабочей Raspberry SD ===${RESET}"
info "Источник только для чтения: $SOURCE_DEVICE"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$SOURCE_DEVICE"
info "Цель только на внешнем диске: $IMAGE_PATH"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$BACKUP_DEVICE"
warn "Рабочая SD не форматируется и не изменяется."

sudo -v
if systemctl is-active --quiet "$SERVICE_NAME"; then
    SERVICE_WAS_ACTIVE=1
    info "Останавливаю VideoT16 на время чтения карты..."
    sudo systemctl stop "$SERVICE_NAME"
fi

sudo sync
info "Создаю сжатый образ всей SD-карты. Это может занять время..."

sudo dd if="$SOURCE_DEVICE" bs=4M iflag=fullblock status=progress \
    | gzip -1 > "$TEMP_PATH"

gzip -t "$TEMP_PATH"
sha256sum "$TEMP_PATH" > "$TEMP_PATH.sha256"
mv "$TEMP_PATH" "$IMAGE_PATH"
mv "$TEMP_PATH.sha256" "$IMAGE_PATH.sha256"
sync

ok "Слепок создан и проверен"
ls -lh "$IMAGE_PATH" "$IMAGE_PATH.sha256"
printf '%b\n' "${GREEN}============================================${RESET}"
printf '%b\n' "${GREEN}РАБОЧАЯ SD СОХРАНЕНА НА ВНЕШНЕМ ДИСКЕ${RESET}"
