#!/usr/bin/env bash
set -Eeuo pipefail

# Сетевой аварийный образ Raspberry Pi на ноутбук.

SOURCE_DEVICE="/dev/mmcblk0"
LAPTOP_USER="oleg"
LAPTOP_HOST="192.168.20.107"
LAPTOP_DIR="/mnt/videot16-backups"
IMAGE_NAME="raspberrypi-videot16-$(date '+%Y-%m-%d_%H-%M-%S').img.gz"
SERVICE_NAME="videot16.service"

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
        info "Возвращаю VideoT16 после копирования..."
        sudo systemctl start "$SERVICE_NAME" || warn "Не удалось автоматически запустить $SERVICE_NAME"
    fi
}
trap cleanup EXIT INT TERM

[[ "$EUID" -ne 0 ]] || fail "Не запускайте скрипт через sudo: он сам запросит права."
command -v ssh >/dev/null || fail "Не найдена команда ssh"
command -v gzip >/dev/null || fail "Не найдена команда gzip"
command -v dd >/dev/null || fail "Не найдена команда dd"
command -v lsblk >/dev/null || fail "Не найдена команда lsblk"
[[ -b "$SOURCE_DEVICE" ]] || fail "Системная карта не найдена: $SOURCE_DEVICE"

printf '%b\n' "${CYAN}=== VideoT16: аварийный образ Raspberry Pi ===${RESET}"
info "Источник только для чтения: $SOURCE_DEVICE"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$SOURCE_DEVICE"
info "Проверка температуры и питания..."
vcgencmd measure_temp 2>/dev/null || true
vcgencmd get_throttled 2>/dev/null || true

if journalctl -k -b --no-pager 2>/dev/null | grep -q 'EXT4-fs error'; then
    warn "В текущей загрузке найдены ошибки ext4. Образ всё равно будет снят, но карту позже нужно проверить офлайн."
fi

info "Проверка SSH-доступа к ноутбуку $LAPTOP_USER@$LAPTOP_HOST..."
ssh -o BatchMode=yes -o ConnectTimeout=8 "$LAPTOP_USER@$LAPTOP_HOST" true \
    || fail "Нет входа по SSH-ключу на ноутбук. Пароль внутри конвейера вводить нельзя."

ssh "$LAPTOP_USER@$LAPTOP_HOST" "mkdir -p '$LAPTOP_DIR' && test -w '$LAPTOP_DIR'" \
    || fail "Каталог на ноутбуке недоступен для записи: $LAPTOP_DIR"

sudo -v
if systemctl is-active --quiet "$SERVICE_NAME"; then
    SERVICE_WAS_ACTIVE=1
    info "Останавливаю $SERVICE_NAME для согласованного снимка..."
    sudo systemctl stop "$SERVICE_NAME"
fi

sudo sync
REMOTE_PATH="$LAPTOP_DIR/$IMAGE_NAME"
REMOTE_TMP="$REMOTE_PATH.part"

info "Передача образа на ноутбук: $REMOTE_PATH"
warn "Читается вся SD-карта 58 ГБ; Raspberry не форматируется и не изменяется."

sudo dd if="$SOURCE_DEVICE" bs=4M iflag=fullblock status=progress \
    | gzip -1 \
    | ssh "$LAPTOP_USER@$LAPTOP_HOST" \
        "cat > '$REMOTE_TMP' && gzip -t '$REMOTE_TMP' && sha256sum '$REMOTE_TMP' > '$REMOTE_TMP.sha256' && mv '$REMOTE_TMP' '$REMOTE_PATH' && mv '$REMOTE_TMP.sha256' '$REMOTE_PATH.sha256'"

ok "Образ передан и проверен gzip на ноутбуке."
ssh "$LAPTOP_USER@$LAPTOP_HOST" "ls -lh '$REMOTE_PATH' '$REMOTE_PATH.sha256'"
printf '%b\n' "${GREEN}============================================${RESET}"
printf '%b\n' "${GREEN}АВАРИЙНЫЙ ОБРАЗ СОЗДАН${RESET}"
printf '%b\n' "${GREEN}$REMOTE_PATH${RESET}"
