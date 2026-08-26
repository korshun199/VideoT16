#!/usr/bin/env bash
set -Eeuo pipefail

# Полное восстановление образа VideoT16 на внешний диск.

IMAGE="/home/work/raspberry-backups/raspberrypi-videot16-2026-08-24.img.zst"
TARGET="/dev/sdb"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() { printf "%b\n" "${CYAN}[INFO]${RESET} $*"; }
ok() { printf "%b\n" "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf "%b\n" "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf "%b\n" "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

[[ "$EUID" -eq 0 ]] || fail "Запустите: sudo $0"
[[ -f "$IMAGE" ]] || fail "Образ не найден: $IMAGE"
[[ -b "$TARGET" ]] || fail "Внешний диск не найден: $TARGET"

model="$(lsblk -dnro MODEL "$TARGET" | xargs | sed 's/x20/ /g')"
size_bytes="$(blockdev --getsize64 "$TARGET")"
[[ "$model" == HGST\ HTS541010A9E680* ]] || fail "Модель цели не совпала: $model"
[[ "$size_bytes" -gt 100000000000 ]] || fail "Цель неожиданно мала: $size_bytes байт"

printf "%b\n" "${CYAN}=== VideoT16: восстановление на внешний диск ===${RESET}"
info "Источник образа: $IMAGE"
info "Целевой диск: $TARGET"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$TARGET"
warn "ВСЕ РАЗДЕЛЫ $TARGET БУДУТ УДАЛЕНЫ."
printf "Введите ERASE_EXTERNAL_VIDEOT16 для продолжения: "
read -r confirmation
[[ "$confirmation" == "ERASE_EXTERNAL_VIDEOT16" ]] || fail "Операция отменена"

info "Проверка сжатого образа..."
zstd -t "$IMAGE"
ok "Образ проверен"

info "Отмонтирование внешнего диска..."
umount "${TARGET}1" 2>/dev/null || true
umount "${TARGET}2" 2>/dev/null || true
ok "Диск подготовлен"

info "Запись образа на внешний диск. Это займёт несколько минут..."
zstd -d -c "$IMAGE" | dd of="$TARGET" bs=16M iflag=fullblock status=progress conv=fsync
sync
partprobe "$TARGET" || true
udevadm settle

printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}ВОССТАНОВЛЕНИЕ НА ВНЕШНИЙ ДИСК ЗАВЕРШЕНО${RESET}"
printf "%b\n" "${GREEN}Цель: $TARGET${RESET}"
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS "$TARGET"
