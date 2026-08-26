#!/usr/bin/env bash
set -Eeuo pipefail

# Клонирование рабочей SD-карты Raspberry на резервную карту.

SOURCE_DEVICE="/dev/sdb"
TARGET_DEVICE="/dev/sda"
BLOCK_SIZE="64M"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() { printf "%b\n" "${CYAN}[INFO]${RESET} $*"; }
ok() { printf "%b\n" "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf "%b\n" "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf "%b\n" "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

cleanup() {
    if [[ -n "${TARGET_PARTITION:-}" ]]; then
        udisksctl mount -b "$TARGET_PARTITION" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

[[ "${EUID}" -eq 0 ]] || fail "Запустите: sudo $0"
command -v lsblk >/dev/null || fail "Не найдена команда lsblk"
command -v sfdisk >/dev/null || fail "Не найдена команда sfdisk"
command -v partprobe >/dev/null || fail "Не найдена команда partprobe"
command -v e2fsck >/dev/null || fail "Не найдена команда e2fsck"

[[ -b "$SOURCE_DEVICE" ]] || fail "Источник не найден: $SOURCE_DEVICE"
[[ -b "$TARGET_DEVICE" ]] || fail "Цель не найдена: $TARGET_DEVICE"
[[ "$SOURCE_DEVICE" != "$TARGET_DEVICE" ]] || fail "Источник и цель совпадают"

SOURCE_PARTITION="${SOURCE_DEVICE}2"
TARGET_PARTITION="${TARGET_DEVICE}1"
[[ -b "$SOURCE_PARTITION" ]] || fail "Не найден раздел источника: $SOURCE_PARTITION"
[[ -b "$TARGET_PARTITION" ]] || fail "Не найден раздел цели: $TARGET_PARTITION"

SOURCE_LAYOUT="$(lsblk -dnro FSTYPE "$SOURCE_PARTITION")"
[[ "$SOURCE_LAYOUT" == "ext4" ]] || fail "Источник $SOURCE_PARTITION не ext4: $SOURCE_LAYOUT"

printf "%b\n" "${BLUE}=== VideoT16: создание аварийной SD-карты ===${RESET}"
info "Источник: $SOURCE_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$SOURCE_DEVICE"
info "Цель: $TARGET_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$TARGET_DEVICE"
warn "ЦЕЛЕВАЯ КАРТА $TARGET_DEVICE БУДЕТ ПЕРЕЗАПИСАНА ПОЛНОСТЬЮ."
warn "Файловая система источника может быть уменьшена примерно на 27 MB."
printf "Введите CLONE для продолжения: "
read -r confirmation
[[ "$confirmation" == "CLONE" ]] || fail "Операция отменена"

info "Отмонтирование резервной карты..."
umount "$TARGET_PARTITION" 2>/dev/null || true
ok "Резервная карта отмонтирована"

info "Подготовка файловой системы источника под размер резервной карты..."
umount "${SOURCE_DEVICE}1" 2>/dev/null || true
umount "$SOURCE_PARTITION" 2>/dev/null || true

target_sectors="$(blockdev --getsz "$TARGET_DEVICE")"
target_disk_bytes="$((target_sectors * 512))"
target_partition_bytes="$(( (target_sectors - 1064960) * 512 ))"
fs_block_size="$(tune2fs -l "$SOURCE_PARTITION" | awk -F: '/Block size/{gsub(/[[:space:]]/, "", $2); print $2}')"
source_blocks="$(tune2fs -l "$SOURCE_PARTITION" | awk -F: '/Block count/{gsub(/[[:space:]]/, "", $2); print $2}')"
target_blocks="$((target_partition_bytes / fs_block_size))"

[[ -n "$fs_block_size" && -n "$source_blocks" ]] || fail "Не удалось определить параметры ext4"
[[ "$target_blocks" -gt 0 ]] || fail "Некорректный размер целевого раздела"

if [[ "$source_blocks" -gt "$target_blocks" ]]; then
    info "Проверка ext4 источника перед уменьшением..."
    set +e
    e2fsck -f -y "$SOURCE_PARTITION"
    source_fsck_status=$?
    set -e
    [[ "$source_fsck_status" -le 1 ]] || fail "Проверка источника завершилась с кодом $source_fsck_status"
    info "Уменьшение ext4: $source_blocks → $target_blocks блоков..."
    resize2fs "$SOURCE_PARTITION" "$target_blocks"
    ok "Файловая система источника подготовлена"
else
    ok "Уменьшение файловой системы не требуется"
fi

info "Копирование рабочей системы. Объём около 62,5 ГБ..."
dd if="$SOURCE_DEVICE" of="$TARGET_DEVICE" \
    bs="$BLOCK_SIZE" count="$target_disk_bytes" \
    iflag=fullblock,count_bytes status=progress conv=fsync
ok "Данные скопированы"

info "Исправление границы последнего раздела под размер резервной карты..."
[[ "$target_sectors" -gt 1064960 ]] || fail "Некорректный размер резервной карты"
sfdisk --no-reread "$TARGET_DEVICE" <<EOF
label: dos
label-id: 0x5a471595
unit: sectors
start=16384, size=1048576, type=c
start=1064960, size=$((target_sectors - 1064960)), type=83
EOF
partprobe "$TARGET_DEVICE"
sleep 2
ok "Разметка резервной карты исправлена"

info "Проверка файловой системы резервной карты..."
set +e
e2fsck -f -y "${TARGET_DEVICE}2"
fsck_status=$?
set -e
[[ "$fsck_status" -le 1 ]] || fail "Проверка ext4 завершилась с кодом $fsck_status"
ok "Файловая система проверена"

sync
printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}КЛОНИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО${RESET}"
printf "%b\n" "${GREEN}Резервная карта: $TARGET_DEVICE${RESET}"
printf "%b\n" "${YELLOW}Безопасно извлеките карту и подпишите: VideoT16-RECOVERY${RESET}"
