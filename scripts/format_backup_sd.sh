#!/usr/bin/env bash
set -Eeuo pipefail

# Форматирование только резервной SD-карты перед файловым клонированием.

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

mapfile -t disks < <(lsblk -dnro NAME,TYPE,RM,TRAN | awk '$2 == "disk" && $3 == "1" && $4 == "usb" {print $1}')
[[ "${#disks[@]}" -eq 2 ]] || fail "Нужно подключить ровно две USB-карты"

SOURCE_DEVICE=""
TARGET_DEVICE=""
for disk_name in "${disks[@]}"; do
    disk_path="/dev/$disk_name"
    [[ -b "${disk_path}1" ]] || fail "У $disk_path нет первого раздела"
    root_type="$(lsblk -dnro FSTYPE "${disk_path}2" 2>/dev/null || true)"
    if [[ "$root_type" == "ext4" ]]; then
        if [[ -z "$SOURCE_DEVICE" || "$(lsblk -bndo SIZE "$disk_path")" -gt "$(lsblk -bndo SIZE "$SOURCE_DEVICE")" ]]; then
            SOURCE_DEVICE="$disk_path"
        fi
    fi
done

[[ -n "$SOURCE_DEVICE" ]] || fail "Не найдена рабочая карта с ext4"

for disk_name in "${disks[@]}"; do
    disk_path="/dev/$disk_name"
    [[ "$disk_path" == "$SOURCE_DEVICE" ]] && continue
    disk_size="$(lsblk -bndo SIZE "$disk_path")"
    if [[ -z "$TARGET_DEVICE" || "$disk_size" -lt "$(lsblk -bndo SIZE "$TARGET_DEVICE")" ]]; then
        TARGET_DEVICE="$disk_path"
    fi
done

[[ -n "$TARGET_DEVICE" ]] || fail "Не найдена резервная карта"
[[ "$SOURCE_DEVICE" != "$TARGET_DEVICE" ]] || fail "Источник и цель совпали"

TARGET_BOOT="${TARGET_DEVICE}1"
TARGET_ROOT="${TARGET_DEVICE}2"

printf "%b\n" "${CYAN}=== VideoT16: форматирование резервной SD-карты ===${RESET}"
info "Рабочая карта НЕ будет изменена: $SOURCE_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$SOURCE_DEVICE"
info "Будет отформатирована только: $TARGET_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$TARGET_DEVICE"
warn "Все данные на $TARGET_DEVICE будут удалены."
printf "Введите FORMAT_BACKUP для продолжения: "
read -r confirmation
[[ "$confirmation" == "FORMAT_BACKUP" ]] || fail "Операция отменена"

if [[ ! -b "$TARGET_ROOT" ]]; then
    info "Создание разметки резервной карты..."
    umount "$TARGET_BOOT" 2>/dev/null || true
    target_sectors="$(blockdev --getsz "$TARGET_DEVICE")"
    [[ "$target_sectors" -gt 1064960 ]] || fail "Резервная карта слишком мала"
    sfdisk --wipe always "$TARGET_DEVICE" <<EOF
label: dos
unit: sectors
start=16384, size=1048576, type=c
start=1064960, size=$((target_sectors - 1064960)), type=83
EOF
    partprobe "$TARGET_DEVICE" || true
    udevadm settle
    sleep 2
    TARGET_BOOT="${TARGET_DEVICE}1"
    TARGET_ROOT="${TARGET_DEVICE}2"
    [[ -b "$TARGET_BOOT" && -b "$TARGET_ROOT" ]] || fail "Ядро не увидело новые разделы"
    ok "Разметка резервной карты создана"
fi

umount "$TARGET_BOOT" 2>/dev/null || true
umount "$TARGET_ROOT" 2>/dev/null || true

info "Форматирование boot-раздела FAT32..."
mkfs.vfat -F 32 -n BOOT "$TARGET_BOOT" >/dev/null
ok "boot-раздел готов"

info "Форматирование root-раздела ext4..."
mkfs.ext4 -F -E lazy_itable_init=0,lazy_journal_init=0 -L rootfs "$TARGET_ROOT" >/dev/null
sync
blockdev --flushbufs "$TARGET_ROOT" || true
udevadm settle
sleep 2

info "Проверка ext4..."
set +e
e2fsck -f -y "$TARGET_ROOT"
fsck_status=$?
set -e
[[ "$fsck_status" -le 1 ]] || fail "Проверка ext4 завершилась с кодом $fsck_status"

printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}РЕЗЕРВНАЯ КАРТА ОТФОРМАТИРОВАНА${RESET}"
printf "%b\n" "${GREEN}Рабочая карта не изменена: $SOURCE_DEVICE${RESET}"
