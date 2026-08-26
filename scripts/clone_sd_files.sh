#!/usr/bin/env bash
set -Eeuo pipefail

# Клонирование файлов Raspberry без изменения рабочей SD-карты.

SOURCE_DEVICE=""
TARGET_DEVICE=""
SOURCE_ROOT=""
SOURCE_BOOT=""
TARGET_ROOT=""
TARGET_BOOT=""

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() { printf "%b\n" "${CYAN}[INFO]${RESET} $*"; }
ok() { printf "%b\n" "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf "%b\n" "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf "%b\n" "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

SRC_ROOT_DIR="$(mktemp -d /tmp/videot16-source.XXXXXX)"
SRC_BOOT_DIR="$(mktemp -d /tmp/videot16-boot.XXXXXX)"
DST_ROOT_DIR="$(mktemp -d /tmp/videot16-target.XXXXXX)"
DST_BOOT_DIR="$(mktemp -d /tmp/videot16-target-boot.XXXXXX)"

cleanup() {
    sync || true
    umount "$DST_BOOT_DIR" 2>/dev/null || true
    umount "$DST_ROOT_DIR" 2>/dev/null || true
    umount "$SRC_BOOT_DIR" 2>/dev/null || true
    umount "$SRC_ROOT_DIR" 2>/dev/null || true
    rmdir "$SRC_ROOT_DIR" "$SRC_BOOT_DIR" "$DST_ROOT_DIR" "$DST_BOOT_DIR" 2>/dev/null || true
}
trap cleanup EXIT

[[ "$EUID" -eq 0 ]] || fail "Запустите: sudo $0"
for command_name in lsblk blkid mkfs.vfat mkfs.ext4 e2fsck mount umount rsync blockdev udevadm; do
    command -v "$command_name" >/dev/null || fail "Не найдена команда: $command_name"
done

# Имена /dev/sdX могут поменяться после перезагрузки USB-кардридеров.
mapfile -t removable_disks < <(lsblk -dnro NAME,TYPE,RM,TRAN | awk '$2 == "disk" && $3 == "1" && $4 == "usb" {print $1}')
[[ "${#removable_disks[@]}" -eq 2 ]] || fail "Нужно подключить ровно две USB-карты памяти"

for disk_name in "${removable_disks[@]}"; do
    disk_path="/dev/$disk_name"
    root_partition="${disk_path}2"
    [[ -b "$root_partition" ]] || fail "У $disk_path нет второго раздела"
    disk_size="$(lsblk -bndo SIZE "$disk_path")"
    root_type="$(lsblk -dnro FSTYPE "$root_partition")"
    if [[ "$root_type" == "ext4" ]]; then
        if [[ -z "$SOURCE_DEVICE" || "$disk_size" -gt "$(lsblk -bndo SIZE "$SOURCE_DEVICE")" ]]; then
            SOURCE_DEVICE="$disk_path"
        fi
    elif [[ -z "$TARGET_DEVICE" || "$disk_size" -lt "$(lsblk -bndo SIZE "$TARGET_DEVICE")" ]]; then
        TARGET_DEVICE="$disk_path"
    fi
done

# После неудачной попытки резервная карта тоже может иметь ext4.
# В текущей паре рабочая карта немного больше резервной.
if [[ -z "$TARGET_DEVICE" && -n "$SOURCE_DEVICE" ]]; then
    for disk_name in "${removable_disks[@]}"; do
        disk_path="/dev/$disk_name"
        [[ "$disk_path" == "$SOURCE_DEVICE" ]] && continue
        disk_size="$(lsblk -bndo SIZE "$disk_path")"
        source_size="$(lsblk -bndo SIZE "$SOURCE_DEVICE")"
        if [[ "$disk_size" -lt "$source_size" ]]; then
            TARGET_DEVICE="$disk_path"
            break
        fi
    done
fi

[[ -n "$SOURCE_DEVICE" ]] || fail "Не найдена рабочая карта с ext4-разделом"
[[ -n "$TARGET_DEVICE" ]] || fail "Не найдена резервная карта"
[[ "$SOURCE_DEVICE" != "$TARGET_DEVICE" ]] || fail "Источник и цель совпали"

SOURCE_ROOT="${SOURCE_DEVICE}2"
SOURCE_BOOT="${SOURCE_DEVICE}1"
TARGET_ROOT="${TARGET_DEVICE}2"
TARGET_BOOT="${TARGET_DEVICE}1"

for device in "$SOURCE_ROOT" "$SOURCE_BOOT" "$TARGET_ROOT" "$TARGET_BOOT"; do
    [[ -b "$device" ]] || fail "Не найдено устройство: $device"
done

printf "%b\n" "${CYAN}=== VideoT16: файловый клон SD-карты ===${RESET}"
info "Источник — рабочая карта: $SOURCE_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$SOURCE_DEVICE"
info "Цель — резервная карта: $TARGET_DEVICE"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$TARGET_DEVICE"
warn "Будут перезаписаны только разделы резервной карты $TARGET_DEVICE."
warn "Рабочая карта $SOURCE_DEVICE не форматируется и не уменьшается."
printf "Введите CLONE_FILES для продолжения: "
read -r confirmation
[[ "$confirmation" == "CLONE_FILES" ]] || fail "Операция отменена"

SOURCE_ROOT_UUID="$(blkid -s UUID -o value "$SOURCE_ROOT")"
[[ -n "$SOURCE_ROOT_UUID" ]] || fail "Не удалось прочитать UUID файловой системы источника"

info "Отмонтирование обеих карт..."
umount "$TARGET_BOOT" 2>/dev/null || true
umount "$TARGET_ROOT" 2>/dev/null || true
umount "$SOURCE_BOOT" 2>/dev/null || true
umount "$SOURCE_ROOT" 2>/dev/null || true
ok "Карты подготовлены"

info "Форматирование только резервных разделов..."
mkfs.vfat -F 32 -n BOOT "$TARGET_BOOT" >/dev/null
# Для этой резервной карты journal повреждался при создании.
# Без journal пустой раздел стабильно монтируется и проверяется.
mkfs.ext4 -F -O ^has_journal -E lazy_itable_init=0,lazy_journal_init=0 -L rootfs "$TARGET_ROOT" >/dev/null
sync
blockdev --flushbufs "$TARGET_BOOT" || true
blockdev --flushbufs "$TARGET_ROOT" || true
udevadm settle
sleep 2
info "Проверка пустой ext4 на резервной карте..."
set +e
e2fsck -f -y "$TARGET_ROOT"
target_fsck_status=$?
set -e
[[ "$target_fsck_status" -le 1 ]] || fail "Проверка резервной ext4 завершилась с кодом $target_fsck_status"
ok "Резервные разделы подготовлены"

info "Подключение файловых систем..."
mount -o ro "$SOURCE_ROOT" "$SRC_ROOT_DIR"
mount -o ro "$SOURCE_BOOT" "$SRC_BOOT_DIR"
mount -t ext4 "$TARGET_ROOT" "$DST_ROOT_DIR"
mount "$TARGET_BOOT" "$DST_BOOT_DIR"
ok "Файловые системы подключены"

TARGET_ROOT_UUID="$(blkid -s UUID -o value "$TARGET_ROOT")"
[[ -n "$TARGET_ROOT_UUID" ]] || fail "Не удалось прочитать UUID резервной файловой системы"

info "Копирование rootfs, занятые данные вместо пустых 60 ГБ..."
rsync -aHAXx --numeric-ids --delete \
    --exclude=/dev/* \
    --exclude=/proc/* \
    --exclude=/sys/* \
    --exclude=/run/* \
    --exclude=/tmp/* \
    --exclude=/mnt/* \
    --exclude=/media/* \
    "$SRC_ROOT_DIR/" "$DST_ROOT_DIR/"
if [[ -f "$DST_ROOT_DIR/etc/fstab" ]]; then
    sed -i "s/$SOURCE_ROOT_UUID/$TARGET_ROOT_UUID/g" "$DST_ROOT_DIR/etc/fstab"
fi
ok "rootfs скопирован"

info "Копирование загрузочного раздела..."
rsync -rt --delete "$SRC_BOOT_DIR/" "$DST_BOOT_DIR/"
ok "bootfs скопирован"

[[ -f "$DST_ROOT_DIR/etc/os-release" ]] || fail "На резервной карте нет /etc/os-release"
[[ -f "$DST_BOOT_DIR/config.txt" || -f "$DST_BOOT_DIR/config.txt" ]] || fail "На резервной карте нет boot/config.txt"
sync

printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}ФАЙЛОВЫЙ КЛОН УСПЕШНО ЗАВЕРШЁН${RESET}"
printf "%b\n" "${GREEN}Источник не изменён: $SOURCE_DEVICE${RESET}"
printf "%b\n" "${GREEN}Резервная карта: $TARGET_DEVICE${RESET}"
printf "%b\n" "${YELLOW}Теперь можно безопасно извлечь резервную карту.${RESET}"
