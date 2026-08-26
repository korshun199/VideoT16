#!/usr/bin/env bash
set -Eeuo pipefail

# Копирование VideoT16 на заранее подготовленную SD-карту.

IMAGE="/home/work/raspberry-backups/raspberrypi-videot16-2026-08-24.img.zst"
TARGET="/dev/sda"

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() { printf "%b\n" "${CYAN}[INFO]${RESET} $*"; }
ok() { printf "%b\n" "${GREEN}[ OK ]${RESET} $*"; }
fail() { printf "%b\n" "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

[[ "$EUID" -eq 0 ]] || fail "Запустите: sudo $0"
[[ -f "$IMAGE" ]] || fail "Образ не найден: $IMAGE"
[[ -b "$TARGET" ]] || fail "Карта не найдена: $TARGET"

WORK_DIR="$(mktemp -d /home/work/raspberry-backups/videot16-copy.XXXXXX)"
LOOP_DEVICE=""
SRC_ROOT_DIR="$(mktemp -d /tmp/videot16-image-root.XXXXXX)"
SRC_BOOT_DIR="$(mktemp -d /tmp/videot16-image-boot.XXXXXX)"
DST_ROOT_DIR="$(mktemp -d /tmp/videot16-copy-root.XXXXXX)"
DST_BOOT_DIR="$(mktemp -d /tmp/videot16-copy-boot.XXXXXX)"

cleanup() {
    sync || true
    umount "$DST_BOOT_DIR" 2>/dev/null || true
    umount "$DST_ROOT_DIR" 2>/dev/null || true
    umount "$SRC_BOOT_DIR" 2>/dev/null || true
    umount "$SRC_ROOT_DIR" 2>/dev/null || true
    [[ -z "$LOOP_DEVICE" ]] || losetup -d "$LOOP_DEVICE" 2>/dev/null || true
    rm -rf "$WORK_DIR" "$SRC_ROOT_DIR" "$SRC_BOOT_DIR" "$DST_ROOT_DIR" "$DST_BOOT_DIR"
}
trap cleanup EXIT

printf "%b\n" "${CYAN}=== VideoT16: копирование на подготовленную SD-карту ===${RESET}"
lsblk -f "$TARGET"
printf "Введите COPY_VIDEOT16 для продолжения: "
read -r confirmation
[[ "$confirmation" == "COPY_VIDEOT16" ]] || fail "Операция отменена"

info "Проверка разделов SD-карты..."
boot_type="$(blkid -s TYPE -o value "${TARGET}1")"
root_type="$(blkid -s TYPE -o value "${TARGET}2")"
[[ "$boot_type" == "vfat" ]] || fail "${TARGET}1 должен быть vfat, найдено: $boot_type"
[[ "$root_type" == "ext4" ]] || fail "${TARGET}2 должен быть ext4, найдено: $root_type"
ok "Разделы имеют правильные типы"

info "Разворачивание образа во временный sparse-файл..."
zstd -d -c "$IMAGE" | dd of="$WORK_DIR/source.img" bs=16M iflag=fullblock conv=sparse status=progress
LOOP_DEVICE="$(losetup --find --show --partscan "$WORK_DIR/source.img")"
udevadm settle
SRC_BOOT="${LOOP_DEVICE}p1"
SRC_ROOT="${LOOP_DEVICE}p2"
[[ -b "$SRC_BOOT" && -b "$SRC_ROOT" ]] || fail "Разделы образа не обнаружены"
ok "Образ подключён: $LOOP_DEVICE"

umount "${TARGET}1" 2>/dev/null || true
umount "${TARGET}2" 2>/dev/null || true
mount -o ro,noload "$SRC_ROOT" "$SRC_ROOT_DIR"
mount -o ro "$SRC_BOOT" "$SRC_BOOT_DIR"
mount "${TARGET}2" "$DST_ROOT_DIR"
mount "${TARGET}1" "$DST_BOOT_DIR"
ok "Разделы подключены"

info "Копирование rootfs..."
rsync -aHAXx --numeric-ids --delete \
    --exclude=/dev/* --exclude=/proc/* --exclude=/sys/* --exclude=/run/* \
    --exclude=/tmp/* --exclude=/mnt/* --exclude=/media/* \
    "$SRC_ROOT_DIR/" "$DST_ROOT_DIR/"

info "Копирование boot-раздела..."
rsync -rt --delete "$SRC_BOOT_DIR/" "$DST_BOOT_DIR/"

source_root_uuid="$(blkid -s UUID -o value "$SRC_ROOT")"
target_root_uuid="$(blkid -s UUID -o value "${TARGET}2")"
target_root_partuuid="$(blkid -s PARTUUID -o value "${TARGET}2")"

if [[ -f "$DST_ROOT_DIR/etc/fstab" ]]; then
    sed -i "s/$source_root_uuid/$target_root_uuid/g" "$DST_ROOT_DIR/etc/fstab"
fi
if [[ -f "$DST_BOOT_DIR/cmdline.txt" ]]; then
    sed -i "s#root=PARTUUID=[^ ]*#root=PARTUUID=$target_root_partuuid#g" "$DST_BOOT_DIR/cmdline.txt"
fi

sync
printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}КОПИРОВАНИЕ VideoT16 НА SD ЗАВЕРШЕНО${RESET}"
printf "%b\n" "${YELLOW}Извлеките карту и проверьте загрузку Raspberry Pi.${RESET}"
