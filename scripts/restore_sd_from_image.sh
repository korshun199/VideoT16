#!/usr/bin/env bash
set -Eeuo pipefail

# Восстановление Raspberry Pi из сохранённого образа на резервную SD-карту.

IMAGE="/home/work/raspberry-backups/raspberrypi-videot16-2026-08-24.img.zst"
TARGET="/dev/sda"

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
[[ -b "$TARGET" ]] || fail "Резервная карта не найдена: $TARGET"

WORK_DIR="$(mktemp -d /home/work/raspberry-backups/videot16-restore.XXXXXX)"
LOOP_DEVICE=""
SRC_ROOT_DIR="$(mktemp -d /tmp/videot16-image-root.XXXXXX)"
SRC_BOOT_DIR="$(mktemp -d /tmp/videot16-image-boot.XXXXXX)"
DST_ROOT_DIR="$(mktemp -d /tmp/videot16-restore-root.XXXXXX)"
DST_BOOT_DIR="$(mktemp -d /tmp/videot16-restore-boot.XXXXXX)"

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

printf "%b\n" "${CYAN}=== VideoT16: восстановление SD-карты из образа ===${RESET}"
info "Источник: $IMAGE"
info "Цель: $TARGET"
lsblk -dnro NAME,MODEL,SERIAL,SIZE,TYPE "$TARGET"
warn "Карта $TARGET будет полностью подготовлена заново."
printf "Введите RESTORE_VIDEOT16 для продолжения: "
read -r confirmation
[[ "$confirmation" == "RESTORE_VIDEOT16" ]] || fail "Операция отменена"

info "Разворачивание сжатого образа во временный sparse-файл..."
zstd -d -c "$IMAGE" | dd of="$WORK_DIR/source.img" bs=16M iflag=fullblock conv=sparse status=progress
ok "Образ развернут"

info "Подключение разделов образа..."
LOOP_DEVICE="$(losetup --find --show --partscan "$WORK_DIR/source.img")"
udevadm settle
SRC_BOOT="${LOOP_DEVICE}p1"
SRC_ROOT="${LOOP_DEVICE}p2"
[[ -b "$SRC_BOOT" && -b "$SRC_ROOT" ]] || fail "Разделы образа не обнаружены"
ok "Образ подключен: $LOOP_DEVICE"

info "Подготовка разделов резервной карты..."
umount "${TARGET}1" 2>/dev/null || true
umount "${TARGET}2" 2>/dev/null || true
target_sectors="$(blockdev --getsz "$TARGET")"
root_sectors=$((32 * 1024 * 1024 * 1024 / 512))
[[ "$target_sectors" -gt $((1064960 + root_sectors)) ]] || fail "Карта слишком мала для 32 ГБ root-раздела"
sfdisk --wipe always "$TARGET" <<EOF
label: dos
unit: sectors
start=16384, size=1048576, type=c
start=1064960, size=$root_sectors, type=83
EOF
partx -d "$TARGET" 2>/dev/null || true
partprobe "$TARGET" || true
blockdev --rereadpt "$TARGET" 2>/dev/null || true
partx -a "$TARGET" 2>/dev/null || true
udevadm settle
sleep 2
TARGET_BOOT="${TARGET}1"
TARGET_ROOT="${TARGET}2"
[[ -b "$TARGET_BOOT" && -b "$TARGET_ROOT" ]] || fail "Разделы резервной карты не обнаружены"

mkfs.vfat -F 32 -n BOOT "$TARGET_BOOT" >/dev/null
# Явно задаём размер ФС: кардридер иногда временно сообщает старый размер раздела.
root_blocks=$((32 * 1024 * 1024 * 1024 / 4096))
mkfs.ext4 -F -b 4096 -O ^has_journal -E lazy_itable_init=0,lazy_journal_init=0 \
    -L rootfs "$TARGET_ROOT" "$root_blocks" >/dev/null
sync
blockdev --flushbufs "$TARGET_BOOT" || true
blockdev --flushbufs "$TARGET_ROOT" || true
udevadm settle
sleep 2

info "Проверка размера новой файловой системы..."
actual_root_blocks="$(dumpe2fs -h "$TARGET_ROOT" 2>/dev/null | awk -F: '/^Block count:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
[[ "$actual_root_blocks" == "$root_blocks" ]] || fail "Кардридер вернул старый суперблок: $actual_root_blocks блоков вместо $root_blocks"
ok "Новая ext4 имеет правильный размер: $actual_root_blocks блоков"

info "Проверка структуры новой ext4..."
set +e
e2fsck -f -y "$TARGET_ROOT"
fsck_status=$?
set -e
[[ "$fsck_status" -le 1 ]] || fail "Проверка новой ext4 завершилась с кодом $fsck_status"

info "Подключение образа и резервной карты..."
mount -o ro,noload "$SRC_ROOT" "$SRC_ROOT_DIR"
mount -o ro "$SRC_BOOT" "$SRC_BOOT_DIR"
mount -t ext4 "$TARGET_ROOT" "$DST_ROOT_DIR"
mount "$TARGET_BOOT" "$DST_BOOT_DIR"
ok "Файловые системы подключены"

info "Копирование rootfs из образа..."
rsync -aHAXx --numeric-ids --delete \
    --exclude=/dev/* \
    --exclude=/proc/* \
    --exclude=/sys/* \
    --exclude=/run/* \
    --exclude=/tmp/* \
    --exclude=/mnt/* \
    --exclude=/media/* \
    "$SRC_ROOT_DIR/" "$DST_ROOT_DIR/"
ok "rootfs восстановлен"

info "Копирование boot-раздела..."
rsync -rt --delete "$SRC_BOOT_DIR/" "$DST_BOOT_DIR/"

source_root_uuid="$(blkid -s UUID -o value "$SRC_ROOT")"
target_root_uuid="$(blkid -s UUID -o value "$TARGET_ROOT")"
source_root_partuuid="$(blkid -s PARTUUID -o value "$SRC_ROOT")"
target_root_partuuid="$(blkid -s PARTUUID -o value "$TARGET_ROOT")"

if [[ -f "$DST_ROOT_DIR/etc/fstab" ]]; then
    sed -i "s/$source_root_uuid/$target_root_uuid/g" "$DST_ROOT_DIR/etc/fstab"
fi
if [[ -f "$DST_BOOT_DIR/cmdline.txt" ]]; then
    sed -i "s#root=PARTUUID=[^ ]*#root=PARTUUID=$target_root_partuuid#g" "$DST_BOOT_DIR/cmdline.txt"
fi

sync
umount "$DST_BOOT_DIR"
umount "$DST_ROOT_DIR"

printf "%b\n" "${GREEN}============================================${RESET}"
printf "%b\n" "${GREEN}ВОССТАНОВЛЕНИЕ VideoT16 ЗАВЕРШЕНО${RESET}"
printf "%b\n" "${GREEN}Резервная карта: $TARGET${RESET}"
printf "%b\n" "${YELLOW}Извлеките карту и проверьте загрузку Raspberry Pi.${RESET}"
