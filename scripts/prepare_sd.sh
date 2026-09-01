#!/usr/bin/env bash
set -euo pipefail

# Устройство microSD в USB-картридере.
DEVICE="/dev/sda"

# Проверяем, что устройство существует и является съёмным USB-диском.
if [[ ! -b "$DEVICE" ]]; then
    echo "Ошибка: устройство $DEVICE не найдено."
    exit 1
fi

read -r removable transport size model < <(
    lsblk -dnro RM,TRAN,SIZE,MODEL "$DEVICE"
)

if [[ "$removable" != "1" || "$transport" != "usb" ]]; then
    echo "Ошибка: $DEVICE не выглядит как съёмный USB-диск."
    echo "Определено: RM=$removable TRAN=$transport SIZE=$size MODEL=$model"
    exit 1
fi

echo "Найдено устройство: $DEVICE"
echo "Модель: $model"
echo "Размер: $size"
echo
echo "ВНИМАНИЕ: все данные на $DEVICE будут удалены."
read -r -p "Введите ERASE для подтверждения: " confirmation

if [[ "$confirmation" != "ERASE" ]]; then
    echo "Отменено."
    exit 0
fi

# Размонтируем разделы карты перед очисткой.
while read -r mountpoint; do
    [[ -n "$mountpoint" ]] || continue
    pkexec umount "$mountpoint"
done < <(lsblk -nrpo MOUNTPOINT "$DEVICE" | sed '/^$/d')

# Удаляем сигнатуры файловых систем и таблицу разделов.
pkexec wipefs --all --force "$DEVICE"
pkexec blockdev --rereadpt "$DEVICE" || true

echo
echo "Карта очищена: $DEVICE"
echo "Теперь её можно выбрать в Raspberry Pi Imager и записать образ."
