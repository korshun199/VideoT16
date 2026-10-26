#!/usr/bin/env bash
set -Eeuo pipefail

# Путь к официальному ISO Lubuntu, который предварительно скачивается отдельно.
iso_path="/home/work/tmp/lubuntu-26.04.1-desktop-amd64.iso"
# Путь к контрольным суммам, полученным с официального сервера Ubuntu.
checksums_path="/home/work/tmp/lubuntu-26.04.1-SHA256SUMS"
# Официальная SHA256-сумма ISO, сверенная со списком Ubuntu.
expected_sha256="a8038585420f2270845549c6f493e0ec98e68d4b2c49571cbe500a4b5fab3062"
# Имя ISO-файла для поиска его строки в списке контрольных сумм.
iso_name="lubuntu-26.04.1-desktop-amd64.iso"
# Серийный номер именно маленькой установочной флешки Silicon-Power.
expected_serial="13010368032000001606"
# Ожидаемая модель маленькой установочной флешки.
expected_model="Silicon-Power8G"
# Флаг проверки без записи; включается параметром --check.
check_only="false"

if [[ "${1:-}" == "--check" ]]; then
    check_only="true"
elif [[ "$#" -ne 0 ]]; then
    printf 'Использование: %s [--check]\n' "$0" >&2
    exit 2
fi

if [[ ! -f "$iso_path" || ! -f "$checksums_path" ]]; then
    printf 'ОШИБКА: не найдены ISO или файл SHA256SUMS.\nISO: %s\nСуммы: %s\n' \
        "$iso_path" "$checksums_path" >&2
    exit 1
fi

# Фактическая сумма скачанного ISO для независимой проверки перед записью.
actual_sha256="$(sha256sum "$iso_path" | awk '{print $1}')"
# Сумма ISO, извлечённая из официального списка контрольных сумм.
listed_sha256="$(awk -v name="$iso_name" '$2 == "*" name {print $1}' "$checksums_path")"

if [[ "$listed_sha256" != "$expected_sha256" || "$actual_sha256" != "$expected_sha256" ]]; then
    printf 'ОШИБКА: SHA256 не совпадает; USB не будет изменена.\nОжидалось: %s\nВ списке: %s\nФайл: %s\n' \
        "$expected_sha256" "$listed_sha256" "$actual_sha256" >&2
    exit 1
fi

# Имя блочного устройства, найденное по уникальному серийному номеру.
device_name="$(lsblk -dn -o NAME,SERIAL | awk -v serial="$expected_serial" '$2 == serial {print $1}')"
if [[ -z "$device_name" || "$device_name" == *$'\n'* ]]; then
    printf 'ОШИБКА: не нашла ровно одну флешку с серийным номером %s.\n' "$expected_serial" >&2
    exit 1
fi

# Полный путь к найденному USB-диску; разделы намеренно не выбираются.
target_device="/dev/$device_name"
# Модель устройства, проверяемая вместе с серийным номером.
actual_model="$(lsblk -dn -o MODEL "$target_device" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
# Признак съёмного устройства из системного списка блоковых дисков.
removable_flag="$(lsblk -dn -o RM "$target_device" | tr -d '[:space:]')"
# Точки монтирования самого диска и всех его разделов.
mounted_partitions="$(lsblk -nr -o MOUNTPOINT "$target_device" | sed '/^[[:space:]]*$/d')"
# Размер ISO в байтах для проверки, что он полностью помещается на носитель.
iso_bytes="$(stat -c '%s' "$iso_path")"
# Полный размер целевой флешки в байтах; lsblk читает размер без доступа к устройству через blockdev.
device_bytes="$(lsblk -bdn -o SIZE "$target_device" | tr -d '[:space:]')"

if [[ "$actual_model" != "$expected_model" || "$removable_flag" != "1" ]]; then
    printf 'ОШИБКА: найдено неожиданное устройство; запись запрещена.\n' >&2
    lsblk -o NAME,PATH,MODEL,SERIAL,SIZE,RM,MOUNTPOINTS "$target_device" >&2
    exit 1
fi

if [[ -n "$mounted_partitions" ]]; then
    printf 'ОШИБКА: раздел целевой флешки смонтирован; запись запрещена.\n%s\n' "$mounted_partitions" >&2
    exit 1
fi

if (( device_bytes < iso_bytes + 67108864 )); then
    printf 'ОШИБКА: ISO не помещается на целевую флешку с запасом.\n' >&2
    exit 1
fi

printf 'Проверка образа: OK (SHA256 %s)\n' "$actual_sha256"
printf 'Точное целевое устройство для УСТАНОВОЧНОГО образа:\n'
lsblk -o NAME,PATH,MODEL,SERIAL,SIZE,RM,MOUNTPOINTS "$target_device"
printf 'ВНИМАНИЕ: запись сотрёт всё содержимое только этого USB-диска.\n'

if [[ "$check_only" == "true" ]]; then
    printf 'РЕЖИМ ПРОВЕРКИ: запись не выполнялась.\n'
    exit 0
fi

# Подтверждение владельца предотвращает случайную запись не на ту флешку.
read -r -p "Для стирания именно $target_device введите ERASE-$expected_model: " confirmation
if [[ "$confirmation" != "ERASE-$expected_model" ]]; then
    printf 'Отменено: подтверждение не совпало; носитель не изменён.\n'
    exit 1
fi

printf 'Запрашиваю sudo и записываю установочный образ; целевой диск перепроверен по серийному номеру.\n'
sudo dd if="$iso_path" of="$target_device" bs=4M iflag=fullblock status=progress conv=fsync
sync

printf 'Сверяю записанные байты с ISO; чтение флешки может занять несколько минут.\n'
sudo cmp -n "$iso_bytes" "$iso_path" "$target_device"
printf 'ГОТОВО: установочная флешка создана и побайтно проверена. Перезагружать ноутбук можно после безопасного извлечения носителя.\n'
