#!/bin/sh
# Выводит температуру именно процессорного пакета, находя датчик по его типу.
set -eu

# Путь к очередной зоне sysfs, которую проверяем на тип датчика CPU.
for thermal_zone in /sys/class/thermal/thermal_zone*; do
    # Пропускаем зоны, для которых система не публикует тип датчика.
    test -r "$thermal_zone/type" || continue

    # Тип датчика, например x86_pkg_temp для температуры процессорного пакета.
    IFS= read -r sensor_type < "$thermal_zone/type" || continue
    test "$sensor_type" = 'x86_pkg_temp' || continue

    # Температура из sysfs хранится в тысячных долях градуса Цельсия.
    IFS= read -r milli_celsius < "$thermal_zone/temp" || exit 1
    printf '%s °C' "$((milli_celsius / 1000))"
    exit 0
done

printf '%s' 'датчик недоступен'
exit 1
