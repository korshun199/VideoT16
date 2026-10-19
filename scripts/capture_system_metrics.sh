#!/usr/bin/env bash

# Снимает повторяемый профиль нагрузки Linux для сравнения режимов проекта.

set -u
set -o pipefail

# Единая локаль делает числовой вывод пригодным для последующего сравнения.
export LC_ALL=C

# Метка отличает чистую систему от профиля с запущенным распознаванием.
profile_name="${1:-measurement}"
# Время входит в имя, поэтому прежние измерения никогда не перезаписываются.
timestamp="$(date '+%Y%m%d_%H%M%S')"
# Все отчёты хранятся рядом и остаются переносимыми вместе с проектом.
report_dir="reports/performance"
report_file="${2:-${report_dir}/${timestamp}_${profile_name}.txt}"

mkdir -p "$(dirname "$report_file")"
umask 022

# Одновременно показывает измерения в терминале и сохраняет их в отчёт.
exec > >(tee "$report_file") 2>&1

# Печатает заметный заголовок раздела.
section() {
    printf '\n===== %s =====\n' "$1"
}

# Запускает необязательную команду, если она установлена в системе.
run_if_available() {
    local command_name="$1"
    shift
    if command -v "$command_name" >/dev/null 2>&1; then
        "$command_name" "$@" || true
    else
        printf 'Команда %s не установлена.\n' "$command_name"
    fi
}

# Показывает значения всех существующих файлов по переданной маске sysfs.
show_sysfs_values() {
    local description="$1"
    shift
    local value_file
    local found=0

    printf '%s\n' "$description"
    for value_file in "$@"; do
        if [[ -r "$value_file" ]]; then
            printf '%s: ' "$value_file"
            cat "$value_file"
            found=1
        fi
    done
    if (( found == 0 )); then
        printf 'Доступных счётчиков нет.\n'
    fi
}

section "Паспорт измерения"
printf 'Профиль: %s\n' "$profile_name"
printf 'Файл: %s\n' "$report_file"
printf 'Начало: %s\n' "$(date --iso-8601=seconds)"
printf 'Пользователь: %s\n' "$(id -un)"
printf 'Ядро: %s\n' "$(uname -srmo)"
if [[ -r /etc/os-release ]]; then
    grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true
fi
uptime || true

section "Процессор и топология"
run_if_available lscpu
show_sysfs_values \
    "Регуляторы частоты CPU:" \
    /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
show_sysfs_values \
    "Текущие частоты CPU, кГц:" \
    /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq
show_sysfs_values \
    "Предпочтения энергопотребления CPU:" \
    /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference

section "Нагрузка и ожидания ядра"
cat /proc/loadavg
for pressure_file in /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io; do
    if [[ -r "$pressure_file" ]]; then
        printf '\n%s\n' "$pressure_file"
        cat "$pressure_file"
    fi
done

section "Короткий образец CPU"
run_if_available vmstat 1 6

section "Загрузка каждого логического ядра"
run_if_available mpstat -P ALL 1 4

section "Память и swap"
run_if_available free -h
run_if_available swapon --show --bytes

section "Процессы с наибольшей загрузкой CPU"
ps -eo pid,ppid,comm,stat,psr,%cpu,%mem,rss,etime --sort=-%cpu | head -n 26 || true

section "Процессы с наибольшим потреблением памяти"
ps -eo pid,ppid,comm,stat,psr,%cpu,%mem,rss,etime --sort=-rss | head -n 26 || true

section "Активность процессов за три секунды"
run_if_available pidstat -u -r -d 1 3

section "Температуры и вентиляторы"
run_if_available sensors
show_sysfs_values \
    "Счётчики температурного троттлинга:" \
    /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count \
    /sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count

section "Диски и ввод-вывод"
run_if_available iostat -xz 1 3

section "Intel GPU и графическая подсистема"
run_if_available lspci -nnk
run_if_available glxinfo -B
show_sysfs_values \
    "Счётчики занятости и частоты GPU:" \
    /sys/class/drm/card*/device/gpu_busy_percent \
    /sys/class/drm/card*/gt/gt*/rps_cur_freq_mhz \
    /sys/class/drm/card*/gt_cur_freq_mhz

section "Питание и аккумулятор"
run_if_available upower -d

section "Подключённые USB-устройства"
run_if_available lsusb

section "Видеокамеры V4L2"
run_if_available v4l2-ctl --list-devices

section "Порты и процессы проекта"
ls -l /dev/video* /dev/ttyACM* 2>/dev/null || true
if pgrep -af 'src\.local_object_detection|run_modeling\.sh|run_military\.sh' >/dev/null; then
    pgrep -af 'src\.local_object_detection|run_modeling\.sh|run_military\.sh' || true
else
    printf 'Процессы VideoT16 не запущены.\n'
fi

section "Дополнительный счётчик Intel"
if command -v turbostat >/dev/null 2>&1; then
    timeout 5s turbostat --quiet --Summary --interval 1 --num_iterations 3 || true
else
    printf 'Команда turbostat не установлена.\n'
fi

section "Завершение"
printf 'Окончание: %s\n' "$(date --iso-8601=seconds)"
printf 'Отчёт сохранён: %s\n' "$report_file"
