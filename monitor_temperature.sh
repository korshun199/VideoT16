#!/usr/bin/env bash

set -Eeuo pipefail

# Интервал наблюдения в секундах.
INTERVAL="5"

printf 'Монитор температуры Raspberry Pi. Для выхода нажмите Ctrl+C.\n'

while true; do
    temperature="$(vcgencmd measure_temp 2>/dev/null | tr -cd '0-9.')"
    throttle="$(vcgencmd get_throttled 2>/dev/null | tr -d '\r')"
    load="$(cut -d ' ' -f 1-3 /proc/loadavg)"

    green=$'\033[32m'
    yellow=$'\033[33m'
    red=$'\033[31m'
    cyan=$'\033[36m'
    reset=$'\033[0m'
    temperature_color="$green"
    temperature_marker="🟢"
    if [[ -n "$temperature" ]] && awk "BEGIN { exit !($temperature >= 70) }"; then
        temperature_color="$red"
        temperature_marker="🔴"
    elif [[ -n "$temperature" ]] && awk "BEGIN { exit !($temperature >= 55) }"; then
        temperature_color="$yellow"
        temperature_marker="🟡"
    fi
    throttle_color="$green"
    [[ "$throttle" != "throttled=0x0" && "$throttle" != "throttled=0x50000" ]] && throttle_color="$red"

    printf '%s%s [%s]%s Температура: %s%s%s°C%s | Питание: %s%s%s | Load: %s%s%s\n' \
        "$temperature_marker" \
        "$cyan" "$(date '+%d.%m.%Y %H:%M:%S')" "$reset" \
        "$temperature_color" "$temperature" "$reset" \
        "$throttle_color" "$throttle" "$reset" "$yellow" "$load" "$reset"

    if [[ -n "$temperature" ]] && awk "BEGIN { exit !($temperature >= 70) }"; then
        printf 'ПРЕДУПРЕЖДЕНИЕ: температура выше 70°C.\n' >&2
    fi
    sleep "$INTERVAL"
done
