#!/usr/bin/env bash

set -Eeuo pipefail

# Включает или выключает наземную Wi-Fi-точку VideoT16.
SETTINGS="/home/oleg/VideoT16/config/wifi_settings.json"
MARKER="/home/oleg/VideoT16/config/wifi_finish.request"

if [[ ! -f "$SETTINGS" ]]; then
  echo "Не найден файл настроек Wi-Fi: $SETTINGS" >&2
  exit 1
fi

CONNECTION_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["connection_name"])' "$SETTINGS")"

case "${1:-}" in
  start)
    rm -f "$MARKER"
    nmcli connection up "$CONNECTION_NAME"
    ;;
  stop)
    nmcli connection down "$CONNECTION_NAME" >/dev/null 2>&1 || true
    rm -f "$MARKER"
    ;;
  *)
    echo "Использование: $0 start|stop" >&2
    exit 2
    ;;
esac
