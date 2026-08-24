#!/usr/bin/env bash

set -Eeuo pipefail

# Включает или выключает наземную Wi-Fi-точку VideoT16.
CONNECTION_NAME="VideoT16-Setup"
MARKER="/home/oleg/VideoT16/config/wifi_finish.request"

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
