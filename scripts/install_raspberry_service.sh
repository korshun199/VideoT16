#!/usr/bin/env bash

set -Eeuo pipefail

# Устанавливает автозапуск рабочей системы VideoT16 на Raspberry Pi.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SOURCE="$PROJECT_DIR/deploy/videot16.service"
SERVICE_TARGET="/etc/systemd/system/videot16.service"

if [[ "$(id -u)" -ne 0 ]]; then
    echo "Запустите скрипт через sudo: sudo $0" >&2
    exit 1
fi

install -m 0644 "$SERVICE_SOURCE" "$SERVICE_TARGET"
systemctl daemon-reload
systemctl enable videot16.service
systemctl restart videot16.service

echo "VideoT16 установлен и запущен как системный сервис."
echo "Проверка: systemctl status videot16.service"
echo "Журнал: journalctl -u videot16.service -f"
