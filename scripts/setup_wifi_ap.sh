#!/usr/bin/env bash

set -Eeuo pipefail

# Однократно настраивает наземную Wi-Fi-точку Raspberry Pi.
if [[ "${EUID}" -ne 0 ]]; then
  echo "Запустите скрипт через sudo: sudo $0 ИМЯ_СЕТИ ПАРОЛЬ" >&2
  exit 1
fi

SSID="${1:-}"
PASSWORD="${2:-}"
CONNECTION_NAME="VideoT16-Setup"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$SSID" || ${#PASSWORD} -lt 8 ]]; then
  echo "Нужно указать имя сети и пароль минимум из 8 символов." >&2
  echo "Пример: sudo $0 VideoT16 БезопасныйПароль" >&2
  exit 2
fi

nmcli connection delete "$CONNECTION_NAME" >/dev/null 2>&1 || true
nmcli connection add type wifi ifname wlan0 con-name "$CONNECTION_NAME" ssid "$SSID"
nmcli connection modify "$CONNECTION_NAME" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel 6 \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses 192.168.50.1/24 \
  ipv6.method disabled \
  connection.autoconnect no

install -m 0755 "$PROJECT_DIR/deploy/videot16-wifi-control.sh" /usr/local/sbin/videot16-wifi-control.sh
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi.service" /etc/systemd/system/videot16-wifi.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-timeout.service" /etc/systemd/system/videot16-wifi-timeout.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-timeout.timer" /etc/systemd/system/videot16-wifi-timeout.timer
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-finish.service" /etc/systemd/system/videot16-wifi-finish.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-finish.path" /etc/systemd/system/videot16-wifi-finish.path
systemctl daemon-reload
systemctl enable videot16-wifi.service videot16-wifi-timeout.timer videot16-wifi-finish.path
systemctl start videot16-wifi.service
systemctl start videot16-wifi-timeout.timer videot16-wifi-finish.path

echo "Wi-Fi-точка VideoT16 настроена: SSID=$SSID, адрес=192.168.50.1"
echo "Таймер отключит её через 15 минут после загрузки."
