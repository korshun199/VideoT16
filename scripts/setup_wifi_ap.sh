#!/usr/bin/env bash

set -Eeuo pipefail

# Однократно настраивает наземную Wi-Fi-точку Raspberry Pi.
if [[ "${EUID}" -ne 0 ]]; then
  echo "Запустите скрипт через sudo: sudo $0 ИМЯ_СЕТИ ПАРОЛЬ" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETTINGS_PATH="$PROJECT_DIR/config/wifi_settings.json"

if [[ $# -ge 2 ]]; then
  python3 - "$SETTINGS_PATH" "$1" "$2" <<'PY'
import json
import os
import pwd
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
data = {
    "connection_name": "VideoT16-Setup",
    "ssid": sys.argv[2],
    "password": sys.argv[3],
    "interface": "wlan0",
    "address": "192.168.50.1/24",
    "channel": 6,
    "timeout_minutes": 15,
    "web_port": 8080,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="wifi_settings.", suffix=".tmp", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    json.dump(data, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
os.replace(temporary, path)
owner = pwd.getpwnam("oleg")
os.chown(path, owner.pw_uid, owner.pw_gid)
os.chmod(path, 0o600)
PY
elif [[ ! -f "$SETTINGS_PATH" ]]; then
  echo "Нужны SSID и пароль либо существующий $SETTINGS_PATH" >&2
  echo "Пример: sudo $0 VideoT16 'пароль-минимум-8-символов'" >&2
  exit 2
fi

mapfile -t WIFI_VALUES < <(python3 - "$SETTINGS_PATH" <<'PY'
import json
import sys

settings = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("connection_name", "ssid", "password", "interface", "address", "channel", "timeout_minutes"):
    print(settings[key])
PY
)
CONNECTION_NAME="${WIFI_VALUES[0]}"
SSID="${WIFI_VALUES[1]}"
PASSWORD="${WIFI_VALUES[2]}"
INTERFACE="${WIFI_VALUES[3]}"
ADDRESS="${WIFI_VALUES[4]}"
CHANNEL="${WIFI_VALUES[5]}"
TIMEOUT_MINUTES="${WIFI_VALUES[6]}"

if [[ ${#PASSWORD} -lt 8 ]]; then
  echo "Пароль Wi-Fi должен содержать минимум 8 символов." >&2
  exit 2
fi

nmcli connection delete "$CONNECTION_NAME" >/dev/null 2>&1 || true
nmcli connection add type wifi ifname "$INTERFACE" con-name "$CONNECTION_NAME" ssid "$SSID"
nmcli connection modify "$CONNECTION_NAME" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel "$CHANNEL" \
  802-11-wireless-security.key-mgmt wpa-psk \
  802-11-wireless-security.psk "$PASSWORD" \
  ipv4.method shared \
  ipv4.addresses "$ADDRESS" \
  ipv6.method disabled \
  connection.autoconnect no

install -m 0755 "$PROJECT_DIR/deploy/videot16-wifi-control.sh" /usr/local/sbin/videot16-wifi-control.sh
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi.service" /etc/systemd/system/videot16-wifi.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-timeout.service" /etc/systemd/system/videot16-wifi-timeout.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-timeout.timer" /etc/systemd/system/videot16-wifi-timeout.timer
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-finish.service" /etc/systemd/system/videot16-wifi-finish.service
install -m 0644 "$PROJECT_DIR/deploy/videot16-wifi-finish.path" /etc/systemd/system/videot16-wifi-finish.path
mkdir -p /etc/systemd/system/videot16-wifi-timeout.timer.d
python3 - "$TIMEOUT_MINUTES" <<'PY'
from pathlib import Path
import sys

minutes = int(sys.argv[1])
Path("/etc/systemd/system/videot16-wifi-timeout.timer.d/settings.conf").write_text(
    f"[Timer]\nOnActiveSec={minutes}min\n", encoding="utf-8"
)
PY
systemctl daemon-reload
systemctl enable videot16-wifi.service videot16-wifi-timeout.timer videot16-wifi-finish.path
systemctl stop videot16-wifi-finish.path >/dev/null 2>&1 || true
systemctl restart videot16-wifi.service
systemctl restart videot16-wifi-timeout.timer
systemctl start videot16-wifi-finish.path

echo "Wi-Fi-точка VideoT16 настроена: SSID=$SSID, адрес=$ADDRESS"
echo "Таймер отключит её через $TIMEOUT_MINUTES минут."
