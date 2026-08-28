#!/usr/bin/env bash

set -Eeuo pipefail

# Устанавливает только веб-панель оператора и не изменяет распознавание.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="videot16-web.service"
SERVICE_SOURCE="$PROJECT_DIR/deploy/$SERVICE_NAME"
SERVICE_TARGET="/etc/systemd/system/$SERVICE_NAME"

if [[ ! -f "$SERVICE_SOURCE" ]]; then
  echo "Ошибка: не найден файл службы $SERVICE_SOURCE" >&2
  exit 1
fi

if [[ ! -x "$PROJECT_DIR/scripts/run_config_web.sh" ]]; then
  echo "Ошибка: отсутствует исполняемый запуск веб-панели" >&2
  exit 1
fi

echo "Устанавливаю только $SERVICE_NAME"
sudo install -m 0644 "$SERVICE_SOURCE" "$SERVICE_TARGET"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"
sudo systemctl --no-pager --full status "$SERVICE_NAME"

echo
echo "Панель доступна по адресу: http://$(hostname -I | awk '{print $1}'):80"
echo "Автономная служба распознавания не изменялась."
