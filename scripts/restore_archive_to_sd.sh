#!/usr/bin/env bash
set -Eeuo pipefail

# Восстановление системы VideoT16 на Raspberry после установки базовой ОС.
# Запускается на ноутбуке и не записывает побайтный образ SD-карты.

ARCHIVE="/media/oleg/VideoT16_Backups/ВЫБЕРИ_АРХИВ.tar.gz"
PI_USER="oleg"
PI_HOST="192.168.20.109"
PI_SERVICE="videot16.service"

RED='\033[1;31m'; GREEN='\033[1;32m'; YELLOW='\033[1;33m'; CYAN='\033[1;36m'; RESET='\033[0m'
info() { printf '%b\n' "${CYAN}[INFO]${RESET} $*"; }
ok() { printf '%b\n' "${GREEN}[ OK ]${RESET} $*"; }
warn() { printf '%b\n' "${YELLOW}[ВНИМАНИЕ]${RESET} $*"; }
fail() { printf '%b\n' "${RED}[ОШИБКА]${RESET} $*" >&2; exit 1; }

[[ "$EUID" -ne 0 ]] || fail "Запускайте без sudo: скрипт сам запросит права."
[[ -f "$ARCHIVE" ]] || fail "Архив не найден: $ARCHIVE"
command -v ssh >/dev/null || fail "Не найдена команда ssh"
command -v gzip >/dev/null || fail "Не найдена команда gzip"

printf '%b\n' "${CYAN}=== VideoT16: восстановление системы на Raspberry ===${RESET}"
info "Архив: $ARCHIVE"
info "Цель: $PI_USER@$PI_HOST"
warn "Будут заменены только файлы VideoT16 и сохранённые системные настройки."

gzip -t "$ARCHIVE" || fail "Архив повреждён"
ssh -o ConnectTimeout=8 "$PI_USER@$PI_HOST" true \
    || fail "Raspberry недоступна по SSH"

printf 'Введите RESTORE_VIDEOT16 для продолжения: '
read -r confirmation
[[ "$confirmation" == "RESTORE_VIDEOT16" ]] || fail "Операция отменена"

info "Подтвердите sudo на Raspberry..."
ssh -tt "$PI_USER@$PI_HOST" 'sudo -v'

ssh "$PI_USER@$PI_HOST" "sudo systemctl stop '$PI_SERVICE' 2>/dev/null || true"
info "Передаю архив и разворачиваю его на Raspberry..."
gzip -dc "$ARCHIVE" \
    | ssh "$PI_USER@$PI_HOST" "sudo tar -xzf - -C / --numeric-owner --xattrs --acls"

info "Включаю автозапуск VideoT16..."
ssh "$PI_USER@$PI_HOST" "sudo systemctl unmask '$PI_SERVICE' 2>/dev/null || true; sudo systemctl daemon-reload; sudo systemctl enable '$PI_SERVICE'; sudo systemctl start '$PI_SERVICE'"

ok "Система VideoT16 восстановлена на Raspberry."
ssh "$PI_USER@$PI_HOST" "systemctl status '$PI_SERVICE' --no-pager -l"
