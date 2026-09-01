#!/usr/bin/env bash

set -Eeuo pipefail

# Адрес и пользователь Raspberry Pi.
RASPBERRY_HOST="192.168.20.121"
RASPBERRY_USER="oleg"
REMOTE_PROJECT="/home/oleg/VideoT16"
SERVICE_NAME="videot16.service"

# Проверяет процент и передаёт его на Raspberry.
if [[ $# -ne 1 ]]; then
    printf 'Использование: %s ПРОЦЕНТ\n' "$0" >&2
    printf 'Пример: %s 50\n' "$0" >&2
    exit 2
fi

CONFIDENCE_PERCENT="$1"
if ! [[ "$CONFIDENCE_PERCENT" =~ ^[0-9]+$ ]] || (( CONFIDENCE_PERCENT < 1 || CONFIDENCE_PERCENT > 99 )); then
    printf '[ОШИБКА] Процент должен быть целым числом от 1 до 99: %s\n' "$CONFIDENCE_PERCENT" >&2
    exit 2
fi

printf '[INFO] Raspberry: %s@%s\n' "$RASPBERRY_USER" "$RASPBERRY_HOST"
printf '[INFO] Новый порог распознавания: %s%%\n' "$CONFIDENCE_PERCENT"
printf '[INFO] Raspberry запросит пароль sudo для перезапуска службы.\n'

# Меняет только порог и перезапускает службу распознавания.
ssh -tt "${RASPBERRY_USER}@${RASPBERRY_HOST}" \
    "CONFIDENCE_PERCENT='$CONFIDENCE_PERCENT' REMOTE_PROJECT='$REMOTE_PROJECT' SERVICE_NAME='$SERVICE_NAME' bash -s" <<'REMOTE_SCRIPT'
set -Eeuo pipefail
cd "$REMOTE_PROJECT"

timestamp=$(date +%Y%m%d_%H%M%S)
cp -a run_raspberry.sh "run_raspberry.sh.before_confidence.$timestamp"
cp -a config/runtime_settings.json "config/runtime_settings.json.before_confidence.$timestamp"

python3 - "$CONFIDENCE_PERCENT" run_raspberry.sh config/runtime_settings.json <<'PY'
from pathlib import Path
import json
import sys

percent = int(sys.argv[1])
run_path = Path(sys.argv[2])
settings_path = Path(sys.argv[3])

run_text = run_path.read_text()
lines = []
changed = False
for line in run_text.splitlines(keepends=True):
    if line.startswith('CONFIDENCE_PERCENT='):
        newline = '\n' if line.endswith('\n') else ''
        line = f'CONFIDENCE_PERCENT="{percent}"{newline}'
        changed = True
    lines.append(line)
if not changed:
    raise SystemExit('Не найдена переменная CONFIDENCE_PERCENT в run_raspberry.sh')
run_path.write_text(''.join(lines))

settings = json.loads(settings_path.read_text())
settings.setdefault('detection', {})['confidence_percent'] = percent
settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + '\n')
PY

printf '[OK] Порог сохранён: %s%%\n' "$CONFIDENCE_PERCENT"
sudo systemctl restart "$SERVICE_NAME"
sleep 2
systemctl is-active --quiet "$SERVICE_NAME"
printf '[OK] Служба %s перезапущена\n' "$SERVICE_NAME"
printf '[INFO] Проверка: journalctl -u %s -n 20 --no-pager\n' "$SERVICE_NAME"
REMOTE_SCRIPT
