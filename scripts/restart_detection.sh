#!/usr/bin/env bash

set -Eeuo pipefail

# Путь к проекту и имя службы на этой Raspberry Pi.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="videot16.service"

# Порог уверенности передаётся первым параметром при запуске.
if [[ $# -ne 1 ]]; then
    printf 'Использование: %s ПРОЦЕНТ\n' "$0" >&2
    printf 'Пример: %s 50\n' "$0" >&2
    exit 2
fi

CONFIDENCE_PERCENT="$1"

# Проверяет значение настройки перед изменением конфигурации.
if ! [[ "$CONFIDENCE_PERCENT" =~ ^[0-9]+$ ]] || (( CONFIDENCE_PERCENT < 1 || CONFIDENCE_PERCENT > 99 )); then
    printf '[ОШИБКА] Процент должен быть целым числом от 1 до 99: %s\n' "$CONFIDENCE_PERCENT" >&2
    exit 2
fi

cd "$PROJECT_DIR"
timestamp=$(date +%Y%m%d_%H%M%S)

# Сохраняет настройки перед изменением.
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
    if line.startswith("CONFIDENCE_PERCENT="):
        newline = "\n" if line.endswith("\n") else ""
        line = f'CONFIDENCE_PERCENT="{percent}"{newline}'
        changed = True
    lines.append(line)
if not changed:
    raise SystemExit("Не найдена переменная CONFIDENCE_PERCENT")
run_path.write_text("".join(lines))

settings = json.loads(settings_path.read_text())
settings.setdefault("detection", {})["confidence_percent"] = percent
settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
PY

printf '[OK] Порог сохранён: %s%%\n' "$CONFIDENCE_PERCENT"
sudo systemctl restart "$SERVICE_NAME"
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    printf '[OK] Служба %s работает\n' "$SERVICE_NAME"
else
    printf '[ОШИБКА] Служба не запустилась\n' >&2
    systemctl status "$SERVICE_NAME" -l --no-pager || true
    exit 1
fi
