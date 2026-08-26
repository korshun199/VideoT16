#!/usr/bin/env bash

set -Eeuo pipefail

# Запускает локальную панель настройки оператора.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
exec "$PROJECT_DIR/.venv/bin/python" -m web_config.server "$@"
