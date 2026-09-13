# Документация VideoT16

## Читать в первую очередь

1. [`WORKING_SCHEME.md`](../WORKING_SCHEME.md) — единая текущая схема,
   проводка, состав файлов и порядок запуска.
2. [`SECURITY.md`](../SECURITY.md) — правила защиты проекта и production-
   развертывания.
3. [`SESSION_HANDOFF.md`](../SESSION_HANDOFF.md) — краткая передача контекста.
4. [`RESTORE_RASPBERRY.md`](../RESTORE_RASPBERRY.md) — восстановление на новой
   Raspberry и SD-карте.

## Актуальные документы

| Файл | Назначение |
|---|---|
| [`osd.md`](osd.md) | Цифровой OSD-прокси и наши добавления |
| [`camera.md`](camera.md) | Проверенная камера EasyCap |
| [`WEB_CONFIG.md`](WEB_CONFIG.md) | Необязательная веб-панель настройки |

## Рабочие файлы рядом с документацией

- `../src/displayport_proxy.py` — прозрачная пересылка FC/VTX и наши команды;
- `../src/local_object_detection.py` — камера, распознавание и рамка;
- `../src/realtime.py` — температура и загрузка CPU;
- `../config/runtime_settings.json` — пути и параметры;
- `../run_raspberry.sh` — фактическая сборка аргументов запуска;
- `../scripts/conf` — изменение порога;
- `../scripts/vtest` — самопроверка Raspberry.

Исторические версии документов сохраняются только в истории Git и не хранятся
в рабочем каталоге.
