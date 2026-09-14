# Цифровой OSD VideoT16

## Текущая схема

```text
FC TX -> Pi pin 10 (/dev/ttyAMA0 RX)
Pi pin 8 (/dev/ttyAMA0 TX) -> VTX RX
VTX TX -> Pi pin 29 (/dev/ttyAMA2 RX)
Pi pin 7 (/dev/ttyAMA2 TX) -> FC RX
Общая земля FC, Raspberry и VTX
```

Полётник самостоятельно формирует штатный цифровой MSP DisplayPort OSD.
Raspberry не читает, не декодирует, не фильтрует и не изменяет эти данные.
Прокси только прозрачно пересылает байты между `/dev/ttyAMA0` и
`/dev/ttyAMA2`.

## Что добавляет Raspberry

В сторону VTX отправляются только собственные команды DisplayPort:

- рамка вокруг обнаруженного объекта;
- подпись `FPV-DRON`;
- нижняя строка `TEMP xx.xC CPU yy%`.

Входящие строки `BAT`, `DISARMED`, режим полёта и остальные элементы OSD
остаются полностью под управлением полётника и его настройки пилотом.

## Источники

- `src/displayport_proxy.py` — MSP DisplayPort и прозрачный двунаправленный
  мост;
- `src/local_object_detection.py` — рамка, подпись и вызов добавлений OSD;
- `src/realtime.py` — чтение температуры Linux и загрузки CPU;
- `config/osd_config.py` — внешний вид графического диагностического кадра;
- `config/runtime_settings.json` — UART-порты и включение proxy.

## Проверка без полёта

```bash
python3 -m unittest discover -s tests -p '*.py' -q
```

На Raspberry после запуска службы проверить журнал:

```bash
journalctl -u videot16.service -n 50 --no-pager
```

В рабочей конфигурации не должны использоваться HDMI или J7. Веб-просмотр
остаётся только диагностическим изображением EasyCap и не заменяет изображение
в очках пилота.

## Тест выхода без камеры и полётника

```bash
python3 scripts/osd_test.py
```

Скрипт использует только `/dev/ttyAMA2` (Pi pin 7 -> Ascent RX) и выводит
тестовую рамку, `V16 OSD TEST` и строку `TEMP CPU`. Камера и FC для этого теста
не нужны.
