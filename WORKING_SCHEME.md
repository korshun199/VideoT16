# VideoT16 — рабочая схема

Этот файл — краткая точка истины перед продолжением проекта. Сначала читать
его, затем `SESSION_HANDOFF.md` и `SECURITY.md`.

## 1. Назначение

Raspberry Pi получает видеокадры с EasyCap, локально выполняет распознавание
и добавляет рамку с подписью `FPV-DRON` в цифровой MSP DisplayPort OSD.
Дрон, полётный контроллер и видеопередача работают самостоятельно: VideoT16
не управляет дроном и не выводит видео через HDMI/J7.

```text
EasyCap -> Raspberry: видеокадр -> локальный YOLO -> рамка/подпись
FC -> Raspberry: цифровой MSP DisplayPort OSD
Raspberry: прозрачная пересылка FC <-> VTX + собственные команды рамки/подписи
Raspberry -> VTX: обновлённый цифровой OSD
VTX -> пилот: штатное изображение с OSD
```

## 2. Подключение Raspberry

Подробная карта сторон и контактов хранится в
[`docs/WIRING_SOURCE_OF_TRUTH.md`](docs/WIRING_SOURCE_OF_TRUTH.md).

На Pi включены `dtoverlay=uart0-pi5` и `dtoverlay=uart2-pi5`; консоль на
UART отключена. Оба канала работают 115200, 8N1, без аппаратного управления.

Камера выбирается по стабильному пути:
`/dev/v4l/by-id/usb-MACROSILICON_AV_TO_USB2.0_20200909-video-index0`.

## 3. Текущая рабочая конфигурация

- модель: `models/fpv_drone_custom_320.onnx`;
- вход модели: 320x320, CPU, офлайн;
- EasyCap: 640x480, 25 FPS;
- DisplayPort: `/dev/ttyAMA0` (FC) -> `/dev/ttyAMA2` (VTX);
- HDMI/J7 в рабочем режиме выключены;
- цифровой OSD дополнительно получает строку `TEMP xx.xC CPU yy% CONF zz%`;
- веб-просмотр — только диагностический поток EasyCap, не поток VTX.

Запрос к веб-просмотру с ноутбука:

```bash
ssh -N -L 8081:127.0.0.1:8081 oleg@192.168.20.126
```

Затем открыть `http://127.0.0.1:8081/`.

## 4. Распознавание и порог

Модель содержит классы `quadcopter` и `fixed-wing`; в OSD они отображаются
как `FPV-DRON` и `AVIA DRON`. В консоль выводится каждый сырой лучший процент
уверенности. Порог `conf X` влияет только на показ рамки пилоту:
рамка появляется, когда `confidence >= X`; консольные значения от порога
не скрываются.

На текущем стандартном наборе возможны ложные срабатывания на лампу. Перед
обучением нужно добавить реальные кадры EasyCap и фоновые кадры в источники
обучения; не считать текущую модель окончательной.

## 5. Защищенный запуск Raspberry

Ноутбук собирает payload и activation под ID платы. Raspberry получает
зашифрованный пакет, launcher читает собственный ID, проверяет ID и ключ,
расшифровывает только в `/dev/shm`, загружает `raspberry.env` и запускает
VideoT16 из оперативного каталога. Исходный payload и секреты в Git не входят.

Основные файлы безопасности: `security/pack.py`, `security/auto.py`,
`security/launcher.py` (или переданный `launcher.py`), `security/dec.c`,
`security/view`, `security/production_payload.v16e` и activation-файл.

## 6. Файлы проекта

- `src/local_object_detection.py` — цикл камеры, инференс, рамка и веб;
- `src/onnx_detector.py` — загрузка ONNX и детекция;
- `src/displayport_proxy.py` — прозрачный MSP DisplayPort proxy и наши добавления;
- `src/preview.py` — диагностический веб-поток;
- `src/realtime.py` — `TEMP CPU` и системная телеметрия;
- `config/runtime_settings.json` — пути и параметры запуска;
- `run_raspberry.sh` — запуск Raspberry;
- `scripts/conf` — изменить порог и перезапустить сервис;
- `scripts/vtest` — короткая самопроверка, `vtest --подробно` — подробная;
- `scripts/train_fpv.py` — обучение;
- `tests/` — автоматические тесты;
- `models/` — утвержденные модели;
- `dataset/fpv/` — локальные материалы обучения, не переносить на Pi;
- `runs/` — результаты обучения, не переносить на Pi без необходимости.

## 7. Проверка

```bash
python3 -m unittest discover -s tests -p '*.py' -q
```

На Raspberry после передачи пакета перезапуск выполняется командой:

```bash
sudo systemctl restart videot16.service
journalctl -u videot16.service -n 50 --no-pager
```

Если `sudo` запрашивает пароль, это действие выполняется вручную на Pi.
