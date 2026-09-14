# Восстановление VideoT16 на другой Raspberry Pi

Инструкция для новой платы и новой SD-карты. Секреты, пароли и production-
payload в этот файл не записываются.

## 1. Подготовить SD-карту

Записать чистую Raspberry Pi OS Lite 64-bit и загрузить Raspberry Pi.
Подключить сеть. Не переносить старый `.venv` или образ Linux: систему и
зависимости ставим заново.

## 2. Получить ID новой платы

На новой Raspberry выполнить:

```bash
grep Serial /proc/cpuinfo
hostname -I
```

Передать Машеньке значение `Serial`. Оно понадобится для выпуска payload,
привязанного только к этой плате.

## 3. Скачать разработку

На ноутбуке получить актуальную ветку из GitHub:

```bash
git clone --branch test-uart-video-osd git@github.com:korshun199/VideoT16.git
cd VideoT16
```

Если каталог уже существует:

```bash
git fetch origin
git switch test-uart-video-osd
git pull --ff-only origin test-uart-video-osd
```

## 4. Подготовить Raspberry

На Raspberry установить системные компоненты, Python и создать новое
виртуальное окружение согласно `requirements-raspberry.txt`. Затем проверить
доступ к камере EasyCap и UART. Не подключать сигнальные провода до проверки
земли и уровней напряжения.

## 5. Выпустить новый production-пакет

На ноутбуке в корне проекта:

1. заменить `DEVICE_ID` в локальном `security/auto.py` на ID новой платы;
2. оставить фразу пустой для ручного запроса;
3. выполнить:

```bash
python3 security/auto.py
```

Скрипт соберёт только runtime-файлы, добавит SHA-256 манифест, зашифрует
архив AES-256-GCM, привяжет его к ID платы и передаст payload/launcher по SSH.
Секретная папка `security/` и production-файлы в Git не публикуются.

## 6. Установить автозапуск

На Raspberry установить переданный systemd-файл и включить службу. После
этого проверить:

```bash
sudo systemctl enable videot16.service
sudo systemctl restart videot16.service
systemctl is-enabled videot16.service
systemctl is-active videot16.service
journalctl -u videot16.service -n 50 --no-pager
```

Ожидается запуск `launcher.py`, расшифровка только в `/dev/shm`, EasyCap,
модель `models/fpv_drone_custom_320.onnx`, вход 320x320, UART `/dev/ttyAMA0` -> `/dev/ttyAMA2`,
а также веб-порт `127.0.0.1:8081`.

## 7. Проверить систему

```bash
./vtest
```

Затем проверить веб-просмотр, штатный OSD полётника, прозрачную передачу OSD,
рамку `FPV-DRON` и строку `TEMP xx.xC CPU yy%` в очках пилота.

## 8. Веб-просмотр

На ноутбуке:

```bash
ssh -N -L 8081:127.0.0.1:8081 oleg@АДРЕС_RASPBERRY
```

Открыть `http://127.0.0.1:8081/`. Это диагностический поток EasyCap, а не
замена цифрового изображения в очках.

## Что не восстанавливается из Git

- Raspberry Pi OS и системные пакеты;
- `.venv`, датасет и результаты обучения;
- production payload, activation, launcher и фразы;
- локальные backup-архивы.

Они создаются или передаются отдельно, чтобы новая плата получила только свой
пакет и свой ID.
