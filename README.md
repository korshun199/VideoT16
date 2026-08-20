# Локальное распознавание объектов с цифровой камеры

Проект для Ubuntu 24.04: захватывает видео с USB-камеры или RTSP-потока и распознаёт объекты локальной моделью YOLO. Кадры не отправляются в облачные сервисы и не требуют подключения к ИИ-сервисам. Инференс настроен на CPU и работает с интегрированной графикой Intel без NVIDIA.

## Требования

- Ubuntu 24.04;
- Python 3.12+;
- цифровая USB-камера или доступный RTSP-поток;
- видеодрайвер V4L2 для USB-камеры.

## Установка

Автоматический вариант:

```bash
chmod +x scripts/setup_ubuntu.sh
./scripts/setup_ubuntu.sh
```

Или вручную:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip v4l-utils
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt
```

Скачайте веса модели заранее на компьютер с интернетом и положите файл в `models/yolov8n.pt`. После этого приложение работает офлайн. Маленькая модель подходит для первого запуска; для более точного распознавания можно заменить её на другую совместимую модель.

## Проверка камеры

```bash
v4l2-ctl --list-devices
python3 -m src.local_object_detection --list-cameras
```

Для внешней Logitech Brio используйте `--source logitech`: проект выберет её по имени `Brio` и не откроет встроенную камеру ноутбука. Также можно указать `--source /dev/video4` или `/dev/video5`. Для RTSP укажите полный адрес источника.

Для проверенной Logitech Brio 90 рабочий видеопоток: `/dev/video4`.

## Запуск

Быстрый запуск внешней Logitech Brio:

```bash
./run_camera.sh
```

Отдельные команды:

```bash
./run_base.sh   # базовые классы YOLO
./run_fpv.sh    # готовая модель обнаружения квадрокоптеров и самолётных дронов
./run_military.sh # модель военной техники
./run_orangepi4.sh # базовый запуск с ограничениями Orange Pi 4
```

Модель FPV хранится локально в `models/fpv_drone_best.pt` и запускается без облачного сервиса. Она содержит классы `квадрокоптер` и `самолётный дрон`. Если модель отсутствует, скачайте её отдельно с Hugging Face и положите в этот путь.

## Модель военной техники

Для отдельного запуска используйте:

```bash
./run_military.sh
./run_military.sh -o                         # режим эмуляции Orange Pi 4
./run_military.sh -f                         # обычный режим без ограничений
./run_military.sh -f --resolution 1920x1080 --fullscreen
./run_military.sh -o --resolution 1280x720 --confidence-percent 50
```

Модель распознаёт танки, бронемашины, РСЗО, САУ, самолёты, грузовики и другие классы. Это локальная учебная модель для компьютерного зрения, а не система оперативного наблюдения или принятия решений. Подробности и лицензия указаны в `models/MILITARY_MODEL.md`.

## Моделирование с полётником INAV

Для ноутбука, камеры и полётного контроллера LongBow F405 Wing используйте:

```bash
./run_modeling.sh
```

По умолчанию программа использует камеру `logitech` и INAV-порт `/dev/ttyACM0`.
Телеметрия читается по MSP в режиме только чтения и выводится поверх HDMI-видео.
Команды управления и изменения настроек полётнику не отправляются.

Порт можно указать явно:

```bash
INAV_PORT=/dev/ttyACM0 CAMERA_SOURCE=/dev/video4 ./run_modeling.sh
```

Для работы нужен `pyserial`, который входит в `requirements.txt`.

В `run_military.sh` все рамки подписываются единым словом `OBJECT` и процентом уверенности.

Параметр `-o` ограничивает процесс двумя медленными ядрами и 4096 МБ адресного пространства. Параметр `-f` запускает модель без этого ограничения. Разрешение задаётся в формате `ШИРИНАxВЫСОТА`, а `--fullscreen` разворачивает окно на весь экран.

На рамке показывается уверенность в процентах. Порог можно менять так:

```bash
./run_military.sh --confidence-percent 50
./run_military.sh -p 50
```

Также поддерживается старый формат: `--confidence 0.50`.
Звуковой сигнал включается только после непрерывного удержания обнаружения 3 секунды. Задержку можно изменить параметром `--alert-delay`.

## Режим проверки Orange Pi 4

Для проверки производительности на компьютере можно ограничить процесс по ядрам и памяти:

```bash
./run_orangepi4.sh
```

По умолчанию используются 2 самых медленных доступных ядра и лимит 4096 МБ. Настройки можно изменить:

```bash
ORANGEPI4_CORES=1 ORANGEPI4_MEMORY_MB=512 ./run_orangepi4.sh
```

Лимит 128 МБ подходит только для очень лёгких скриптов. Для Python, OpenCV и YOLO он обычно недостаточен, поэтому в качестве режима проверки Orange Pi 4 выбран лимит 4096 МБ.

Если модель потерялась, восстановите её с оригинального источника и проверьте SHA-256:

```bash
./scripts/download_fpv_model.sh
```

Резервные ссылки: [оригинальный `best.pt` на Hugging Face](https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt), [копия `.pt` в GitHub](https://raw.githubusercontent.com/korshun199/VideoT16/main/models/fpv_drone_best.pt), [копия `.onnx` в GitHub](https://raw.githubusercontent.com/korshun199/VideoT16/main/models/fpv_drone_best.onnx).

## Подготовка для Orange Pi 5

Orange Pi 5 использует Rockchip RK3588S. Для NPU модель нужно подготовить на этом компьютере с x86 Linux, а затем перенести на Orange Pi 5 в формате RKNN. Сам Orange Pi 5 для экспорта не используется.

На компьютере установите экспортёр ONNX и выполните:

```bash
source .venv/bin/activate
python3 -m pip install -r requirements-export.txt
python3 scripts/export_orangepi5_onnx.py
```

Затем установите совместимый `rknn-toolkit2` в отдельное окружение x86 Linux и выполните:

```bash
python3 scripts/convert_onnx_to_rknn.py \
  models/fpv_drone_best.onnx \
  models/orangepi5/fpv_drone.rknn
```

Скопируйте на Orange Pi 5 проект, файл `models/orangepi5/fpv_drone.rknn` и установите runtime `rknn-toolkit-lite2` для RK3588. Запуск:

```bash
CAMERA_SOURCE=/dev/video0 ./run_fpv_orangepi5.sh
```

Если Logitech получила другой номер устройства:

```bash
CAMERA_SOURCE=/dev/video4 ./run_fpv_orangepi5.sh
```

`.pt` остаётся версией для компьютера, а `.rknn` — версией для NPU Orange Pi 5. Экспорт выполняйте на x86 Linux: официальная документация указывает, что экспорт RKNN на ARM-плате не поддерживается.

Дополнительные параметры передаются в приложение:

```bash
./run_camera.sh --window-width 1600 --window-height 900
```

С окном предпросмотра:

```bash
source .venv/bin/activate
python3 -m src.local_object_detection --source /dev/video4 --model models/yolov8n.pt
```

Окно запускается размером 1280×720. При необходимости размер можно изменить: `--window-width 1600 --window-height 900`.
Подписи объектов по умолчанию русские. Для английских подписей добавьте `--labels en`.

Горячие клавиши окна: `q` или `Esc` — выход, `s` — сохранить текущий кадр.

### Звуковой сигнал

Для локального сигнала при обнаружении положите WAV-файл в `sounds/fpv_detected.wav`. Скрипты `run_fpv.sh` и `run_fpv_orangepi5.sh` подключат его автоматически. Сигнал повторяется не чаще одного раза в 3 секунды. Для Ubuntu установите любой проигрыватель WAV:

```bash
sudo apt install alsa-utils
```

Без окна, с записью результата:

```bash
python3 -m src.local_object_detection \
  --source 0 \
  --model models/yolov8n.pt \
  --headless \
  --output recordings/result.mp4 \
  --max-frames 300
```

Все настройки можно передать через аргументы командной строки. Приложение не скачивает модель само и завершается с понятной ошибкой, если локальные веса отсутствуют.

## Обучение на своём коте

## Съёмка фотографий предметов

Для съёмки предметов с Logitech Brio на компьютере:

```bash
./capture_brio.sh предмет_1
```

Нажимай `Space` для сохранения фотографии, `q` или `Esc` для выхода. Фотографии будут в `dataset/objects/предмет_1/`.

Текущая модель распознаёт любой объект класса `кошка`. Чтобы отличать именно Фипика от других, используется отдельный локальный класс `my_cat`, который в окне подписывается как `Фипик`.

1. Соберите 100–300 кадров кота в разных позах, местах и условиях освещения:

   ```bash
   .venv/bin/python3 scripts/collect_cat_frames.py --source /dev/video4
   ```

   Нажмите `q` для завершения.

2. Разметьте кадры локальным инструментом проекта. На каждом изображении мышью обведите Фипика и нажмите Enter:

   ```bash
   .venv/bin/python3 scripts/annotate_cat.py
   ```

   Программа сама создаст YOLO-разметку и разделит кадры на `train` и `val`.

3. Разделите размеченные файлы между `images/train` и `images/val`, а подписи — между одноимёнными каталогами `labels/train` и `labels/val`.

4. Запустите обучение локально на CPU:

   ```bash
   .venv/bin/python3 scripts/train_my_cat.py --epochs 40
   ```

5. Запустите распознавание обученной моделью:

   ```bash
   ./run_camera.sh --model runs/detect/runs/my_cat/weights/best.pt
   ```

Если модель реагирует на движение, добавьте отрицательные кадры комнаты без Фипика:

```bash
.venv/bin/python3 scripts/collect_cat_frames.py \
  --source /dev/video4 \
  --output dataset/my_cat/images/raw_negative \
  --interval 2
```

Остановите сбор клавишей `q`, затем добавьте их в датасет:

```bash
.venv/bin/python3 scripts/prepare_negative_frames.py
```

Соберите также 100–200 новых резких кадров Фипика: пусть он чаще сидит или лежит, а не бежит. После разметки положительных кадров повторите обучение.

## Приватность и безопасность

- сетевые отправки кадров в коде отсутствуют;
- RTSP-логин и пароль не записывайте в Git;
- перед подключением камеры проверьте её устройство через `v4l2-ctl`, не подавайте питание на неизвестные контакты;
- для производственного режима стоит добавить systemd-службу, журналирование и ограничение доступа к каталогу записей.

## Тесты

```bash
python3 -m unittest discover -s tests -v
```
