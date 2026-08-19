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
```

Модель FPV хранится локально в `models/fpv_drone_best.pt` и запускается без облачного сервиса. Она содержит классы `квадрокоптер` и `самолётный дрон`. Если модель отсутствует, скачайте её отдельно с Hugging Face и положите в этот путь.

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
