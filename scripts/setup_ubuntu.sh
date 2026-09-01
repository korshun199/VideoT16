#!/usr/bin/env bash
set -euo pipefail

# Устанавливает системные пакеты и локальное Python-окружение проекта.
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 v4l-utils

# Picamera2 устанавливается системным пакетом Raspberry Pi; разрешаем .venv
# видеть его без копирования библиотек и запуска системного Python.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
# Сначала ставим CPU-сборки, чтобы не тащить гигабайты CUDA без NVIDIA.
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

echo
echo "Установка завершена. Положите локальную модель в models/yolov8n.pt."
echo "Проверка камеры: source .venv/bin/activate && python3 -m src.local_object_detection --list-cameras"
echo "Проверка Picamera2: source .venv/bin/activate && python3 -c 'from picamera2 import Picamera2; print(\"Picamera2 работает в .venv\")'"
