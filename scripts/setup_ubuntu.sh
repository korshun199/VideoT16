#!/usr/bin/env bash
set -euo pipefail

# Устанавливает системные пакеты и локальное Python-окружение проекта.
sudo apt update
sudo apt install -y python3-venv python3-pip v4l-utils

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
# Сначала ставим CPU-сборки, чтобы не тащить гигабайты CUDA без NVIDIA.
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt

echo
echo "Установка завершена. Положите локальную модель в models/yolov8n.pt."
echo "Проверка камеры: source .venv/bin/activate && python3 -m src.local_object_detection --list-cameras"
