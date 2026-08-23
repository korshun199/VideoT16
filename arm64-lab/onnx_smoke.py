#!/usr/bin/env python3

"""Проверяет загрузку и один прямой инференс ONNX через OpenCV DNN."""

import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Загружает ONNX и получает выходной тензор на пустом кадре."""
    model_path = Path(sys.argv[1])
    net = cv2.dnn.readNetFromONNX(str(model_path))
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True)
    net.setInput(blob)
    output = net.forward()
    print(f"ONNX_OK shape={tuple(output.shape)} dtype={output.dtype}")


if __name__ == "__main__":
    main()

