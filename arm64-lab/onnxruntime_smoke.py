#!/usr/bin/env python3

"""Проверяет ONNX-инференс через ONNX Runtime на ARM64 CPU."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> None:
    """Запускает один инференс и печатает форму выходных тензоров."""
    model_path = Path(sys.argv[1])
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    input_shape = tuple(int(value) for value in input_info.shape)
    frame = np.zeros((input_shape[2], input_shape[3], 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (input_shape[3], input_shape[2]), swapRB=True)
    feed = {input_info.name: blob}
    for _ in range(2):
        session.run(None, feed)
    started = time.perf_counter()
    outputs = None
    for _ in range(3):
        outputs = session.run(None, feed)
    elapsed_ms = (time.perf_counter() - started) * 1000 / 3
    shapes = [tuple(output.shape) for output in outputs]
    print(f"ORT_OK providers={session.get_providers()} outputs={shapes} avg_ms={elapsed_ms:.1f}")


if __name__ == "__main__":
    main()
