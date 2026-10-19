#!/usr/bin/env python3

"""Проверяет реальный ONNX-инференс на кадрах видеопотока."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> None:
    """Читает несколько кадров, запускает модель и печатает задержку."""
    video_path = Path(sys.argv[1])
    model_path = Path(sys.argv[2])
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    input_height = int(input_info.shape[2])
    input_width = int(input_info.shape[3])
    processed = 0
    elapsed = 0.0
    output_shape = None

    while processed < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        blob = cv2.dnn.blobFromImage(
            frame,
            1 / 255.0,
            (input_width, input_height),
            swapRB=True,
        )
        started = time.perf_counter()
        outputs = session.run(None, {input_info.name: blob})
        elapsed += time.perf_counter() - started
        output_shape = tuple(outputs[0].shape)
        processed += 1
    capture.release()
    if processed == 0:
        raise RuntimeError("Видео не содержит кадров")
    average_ms = elapsed * 1000 / processed
    print(
        f"ORT_VIDEO_OK frames={processed} output={output_shape} "
        f"avg_ms={average_ms:.1f} fps={1000 / average_ms:.3f}"
    )


if __name__ == "__main__":
    main()

