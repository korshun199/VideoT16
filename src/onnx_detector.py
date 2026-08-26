"""Лёгкий инференс YOLO-детектора через ONNX Runtime."""

from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.realtime import Detection


class OnnxDetector:
    """Запускает экспортированную YOLO-модель без Torch и Ultralytics."""

    def __init__(self, path: Path, confidence: float, generic_label: bool, size: int) -> None:
        self.confidence = confidence
        self.generic_label = generic_label
        self.size = size
        # Ограничиваем параллелизм: поток камеры и SSH должны оставаться отзывчивыми.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        try:
            parsed = ast.literal_eval(metadata.get("names", "{}"))
            self.names = {int(key): str(value) for key, value in parsed.items()}
        except (ValueError, SyntaxError):
            self.names = {}

    def __call__(self, frame: np.ndarray) -> tuple[Detection, ...]:
        height, width = frame.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized = cv2.resize(frame, (round(width * scale), round(height * scale)))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        pad_x = (self.size - resized.shape[1]) // 2
        pad_y = (self.size - resized.shape[0]) // 2
        canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0][0].T
        scores = output[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        boxes, kept_scores, kept_ids = [], [], []
        for row, score, class_id in zip(output, confidences, class_ids):
            if float(score) < self.confidence:
                continue
            cx, cy, box_width, box_height = row[:4]
            x = (cx - box_width / 2 - pad_x) / scale
            y = (cy - box_height / 2 - pad_y) / scale
            w = box_width / scale
            h = box_height / scale
            boxes.append([int(max(0, x)), int(max(0, y)), int(w), int(h)])
            kept_scores.append(float(score))
            kept_ids.append(int(class_id))
        indices = cv2.dnn.NMSBoxes(boxes, kept_scores, self.confidence, 0.45)
        result = []
        for index in np.array(indices).reshape(-1):
            x, y, w, h = boxes[int(index)]
            name = "OBJECT" if self.generic_label else self.names.get(kept_ids[int(index)], "OBJECT")
            result.append(Detection(x, y, min(width, x + w), min(height, y + h), name, kept_scores[int(index)]))
        return tuple(result)
