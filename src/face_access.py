"""Локальные функции обнаружения и сравнения лица через OpenCV Zoo."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path("models/face")
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def download_model(path: Path, url: str) -> None:
    """Скачивает отсутствующую локальную модель."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "VideoT16"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as target:
        target.write(response.read())


def create_models() -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    """Загружает YuNet и SFace на CPU."""
    download_model(YUNET_PATH, YUNET_URL)
    download_model(SFACE_PATH, SFACE_URL)
    detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.85, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
    return detector, recognizer


def find_largest_face(detector: cv2.FaceDetectorYN, image: np.ndarray) -> np.ndarray | None:
    """Возвращает самое крупное найденное лицо или None."""
    detector.setInputSize((image.shape[1], image.shape[0]))
    _height, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda face: float(face[2] * face[3]))


def make_embedding(recognizer: cv2.FaceRecognizerSF, image: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Выравнивает лицо и создаёт нормированный вектор признаков."""
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return cv2.normalize(feature, None).flatten().astype(np.float32)
