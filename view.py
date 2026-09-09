#!/usr/bin/env python3
"""Pokazyvaet realnye rezultaty YOLO na fotografiyah po ocheredi."""

from pathlib import Path

import cv2

from src.onnx_detector import OnnxDetector


# Intervall avtomaticheskoj smeny kadrov.
INTERVAL_SECONDS = 1.0
# Razmer okna prosmotra.
WINDOW_NAME = "FPV object photos"
# Model dlya proverki realnyh obnaruzhenij.
MODEL_PATH = Path(__file__).resolve().parent / "runs/fpv_training/fpv_quadcopter_merged/weights/best.onnx"
# Porog dlya prosmotra modely.
CONFIDENCE = 0.20
# Razmer vhoda modeli.
INFERENCE_SIZE = 640


def draw_detections(image, detections):
    """Risuyet tolko ramki, kotorye nashla model, s uverennostyu."""
    for detection in detections:
        cv2.rectangle(image, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 255, 0), 3)
        text = f"{detection.name} {detection.confidence * 100:.0f}%"
        cv2.putText(image, text, (detection.x1, max(25, detection.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return image


def main() -> int:
    """Pokazyvaet vse fotografii s validnoj razmetkoj."""
    project_dir = Path(__file__).resolve().parent
    root = project_dir / "dataset/fpv/object_photos"
    image_dir = root / "images"
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        print("Fotografii s obektom ne najdeny")
        return 1
    if not MODEL_PATH.is_file():
        print(f"Model ne najdena: {MODEL_PATH}")
        return 1
    detector = OnnxDetector(MODEL_PATH, CONFIDENCE, False, INFERENCE_SIZE, "FPV-DRON")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    for number, image_path in enumerate(images, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        image = draw_detections(image, detector(image))
        cv2.putText(image, f"{number}/{len(images)}  {image_path.name}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(max(1, int(INTERVAL_SECONDS * 1000))) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            while True:
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("p"), ord("P"), 27, ord("q"), ord("Q")):
                    break
            if key in (27, ord("q"), ord("Q")):
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
