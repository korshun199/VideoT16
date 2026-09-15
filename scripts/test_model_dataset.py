#!/usr/bin/env python3
"""Проверяет ONNX-модель на размеченных кадрах датасета VideoT16."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.onnx_detector import Detection, OnnxDetector


# Каталог проверенного компактного набора.
DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset/fpv/compact_training"
# Модель, которую сейчас готовим для Raspberry.
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/fpv_drone_custom_320.onnx"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def arguments() -> argparse.Namespace:
    """Читает параметры теста без изменения датасета."""
    parser = argparse.ArgumentParser(description="Тест ONNX-модели на датасете FPV")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25, help="порог 0..1")
    parser.add_argument("--size", type=int, default=320, help="размер входа модели")
    parser.add_argument("--limit", type=int, default=0, help="максимум кадров; 0 — все")
    parser.add_argument("--interval", type=int, default=700, help="пауза показа в миллисекундах")
    parser.add_argument("--no-window", action="store_true", help="только отчёт без окна")
    parser.add_argument("--show-truth", action="store_true", help="показывать эталонную разметку синим")
    parser.add_argument("--with-backgrounds", action="store_true", help="добавить кадры без объектов")
    return parser.parse_args()


def image_paths(split: str, with_backgrounds: bool) -> list[Path]:
    """Возвращает кадры с объектами, либо весь выбранный набор по запросу."""
    splits = ("train", "val") if split == "all" else (split,)
    result = []
    for current in splits:
        directory = DATASET_DIR / current / "images"
        label_dir = DATASET_DIR / current / "labels"
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label = label_dir / f"{path.stem}.txt"
            if with_backgrounds or (label.is_file() and label.read_text(encoding="utf-8").strip()):
                result.append(path)
    if not result:
        raise RuntimeError(f"В датасете нет кадров: split={split}")
    return sorted(result)


def ground_truth(path: Path) -> list[tuple[float, float, float, float]]:
    """Читает эталонные YOLO-рамки и возвращает их в нормированных координатах."""
    label = DATASET_DIR / path.parent.parent.name / "labels" / f"{path.stem}.txt"
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            continue
        boxes.append(tuple(float(value) for value in fields[1:]))
    return boxes


def to_pixels(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Переводит нормированную YOLO-рамку в пиксельные координаты."""
    center_x, center_y, box_width, box_height = box
    return (
        round((center_x - box_width / 2) * width),
        round((center_y - box_height / 2) * height),
        round((center_x + box_width / 2) * width),
        round((center_y + box_height / 2) * height),
    )


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Считает пересечение двух рамок."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def draw_frame(frame, detections: tuple[Detection, ...], truth: list[tuple[int, int, int, int]], number: int, total: int, show_truth: bool):
    """Рисует предсказания модели зелёным, а эталонные рамки — по запросу."""
    result = frame.copy()
    if show_truth:
        for x1, y1, x2, y2 in truth:
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 120, 0), 2)
    for detection in detections:
        cv2.rectangle(result, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 255, 0), 3)
        cv2.putText(result, f"{detection.name} {detection.confidence * 100:.0f}%", (detection.x1, max(25, detection.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result, f"{number}/{total}  GT: {len(truth)}  DET: {len(detections)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return result


class DatasetViewer:
    """Оконный просмотрщик с режимами распознавания, старта и остановки."""

    def __init__(self, paths: list[Path], detector: OnnxDetector, interval: int, show_truth: bool) -> None:
        """Создаёт интерфейс и загружает первый кадр."""
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.paths = paths
        self.detector = detector
        self.interval = max(100, interval)
        self.show_truth = show_truth
        self.position = 0
        self.running = False
        self.recognition = True
        self.root = tk.Tk()
        self.root.title("VideoT16 — тест модели на фотографиях")
        self.root.geometry("1200x850")
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.status = tk.Label(self.root, text="Готово к запуску", anchor="w", font=("DejaVu Sans", 11))
        self.status.pack(fill=tk.X, padx=10)
        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(controls, text="С РАСПОЗНАВАНИЕМ", command=self.enable_recognition).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="БЕЗ РАСПОЗНАВАНИЯ", command=self.disable_recognition).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="СТАРТ", command=self.start).pack(side=tk.LEFT, padx=18)
        ttk.Button(controls, text="СТОП", command=self.stop).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="ВЫХОД", command=self.close).pack(side=tk.RIGHT, padx=3)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.render()

    def enable_recognition(self) -> None:
        """Включает обработку кадров моделью."""
        self.recognition = True
        self.render()

    def disable_recognition(self) -> None:
        """Показывает исходные кадры без рамок модели."""
        self.recognition = False
        self.render()

    def start(self) -> None:
        """Запускает автоматический перебор фотографий."""
        if not self.running:
            self.running = True
            self.tick()

    def stop(self) -> None:
        """Останавливает перебор, сохраняя текущую фотографию."""
        self.running = False

    def render(self) -> None:
        """Рисует текущий кадр и результат модели в окне."""
        from PIL import Image, ImageTk

        path = self.paths[self.position]
        frame = cv2.imread(str(path))
        if frame is None:
            self.status.configure(text=f"Повреждённый кадр: {path.name}")
            return
        height, width = frame.shape[:2]
        truth = [to_pixels(box, width, height) for box in ground_truth(path)]
        detections = self.detector(frame) if self.recognition else ()
        shown = draw_frame(frame, detections, truth, self.position + 1, len(self.paths), self.show_truth)
        shown = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(shown)
        image.thumbnail((1160, 730), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        mode = "с распознаванием" if self.recognition else "без распознавания"
        self.status.configure(text=f"{self.position + 1}/{len(self.paths)}  {path.name}  |  режим: {mode}  |  найдено: {len(detections)}")

    def tick(self) -> None:
        """Показывает кадр и планирует следующий, пока включён старт."""
        if not self.running:
            return
        self.render()
        self.position = (self.position + 1) % len(self.paths)
        self.root.after(self.interval, self.tick)

    def close(self) -> None:
        """Закрывает окно просмотра."""
        self.running = False
        self.root.destroy()

    def run(self) -> None:
        """Запускает цикл обработки событий Tkinter."""
        self.root.mainloop()


def main() -> int:
    """Прогоняет модель по кадрам, показывает результат и печатает отчёт."""
    args = arguments()
    if not 0.0 < args.confidence <= 1.0:
        raise SystemExit("--confidence должен быть от 0 до 1")
    if args.size < 32 or args.size % 32:
        raise SystemExit("--size должен быть не меньше 32 и кратен 32")
    if not args.model.is_file():
        raise SystemExit(f"Модель не найдена: {args.model}")
    paths = image_paths(args.split, args.with_backgrounds)
    if args.limit > 0:
        paths = paths[:args.limit]
    detector = OnnxDetector(args.model, args.confidence, False, args.size, "FPV-DRON")
    true_positive = false_positive = false_negative = 0
    confidence_values: list[float] = []
    inference_seconds = 0.0
    window = "VideoT16 — тест модели на датасете"
    if not args.no_window:
        DatasetViewer(paths, detector, args.interval, args.show_truth).run()
        return 0
    try:
        for number, path in enumerate(paths, 1):
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"Пропуск повреждённого кадра: {path.name}")
                continue
            height, width = frame.shape[:2]
            truth = [to_pixels(box, width, height) for box in ground_truth(path)]
            started = time.perf_counter()
            detections = detector(frame)
            inference_seconds += time.perf_counter() - started
            confidence_values.extend(detection.confidence for detection in detections)
            matched_truth: set[int] = set()
            for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
                detection_box = (detection.x1, detection.y1, detection.x2, detection.y2)
                matches = [(iou(detection_box, expected), index) for index, expected in enumerate(truth) if index not in matched_truth]
                best = max(matches, default=(0.0, -1))
                if best[0] >= 0.30:
                    true_positive += 1
                    matched_truth.add(best[1])
                else:
                    false_positive += 1
            false_negative += len(truth) - len(matched_truth)
    finally:
        pass
    average = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    print(f"Кадров проверено: {len(paths)}")
    print(f"TP={true_positive} FP={false_positive} FN={false_negative}")
    print(f"Precision={precision:.1%} Recall={recall:.1%} Средняя уверенность={average:.1%}")
    if inference_seconds:
        print(
            f"Скорость={inference_seconds * 1000 / len(paths):.0f} мс/кадр "
            f"({len(paths) / inference_seconds:.1f} кадр/с)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
