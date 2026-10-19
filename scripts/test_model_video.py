#!/usr/bin/env python3
"""Проверяет локальную ONNX-модель на видеофайле и сохраняет результат."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.local_object_detection import draw_detections
from src.onnx_detector import OnnxDetector
from src.realtime import Detection


class VideoViewer:
    """Показывает видео с кнопками запуска, паузы и остановки."""

    def __init__(self, args: argparse.Namespace, detector: OnnxDetector) -> None:
        """Открывает видео и создаёт компактное окно ноутбука."""
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk
        self.args = args
        self.detector = detector
        self.capture = cv2.VideoCapture(str(args.video))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if args.rotate != "none":
            width, height = height, width
        args.output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        self.root = tk.Tk()
        self.root.title("VideoT16 — тест модели")
        self.root.geometry("1280x800")
        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(self.root, text="Готово", anchor="w")
        self.status.pack(fill=tk.X, padx=8)
        controls = tk.Frame(self.root)
        controls.pack(fill=tk.X, padx=8, pady=8)
        tk.Button(controls, text="ПУСК", width=12, command=self.start).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="ПАУЗА", width=12, command=self.pause).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="СТОП", width=12, command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="ВЫХОД", width=12, command=self.close).pack(side=tk.RIGHT, padx=4)
        self.running = False
        self.stopped = False
        self.frame_number = 0
        self.detections: tuple[Detection, ...] = ()
        self.render_message("Нажмите ПУСК")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def render_message(self, message: str) -> None:
        """Показывает служебное сообщение до появления кадра."""
        self.status.configure(text=message)

    def start(self) -> None:
        """Запускает или продолжает воспроизведение."""
        if not self.stopped and not self.running:
            self.running = True
            self.tick()

    def pause(self) -> None:
        """Приостанавливает воспроизведение на текущем кадре."""
        self.running = False
        self.render_message(f"Пауза: кадр {self.frame_number}")

    def stop(self) -> None:
        """Останавливает воспроизведение и возвращает видео в начало."""
        self.running = False
        self.stopped = True
        self.capture.release()
        self.writer.release()
        self.render_message(f"Стоп: обработано кадров {self.frame_number}")

    def tick(self) -> None:
        """Обрабатывает один кадр и планирует следующий."""
        if not self.running:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.running = False
            self.render_message(f"Готово: обработано кадров {self.frame_number}")
            return
        self.frame_number += 1
        if self.args.rotate == "ccw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.args.rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.frame_number % self.args.every == 0:
            self.detections = self.detector(frame)
        annotated = draw_detections(frame, self.detections)
        self.writer.write(annotated)
        shown = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(shown)
        image.thumbnail((1240, 700), self.Image.Resampling.LANCZOS)
        self.photo = self.ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        confidence = max((item.confidence for item in self.detections), default=0.0)
        self.status.configure(text=f"Кадр {self.frame_number} | обнаружений: {len(self.detections)} | уверенность: {confidence:.0%}")
        delay = max(1, round(1000 / self.fps / self.args.speed))
        self.root.after(delay, self.tick)

    def close(self) -> None:
        """Закрывает окно и освобождает видеофайлы."""
        self.running = False
        if self.capture.isOpened():
            self.capture.release()
        self.writer.release()
        self.root.destroy()

    def run(self) -> int:
        """Запускает цикл графического интерфейса."""
        self.root.mainloop()
        return 0


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры офлайн-проверки видео."""
    parser = argparse.ArgumentParser(description="Тест VideoT16 на видеофайле")
    parser.add_argument("video", type=Path, help="Входной видеофайл")
    parser.add_argument("--model", type=Path, required=True, help="Локальная ONNX-модель")
    parser.add_argument("--output", type=Path, required=True, help="Обработанное видео")
    parser.add_argument("--confidence", type=float, default=0.25, help="Порог 0..1")
    parser.add_argument("--size", type=int, default=320, help="Размер входа модели")
    parser.add_argument("--threads", type=int, default=3, help="Потоки ONNX Runtime")
    parser.add_argument("--every", type=int, default=1, help="Распознавать каждый N-й кадр")
    parser.add_argument("--show", action="store_true", help="Показывать обработанное видео в окне")
    parser.add_argument("--rotate", choices=("none", "ccw", "cw"), default="none", help="Поворот кадра")
    parser.add_argument("--speed", type=float, default=1.0, help="Скорость просмотра")
    return parser


def main() -> int:
    """Последовательно читает видео, распознаёт кадры и печатает статистику."""
    args = build_parser().parse_args()
    is_camera = str(args.video).startswith("/dev/video")
    if not args.video.is_file() and not is_camera:
        raise SystemExit(f"Видео не найдено: {args.video}")
    if not args.model.is_file():
        raise SystemExit(f"Модель не найдена: {args.model}")
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence должен быть от 0 до 1")
    if args.speed <= 0:
        raise SystemExit("--speed должен быть больше 0")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Не удалось открыть видео: {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.rotate != "none":
        width, height = height, width
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Не удалось создать результат: {args.output}")

    detector = OnnxDetector(args.model, args.confidence, False, args.size, "FPV-DRON", args.threads)
    if args.show:
        # Освобождаем предварительную проверку файла перед окном управления.
        capture.release()
        writer.release()
        return VideoViewer(args, detector).run()
    frame_number = 0
    detected_frames = 0
    detections_total = 0
    max_confidence = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if args.rotate == "ccw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif args.rotate == "cw":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            detections = detector(frame) if frame_number % args.every == 0 else ()
            if detections:
                detected_frames += 1
                detections_total += len(detections)
                max_confidence = max(max_confidence, max(item.confidence for item in detections))
            annotated = draw_detections(frame, detections)
            writer.write(annotated)
            if args.show:
                cv2.imshow("VideoT16 — тест модели", annotated)
                key = cv2.waitKey(max(1, round(1000 / fps / args.speed))) & 0xFF
                if key in (ord("q"), 27):
                    print("Остановлено пользователем.", flush=True)
                    break
            if frame_number % max(1, round(fps * 5)) == 0:
                print(
                    f"Кадр {frame_number}: обнаружений={len(detections)}, "
                    f"сырая уверенность={detector.last_best_confidence:.1%}",
                    flush=True,
                )
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    print(f"Готово: кадров={frame_number}, кадров с объектом={detected_frames}")
    print(f"Всего рамок={detections_total}, максимальная уверенность={max_confidence:.1%}")
    print(f"Результат: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
