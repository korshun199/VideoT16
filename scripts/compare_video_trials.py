#!/usr/bin/env python3
"""Сравнивает готовые ONNX-модели на вручную отобранных кадрах видео."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REPORT_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
DEFAULT_MODELS = (
    PROJECT_ROOT / "models" / "fpv_drone_best.onnx",
    PROJECT_ROOT / "models" / "anti_uav_fast_320_int8_pc.onnx",
)

# Позволяет запускать файл напрямую из каталога scripts без установки проекта.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.onnx_detector import OnnxDetector  # noqa: E402
from src.realtime import Detection  # noqa: E402


def model_input_size(model_path: Path) -> int:
    """Читает и проверяет квадратный размер входа ONNX-модели."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"Ожидался один вход модели, найдено: {len(inputs)} ({model_path})")
    shape = inputs[0].shape
    if len(shape) != 4 or shape[1] != 3:
        raise ValueError(f"Ожидался вход NCHW с тремя каналами, получено {shape}: {model_path}")
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int) or height != width:
        raise ValueError(f"Нужен фиксированный квадратный вход модели, получено {shape}: {model_path}")
    return height


def image_paths(sample_dir: Path) -> list[Path]:
    """Возвращает только изображения непосредственно из выбранной папки."""
    if not sample_dir.is_dir():
        raise ValueError(f"Папка кадров не найдена: {sample_dir}")
    return sorted(
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_name(value: str) -> str:
    """Удаляет из имени файла символы, неудобные для разных ОС."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "model"


def load_overlay_settings(profile_name: str = "sky") -> dict:
    """Читает параметры рамки и подписи из JSON выбранного профиля."""
    profile_path = PROJECT_ROOT / "config" / "live_drone_profiles.json"
    try:
        with profile_path.open("r", encoding="utf-8") as profile_file:
            document = json.load(profile_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Не удалось прочитать стиль оверлея из {profile_path}: {error}") from error
    profiles = document.get("profiles") if isinstance(document, dict) else None
    profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
    style = profile.get("overlay") if isinstance(profile, dict) else None
    if not isinstance(style, dict):
        raise ValueError(f"В профиле '{profile_name}' отсутствует секция overlay")
    return style


def draw_overlay(
    frame,
    detections: Iterable,
    threshold: float,
    style: dict | None = None,
):
    """Рисует рамки выше порога по геометрии и стилю из профиля JSON."""
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    style = style or load_overlay_settings()
    box_style = style["box"]
    label_style = style["label"]
    near_threshold_margin = style.get("near_threshold_margin", 0.0)
    near_threshold_rgb = style.get("near_threshold_color_rgb", box_style["color_rgb"])
    near_threshold_color = tuple(reversed(near_threshold_rgb))
    label_font = getattr(cv2, label_style["font"])
    for detection in detections:
        if detection.confidence >= threshold:
            box_color = tuple(reversed(box_style["color_rgb"]))
            label_color = tuple(reversed(label_style["color_rgb"]))
        elif (
            near_threshold_margin > 0
            and detection.confidence >= threshold - near_threshold_margin
        ):
            box_color = near_threshold_color
            label_color = near_threshold_color
        else:
            continue
        center_x = (detection.x1 + detection.x2) / 2
        center_y = (detection.y1 + detection.y2) / 2
        half_width = max(1.0, (detection.x2 - detection.x1) * box_style["scale_x"] / 2)
        half_height = max(1.0, (detection.y2 - detection.y1) * box_style["scale_y"] / 2)
        if width > 1:
            x1 = max(0, min(width - 2, round(center_x - half_width)))
            x2 = max(x1 + 1, min(width - 1, round(center_x + half_width)))
        else:
            x1 = x2 = 0
        if height > 1:
            y1 = max(0, min(height - 2, round(center_y - half_height)))
            y2 = max(y1 + 1, min(height - 1, round(center_y + half_height)))
        else:
            y1 = y2 = 0
        first = (x1, y1)
        second = (x2, y2)
        cv2.rectangle(overlay, first, second, box_color, box_style["thickness"], cv2.LINE_AA)
        label = f"{detection.name} {detection.confidence * 100:.0f}%"
        text_y = max(label_style["min_y"], first[1] - label_style["offset_y"])
        cv2.putText(
            overlay,
            label,
            (max(0, first[0]), text_y),
            label_font,
            label_style["font_scale"],
            label_color,
            label_style["thickness"],
            cv2.LINE_AA,
        )
    return overlay


def generate_tiles(
    frame: np.ndarray,
    tile_size: int,
    overlap: float = 0.20,
) -> list[tuple[int, int, np.ndarray]]:
    """Делит кадр на одинаковые для всех моделей фрагменты с перекрытием."""
    if tile_size < 1:
        raise ValueError("Размер фрагмента в исходном кадре должен быть положительным")
    if not 0.0 <= overlap < 0.5:
        raise ValueError("Перекрытие фрагментов должно быть от 0 до 0.5")

    height, width = frame.shape[:2]
    columns = max(1, math.ceil(width / tile_size))
    rows = max(1, math.ceil(height / tile_size))
    tile_width = min(
        width,
        max(1, round(width / (columns - (columns - 1) * overlap))),
    )
    tile_height = min(
        height,
        max(1, round(height / (rows - (rows - 1) * overlap))),
    )

    def positions(length: int, tile_length: int, count: int) -> list[int]:
        """Распределяет начало фрагментов равномерно, включая край кадра."""
        if count == 1:
            return [0]
        return [round(index * (length - tile_length) / (count - 1)) for index in range(count)]

    return [
        (x, y, frame[y:y + tile_height, x:x + tile_width])
        for y in positions(height, tile_height, rows)
        for x in positions(width, tile_width, columns)
    ]


def merge_tile_detections(
    detections: Iterable[Detection],
    overlap_threshold: float = 0.45,
) -> tuple[Detection, ...]:
    """Удаляет повторные рамки одного класса из перекрывающихся фрагментов."""
    ordered = sorted(detections, key=lambda detection: detection.confidence, reverse=True)
    retained: list[Detection] = []
    for candidate in ordered:
        duplicate = False
        for selected in retained:
            if selected.name != candidate.name:
                continue
            intersection_width = max(
                0, min(selected.x2, candidate.x2) - max(selected.x1, candidate.x1)
            )
            intersection_height = max(
                0, min(selected.y2, candidate.y2) - max(selected.y1, candidate.y1)
            )
            intersection = intersection_width * intersection_height
            selected_area = max(0, selected.x2 - selected.x1) * max(0, selected.y2 - selected.y1)
            candidate_area = max(0, candidate.x2 - candidate.x1) * max(0, candidate.y2 - candidate.y1)
            union = selected_area + candidate_area - intersection
            if union and intersection / union > overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            retained.append(candidate)
    return tuple(retained)


def infer_full_frame(detector, frame: np.ndarray) -> tuple[tuple[Detection, ...], float, float]:
    """Обрабатывает полный кадр и возвращает рамки, время и сырую уверенность."""
    started = time.perf_counter()
    detections = tuple(detector(frame))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    raw_confidence = float(getattr(detector, "last_best_confidence", 0.0))
    return detections, elapsed_ms, raw_confidence


def infer_tiles(
    detector,
    frame: np.ndarray,
    tile_size: int,
    overlap: float,
) -> tuple[tuple[Detection, ...], float, float]:
    """Обрабатывает перекрывающиеся фрагменты и переводит рамки в координаты кадра."""
    height, width = frame.shape[:2]
    detections: list[Detection] = []
    total_ms = 0.0
    best_raw_confidence = 0.0
    for x, y, tile in generate_tiles(frame, tile_size, overlap):
        started = time.perf_counter()
        tile_detections = tuple(detector(tile))
        total_ms += (time.perf_counter() - started) * 1000.0
        best_raw_confidence = max(
            best_raw_confidence,
            float(getattr(detector, "last_best_confidence", 0.0)),
        )
        for detection in tile_detections:
            x1 = max(0, min(width, detection.x1 + x))
            y1 = max(0, min(height, detection.y1 + y))
            x2 = max(0, min(width, detection.x2 + x))
            y2 = max(0, min(height, detection.y2 + y))
            if x2 > x1 and y2 > y1:
                detections.append(
                    Detection(x1, y1, x2, y2, detection.name, detection.confidence)
                )
    return merge_tile_detections(detections), total_ms, best_raw_confidence


def compare_samples(
    sample_dir: Path,
    model_paths: Iterable[Path],
    output_dir: Path,
    minimum_confidence: float = 0.01,
    draw_threshold: float = 0.15,
    threads: int = 2,
    input_size_reader: Callable[[Path], int] | None = None,
    detector_factory: Callable | None = None,
    include_tiles: bool = False,
    tile_overlap: float = 0.20,
    tile_size: int = 640,
    overlay_style: dict | None = None,
) -> dict:
    """Сравнивает модели на одинаковых выбранных кадрах и сохраняет отчёт."""
    samples = image_paths(Path(sample_dir))
    if not samples:
        raise ValueError(f"Нет изображений для сравнения в папке: {sample_dir}")

    candidates = [Path(path) for path in model_paths]
    if not candidates:
        raise ValueError("Не указаны модели для сравнения")
    if not 0.0 < minimum_confidence <= 1.0:
        raise ValueError("Минимальная уверенность должна быть больше 0 и не больше 1")
    if not 0.0 <= draw_threshold <= 1.0:
        raise ValueError("Порог рамки должен быть от 0 до 1")
    if threads < 1:
        raise ValueError("Количество потоков должно быть не меньше 1")
    if not 0.0 <= tile_overlap < 0.5:
        raise ValueError("Перекрытие фрагментов должно быть от 0 до 0.5")
    if tile_size < 1:
        raise ValueError("Размер фрагмента в исходном кадре должен быть положительным")
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
    if Path(output_dir).exists():
        raise FileExistsError(f"Каталог результата уже существует; перезапись запрещена: {output_dir}")

    read_size = input_size_reader or model_input_size
    create_detector = detector_factory or OnnxDetector
    overlay_style = overlay_style or load_overlay_settings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True)
    model_runs = []
    for model_number, model_path in enumerate(candidates, start=1):
        input_size = read_size(model_path)
        detector = create_detector(
            model_path,
            minimum_confidence,
            False,
            input_size,
            object_label=None,
            threads=threads,
        )
        model_output = output_dir / f"{model_number:02d}_{safe_name(model_path.stem)}"
        model_output.mkdir()
        model_runs.append(
            {
                "path": model_path,
                "name": model_path.name,
                "input_size": input_size,
                "detector": detector,
                "output_dir": model_output,
                "mode_summaries": {},
            }
        )

    report_path = output_dir / "results.csv"
    fields = [
        "frame_file",
        "model_file",
        "inference_mode",
        "input_size",
        "inference_ms",
        "raw_best_confidence_percent",
        "detection_count",
        *(f"count_at_{round(value * 100)}_percent" for value in REPORT_THRESHOLDS),
        "detections_json",
        "overlay_file",
    ]
    rows_written = 0
    with report_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame_number, image_path in enumerate(samples, start=1):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise OSError(f"Не удалось прочитать изображение: {image_path}")
            for run in model_runs:
                inference_results = [
                    ("full", *infer_full_frame(run["detector"], frame))
                ]
                if include_tiles:
                    inference_results.append(
                        (
                            "tiles",
                            *infer_tiles(
                                run["detector"],
                                frame,
                                tile_size,
                                tile_overlap,
                            ),
                        )
                    )
                for mode, detections, elapsed_ms, raw_confidence in inference_results:
                    mode_summary = run["mode_summaries"].setdefault(
                        mode,
                        {
                            "latency_ms": 0.0,
                            "detections_by_threshold": {
                                value: 0 for value in REPORT_THRESHOLDS
                            },
                        },
                    )
                    mode_summary["latency_ms"] += elapsed_ms
                    for threshold in REPORT_THRESHOLDS:
                        mode_summary["detections_by_threshold"][threshold] += sum(
                            detection.confidence >= threshold for detection in detections
                        )

                    overlay_dir = run["output_dir"]
                    if mode == "tiles":
                        overlay_dir = overlay_dir / "tiles"
                        overlay_dir.mkdir(exist_ok=True)
                    overlay = draw_overlay(frame, detections, draw_threshold, overlay_style)
                    overlay_path = overlay_dir / (
                        f"frame_{frame_number:04d}_{safe_name(image_path.stem)}.png"
                    )
                    if not cv2.imwrite(str(overlay_path), overlay):
                        raise OSError(f"Не удалось сохранить результат: {overlay_path}")
                    row = {
                        "frame_file": image_path.name,
                        "model_file": run["name"],
                        "inference_mode": mode,
                        "input_size": run["input_size"],
                        "inference_ms": f"{elapsed_ms:.2f}",
                        "raw_best_confidence_percent": f"{raw_confidence * 100:.2f}",
                        "detection_count": len(detections),
                        "detections_json": json.dumps(
                            [
                                {
                                    "name": detection.name,
                                    "confidence_percent": round(detection.confidence * 100, 2),
                                    "box_xyxy": [
                                        detection.x1,
                                        detection.y1,
                                        detection.x2,
                                        detection.y2,
                                    ],
                                }
                                for detection in detections
                            ],
                            ensure_ascii=False,
                        ),
                        "overlay_file": str(overlay_path.relative_to(output_dir)),
                    }
                    for threshold in REPORT_THRESHOLDS:
                        row[f"count_at_{round(threshold * 100)}_percent"] = sum(
                            detection.confidence >= threshold for detection in detections
                        )
                    writer.writerow(row)
                    rows_written += 1
        stream.flush()

    return {
        "frames": len(samples),
        "models": len(model_runs),
        "rows": rows_written,
        "report": report_path,
        "model_summaries": [
            {
                "model": run["name"],
                "mode": mode,
                "input_size": run["input_size"],
                "average_inference_ms": values["latency_ms"] / len(samples),
                "detections_by_threshold": values["detections_by_threshold"],
            }
            for run in model_runs
            for mode, values in run["mode_summaries"].items()
        ],
    }


def parse_arguments() -> argparse.Namespace:
    """Читает параметры локального сравнения и не трогает рабочий конфиг."""
    parser = argparse.ArgumentParser(
        description="Сравнить готовые модели на выбранных PNG/JPG кадрах без обучения."
    )
    parser.add_argument("samples", type=Path, help="папка с выбранными кадрами")
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        dest="models",
        help="модель ONNX; параметр можно повторять (по умолчанию сравниваются две модели)",
    )
    parser.add_argument("--output-dir", type=Path, help="каталог результата, должен быть новым")
    parser.add_argument("--min-confidence", type=float, default=0.01, help="нижняя граница кандидатов, 0..1")
    parser.add_argument("--draw-threshold", type=float, default=0.15, help="порог рамки на картинке, 0..1")
    parser.add_argument("--threads", type=int, default=2, help="потоки CPU на модель")
    parser.add_argument(
        "--tiles",
        action="store_true",
        help="дополнительно проверить перекрывающиеся увеличенные фрагменты",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.20,
        help="доля перекрытия соседних фрагментов, 0..0.49 (по умолчанию 0.20)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="размер фрагмента исходного кадра; одинаков для всех моделей (по умолчанию 640 px)",
    )
    return parser.parse_args()


def main() -> int:
    """Запускает сравнение и печатает короткую сводку по результатам."""
    args = parse_arguments()
    models = args.models or list(DEFAULT_MODELS)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.samples / f"comparison_{timestamp}"
        suffix = 1
        while output_dir.exists():
            output_dir = args.samples / f"comparison_{timestamp}_{suffix:02d}"
            suffix += 1
    else:
        output_dir = args.output_dir
    try:
        sample_count = len(image_paths(args.samples))
    except ValueError as error:
        print(f"ОШИБКА: {error}", file=sys.stderr, flush=True)
        return 2
    print(f"Кадров: {sample_count}; моделей: {len(models)}", flush=True)
    print("Это сравнение готовых моделей на кадрах, не обучение и не оценка точности по эталонной разметке.", flush=True)
    try:
        summary = compare_samples(
            args.samples,
            models,
            output_dir,
            minimum_confidence=args.min_confidence,
            draw_threshold=args.draw_threshold,
            threads=args.threads,
            include_tiles=args.tiles,
            tile_overlap=args.tile_overlap,
            tile_size=args.tile_size,
        )
    except Exception as error:
        print(f"ОШИБКА ({type(error).__name__}): {error}", file=sys.stderr, flush=True)
        return 2

    for model in summary["model_summaries"]:
        threshold_counts = " | ".join(
            f"≥{round(threshold * 100)}%: {model['detections_by_threshold'][threshold]}"
            for threshold in REPORT_THRESHOLDS
        )
        print(
            f"{model['model']} ({model['input_size']}×{model['input_size']}): "
            f"{model['mode']}: среднее {model['average_inference_ms']:.1f} мс/кадр; {threshold_counts}",
            flush=True,
        )
    print(f"Сравнение готово: {summary['report']}", flush=True)
    print(f"Картинки с рамками: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
