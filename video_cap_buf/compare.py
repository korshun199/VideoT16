#!/usr/bin/env python3
"""Сравнивает модели и масштабы на одной выборке кадров с USB-видеопотока."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np

from capture import load_settings, resolve_model_path
from detector import Detection, OnnxRuntimeDetector


PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PROGRAM_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "tmp" / "video_cap_buf"


@dataclass(frozen=True)
class ModelMode:
    """Хранит название сравниваемого режима и ONNX-файл с размером входа."""

    key: str
    title: str
    model_path: Path
    input_size: int
    tiled: bool = False
    motion_roi: bool = False


@dataclass(frozen=True)
class SampleFrame:
    """Хранит выбранный кадр, его номер и измерения качества видеосигнала."""

    number: int
    captured_ns: int
    image: np.ndarray
    mean_luma: float
    luma_stddev: float
    near_black_percent: float


@dataclass(frozen=True)
class FrameEvaluation:
    """Сохраняет ответы модели и длительность обработки одного кадра."""

    candidates: tuple[Detection, ...]
    detections: tuple[Detection, ...]
    elapsed_ms: float
    diagnostics: dict[str, Any] | None = None


@dataclass(frozen=True)
class MotionAnalysis:
    """Сохраняет области локального движения и качество компенсации фона."""

    regions: tuple[tuple[int, int, int, int], ...]  # Квадраты для проверки моделью в координатах полного кадра.
    components: tuple[tuple[int, int, int, int], ...]  # Найденные локальные изменения до расширения области.
    status: str  # Причина успешного поиска или безопасного пропуска кадра.
    tracked_points: int  # Число точек, которые удалось проследить между кадрами.
    inliers: int  # Число точек, совпавших с движением основного фона.
    changed_pixels: int  # Площадь остаточного движения после компенсации фона.
    residual_median: float  # Медианная разница яркости выровненного фона.
    elapsed_ms: float  # Время анализа движения без работы модели.


class MotionRoiProposer:
    """Предлагает крупные области на основе движения после стабилизации фона."""

    def __init__(self, roi_size: int = 640, max_regions: int = 4) -> None:
        """Сохраняет ограничители прототипа и ждёт первый кадр для базовой линии."""
        self.roi_size = max(64, int(roi_size))
        self.max_regions = max(1, int(max_regions))
        self.previous_gray: np.ndarray | None = None

    def reset(self) -> None:
        """Сбрасывает опорный кадр после разрыва видеопотока."""
        self.previous_gray = None

    def process(self, image: np.ndarray) -> MotionAnalysis:
        """Оценивает общий перенос фона и находит небольшие локальные изменения."""
        started_ns = time.perf_counter_ns()
        frame_height, frame_width = image.shape[:2]
        working_width = min(960, frame_width)
        working_height = max(1, round(frame_height * working_width / frame_width))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if (working_width, working_height) != (frame_width, frame_height):
            gray = cv2.resize(gray, (working_width, working_height), interpolation=cv2.INTER_AREA)

        if self.previous_gray is None:
            self.previous_gray = gray
            return self._empty_analysis("базовый кадр", started_ns)

        previous = self.previous_gray
        self.previous_gray = gray
        points = cv2.goodFeaturesToTrack(
            previous, maxCorners=800, qualityLevel=0.01, minDistance=7, blockSize=5
        )
        if points is None or len(points) < 8:
            return self._empty_analysis("недостаточно точек фона", started_ns)

        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous, gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if moved is None or status is None:
            return self._empty_analysis("не удалось проследить фон", started_ns, len(points))

        valid = status.reshape(-1) == 1
        source_points = points.reshape(-1, 2)[valid]
        target_points = moved.reshape(-1, 2)[valid]
        if len(source_points) < 8:
            return self._empty_analysis("мало прослеженных точек", started_ns, len(source_points))

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source_points, target_points, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=500, confidence=0.995, refineIters=10,
        )
        inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
        if affine is None or inliers < 6 or not np.isfinite(affine).all():
            return self._empty_analysis(
                "фон не стабилизирован", started_ns, len(source_points), inliers
            )

        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        shift_x, shift_y = float(affine[0, 2]), float(affine[1, 2])
        if not 0.75 <= scale <= 1.25 or np.hypot(shift_x, shift_y) > 120:
            return self._empty_analysis(
                "резкий сдвиг камеры; кадр пропущен", started_ns, len(source_points), inliers
            )

        aligned_previous = cv2.warpAffine(
            previous, affine.astype(np.float32), (working_width, working_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        valid_pixels = cv2.warpAffine(
            np.full(previous.shape, 255, dtype=np.uint8),
            affine.astype(np.float32), (working_width, working_height),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ) > 0
        difference = cv2.absdiff(gray, aligned_previous)
        differences_in_background = difference[valid_pixels]
        if differences_in_background.size == 0:
            return self._empty_analysis(
                "нет общей области фона", started_ns, len(source_points), inliers
            )
        residual_median = float(np.median(differences_in_background))
        threshold = max(16.0, residual_median + 12.0)
        mask = (difference >= threshold) & valid_pixels
        motion_mask = mask.astype(np.uint8) * 255
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        motion_mask = cv2.morphologyEx(
            motion_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
        )

        contours, _ = cv2.findContours(
            motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = working_width * working_height
        scale_x = frame_width / working_width
        scale_y = frame_height / working_height
        components: list[tuple[int, int, int, int, int]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = width * height
            if area < 10 or area > frame_area * 0.06:
                continue
            changed_pixels = int(cv2.countNonZero(
                motion_mask[y:y + height, x:x + width]
            ))
            if changed_pixels < 16:
                continue
            left = max(0, min(frame_width - 1, round(x * scale_x)))
            top = max(0, min(frame_height - 1, round(y * scale_y)))
            right = max(left + 1, min(frame_width, round((x + width) * scale_x)))
            bottom = max(top + 1, min(frame_height, round((y + height) * scale_y)))
            components.append((left, top, right, bottom, changed_pixels))

        components.sort(key=lambda item: item[4], reverse=True)
        regions: list[tuple[int, int, int, int]] = []
        for left, top, right, bottom, _ in components:
            region = self._square_region(
                (left + right) // 2, (top + bottom) // 2,
                self.roi_size,
                frame_width, frame_height,
            )
            if any(self._iou(region, accepted) > 0.35 for accepted in regions):
                continue
            regions.append(region)
            if len(regions) >= self.max_regions:
                break

        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        return MotionAnalysis(
            regions=tuple(regions),
            components=tuple(item[:4] for item in components[:self.max_regions]),
            status="фон стабилизирован",
            tracked_points=len(source_points),
            inliers=inliers,
            changed_pixels=int(cv2.countNonZero(motion_mask)),
            residual_median=round(residual_median, 3),
            elapsed_ms=elapsed_ms,
        )

    def _empty_analysis(
        self, reason: str, started_ns: int, tracked_points: int = 0, inliers: int = 0
    ) -> MotionAnalysis:
        """Возвращает безопасный пустой результат и диагностическую причину."""
        return MotionAnalysis(
            regions=(), components=(), status=reason,
            tracked_points=tracked_points, inliers=inliers,
            changed_pixels=0, residual_median=0.0,
            elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
        )

    @staticmethod
    def _square_region(
        center_x: int, center_y: int, requested_size: int, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        """Строит ограниченную квадратную область вокруг локального движения."""
        size = min(requested_size, frame_width, frame_height)
        left = max(0, min(frame_width - size, center_x - size // 2))
        top = max(0, min(frame_height - size, center_y - size // 2))
        return left, top, left + size, top + size

    @staticmethod
    def _iou(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int]
    ) -> float:
        """Считает пересечение по объединению двух предложенных областей."""
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        area_first = (first[2] - first[0]) * (first[3] - first[1])
        area_second = (second[2] - second[0]) * (second[3] - second[1])
        union = area_first + area_second - intersection
        return intersection / union if union else 0.0


def _capture_samples(
    settings: dict[str, Any], duration_seconds: float, sample_fps: float, max_samples: int
) -> tuple[list[SampleFrame], dict[str, Any]]:
    """Открывает USB-поток, выбирает равномерные кадры и всегда освобождает устройство."""
    source = str(Path(settings["source"]).resolve())
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Не удалось открыть USB-видеоустройство: {settings['source']}")

        requested = (
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*settings["pixel_format"])),
            (cv2.CAP_PROP_FRAME_WIDTH, settings["width"]),
            (cv2.CAP_PROP_FRAME_HEIGHT, settings["height"]),
            (cv2.CAP_PROP_FPS, settings["fps"]),
        )
        accepted = [bool(camera.set(prop, value)) for prop, value in requested]
        actual = {
            "width": camera.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": camera.get(cv2.CAP_PROP_FPS),
            "pixel_format": _fourcc_text(camera.get(cv2.CAP_PROP_FOURCC)),
            "accepted": accepted,
        }

        samples: list[SampleFrame] = []
        started_ns: int | None = None
        next_sample_ns = 0
        frames_read = 0
        sample_interval_ns = int(1_000_000_000 / sample_fps)
        while len(samples) < max_samples:
            ok, image = camera.read()
            captured_ns = time.monotonic_ns()
            if not ok or image is None:
                raise RuntimeError("USB-оцифровщик перестал возвращать кадры")
            frames_read += 1
            if started_ns is None:
                started_ns = captured_ns
                next_sample_ns = started_ns
            if captured_ns - started_ns >= int(duration_seconds * 1_000_000_000):
                break
            if captured_ns >= next_sample_ns:
                # Сразу отмечаем пустой вход: нулевой сигнал нельзя считать провалом YOLO.
                preview_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                preview_gray = cv2.resize(preview_gray, (240, 135), interpolation=cv2.INTER_AREA)
                samples.append(SampleFrame(
                    len(samples) + 1,
                    captured_ns,
                    image.copy(),
                    float(np.mean(preview_gray)),
                    float(np.std(preview_gray)),
                    float(np.count_nonzero(preview_gray < 8) * 100 / preview_gray.size),
                ))
                next_sample_ns = captured_ns + sample_interval_ns
        if not samples:
            raise RuntimeError("За время захвата не удалось выбрать ни одного кадра")
        actual["frames_read"] = frames_read
        actual["sample_count"] = len(samples)
        actual["sample_signal"] = [
            {
                "frame": sample.number,
                "mean_luma": round(sample.mean_luma, 3),
                "luma_stddev": round(sample.luma_stddev, 3),
                "near_black_percent": round(sample.near_black_percent, 3),
            }
            for sample in samples
        ]
        actual["duration_seconds"] = (
            (samples[-1].captured_ns - samples[0].captured_ns) / 1_000_000_000
            if len(samples) > 1 else 0.0
        )
        return samples, actual
    finally:
        camera.release()


def _fourcc_text(value: float) -> str:
    """Преобразует числовой код формата кадров в четыре читаемых символа."""
    code = int(value)
    return "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4)).rstrip("\x00")


def _make_detector(
    model_path: Path, input_size: int, settings: dict[str, Any], threads: int
) -> OnnxRuntimeDetector:
    """Создаёт ONNX Runtime для указанного файла и порогов текущего профиля."""
    return OnnxRuntimeDetector(
        model_path=model_path,
        input_size=input_size,
        confidence_percent=float(settings["confidence_percent"]),
        nms_iou=float(settings["nms_iou_threshold"]),
        class_names=list(settings["class_names"]),
        box_color_bgr=tuple(settings["box_color_bgr"]),
        text_color_bgr=tuple(settings["text_color_bgr"]),
        weak_candidate_color_bgr=tuple(settings.get("weak_candidate_color_bgr", [255, 0, 255])),
        box_thickness=int(settings["box_thickness"]),
        font_scale=float(settings["font_scale"]),
        threads=threads,
        presence_confidence_percent=float(
            settings.get("presence_confidence_percent", settings["confidence_percent"])
        ),
    )


def _tile_regions(
    width: int, height: int, overlap_fraction: float
) -> tuple[tuple[int, int, int, int], ...]:
    """Рассчитывает четыре перекрывающиеся области кадра в исходных координатах."""
    tile_width = min(width, math.ceil(width / (2.0 - overlap_fraction)))
    tile_height = min(height, math.ceil(height / (2.0 - overlap_fraction)))
    x_positions = sorted({0, width - tile_width})
    y_positions = sorted({0, height - tile_height})
    return tuple(
        (x, y, min(x + tile_width, width), min(y + tile_height, height))
        for y in y_positions
        for x in x_positions
    )


def _suppress_cross_tile_duplicates(
    detections: list[Detection], nms_iou: float, confidence_threshold: float
) -> tuple[Detection, ...]:
    """Объединяет рамки соседних областей и подавляет повторные рамки одного класса."""
    by_class: dict[int, list[Detection]] = {}
    for detection in detections:
        by_class.setdefault(detection.class_id, []).append(detection)

    kept: list[Detection] = []
    for class_detections in by_class.values():
        boxes = [
            [item.left, item.top, item.right - item.left, item.bottom - item.top]
            for item in class_detections
        ]
        scores = [item.confidence for item in class_detections]
        selected = cv2.dnn.NMSBoxes(
            boxes, scores, confidence_threshold, nms_iou
        )
        kept.extend(class_detections[int(index)] for index in np.asarray(selected).reshape(-1))
    return tuple(sorted(kept, key=lambda item: item.confidence, reverse=True))


def _detect_tiled(
    detector: OnnxRuntimeDetector,
    image: np.ndarray,
    overlap_fraction: float,
    confidence_threshold: float,
    nms_iou: float,
) -> tuple[tuple[Detection, ...], tuple[Detection, ...]]:
    """Запускает одну и ту же модель на четырёх перекрывающихся участках кадра."""
    image_height, image_width = image.shape[:2]
    candidates: list[Detection] = []
    for left, top, right, bottom in _tile_regions(
        image_width, image_height, overlap_fraction
    ):
        crop = image[top:bottom, left:right]
        output, geometry, _ = detector._prepare_and_infer(crop)
        tile_candidates = detector._decode(
            output, *geometry, confidence_threshold=confidence_threshold
        )
        candidates.extend(
            Detection(
                class_id=item.class_id,
                label=item.label,
                confidence=item.confidence,
                left=item.left + left,
                top=item.top + top,
                right=min(item.right + left, image_width),
                bottom=min(item.bottom + top, image_height),
            )
            for item in tile_candidates
        )

    merged = _suppress_cross_tile_duplicates(candidates, nms_iou, confidence_threshold)
    detections = tuple(
        item for item in merged if item.confidence >= detector.confidence_threshold
    )
    return merged, detections


def _evaluate_mode(
    detector: OnnxRuntimeDetector,
    image: np.ndarray,
    mode: ModelMode,
    settings: dict[str, Any],
    overlap_fraction: float,
    motion_proposer: MotionRoiProposer | None = None,
) -> FrameEvaluation:
    """Обрабатывает полный кадр, плитки или области движения и измеряет время."""
    started_ns = time.perf_counter_ns()
    diagnostics: dict[str, Any] | None = None
    if mode.motion_roi:
        if motion_proposer is None:
            raise ValueError("Для режима локального движения не задан анализатор")
        analysis = motion_proposer.process(image)
        candidates: list[Detection] = []
        detections: list[Detection] = []
        for left, top, right, bottom in analysis.regions:
            crop = image[top:bottom, left:right]
            roi_candidates, roi_detections, _ = detector.detect_with_presence(crop)
            candidates.extend(
                Detection(
                    class_id=item.class_id, label=item.label,
                    confidence=item.confidence,
                    left=item.left + left, top=item.top + top,
                    right=min(item.right + left, image.shape[1]),
                    bottom=min(item.bottom + top, image.shape[0]),
                )
                for item in roi_candidates
            )
            detections.extend(
                Detection(
                    class_id=item.class_id, label=item.label,
                    confidence=item.confidence,
                    left=item.left + left, top=item.top + top,
                    right=min(item.right + left, image.shape[1]),
                    bottom=min(item.bottom + top, image.shape[0]),
                )
                for item in roi_detections
            )
        presence_threshold = float(
            settings.get("presence_confidence_percent", settings["confidence_percent"])
        ) / 100.0
        candidates = list(_suppress_cross_tile_duplicates(
            candidates, float(settings["nms_iou_threshold"]), presence_threshold
        ))
        detections = list(_suppress_cross_tile_duplicates(
            detections, float(settings["nms_iou_threshold"]), detector.confidence_threshold
        ))
        diagnostics = {
            "motion_status": analysis.status,
            "tracked_points": analysis.tracked_points,
            "background_inliers": analysis.inliers,
            "changed_pixels": analysis.changed_pixels,
            "residual_median": analysis.residual_median,
            "motion_ms": round(analysis.elapsed_ms, 3),
            "motion_components": [list(box) for box in analysis.components],
            "motion_regions": [list(box) for box in analysis.regions],
        }
        candidate_result = tuple(candidates)
        detection_result = tuple(detections)
    elif mode.tiled:
        candidates, detections = _detect_tiled(
            detector,
            image,
            overlap_fraction,
            float(settings.get("presence_confidence_percent", settings["confidence_percent"])) / 100.0,
            float(settings["nms_iou_threshold"]),
        )
        candidate_result = candidates
        detection_result = detections
    else:
        candidate_result, detection_result, _ = detector.detect_with_presence(image)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    return FrameEvaluation(candidate_result, detection_result, elapsed_ms, diagnostics)


def _detection_dict(detection: Detection) -> dict[str, Any]:
    """Преобразует рамку в простой JSON-совместимый словарь."""
    return {
        "class_id": detection.class_id,
        "label": detection.label,
        "confidence": round(detection.confidence, 6),
        "box": [detection.left, detection.top, detection.right, detection.bottom],
    }


def _draw_result(
    image: np.ndarray,
    evaluation: FrameEvaluation,
    confidence_threshold: float,
) -> np.ndarray:
    """Создаёт копию кадра для таблицы с кандидатами и подтверждёнными рамками."""
    output = image.copy()
    if evaluation.diagnostics:
        # Оранжевым показаны области, переданные детектору после поиска движения.
        for left, top, right, bottom in evaluation.diagnostics.get("motion_regions", []):
            cv2.rectangle(output, (left, top), (right, bottom), (0, 165, 255), 2)
    for item in evaluation.candidates:
        strong = item.confidence >= confidence_threshold
        color = (0, 255, 0) if strong else (255, 0, 255)
        cv2.rectangle(output, (item.left, item.top), (item.right, item.bottom), color, 2)
        prefix = "" if strong else "? "
        cv2.putText(
            output,
            f"{prefix}{item.label} {item.confidence * 100:.0f}%",
            (item.left, max(20, item.top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def _write_contact_sheets(
    output_dir: Path,
    samples: list[SampleFrame],
    modes: list[ModelMode],
    evaluations: dict[str, list[FrameEvaluation]],
    confidence_threshold: float,
) -> list[str]:
    """Сохраняет таблицы кадров для визуального сравнения всех режимов."""
    page_paths: list[str] = []
    cell_width, cell_height = 480, 270
    page_size = 6
    columns = len(modes) + 1
    titles = ["RAW FRAME", *(mode.key.upper() for mode in modes)]
    for page_number, start in enumerate(range(0, len(samples), page_size), start=1):
        page_samples = samples[start:start + page_size]
        sheet = np.full(
            (cell_height * (len(page_samples) + 1), cell_width * columns, 3),
            (28, 28, 28),
            dtype=np.uint8,
        )
        for column, title in enumerate(titles):
            cv2.putText(
                sheet,
                title[:38],
                (column * cell_width + 8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        for row, sample in enumerate(page_samples, start=1):
            cells = [sample.image]
            for mode in modes:
                evaluation = evaluations[mode.key][start + row - 1]
                cells.append(
                    _draw_result(sample.image, evaluation, confidence_threshold)
                )
            for column, cell in enumerate(cells):
                resized = cv2.resize(cell, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
                y = row * cell_height
                x = column * cell_width
                sheet[y:y + cell_height, x:x + cell_width] = resized
            cv2.putText(
                sheet,
                f"FRAME {sample.number}",
                (8, row * cell_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        path = output_dir / f"comparison_{page_number:02d}.jpg"
        if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise RuntimeError(f"Не удалось записать таблицу кадров: {path}")
        page_paths.append(str(path))
    return page_paths


def _summarize(evaluations: list[FrameEvaluation]) -> dict[str, Any]:
    """Считает наблюдаемые срабатывания и времена без заявления о полноте модели."""
    elapsed = [item.elapsed_ms for item in evaluations]
    motion_results = [item.diagnostics for item in evaluations if item.diagnostics]
    return {
        "sample_frames": len(evaluations),
        "frames_with_model_candidate": sum(bool(item.candidates) for item in evaluations),
        "frames_with_box": sum(bool(item.detections) for item in evaluations),
        "candidate_boxes": sum(len(item.candidates) for item in evaluations),
        "median_full_pass_ms": round(statistics.median(elapsed), 3) if elapsed else None,
        "p95_full_pass_ms": round(float(np.percentile(elapsed, 95)), 3) if elapsed else None,
        "frames_with_motion_region": sum(
            bool(item.get("motion_regions")) for item in motion_results
        ),
        "motion_regions_total": sum(
            len(item.get("motion_regions", [])) for item in motion_results
        ),
        "motion_status_counts": {
            status: sum(item.get("motion_status") == status for item in motion_results)
            for status in sorted({item.get("motion_status", "") for item in motion_results})
        },
        "metric_note": "Счётчики кандидатов; полнота не вычисляется без разметки кадров.",
    }


def _build_modes(
    settings: dict[str, Any], settings_path: Path, motion_only: bool = False
) -> list[ModelMode]:
    """Готовит текущую модель, пару одинаковых весов 320/640 и режим плиток."""
    current_path = resolve_model_path(settings, settings_path)
    if settings["inference_size"] != 320:
        raise ValueError(
            "Сравнение плитками требует текущий baseline с входом 320; "
            "временно выберите модель 320×320 в settings.json"
        )
    if not current_path.is_file():
        raise FileNotFoundError(f"Текущая модель не найдена: {current_path}")
    current_mode = ModelMode("current", f"Текущая модель {current_path.name} 320", current_path, 320)
    motion_mode = ModelMode(
        "motion_roi", "Движение + fast320 в ROI 640", current_path, 320,
        motion_roi=True,
    )
    if motion_only:
        return [current_mode, motion_mode]
    small_path = MODEL_DIR / "anti_uav_best.onnx"
    large_path = MODEL_DIR / "anti_uav_best_from_pt_640.onnx"
    missing = [str(path) for path in (small_path, large_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Не найдены модели сравнения: " + ", ".join(missing))
    return [
        current_mode,
        ModelMode("best320", "anti_uav_best 320", small_path, 320),
        ModelMode("best640", "anti_uav_best 640", large_path, 640),
        ModelMode("tiles320", "anti_uav_best 2×2 плитки 320", small_path, 320, True),
        motion_mode,
    ]


def _run_comparison(
    settings: dict[str, Any],
    settings_path: Path,
    duration_seconds: float,
    sample_fps: float,
    max_samples: int,
    overlap_fraction: float,
    threads: int,
    motion_only: bool,
) -> Path:
    """Захватывает один набор кадров, сравнивает режимы и записывает отчёт в tmp."""
    modes = _build_modes(settings, settings_path, motion_only)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_DIR / f"compare_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "comparison.jsonl"
    settings_copy = dict(settings)
    with log_path.open("w", encoding="utf-8") as log_file:
        _write_jsonl(log_file, "comparison_started", {
            "run_id": run_id,
            "duration_seconds": duration_seconds,
            "sample_fps": sample_fps,
            "max_samples": max_samples,
            "tile_overlap": overlap_fraction,
            "motion_only": motion_only,
            "motion_roi_size": 640,
            "motion_max_regions": 4,
            "motion_working_width": 960,
            "motion_difference_threshold": 16,
            "motion_min_component_area": 10,
            "motion_min_changed_pixels": 16,
            "motion_max_component_area_fraction": 0.06,
            "threads": threads,
            "threshold_percent": settings["confidence_percent"],
            "presence_threshold_percent": settings.get(
                "presence_confidence_percent", settings["confidence_percent"]
            ),
            "modes": [asdict(mode) | {"model_path": str(mode.model_path)} for mode in modes],
        })
        print(f"Сниму выборку с USB: до {duration_seconds:g} с, {sample_fps:g} кадр/с…", flush=True)
        try:
            samples, capture_info = _capture_samples(
                settings, duration_seconds, sample_fps, max_samples
            )
            _write_jsonl(log_file, "sample_captured", capture_info)
            print(
                f"Захвачено одинаковых для всех режимов кадров: {len(samples)}; "
                f"источник {capture_info['width']:.0f}×{capture_info['height']:.0f} "
                f"@ {capture_info['fps']:.1f} FPS, {capture_info['pixel_format']}",
                flush=True,
            )
            print(
                "Сигнал выборки: яркость в среднем "
                f"{statistics.median(sample.mean_luma for sample in samples):.1f}/255, "
                "контраст "
                f"{statistics.median(sample.luma_stddev for sample in samples):.1f}/255.",
                flush=True,
            )

            black_samples = [
                sample for sample in samples
                if sample.mean_luma < 3.0 and sample.luma_stddev < 2.0
            ]
            if len(black_samples) == len(samples):
                summary_path = output_dir / "summary.json"
                invalid_signal_summary = {
                    "run_id": run_id,
                    "status": "no_video_signal_or_all_black",
                    "capture": capture_info,
                    "sample_frames": len(samples),
                    "message": (
                        "Все выбранные изображения почти полностью чёрные; "
                        "сравнение моделей пропущено, поскольку входной видеосигнал пуст."
                    ),
                }
                summary_path.write_text(
                    json.dumps(invalid_signal_summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                _write_jsonl(log_file, "video_signal_invalid", invalid_signal_summary)
                print(
                    "ОШИБКА ВХОДА: все выбранные кадры почти чёрные; "
                    "распознавание пропущено. Сначала восстановите видеосигнал.",
                    flush=True,
                )
                print(f"Диагностика сигнала: {summary_path} и {log_path}", flush=True)
                raise RuntimeError(
                    f"Видеосигнал отсутствует или кадры чёрные; отчёт: {summary_path}"
                )

            evaluations: dict[str, list[FrameEvaluation]] = {}
            model_groups: dict[tuple[Path, int], list[ModelMode]] = {}
            for mode in modes:
                model_groups.setdefault((mode.model_path.resolve(), mode.input_size), []).append(mode)
            mode_load_times: dict[str, float] = {}
            for (model_path, input_size), grouped_modes in model_groups.items():
                detector = None
                try:
                    load_started_ns = time.perf_counter_ns()
                    detector = _make_detector(
                        model_path, input_size, settings_copy, threads
                    )
                    load_ms = (time.perf_counter_ns() - load_started_ns) / 1_000_000
                    # Прогреваем реальным первым кадром; сам кадр всё равно попадёт в сравнение.
                    detector.detect_with_presence(samples[0].image)
                    for mode in grouped_modes:
                        print(f"Сравниваю: {mode.title}…", flush=True)
                        mode_results: list[FrameEvaluation] = []
                        motion_proposer = MotionRoiProposer() if mode.motion_roi else None
                        for sample in samples:
                            result = _evaluate_mode(
                                detector, sample.image, mode, settings_copy,
                                overlap_fraction, motion_proposer,
                            )
                            mode_results.append(result)
                            _write_jsonl(log_file, "frame_evaluated", {
                                "mode": mode.key,
                                "frame": sample.number,
                                "captured_ns": sample.captured_ns,
                                "elapsed_ms": round(result.elapsed_ms, 3),
                                "model_candidates": [
                                    _detection_dict(item) for item in result.candidates
                                ],
                                "display_detections": [
                                    _detection_dict(item) for item in result.detections
                                ],
                                "diagnostics": result.diagnostics,
                            })
                        evaluations[mode.key] = mode_results
                        summary = _summarize(mode_results)
                        mode_load_times[mode.key] = round(load_ms, 3)
                        _write_jsonl(log_file, "mode_finished", {
                            "mode": mode.key,
                            "model_load_ms": round(load_ms, 3),
                            **summary,
                        })
                        print(
                            f"  Кандидат есть на {summary['frames_with_model_candidate']}/"
                            f"{len(samples)} кадрах; зелёная рамка на "
                            f"{summary['frames_with_box']}/{len(samples)}; "
                            f"медиана {summary['median_full_pass_ms']} мс, "
                            f"p95 {summary['p95_full_pass_ms']} мс.",
                            flush=True,
                        )
                finally:
                    if detector is not None:
                        detector.session = None
                        del detector

            pages = _write_contact_sheets(
                output_dir, samples, modes, evaluations,
                float(settings["confidence_percent"]) / 100.0,
            )
            summary = {
                "run_id": run_id,
                "capture": capture_info,
                "threshold_percent": settings["confidence_percent"],
                "presence_threshold_percent": settings.get(
                    "presence_confidence_percent", settings["confidence_percent"]
                ),
                "tile_overlap": overlap_fraction,
                "sample_fps": sample_fps,
                "modes": {
                    mode.key: {
                        "title": mode.title,
                        "model": str(mode.model_path),
                        "input_size": mode.input_size,
                        "tiled": mode.tiled,
                        "motion_roi": mode.motion_roi,
                        "model_load_ms": mode_load_times[mode.key],
                        **_summarize(evaluations[mode.key]),
                    }
                    for mode in modes
                },
                "contact_sheets": pages,
                "motion_proposal_settings": {
                    "algorithm": "LK optical flow + affine RANSAC stabilization + residual threshold",
                    "working_width": 960,
                    "max_motion_regions": 4,
                    "roi_size_px": 640,
                    "difference_threshold_min": 16,
                    "minimum_component_area_working_pixels": 10,
                    "minimum_changed_pixels": 16,
                    "maximum_component_area_fraction": 0.06,
                    "fail_closed_without_stable_background": True,
                },
                "warning": "Без покадровой истины показатели отражают выходы моделей, не точность/полноту.",
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _write_jsonl(log_file, "comparison_finished", summary)
        except Exception as error:
            _write_jsonl(log_file, "comparison_error", {
                "error_type": type(error).__name__, "message": str(error),
            })
            raise
    print(f"Таблица кадров: {', '.join(pages)}", flush=True)
    print(f"Сводка и журнал: {summary_path} и {log_path}", flush=True)
    return summary_path


def _write_jsonl(stream: Any, event: str, fields: dict[str, Any]) -> None:
    """Добавляет одну UTF-8 запись JSONL и сразу сбрасывает её на диск."""
    row = {"event": event, "time": datetime.now().astimezone().isoformat(), **fields}
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    """Разбирает параметры и запускает ограниченный сравнительный опыт."""
    parser = argparse.ArgumentParser(
        description="Сравнение моделей на одних и тех же кадрах USB-потока"
    )
    parser.add_argument("--settings", type=Path, default=PROGRAM_DIR / "settings.json",
                        help="путь к JSON-настройкам основной программы")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="сколько секунд собирать кадры (по умолчанию 8)")
    parser.add_argument("--sample-fps", type=float, default=2.0,
                        help="частота выборки кадров (по умолчанию 2) ")
    parser.add_argument("--max-samples", type=int, default=24,
                        help="жёсткий предел числа кадров (по умолчанию 24)")
    parser.add_argument("--tile-overlap", type=float, default=0.2,
                        help="доля перекрытия соседних плиток, 0–0.49")
    parser.add_argument("--motion-only", action="store_true",
                        help="сравнить только полный кадр и экспериментальный поиск по движению")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 0 < args.duration <= 120:
        parser.error("--duration должен быть в диапазоне 0–120 секунд")
    if not math.isfinite(args.sample_fps) or not 0 < args.sample_fps <= 30:
        parser.error("--sample-fps должен быть в диапазоне 0–30")
    if not 1 <= args.max_samples <= 120:
        parser.error("--max-samples должен быть от 1 до 120")
    if not math.isfinite(args.tile_overlap) or not 0 <= args.tile_overlap < 0.5:
        parser.error("--tile-overlap должен быть от 0 до 0.49")

    try:
        settings = load_settings(args.settings)
        threads = settings["inference_threads_per_worker"]
        summary = _run_comparison(
            settings,
            args.settings,
            args.duration,
            args.sample_fps,
            args.max_samples,
            args.tile_overlap,
            threads,
            args.motion_only,
        )
    except Exception as error:
        print(f"Ошибка сравнительного прогона: {error}", file=sys.stderr)
        return 1
    print(f"Сравнение завершено: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
