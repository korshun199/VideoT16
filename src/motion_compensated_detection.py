"""Поиск движения и цветовых отличий после компенсации движения камеры."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionBox:
    """Кандидат движущегося объекта в координатах вырезанной области."""

    x1: int
    y1: int
    x2: int
    y2: int
    changed_pixels: int
    mean_luma_delta: float
    mean_chroma_delta: float
    local_color_contrast: float
    frames_seen: int = 1


@dataclass(frozen=True)
class MotionResult:
    """Результат анализа одного кадра и параметры найденного движения фона."""

    roi: tuple[int, int, int, int]
    mask: np.ndarray
    boxes: tuple[MotionBox, ...]
    background_shift: tuple[float, float] | None
    background_inliers: int
    compensated: bool
    status: str
    luma_pixels: int
    chroma_pixels: int


def center_roi_bounds(frame_shape: tuple[int, ...], settings: dict[str, Any]) -> tuple[int, int, int, int]:
    """Вычисляет прямоугольник области поиска относительно кадра."""
    frame_height, frame_width = frame_shape[:2]
    roi_width = max(1, min(frame_width, round(frame_width * float(settings["width_percent"]) / 100)))
    roi_height = max(1, min(frame_height, round(frame_height * float(settings["height_percent"]) / 100)))
    center_x = round(frame_width * float(settings["center_x_percent"]) / 100)
    center_y = round(frame_height * float(settings["center_y_percent"]) / 100)
    left = max(0, min(frame_width - roi_width, center_x - roi_width // 2))
    top = max(0, min(frame_height - roi_height, center_y - roi_height // 2))
    return left, top, roi_width, roi_height


def box_iou(first: MotionBox, second: MotionBox) -> float:
    """Считает пересечение по объединению двух рамок движения."""
    intersection_width = max(0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = max(0, first.x2 - first.x1) * max(0, first.y2 - first.y1)
    second_area = max(0, second.x2 - second.x1) * max(0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class MotionAnalyzer:
    """Сравнивает подряд идущие кропы, удаляя общее движение камеры."""

    def __init__(self, roi_settings: dict[str, Any], motion_settings: dict[str, Any]) -> None:
        """Сохраняет параметры поиска и начинает без кадра-предшественника."""
        self.roi_settings = dict(roi_settings)
        self.settings = dict(motion_settings)
        self._previous_gray: np.ndarray | None = None
        self._previous_lab: np.ndarray | None = None
        self._tracks: list[MotionBox] = []

    def reset(self) -> None:
        """Сбрасывает фон и историю кандидатов после смены источника."""
        self._previous_gray = None
        self._previous_lab = None
        self._tracks.clear()

    def process(self, frame: np.ndarray) -> MotionResult:
        """Выделяет остаточное движение и цветовую разницу в центральной области."""
        left, top, width, height = center_roi_bounds(frame.shape, self.roi_settings)
        crop = frame[top : top + height, left : left + width]
        current_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        current_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        empty_mask = np.zeros((height, width), dtype=np.uint8)

        if self._previous_gray is not None and self._previous_gray.shape != current_gray.shape:
            self.reset()

        if self._previous_gray is None or self._previous_lab is None:
            self._remember(current_gray, current_lab)
            return MotionResult(
                (left, top, width, height), empty_mask, (), None, 0, False,
                "Калибровка фона: жду следующий кадр", 0, 0,
            )

        transform, shift, inliers, compensated, status = self._estimate_background_motion(
            self._previous_gray, current_gray
        )
        previous_lab = cv2.warpAffine(
            self._previous_lab,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid_pixels = cv2.warpAffine(
            np.full((height, width), 255, dtype=np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ) > 0

        difference = current_lab.astype(np.int16) - previous_lab.astype(np.int16)
        if np.any(valid_pixels):
            # Медиана вычитает общую перемену экспозиции, не связанную с движением.
            illumination_delta = np.median(difference[valid_pixels], axis=0).astype(np.int16)
        else:
            illumination_delta = np.zeros(3, dtype=np.int16)
        residual = difference - illumination_delta
        luma_map = np.clip(np.abs(residual[:, :, 0]), 0, 255).astype(np.uint8)
        chroma_map = np.sqrt(
            residual[:, :, 1].astype(np.float32) ** 2
            + residual[:, :, 2].astype(np.float32) ** 2
        )
        luma_mask = (luma_map >= int(self.settings["luma_threshold"])) & valid_pixels
        chroma_mask = (chroma_map >= int(self.settings["chroma_threshold"])) & valid_pixels
        motion_mask = (luma_mask | chroma_mask) & valid_pixels

        if bool(self.settings["suppress_specular"]):
            motion_mask &= ~self._specular_highlights(crop)

        radius = int(self.settings["morphology_radius"])
        if radius > 0:
            kernel_width = radius * 2 + 1
            kernel = np.ones((kernel_width, kernel_width), dtype=np.uint8)
            motion_mask = cv2.morphologyEx(
                motion_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
            ) > 0

        mask = motion_mask.astype(np.uint8) * 255
        boxes = self._find_motion_boxes(mask, luma_map, chroma_map, current_lab)
        boxes = self._update_persistence(boxes)
        self._remember(current_gray, current_lab)
        return MotionResult(
            (left, top, width, height), mask, boxes, shift, inliers, compensated,
            status, int(np.count_nonzero(luma_mask)), int(np.count_nonzero(chroma_mask)),
        )

    def _estimate_background_motion(
        self, previous_gray: np.ndarray, current_gray: np.ndarray
    ) -> tuple[np.ndarray, tuple[float, float] | None, int, bool, str]:
        """Оценивает общий перенос камеры по устойчивым фоновым точкам RANSAC."""
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=int(self.settings["max_features"]),
            qualityLevel=float(self.settings["feature_quality"]),
            minDistance=float(self.settings["min_feature_distance"]),
            blockSize=3,
        )
        minimum_points = int(self.settings["min_background_features"])
        if points is None or len(points) < minimum_points:
            return identity, None, 0, False, "Мало точек фона: показано нескомпенсированное движение"

        moved_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None)
        if moved_points is None or status is None:
            return identity, None, 0, False, "Поток фона не найден: показано нескомпенсированное движение"
        valid = status.reshape(-1) == 1
        if int(np.count_nonzero(valid)) < minimum_points:
            return identity, None, 0, False, "Мало прослеженных точек: показано нескомпенсированное движение"

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            points.reshape(-1, 2)[valid],
            moved_points.reshape(-1, 2)[valid],
            method=cv2.RANSAC,
            ransacReprojThreshold=float(self.settings["ransac_threshold_px"]),
            maxIters=200,
            confidence=0.99,
            refineIters=10,
        )
        inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
        if affine is None or inliers < int(self.settings["min_background_inliers"]):
            return identity, None, inliers, False, "Фон нестабилен: показано нескомпенсированное движение"

        scale = float(np.hypot(affine[0, 0], affine[1, 0]))
        shift_x, shift_y = float(affine[0, 2]), float(affine[1, 2])
        shift = (shift_x, shift_y)
        if (
            not np.isfinite(affine).all()
            or not 0.7 <= scale <= 1.3
            or np.hypot(shift_x, shift_y) > float(self.settings["max_background_shift_px"])
        ):
            return identity, shift, inliers, False, "Скачок камеры вне диапазона: компенсация пропущена"

        return affine.astype(np.float32), shift, inliers, True, "Фон скомпенсирован"

    def _specular_highlights(self, crop: np.ndarray) -> np.ndarray:
        """Находит яркие малонасыщенные пиксели по настраиваемым порогам."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return (
            (hsv[:, :, 2] >= int(self.settings["specular_value_min"]))
            & (hsv[:, :, 1] <= int(self.settings["specular_saturation_max"]))
        )

    def _find_motion_boxes(
        self,
        mask: np.ndarray,
        luma_map: np.ndarray,
        chroma_map: np.ndarray,
        current_lab: np.ndarray,
    ) -> tuple[MotionBox, ...]:
        """Формирует рамки движущихся областей и измеряет их цветовой контраст."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crop_height, crop_width = mask.shape
        max_area = crop_width * crop_height * float(self.settings["max_blob_area_percent"]) / 100
        minimum_area = int(self.settings["min_blob_area"])
        boxes: list[MotionBox] = []

        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box_area = width * height
            if box_area < minimum_area or box_area > max_area:
                continue

            component = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
            component_mask = component > 0
            changed_pixels = int(np.count_nonzero(component_mask & (mask > 0)))
            if changed_pixels < minimum_area:
                continue

            expanded_x1 = max(0, x - width)
            expanded_y1 = max(0, y - height)
            expanded_x2 = min(crop_width, x + width * 2)
            expanded_y2 = min(crop_height, y + height * 2)
            ring = np.zeros(mask.shape, dtype=bool)
            ring[expanded_y1:expanded_y2, expanded_x1:expanded_x2] = True
            ring &= ~component_mask
            if np.any(ring):
                object_lab = np.median(current_lab[component_mask].astype(np.float32), axis=0)
                background_lab = np.median(current_lab[ring].astype(np.float32), axis=0)
                local_color_contrast = float(np.linalg.norm(object_lab - background_lab))
            else:
                local_color_contrast = 0.0

            contrast_threshold = float(self.settings["min_local_color_contrast"])
            if contrast_threshold > 0 and local_color_contrast < contrast_threshold:
                continue

            box_luma = luma_map[y : y + height, x : x + width][component_mask[y : y + height, x : x + width]]
            box_chroma = chroma_map[y : y + height, x : x + width][component_mask[y : y + height, x : x + width]]
            boxes.append(
                MotionBox(
                    x, y, x + width, y + height, changed_pixels,
                    float(np.mean(box_luma)) if box_luma.size else 0.0,
                    float(np.mean(box_chroma)) if box_chroma.size else 0.0,
                    local_color_contrast,
                )
            )

        boxes.sort(key=lambda box: box.changed_pixels, reverse=True)
        return tuple(boxes[: int(self.settings["max_candidates"])])

    def _update_persistence(self, boxes: tuple[MotionBox, ...]) -> tuple[MotionBox, ...]:
        """Отсекает одиночные мерцания, сопоставляя рамки с предыдущим кадром."""
        unmatched = set(range(len(self._tracks)))
        current_tracks: list[MotionBox] = []
        threshold = float(self.settings["track_iou_percent"]) / 100

        for box in boxes:
            best_index: int | None = None
            best_overlap = threshold
            for index in sorted(unmatched):
                overlap = box_iou(box, self._tracks[index])
                if overlap >= best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index is None:
                current_tracks.append(replace(box, frames_seen=1))
            else:
                previous = self._tracks[best_index]
                unmatched.remove(best_index)
                current_tracks.append(replace(box, frames_seen=previous.frames_seen + 1))

        self._tracks = current_tracks
        minimum_frames = int(self.settings["persistence_frames"])
        return tuple(box for box in current_tracks if box.frames_seen >= minimum_frames)

    def _remember(self, gray: np.ndarray, lab: np.ndarray) -> None:
        """Сохраняет только один предыдущий кроп без накопления очереди кадров."""
        self._previous_gray = gray.copy()
        self._previous_lab = lab.copy()
