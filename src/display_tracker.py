"""Лёгкое межкадровое сопровождение рамок только для плавного вывода.

Трекер не распознаёт класс и не заменяет YOLO. Он переносит подтверждённую
YOLO рамку на новые кадры, а при потере признаков прекращает её показывать.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from src.realtime import Detection


@dataclass
class _TrackedBox:
    """Внутреннее состояние одной рамки в координатах уменьшенного кадра."""

    detection: Detection
    box: np.ndarray
    points: np.ndarray | None
    age_frames: int = 0


class OpticalFlowBoxTracker:
    """Переносит рамки YOLO по свежим кадрам методом Lucas–Kanade."""

    def __init__(
        self,
        scale: float,
        max_age_frames: int = 90,
        max_features: int = 40,
        quality_level: float = 0.01,
        min_distance: float = 2.0,
        min_points: int = 3,
        refresh_interval: int = 5,
    ) -> None:
        """Проверяет настройки и готовит пустое состояние трекера."""
        if not 0.1 <= scale <= 1.0:
            raise ValueError("scale должен быть от 0.1 до 1.0")
        if max_age_frames < 1 or max_features < 3 or min_points < 3:
            raise ValueError("параметры трекера слишком малы")
        if not 0.0 < quality_level <= 1.0 or min_distance < 0 or refresh_interval < 1:
            raise ValueError("некорректные параметры поиска признаков")
        self._scale = scale
        self._max_age_frames = max_age_frames
        self._max_features = max_features
        self._quality_level = quality_level
        self._min_distance = min_distance
        self._min_points = min_points
        self._refresh_interval = refresh_interval
        self._previous_gray: np.ndarray | None = None
        self._sequence = 0
        self._boxes: list[_TrackedBox] = []

    @property
    def sequence(self) -> int:
        """Возвращает номер последнего кадра, обработанного трекером."""
        return self._sequence

    def reset(
        self,
        gray: np.ndarray,
        sequence: int,
        detections: Iterable[Detection],
    ) -> None:
        """Создаёт треки из рамок нового результата YOLO на его исходном кадре."""
        if gray.ndim != 2 or sequence < 1:
            raise ValueError("ожидается серый кадр и положительный номер кадра")
        self._previous_gray = gray
        self._sequence = sequence
        self._boxes = []
        for detection in detections:
            box = np.array(
                [
                    detection.x1 * self._scale,
                    detection.y1 * self._scale,
                    detection.x2 * self._scale,
                    detection.y2 * self._scale,
                ],
                dtype=np.float32,
            )
            points = self._find_points(gray, box)
            self._boxes.append(_TrackedBox(detection, box, points))

    def advance(self, gray: np.ndarray, sequence: int) -> tuple[Detection, ...]:
        """Сопровождает все рамки ровно на следующий видеокадр."""
        if gray.ndim != 2:
            raise ValueError("для трекера нужен одноканальный серый кадр")
        if self._previous_gray is None or sequence != self._sequence + 1:
            self._boxes = []
            self._previous_gray = gray
            self._sequence = sequence
            return ()

        current_gray = gray
        kept: list[_TrackedBox] = []
        for track in self._boxes:
            moved = self._move_box(track, current_gray)
            if moved and track.age_frames < self._max_age_frames:
                kept.append(track)
        self._boxes = kept
        self._previous_gray = current_gray
        self._sequence = sequence
        return tuple(self._to_detection(track) for track in kept)

    def current(self) -> tuple[Detection, ...]:
        """Возвращает рамки текущего кадра в исходном разрешении камеры."""
        return tuple(self._to_detection(track) for track in self._boxes)

    def _move_box(self, track: _TrackedBox, current_gray: np.ndarray) -> bool:
        """Сдвигает рамку по устойчивому медианному перемещению точек объекта."""
        previous = self._previous_gray
        points = track.points
        if previous is None or points is None or len(points) < self._min_points:
            return False
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            current_gray,
            points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if next_points is None or status is None:
            return False
        valid = status.reshape(-1).astype(bool)
        old_good = points.reshape(-1, 2)[valid]
        new_good = next_points.reshape(-1, 2)[valid]
        if len(new_good) < self._min_points:
            return False
        flow = new_good - old_good
        dx, dy = np.median(flow, axis=0)
        deviations = np.linalg.norm(flow - np.array([dx, dy], dtype=np.float32), axis=1)
        inliers = deviations <= max(2.0, float(np.median(deviations)) * 2.5)
        if int(np.count_nonzero(inliers)) < self._min_points:
            return False
        dx, dy = np.median(flow[inliers], axis=0)
        track.box += np.array([dx, dy, dx, dy], dtype=np.float32)
        track.age_frames += 1
        if track.age_frames % self._refresh_interval == 0:
            track.points = self._find_points(current_gray, track.box)
        else:
            track.points = new_good[inliers].reshape(-1, 1, 2).astype(np.float32)
        if track.points is None or len(track.points) < self._min_points:
            return False
        height, width = current_gray.shape[:2]
        return (
            track.box[2] > track.box[0]
            and track.box[3] > track.box[1]
            and track.box[2] > 0
            and track.box[3] > 0
            and track.box[0] < width
            and track.box[1] < height
        )

    def _find_points(self, gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        """Находит угловые признаки внутри рамки, не включая её внешний фон."""
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        inset_x = min((x2 - x1) // 6, 4)
        inset_y = min((y2 - y1) // 6, 4)
        cv2.rectangle(mask, (x1 + inset_x, y1 + inset_y), (x2 - 1 - inset_x, y2 - 1 - inset_y), 255, -1)
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self._max_features,
            qualityLevel=self._quality_level,
            minDistance=self._min_distance,
            mask=mask,
            blockSize=3,
        )
        if points is None or len(points) < self._min_points:
            return None
        return points.astype(np.float32)

    def _to_detection(self, track: _TrackedBox) -> Detection:
        """Преобразует текущую уменьшенную рамку обратно в координаты видео."""
        x1, y1, x2, y2 = np.rint(track.box / self._scale).astype(int)
        return Detection(x1, y1, x2, y2, track.detection.name, track.detection.confidence)
