"""Лёгкое сопровождение последнего обнаруженного объекта."""

from __future__ import annotations

from dataclasses import replace

from src.realtime import Detection


class DetectionTracker:
    """Удерживает и плавно перемещает рамку между результатами YOLO."""

    def __init__(self, max_missing: int = 8) -> None:
        """Создаёт трекер с числом кадров терпения при пропадании объекта."""
        self.max_missing = max_missing
        self._last: Detection | None = None
        self._velocity = (0.0, 0.0)
        self._missing = 0

    @staticmethod
    def _center(detection: Detection) -> tuple[float, float]:
        """Возвращает центр рамки."""
        return (detection.x1 + detection.x2) / 2, (detection.y1 + detection.y2) / 2

    @staticmethod
    def _iou(first: Detection, second: Detection) -> float:
        """Считает пересечение двух рамок."""
        x1 = max(first.x1, second.x1)
        y1 = max(first.y1, second.y1)
        x2 = min(first.x2, second.x2)
        y2 = min(first.y2, second.y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        first_area = max(1, first.x2 - first.x1) * max(1, first.y2 - first.y1)
        second_area = max(1, second.x2 - second.x1) * max(1, second.y2 - second.y1)
        return area / (first_area + second_area - area)

    def update(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        """Выбирает продолжение трека; при пропадании объекта сопровождение прекращается."""
        if not detections:
            # Не прогнозируем координаты без подтверждённой детекции:
            # свободный прогноз мог выйти за границы сетки штатного OSD.
            self._last = None
            self._velocity = (0.0, 0.0)
            self._missing = 0
            return ()

        if self._last is None:
            self._last = max(detections, key=lambda item: item.confidence)
            self._missing = 0
            return (self._last,)

        previous_center = self._center(self._last)
        candidate = max(
            detections,
            key=lambda item: self._iou(self._last, item) * 2 + item.confidence,
        )
        candidate_center = self._center(candidate)
        width = max(1, self._last.x2 - self._last.x1)
        height = max(1, self._last.y2 - self._last.y1)
        distance = abs(candidate_center[0] - previous_center[0]) / width
        distance += abs(candidate_center[1] - previous_center[1]) / height
        if self._iou(self._last, candidate) < 0.01 and distance > 3.0:
            return self.update(())

        self._velocity = (
            (candidate_center[0] - previous_center[0]) * 0.5 + self._velocity[0] * 0.5,
            (candidate_center[1] - previous_center[1]) * 0.5 + self._velocity[1] * 0.5,
        )
        self._last = candidate
        self._missing = 0
        return (candidate,)
