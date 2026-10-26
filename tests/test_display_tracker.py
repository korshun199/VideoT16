"""Проверяет межкадровый перенос рамки на синтетическом движении."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.display_tracker import OpticalFlowBoxTracker
from src.realtime import Detection


class OpticalFlowBoxTrackerTests(unittest.TestCase):
    """Проверяет координаты сопровождения и сброс при пропущенном кадре."""

    @staticmethod
    def _translated_frame(shift_x: int) -> np.ndarray:
        """Создаёт серый кадр с контрастной текстурированной целью."""
        image = np.zeros((100, 160), dtype=np.uint8)
        target = np.random.default_rng(71).integers(30, 240, (30, 30), dtype=np.uint8)
        x = 40 + shift_x
        image[35:65, x : x + 30] = target
        cv2.rectangle(image, (x, 35), (x + 29, 64), 255, 1)
        return image

    def test_box_moves_with_target_between_detection_frames(self) -> None:
        """Рамка движется вместе с объектом до следующего результата YOLO."""
        tracker = OpticalFlowBoxTracker(scale=1.0, refresh_interval=4)
        tracker.reset(
            self._translated_frame(0),
            sequence=10,
            detections=(Detection(40, 35, 70, 65, "copter_drone", 0.84),),
        )

        tracked = ()
        for sequence in range(11, 16):
            tracked = tracker.advance(self._translated_frame(sequence - 10), sequence)

        self.assertEqual(len(tracked), 1)
        self.assertGreaterEqual(tracked[0].x1, 43)
        self.assertLessEqual(tracked[0].x1, 46)
        self.assertEqual(tracked[0].name, "copter_drone")
        self.assertEqual(tracked[0].confidence, 0.84)

    def test_skipped_video_frame_drops_track_instead_of_guessing(self) -> None:
        """При разрыве кадров трекер сбрасывает рамку, а не переносит её вслепую."""
        tracker = OpticalFlowBoxTracker(scale=1.0)
        tracker.reset(
            self._translated_frame(0),
            sequence=10,
            detections=(Detection(40, 35, 70, 65, "copter_drone", 0.84),),
        )

        self.assertEqual(tracker.advance(self._translated_frame(2), sequence=12), ())
        self.assertEqual(tracker.current(), ())

    def test_track_expires_after_configured_frame_age(self) -> None:
        """Рамка исчезает по ограничению возраста даже при сохраняющихся признаках."""
        tracker = OpticalFlowBoxTracker(scale=1.0, max_age_frames=2)
        tracker.reset(
            self._translated_frame(0),
            sequence=10,
            detections=(Detection(40, 35, 70, 65, "copter_drone", 0.84),),
        )

        self.assertEqual(len(tracker.advance(self._translated_frame(1), sequence=11)), 1)
        self.assertEqual(tracker.advance(self._translated_frame(2), sequence=12), ())


if __name__ == "__main__":
    unittest.main()
