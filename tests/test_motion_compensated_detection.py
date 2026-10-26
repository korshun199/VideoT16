"""Проверки ROI, компенсации общего движения и цветового остатка."""

import unittest

import cv2
import numpy as np

from src.motion_compensated_detection import MotionAnalyzer, center_roi_bounds


def trial_settings() -> tuple[dict, dict]:
    """Возвращает небольшие детерминированные настройки для искусственных кадров."""
    roi = {
        "width_percent": 50,
        "height_percent": 50,
        "center_x_percent": 50,
        "center_y_percent": 50,
    }
    motion = {
        "max_features": 500,
        "feature_quality": 0.01,
        "min_feature_distance": 4,
        "min_background_features": 8,
        "min_background_inliers": 6,
        "ransac_threshold_px": 2.5,
        "max_background_shift_px": 50,
        "luma_threshold": 18,
        "chroma_threshold": 22,
        "min_blob_area": 4,
        "max_blob_area_percent": 30,
        "morphology_radius": 0,
        "suppress_specular": False,
        "specular_value_min": 245,
        "specular_saturation_max": 35,
        "persistence_frames": 1,
        "track_iou_percent": 10,
        "max_candidates": 40,
        "min_local_color_contrast": 0,
    }
    return roi, motion


def textured_background(seed: int = 17) -> np.ndarray:
    """Создаёт повторяемый текстурный фон для проверки движения камеры."""
    generator = np.random.default_rng(seed)
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    for _ in range(180):
        x = int(generator.integers(0, frame.shape[1]))
        y = int(generator.integers(0, frame.shape[0]))
        radius = int(generator.integers(1, 4))
        level = int(generator.integers(55, 200))
        cv2.circle(frame, (x, y), radius, (level, level, level), -1)
    return frame


class MotionCompensatedDetectionTests(unittest.TestCase):
    """Проверяет наблюдаемое поведение алгоритма без камеры и модели."""

    def test_center_roi_uses_quarter_of_frame_area(self) -> None:
        """Центральная область 50x50 процентов занимает четверть изображения."""
        self.assertEqual(center_roi_bounds((480, 640, 3), trial_settings()[0]), (160, 120, 320, 240))

    def test_dark_moving_object_remains_after_background_camera_shift(self) -> None:
        """RANSAC убирает сдвиг фона и оставляет тёмный объект в маске движения."""
        roi, motion = trial_settings()
        background = textured_background()
        first = background.copy()
        cv2.rectangle(first, (135, 105), (147, 113), (5, 5, 5), -1)

        camera_shift = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        second = cv2.warpAffine(background, camera_shift, (background.shape[1], background.shape[0]))
        object_x = 135 + 3 + 5
        object_y = 105 + 2 - 1
        cv2.rectangle(second, (object_x, object_y), (object_x + 12, object_y + 8), (5, 5, 5), -1)

        analyzer = MotionAnalyzer(roi, motion)
        analyzer.process(first)
        result = analyzer.process(second)

        self.assertTrue(result.compensated, result.status)
        self.assertIsNotNone(result.background_shift)
        self.assertAlmostEqual(result.background_shift[0], 3.0, delta=1.0)
        self.assertAlmostEqual(result.background_shift[1], 2.0, delta=1.0)
        local_x = object_x - result.roi[0]
        local_y = object_y - result.roi[1]
        self.assertTrue(
            any(box.x1 <= local_x + 4 <= box.x2 and box.y1 <= local_y + 3 <= box.y2 for box in result.boxes),
            f"движущийся объект не попал в кандидаты: {result.boxes}",
        )

    def test_static_background_shift_does_not_create_candidate_boxes(self) -> None:
        """Один только перенос камеры с текстурным фоном не выдаётся за объект."""
        roi, motion = trial_settings()
        background = textured_background()
        camera_shift = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, -3.0]], dtype=np.float32)
        shifted = cv2.warpAffine(background, camera_shift, (background.shape[1], background.shape[0]))
        analyzer = MotionAnalyzer(roi, motion)
        analyzer.process(background)

        result = analyzer.process(shifted)

        self.assertTrue(result.compensated, result.status)
        self.assertEqual(result.boxes, ())

    def test_global_exposure_change_is_removed_by_robust_photometric_offset(self) -> None:
        """Общее увеличение яркости не превращает весь кадр в движущийся объект."""
        roi, motion = trial_settings()
        background = textured_background()
        brighter = np.clip(background.astype(np.int16) + 20, 0, 255).astype(np.uint8)
        analyzer = MotionAnalyzer(roi, motion)
        analyzer.process(background)
        result = analyzer.process(brighter)

        self.assertLess(result.luma_pixels, 20)
        self.assertEqual(result.boxes, ())

    def test_large_residual_is_clipped_instead_of_wrapping_as_uint8(self) -> None:
        """Резкая локальная перемена на фоне общей вспышки не теряется из-за переполнения."""
        roi, motion = trial_settings()
        motion["luma_threshold"] = 255
        motion["chroma_threshold"] = 255
        first = np.full((240, 320, 3), 255, dtype=np.uint8)
        second = np.zeros_like(first)
        cv2.rectangle(first, (145, 105), (157, 113), (0, 0, 0), -1)
        cv2.rectangle(second, (145, 105), (157, 113), (255, 255, 255), -1)

        analyzer = MotionAnalyzer(roi, motion)
        analyzer.process(first)
        result = analyzer.process(second)

        self.assertGreater(result.luma_pixels, 0)
        self.assertTrue(result.boxes)

    def test_chroma_change_can_trigger_candidate_without_large_luma_change(self) -> None:
        """Цветовое отличие обнаруживается отдельно от порога яркости."""
        roi, motion = trial_settings()
        motion["luma_threshold"] = 80
        motion["chroma_threshold"] = 12
        background = textured_background()
        first = background.copy()
        second = background.copy()
        cv2.rectangle(first, (135, 105), (147, 113), (125, 125, 125), -1)
        cv2.rectangle(second, (135, 105), (147, 113), (200, 115, 115), -1)

        analyzer = MotionAnalyzer(roi, motion)
        analyzer.process(first)
        result = analyzer.process(second)

        self.assertGreater(result.chroma_pixels, 0)
        self.assertTrue(result.boxes)


if __name__ == "__main__":
    unittest.main()
