"""Проверяет конфигурацию и отрисовку прототипа без открытия камеры."""

import copy
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.motion_trial import CONFIG_PATH, draw_frame, load_config, resolve_camera, validate_config
from src.motion_compensated_detection import MotionBox, MotionResult
from src.realtime import Detection, InferenceSnapshot


class MotionTrialIntegrationTests(unittest.TestCase):
    """Проверяет безопасные входы и демонстрационную отрисовку на памяти."""

    def test_prototype_json_has_valid_editable_parameters(self) -> None:
        """Читает отдельный JSON прототипа и принимает его текущие стартовые значения."""
        settings = load_config(CONFIG_PATH)
        self.assertEqual(settings["roi"]["width_percent"], 50)
        self.assertEqual(settings["roi"]["height_percent"], 50)
        self.assertEqual(settings["detector"]["threads"], 0)

    def test_invalid_background_point_limits_fail_before_camera_access(self) -> None:
        """Выявляет противоречивые значения RANSAC до запуска видеоустройства."""
        settings = load_config(CONFIG_PATH)
        settings = copy.deepcopy(settings)
        settings["motion"]["min_background_features"] = settings["motion"]["max_features"] + 1
        with self.assertRaisesRegex(ValueError, "min_background_features"):
            validate_config(settings)

    def test_v4l2_alias_is_converted_to_numeric_opencv_source(self) -> None:
        """Символический alias USB-камеры превращается в индекс, не открывая устройство."""
        with patch("scripts.motion_trial.Path.resolve", return_value=Path("/dev/video4")):
            self.assertEqual(resolve_camera("/dev/v4l/by-id/usb-example-video-index0"), 4)

    def test_draw_frame_keeps_motion_and_yolo_candidates_distinct(self) -> None:
        """Рисует оба вида рамок на синтетическом кадре без камеры и инференса."""
        settings = load_config(CONFIG_PATH)
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        mask[40:48, 50:60] = 255
        result = MotionResult(
            roi=(80, 60, 160, 120),
            mask=mask,
            boxes=(MotionBox(50, 40, 60, 48, 80, 22.0, 3.0, 16.0),),
            background_shift=(1.0, 0.5),
            background_inliers=24,
            compensated=True,
            status="Фон скомпенсирован",
            luma_pixels=80,
            chroma_pixels=0,
        )
        snapshot = InferenceSnapshot(
            sequence=1,
            completed_at=time.monotonic(),
            detections=(Detection(48, 38, 64, 50, "FPV-DRON", 0.78),),
        )

        output = draw_frame(frame, result, snapshot, settings, "TEMP 45.0C CPU 20%", 0.78)

        self.assertEqual(output.shape, frame.shape)
        self.assertFalse(np.array_equal(output, frame))


if __name__ == "__main__":
    unittest.main()
