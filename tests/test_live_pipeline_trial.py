"""Проверяет профиль и раннюю валидацию V2 без открытия камеры."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.live_pipeline_trial import (
    _camera_source,
    compose_frame,
    load_profile,
    prepare_analysis_frame,
    validate_profile,
)


class LivePipelineTrialTests(unittest.TestCase):
    """Проверяет отдельные настройки V2 и независимость от старого two_stage."""

    def test_v2_loads_pipeline_settings_and_does_not_enable_legacy_two_stage(self) -> None:
        """Профиль V2 использует pipeline_v2 и игнорирует старую экспериментальную схему."""
        project_root = Path(__file__).resolve().parents[1]
        profile = load_profile("sky")

        self.assertEqual(profile.model, project_root / "models/anti_uav_fast_320_int8_pc.onnx")
        self.assertEqual(profile.source, "/dev/v4l/by-id/usb-UltraSemi_USB3_Video_73908951-video-index0")
        self.assertEqual(profile.pipeline["display_fps"], 60)
        self.assertEqual(profile.threads, 8)
        self.assertNotIn("display_tracker", profile.pipeline)
        self.assertNotIn("show_analysis_inset", profile.pipeline)
        validate_profile(profile)

    def test_v2_uses_defaults_when_profile_has_no_v2_section(self) -> None:
        """Отсутствующий раздел V2 получает безопасные начальные значения."""
        document = {
            "profiles": {
                "test": {
                    "settings": {"source": "0", "model": "model.onnx"},
                    "two_stage": {"enabled": True, "roi_width_percent": 1},
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            profile = load_profile("test", path)

        self.assertEqual(profile.source, 0)
        self.assertEqual(profile.pipeline["submission_interval_frames"], 1)
        self.assertEqual(profile.pipeline["display_fps"], 60)
        self.assertEqual(profile.threads, 2)
        validate_profile(profile)

    def test_v2_rejects_invalid_pipeline_values_before_camera_open(self) -> None:
        """Ошибочные настройки потока отклоняются на этапе чтения профиля."""
        profile = load_profile("sky")
        profile.pipeline["submission_interval_frames"] = 0

        with self.assertRaisesRegex(ValueError, "submission_interval_frames"):
            validate_profile(profile)

    def test_v2_rejects_invalid_display_rate_before_camera_open(self) -> None:
        """Непригодная частота окна отклоняется при чтении настроек."""
        profile = load_profile("sky")
        profile.pipeline["display_fps"] = 0

        with self.assertRaisesRegex(ValueError, "pipeline_v2.display_fps"):
            validate_profile(profile)

    def test_single_window_uses_live_frame_before_first_yolo_result(self) -> None:
        """До первого результата окно показывает живой кадр камеры."""
        profile = load_profile("sky")
        current_bgr = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
        packet = SimpleNamespace(sequence=123, bgr=current_bgr)

        canvas = compose_frame(packet, profile, "TEMP 40C CPU 25%", None)
        expected_height = profile.display["max_frame_height"] + profile.status_panel["height"]
        self.assertEqual(canvas.shape[:2], (expected_height, profile.display["window_width"]))
        np.testing.assert_array_equal(canvas[540, 740], np.array([10, 20, 30], dtype=np.uint8))

    def test_yolo_source_frame_becomes_the_only_main_window(self) -> None:
        """Главное окно показывает ровно исходный кадр с рамкой YOLO."""
        profile = load_profile("sky")
        analyzed_bgr = np.full((480, 640, 3), (70, 80, 90), dtype=np.uint8)
        current_bgr = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
        detection = SimpleNamespace(x1=40, y1=50, x2=100, y2=120, name="drone", confidence=0.8)
        result = SimpleNamespace(
            source=SimpleNamespace(bgr=analyzed_bgr),
            detections=(detection,),
            source_sequence=12,
            raw_confidence=0.8,
            inference_ms=500.0,
            source_captured_at_ns=0,
        )

        with patch("scripts.live_pipeline_trial.draw_overlay", side_effect=lambda frame, *_args: frame.copy()) as draw:
            analyzed_frame = prepare_analysis_frame(result, profile)

        self.assertIs(draw.call_args.args[0], analyzed_bgr)
        self.assertEqual(draw.call_args.args[1], (detection,))
        packet = SimpleNamespace(sequence=99, bgr=current_bgr)
        canvas = compose_frame(packet, profile, "TEMP 40C CPU 25%", result, analyzed_frame)
        np.testing.assert_array_equal(
            canvas[540, 740],
            np.array([70, 80, 90], dtype=np.uint8),
        )

    def test_linux_by_id_camera_alias_resolves_to_video_node(self) -> None:
        """OpenCV получает конечный video-узел вместо ссылки /dev/v4l/by-id."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "video4"
            node.touch()
            alias_dir = root / "by-id"
            alias_dir.mkdir()
            alias = alias_dir / "usb-capture-video-index0"
            alias.symlink_to(node)

            with patch("scripts.live_pipeline_trial.sys.platform", "linux"):
                self.assertEqual(_camera_source(str(alias)), str(node))


if __name__ == "__main__":
    unittest.main()
