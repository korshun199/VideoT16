"""Проверяет профиль и раннюю валидацию V2 без открытия камеры."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts.live_pipeline_trial import (
    CameraCaptureWorker,
    DelayedOverlaySelector,
    _camera_source,
    compose_frame,
    load_profile,
    validate_profile,
)
from src.live_pipeline import DetectionSnapshot, FrameHub
from src.realtime import Detection


class LivePipelineTrialTests(unittest.TestCase):
    """Проверяет отдельные настройки V2 и независимость от старого two_stage."""

    def test_v2_loads_pipeline_settings_and_does_not_enable_legacy_two_stage(self) -> None:
        """Профиль V2 использует pipeline_v2 и игнорирует старую экспериментальную схему."""
        project_root = Path(__file__).resolve().parents[1]
        profile = load_profile("sky")

        self.assertEqual(profile.model, project_root / "models/anti_uav_fast_320_int8_pc.onnx")
        self.assertEqual(profile.source, "/dev/v4l/by-id/usb-UltraSemi_USB3_Video_73908951-video-index0")
        self.assertEqual(profile.pipeline["display_fps"], 60)
        self.assertEqual(profile.pipeline["output_delay_ms"], 1200)
        self.assertEqual(profile.pipeline["output_buffer_max_mb"], 384)
        self.assertEqual(profile.pipeline["box_hold_ms"], 1000)
        self.assertEqual(profile.threads, 8)
        self.assertNotIn("tracking_scale", profile.pipeline)
        self.assertNotIn("tracking_max_age_frames", profile.pipeline)
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

    def test_v2_rejects_output_buffer_smaller_than_delay_requires(self) -> None:
        """Настройка не запускает видео, если RAM-буфера не хватит на задержку."""
        profile = load_profile("sky")
        profile.pipeline["output_buffer_max_mb"] = 64

        with self.assertRaisesRegex(ValueError, "нужно не менее"):
            validate_profile(profile)

    def test_single_window_uses_live_frame_before_first_yolo_result(self) -> None:
        """До первого результата окно показывает кадр из задержанного буфера без рамок."""
        profile = load_profile("sky")
        current_bgr = np.full((480, 640, 3), (10, 20, 30), dtype=np.uint8)
        packet = SimpleNamespace(sequence=123, bgr=current_bgr)

        canvas = compose_frame(packet, profile, "TEMP 40C CPU 25%", None)
        expected_height = profile.display["max_frame_height"] + profile.status_panel["height"]
        self.assertEqual(canvas.shape[:2], (expected_height, profile.display["window_width"]))
        np.testing.assert_array_equal(canvas[540, 740], np.array([10, 20, 30], dtype=np.uint8))

    def test_live_capture_frame_uses_latest_yolo_boxes(self) -> None:
        """Плавный вывод берёт текущий кадр, а координаты рамок — из последнего ответа YOLO."""
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

        packet = SimpleNamespace(sequence=99, bgr=current_bgr)
        with patch("scripts.live_pipeline_trial.draw_overlay", side_effect=lambda frame, *_args: frame.copy()) as draw:
            canvas = compose_frame(packet, profile, "TEMP 40C CPU 25%", result)

        self.assertIs(draw.call_args.args[0], current_bgr)
        self.assertEqual(draw.call_args.args[1], (detection,))
        np.testing.assert_array_equal(
            canvas[540, 740],
            np.array([10, 20, 30], dtype=np.uint8),
        )

    def test_yolo_box_is_held_then_expires_without_optical_tracker(self) -> None:
        """Рамка не теряется трекером, удерживается и затем гаснет по возрасту."""
        hub = FrameHub(output_buffer_max_bytes=1_000_000)
        first = hub.publish(np.zeros((8, 8, 3), dtype=np.uint8), None, captured_at_ns=1_000_000_000)
        second = hub.publish(np.zeros((8, 8, 3), dtype=np.uint8), None, captured_at_ns=1_050_000_000)
        expired = hub.publish(np.zeros((8, 8, 3), dtype=np.uint8), None, captured_at_ns=1_150_000_001)
        detection = Detection(1, 2, 5, 6, "copter_drone", 0.84)

        def snapshot(packet, detections):
            """Создаёт детерминированный результат YOLO с ID исходного кадра."""
            return DetectionSnapshot(
                source_sequence=packet.sequence,
                source_captured_at_ns=packet.captured_at_ns,
                submitted_at_ns=packet.captured_at_ns,
                started_at_ns=packet.captured_at_ns,
                finished_at_ns=packet.captured_at_ns + 1_000_000,
                mode="full_scan",
                detections=detections,
                raw_confidence=max((item.confidence for item in detections), default=0.0),
            )

        selector = DelayedOverlaySelector(100)
        first_selection = selector.select(first, snapshot(first, (detection,)), 0.4, 0.1)
        held_selection = selector.select(second, snapshot(second, ()), 0.4, 0.1)
        expired_selection = selector.select(expired, None, 0.4, 0.1)

        self.assertEqual(first_selection.mode, "YOLO")
        self.assertEqual(first_selection.source_sequence, first.sequence)
        self.assertEqual(held_selection.mode, "HOLD")
        self.assertEqual(held_selection.source_sequence, first.sequence)
        self.assertEqual(held_selection.detections, (detection,))
        self.assertEqual(expired_selection.mode, "EXPIRED")
        self.assertEqual(expired_selection.detections, ())

    def test_capture_stage_fills_frame_and_inference_buffers_together(self) -> None:
        """Первый из трёх рабочих потоков пишет кадр в оба буфера с одним ID."""
        hub = FrameHub(output_buffer_max_bytes=1_000_000)
        submitted = threading.Event()
        requests = []

        class RequestRecorder:
            """Запоминает запрос вместо реального запуска ONNX-модели."""

            def submit(self, request):
                requests.append(request)
                submitted.set()

        class OneFrameCapture:
            """Отдаёт один искусственный кадр и ждёт закрытия устройства."""

            def __init__(self):
                self.released = threading.Event()
                self.sent = False

            def read(self):
                if not self.sent:
                    self.sent = True
                    return True, np.zeros((8, 8, 3), dtype=np.uint8)
                self.released.wait(timeout=1.0)
                return False, None

            def release(self):
                self.released.set()

        enabled = threading.Event()
        enabled.set()
        capture = OneFrameCapture()
        camera = CameraCaptureWorker(capture, hub, RequestRecorder(), 1, enabled)
        try:
            self.assertTrue(submitted.wait(timeout=1.0))
            packet = hub.latest()
            self.assertIsNotNone(packet)
            assert packet is not None
        finally:
            camera.close()

        self.assertEqual(requests[0].source.sequence, packet.sequence)
        self.assertIs(requests[0].image, packet.bgr)
        self.assertEqual(hub.buffer_status()["newest_output_sequence"], packet.sequence)
        self.assertEqual(camera.statistics()["submitted_sequence"], packet.sequence)

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
