"""Проверяет профиль и раннюю валидацию V2 без открытия камеры."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.live_pipeline_trial import (
    CameraCaptureWorker,
    DelayedOverlaySelector,
    FastScanPlanner,
    _camera_source,
    configure_window,
    compose_frame,
    configure_capture,
    load_profile,
    make_logger,
    validate_profile,
)
from src.live_pipeline import DetectionSnapshot, FrameHub
from src.realtime import Detection


class LivePipelineTrialTests(unittest.TestCase):
    """Проверяет основной профиль и настройки настольного V2-конвейера."""

    def test_v2_loads_saved_profile_and_pipeline_settings(self) -> None:
        """Рабочий профиль берёт параметры захвата и обработки из settings.json."""
        project_root = Path(__file__).resolve().parents[1]
        profile = load_profile("sky")

        self.assertEqual(profile.model, project_root / "models/anti_uav_fast_320_int8_pc.onnx")
        self.assertEqual(profile.source, "/dev/v4l/by-id/usb-UltraSemi_USB3_Video_73908951-video-index0")
        self.assertEqual(profile.pipeline["display_fps"], 60)
        self.assertEqual(profile.pipeline["output_delay_ms"], 2200)
        self.assertEqual(profile.pipeline["output_buffer_max_mb"], 512)
        self.assertEqual(profile.pipeline["box_hold_ms"], 1000)
        self.assertEqual(profile.pipeline["max_match_distance_fraction"], 0.25)
        self.assertEqual(profile.capture["pixel_format"], "YUYV")
        self.assertTrue(profile.display["fullscreen"])
        self.assertTrue(profile.fast_scan["enabled"])
        self.assertEqual(profile.fast_scan["focus_crop_width"], 640)
        self.assertEqual(profile.fast_scan["focus_lost_after_results"], 3)
        self.assertEqual(profile.threads, 8)
        validate_profile(profile)

    def test_v2_uses_defaults_when_optional_sections_are_missing(self) -> None:
        """Отсутствующий раздел V2 получает безопасные начальные значения."""
        document = {
            "profiles": {
                "test": {
                    "settings": {"source": "0", "model": "model.onnx"},
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
        self.assertEqual(profile.capture["pixel_format"], "YUYV")
        self.assertFalse(profile.fast_scan["enabled"])
        self.assertEqual(profile.threads, 2)
        validate_profile(profile)

    def test_fast_scan_cycles_one_tile_at_a_time_before_candidate(self) -> None:
        """До кандидата быстрый режим не запускает весь набор плиток за один ответ."""
        planner = FastScanPlanner(
            {
                "enabled": True,
                "focus_min_confidence": 0.15,
                "focus_crop_width": 640,
                "focus_crop_height": 640,
                "focus_lost_after_results": 3,
            },
            tile_size=640,
            tile_overlap=0.20,
        )
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)

        crops = [planner.next_crop(frame) for _ in range(5)]

        self.assertTrue(all(crop.mode == "SWEEP" for crop in crops))
        self.assertEqual(len({(crop.x, crop.y) for crop in crops[:4]}), 4)
        self.assertEqual((crops[0].x, crops[0].y), (crops[4].x, crops[4].y))

    def test_fast_scan_focuses_on_confident_candidate_then_returns_to_sweep(self) -> None:
        """Кандидат переводит следующий запрос в вырез, а три промаха сбрасывают вырез."""
        planner = FastScanPlanner(
            {
                "enabled": True,
                "focus_min_confidence": 0.15,
                "focus_crop_width": 640,
                "focus_crop_height": 640,
                "focus_lost_after_results": 3,
            },
            tile_size=640,
            tile_overlap=0.20,
        )
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        target = Detection(900, 200, 940, 240, "copter_drone", 0.76)

        planner.observe((target,))
        crop = planner.next_crop(frame)

        self.assertEqual(crop.mode, "FOCUS")
        self.assertEqual(crop.image.shape[:2], (640, 640))
        self.assertLessEqual(crop.x, 920)
        self.assertLess(920, crop.x + crop.image.shape[1])
        self.assertLessEqual(crop.y, 220)
        self.assertLess(220, crop.y + crop.image.shape[0])

        planner.observe(())
        planner.observe(())
        self.assertEqual(planner.next_crop(frame).mode, "FOCUS")
        planner.observe(())
        self.assertEqual(planner.next_crop(frame).mode, "SWEEP")

    def test_fast_scan_ignores_candidate_below_focus_threshold(self) -> None:
        """Слабая рамка не удерживает вырез и не маскирует полный поиск."""
        planner = FastScanPlanner(
            {
                "enabled": True,
                "focus_min_confidence": 0.50,
                "focus_crop_width": 640,
                "focus_crop_height": 640,
                "focus_lost_after_results": 3,
            },
            tile_size=640,
            tile_overlap=0.20,
        )
        planner.observe((Detection(100, 100, 140, 140, "copter_drone", 0.49),))

        self.assertEqual(planner.next_crop(np.zeros((960, 1280, 3), dtype=np.uint8)).mode, "SWEEP")

    def test_v2_rejects_invalid_pipeline_values_before_camera_open(self) -> None:
        """Ошибочные настройки потока отклоняются на этапе чтения профиля."""
        profile = load_profile("sky")
        profile.pipeline["submission_interval_frames"] = 0

        with self.assertRaisesRegex(ValueError, "submission_interval_frames"):
            validate_profile(profile)

    def test_v2_rejects_invalid_fast_scan_before_camera_open(self) -> None:
        """Ошибка параметра быстрого поиска останавливает запуск до открытия камеры."""
        profile = load_profile("sky")
        profile.fast_scan["focus_lost_after_results"] = 0

        with self.assertRaisesRegex(ValueError, "focus_lost_after_results"):
            validate_profile(profile)

    def test_v2_rejects_invalid_display_rate_before_camera_open(self) -> None:
        """Непригодная частота окна отклоняется при чтении настроек."""
        profile = load_profile("sky")
        profile.pipeline["display_fps"] = 0

        with self.assertRaisesRegex(ValueError, "pipeline_v2.display_fps"):
            validate_profile(profile)

    def test_v2_rejects_non_boolean_fullscreen_before_camera_open(self) -> None:
        """Полноэкранный режим не принимает строковые и числовые значения JSON."""
        profile = load_profile("sky")
        profile.display["fullscreen"] = 1

        with self.assertRaisesRegex(ValueError, "display.fullscreen"):
            validate_profile(profile)

    def test_configure_window_uses_fullscreen_when_profile_requests_it(self) -> None:
        """Рабочий профиль включает полноэкранный флаг окна OpenCV."""
        display = dict(load_profile("sky").display)
        with patch("scripts.live_pipeline_trial.cv2.namedWindow") as named, patch(
            "scripts.live_pipeline_trial.cv2.setWindowProperty"
        ) as fullscreen, patch("scripts.live_pipeline_trial.cv2.resizeWindow") as resize:
            configure_window(display)

        named.assert_called_once_with(
            display["window_title"], cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
        )
        fullscreen.assert_called_once_with(
            display["window_title"], cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        resize.assert_not_called()

    def test_v2_logger_writes_file_without_console_handler(self) -> None:
        """Живой V2-запуск оставляет терминал чистым, сохраняя журнал для разбора."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            logger = make_logger(path)
            try:
                logger.info("строка для файла")
                self.assertFalse(any(type(handler) is logging.StreamHandler for handler in logger.handlers))
            finally:
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                    handler.close()

            self.assertEqual(path.read_text(encoding="utf-8").count("строка для файла"), 1)

    def test_v2_rejects_unknown_capture_pixel_format_before_camera_open(self) -> None:
        """Неподдерживаемый FOURCC не должен доходить до V4L2-устройства."""
        profile = load_profile("sky")
        profile.capture["pixel_format"] = "H264"

        with self.assertRaisesRegex(ValueError, "capture.pixel_format"):
            validate_profile(profile)

    def test_configure_capture_requests_profile_pixel_format_and_reports_driver_value(self) -> None:
        """V2 задаёт MJPEG до разрешения/FPS и сообщает принятый FOURCC."""
        class ConfigurableCapture:
            """Запоминает настройки, которые мог бы принять V4L2-драйвер."""

            def __init__(self) -> None:
                """Создаёт фиктивный вход с возвращаемыми параметрами."""
                self.values = {
                    cv2.CAP_PROP_FRAME_WIDTH: 0.0,
                    cv2.CAP_PROP_FRAME_HEIGHT: 0.0,
                    cv2.CAP_PROP_FPS: 0.0,
                    cv2.CAP_PROP_FOURCC: 0.0,
                }

            def set(self, property_id, value):
                """Принимает одно свойство формата или буфера."""
                self.values[property_id] = float(value)
                return True

            def get(self, property_id):
                """Возвращает последнее заданное значение свойства."""
                return self.values.get(property_id, 0.0)

        capture = ConfigurableCapture()
        actual = configure_capture(
            capture,
            {"pixel_format": "MJPG", "width": 1280, "height": 960, "fps": 60.0, "buffer_size": 1},
            logging.getLogger("videot16.test.capture"),
        )

        self.assertEqual(actual, {"width": 1280, "height": 960, "fps": 60.0, "pixel_format": "MJPG"})

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

    def test_empty_yolo_result_clears_previous_box(self) -> None:
        """Новый результат без уверенных рамок не оставляет устаревшие координаты."""
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
        first_selection = selector.select(first, snapshot(first, (detection,)), None, 0.4, 0.1)
        held_selection = selector.select(second, snapshot(second, ()), None, 0.4, 0.1)
        expired_selection = selector.select(expired, None, None, 0.4, 0.1)

        self.assertEqual(first_selection.mode, "YOLO")
        self.assertEqual(first_selection.source_sequence, first.sequence)
        self.assertEqual(held_selection.mode, "NONE")
        self.assertEqual(held_selection.detections, ())
        self.assertEqual(expired_selection.mode, "NONE")
        self.assertEqual(expired_selection.detections, ())

    def test_interpolates_box_to_displayed_frame_between_yolo_results(self) -> None:
        """Рамка перемещается к ID показываемого кадра между результатами YOLO."""
        hub = FrameHub(output_buffer_max_bytes=1_000_000)
        first = hub.publish(np.zeros((100, 100, 3), dtype=np.uint8), None, captured_at_ns=1_000_000_000)
        middle = hub.publish(np.zeros((100, 100, 3), dtype=np.uint8), None, captured_at_ns=1_050_000_000)
        last = hub.publish(np.zeros((100, 100, 3), dtype=np.uint8), None, captured_at_ns=1_100_000_000)

        def snapshot(packet, detection):
            """Строит результат с кадром-источником для проверки линейного движения."""
            return DetectionSnapshot(
                source_sequence=packet.sequence,
                source_captured_at_ns=packet.captured_at_ns,
                submitted_at_ns=packet.captured_at_ns,
                started_at_ns=packet.captured_at_ns,
                finished_at_ns=packet.captured_at_ns + 1,
                mode="full_scan",
                detections=(detection,),
                raw_confidence=detection.confidence,
            )

        earlier = snapshot(first, Detection(10, 20, 30, 40, "drone", 0.8))
        later = snapshot(last, Detection(30, 40, 50, 60, "drone", 0.9))
        selection = DelayedOverlaySelector(100).select(middle, earlier, later, 0.4, 0.1)

        self.assertEqual(selection.mode, "INTERPOLATED")
        self.assertEqual(selection.source_sequence, middle.sequence)
        self.assertEqual(selection.source_sequence_end, last.sequence)
        self.assertEqual(len(selection.detections), 1)
        self.assertEqual((
            selection.detections[0].x1,
            selection.detections[0].y1,
            selection.detections[0].x2,
            selection.detections[0].y2,
            selection.detections[0].name,
        ), (20, 30, 40, 50, "drone"))
        self.assertAlmostEqual(selection.detections[0].confidence, 0.85)

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
        statistics = camera.statistics()
        self.assertEqual(statistics["submitted_sequence"], packet.sequence)
        self.assertEqual(statistics["capture_samples"], 1)
        self.assertGreaterEqual(statistics["capture_read_p50_ms"], 0.0)

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
