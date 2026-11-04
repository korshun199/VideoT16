"""Проверяет точечный фильтр AIR и сохранение остальных обнаружений."""

from pathlib import Path
import sys
import time
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "video_cap_buf"))

from air_filter import AirFilter, _contains_air
from detector import Detection
from capture import Frame, PipelineSession, load_settings


def box(left: int, top: int, right: int, bottom: int) -> Detection:
    """Создаёт тестовое обнаружение модели в координатах кадра."""
    return Detection(0, "copter_drone", 0.59, left, top, right, bottom)


class AirFilterTests(unittest.TestCase):
    """Проверяет границы совпадения и работу с реальными сохранёнными примерами."""

    def test_overlap_requires_a_compact_complete_air_box(self) -> None:
        """Рамка вокруг всего AIR подавляется, но большая или соседняя — нет."""
        air = (837, 133, 952, 174)
        self.assertTrue(_contains_air(box(820, 93, 1034, 186), air))
        self.assertFalse(_contains_air(box(700, 50, 1200, 300), air))
        self.assertFalse(_contains_air(box(880, 271, 934, 315), air))
        self.assertFalse(_contains_air(box(837, 133, 860, 174), air))

    def test_blank_image_does_not_remove_detection(self) -> None:
        """В отсутствие надписи даже рамка в её обычном месте остаётся."""
        drone = box(820, 93, 1034, 186)
        kept, removed, _ = AirFilter().filter(np.zeros((1080, 1920, 3), np.uint8), (drone,))
        self.assertEqual(kept, (drone,))
        self.assertFalse(removed)

    def test_recognition_pipeline_logs_and_filters_only_when_enabled(self) -> None:
        """Распознавание убирает AIR из обеих групп и записывает причину в журнал."""
        image = cv2.imread(str(ROOT / "tmp/osd-probe-aJpyTM/frame_475.png"))
        if image is None:
            self.skipTest("Локальный кадр испытаний отсутствует")
        air, drone = box(820, 93, 1034, 186), box(881, 271, 934, 315)

        class Log:
            def __init__(self) -> None:
                self.events = []

            def write(self, event, **fields) -> None:
                self.events.append((event, fields))

        class Detector:
            def detect_with_presence(self, _image):
                return (air, drone), (air, drone), 1.0

        for mode in ("on", "off"):
            with self.subTest(mode=mode):
                settings = load_settings(ROOT / "video_cap_buf/settings.json")
                settings["air_osd_filter"] = mode
                log = Log()
                session = PipelineSession(settings, log, log, Detector())
                now = time.monotonic_ns()
                self.assertTrue(session.frame_buffer.append(Frame(1, image, "test", now, now)))
                session.frame_buffer.finish()
                session._recognize(0, session.detector)
                result = session.display_buffer.get(1)
                self.assertIsNotNone(result)
                expected = (drone,) if mode == "on" else (air, drone)
                self.assertEqual(result.presence_candidates, expected)
                self.assertEqual(result.detections, expected)
                event = next(fields for name, fields in log.events if name == "frame_recognized")
                self.assertEqual(len(event["air_suppressed"]), 1 if mode == "on" else 0)

    def test_filter_failure_keeps_original_detections(self) -> None:
        """При ошибке фильтра программа сохраняет находку и сообщает в журнал."""
        drone = box(820, 93, 1034, 186)

        class Log:
            def __init__(self):
                self.events = []

            def write(self, event, **fields):
                self.events.append((event, fields))

        class Detector:
            def detect_with_presence(self, _image):
                return (drone,), (drone,), 1.0

        class BrokenFilter:
            def filter(self, _image, _candidates):
                raise ValueError("Проверочная ошибка")

        settings = load_settings(ROOT / "video_cap_buf/settings.json")
        settings["air_osd_filter"] = "on"
        log = Log()
        session = PipelineSession(settings, log, log, Detector())
        session.air_filter = BrokenFilter()
        now = time.monotonic_ns()
        session.frame_buffer.append(Frame(1, np.zeros((480, 640, 3), np.uint8), "test", now, now))
        session.frame_buffer.finish()
        session._recognize(0, session.detector)
        self.assertEqual(session.display_buffer.get(1).detections, (drone,))
        self.assertIn("air_filter_error", [event for event, _ in log.events])
    def test_saved_air_does_not_hide_separate_real_drone(self) -> None:
        """На реальном кадре удаляется AIR, а рамка другого дрона остаётся."""
        image = cv2.imread(str(ROOT / "tmp/osd-probe-aJpyTM/frame_475.png"))
        if image is None:
            self.skipTest("Локальный кадр испытаний отсутствует")
        air = box(820, 93, 1034, 186)
        drone = box(881, 271, 934, 315)
        kept, removed, score = AirFilter().filter(image, (air, drone))
        self.assertGreaterEqual(score, 0.75)
        self.assertEqual(removed, (air,))
        self.assertEqual(kept, (drone,))

    def test_saved_air_at_three_video_sizes(self) -> None:
        """Одна и та же надпись находится после уменьшения исходного кадра."""
        image = cv2.imread(str(ROOT / "tmp/osd-probe-aJpyTM/frame_280.png"))
        if image is None:
            self.skipTest("Локальный кадр испытаний отсутствует")
        for width in (1920, 1280, 960):
            with self.subTest(width=width):
                scale = width / 1920
                resized = cv2.resize(image, (width, round(1080 * scale)))
                candidate = box(*(round(value * scale) for value in (835, 125, 949, 173)))
                kept, removed, _ = AirFilter().filter(resized, (candidate,))
                self.assertEqual(kept, ())
                self.assertEqual(removed, (candidate,))


if __name__ == "__main__":
    unittest.main()
