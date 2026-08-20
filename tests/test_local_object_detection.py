import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.inav_msp import InavTelemetry
from src.local_object_detection import (
    RUSSIAN_LABELS,
    draw_detections,
    draw_flight_status,
    draw_telemetry,
    extract_detections,
    parse_monitor_geometry,
    parse_source,
)
from src.realtime import Detection


class ParseSourceTests(unittest.TestCase):
    def test_camera_number(self):
        self.assertEqual(parse_source("0"), 0)

    def test_url(self):
        self.assertEqual(parse_source("rtsp://127.0.0.1/stream"), "rtsp://127.0.0.1/stream")

    def test_device_path(self):
        self.assertEqual(parse_source("/dev/video0"), "/dev/video0")

    def test_logitech_alias(self):
        self.assertEqual(parse_source("logitech"), "logitech")

    def test_monitor_geometry_from_xrandr(self):
        output = (
            "eDP-1 connected primary 1920x1200+0+0\n"
            "HDMI-1 connected 1024x600+1920+0\n"
        )
        self.assertEqual(parse_monitor_geometry(output, "HDMI-1"), (1024, 600, 1920, 0))

    def test_inactive_monitor_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "HDMI-1"):
            parse_monitor_geometry("HDMI-1 connected (normal left inverted right x axis y axis)\n", "HDMI-1")

    def test_russian_labels_cover_coco(self):
        self.assertEqual(len(RUSSIAN_LABELS), 80)

    def test_detection_is_extracted_with_generic_label(self):
        box = SimpleNamespace(
            xyxy=np.array([[1, 2, 8, 9]], dtype=float),
            cls=np.array([3]),
            conf=np.array([0.75]),
        )
        result = SimpleNamespace(boxes=[box], names={3: "plane"})
        detections = extract_detections(result, generic_label=True)
        self.assertEqual(detections, (Detection(1, 2, 8, 9, "OBJECT", 0.75),))

    def test_drawing_does_not_change_camera_frame(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        original = frame.copy()
        annotated = draw_detections(
            frame,
            (Detection(5, 10, 30, 30, "OBJECT", 0.8),),
        )
        self.assertTrue(np.array_equal(frame, original))
        self.assertFalse(np.array_equal(annotated, original))

    def test_flight_status_draws_arm_and_throttle(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        telemetry = SimpleNamespace(
            updated_at=1.0,
            armed=True,
            arm_switch=True,
            arm_aux_channel=1,
            arm_switch_value=1800,
            throttle_percent=50,
        )
        annotated = draw_flight_status(frame, telemetry)
        self.assertFalse(np.array_equal(annotated, np.zeros_like(frame)))

    def test_ground_speed_line_is_kept_without_gps_fix(self):
        frame = np.zeros((240, 640, 3), dtype=np.uint8)
        telemetry = InavTelemetry(
            gps_fix=0,
            gps_satellites=0,
            ground_speed=None,
            updated_at=1.0,
        )
        with patch("src.local_object_detection.put_osd_text") as put_text:
            draw_telemetry(frame, telemetry)
        lines = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("GS 0.0 m/s 0.0 km/h", lines)


if __name__ == "__main__":
    unittest.main()
