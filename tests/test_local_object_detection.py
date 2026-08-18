import unittest

from src.local_object_detection import RUSSIAN_LABELS, parse_source


class ParseSourceTests(unittest.TestCase):
    def test_camera_number(self):
        self.assertEqual(parse_source("0"), 0)

    def test_url(self):
        self.assertEqual(parse_source("rtsp://127.0.0.1/stream"), "rtsp://127.0.0.1/stream")

    def test_device_path(self):
        self.assertEqual(parse_source("/dev/video0"), "/dev/video0")

    def test_logitech_alias(self):
        self.assertEqual(parse_source("logitech"), "logitech")

    def test_russian_labels_cover_coco(self):
        self.assertEqual(len(RUSSIAN_LABELS), 80)


if __name__ == "__main__":
    unittest.main()
