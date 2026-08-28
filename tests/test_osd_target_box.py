import unittest

from src.osd_target_box import target_box_from_pixels, update_pointer_from_detection


class TargetBoxTests(unittest.TestCase):
    """Проверяет перевод рамки изображения в сетку OSD."""

    def test_box_corners_are_mapped_to_osd_grid(self):
        box = target_box_from_pixels(0, 0, 1920, 1080, 1920, 1080)
        self.assertEqual(box.top_left, (0, 0))
        self.assertEqual(box.top_right, (29, 0))
        self.assertEqual(box.bottom_left, (0, 15))
        self.assertEqual(box.bottom_right, (29, 15))

    def test_invalid_image_size_is_rejected(self):
        with self.assertRaises(ValueError):
            target_box_from_pixels(0, 0, 10, 10, 0, 1080)

    def test_detection_center_is_sent_to_pointer(self):
        class FakePort:
            def reset_input_buffer(self):
                pass

            def write(self, packet):
                self.packet = packet

            def flush(self):
                pass

        port = FakePort()
        center = update_pointer_from_detection(port, 640, 360, 1280, 900, 1920, 1080)
        self.assertEqual(center, (15, 9))
        self.assertEqual(port.packet[5:8], bytes((2, 0x2F, 0x09)))


if __name__ == "__main__":
    unittest.main()
