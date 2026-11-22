"""Проверки границ экспериментального сопровождения без камеры."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'video_cap_buf'))
from capture import DisplayResult, Frame
from detector import Detection
from observe_kcf import KcfOverlay


class NativeFunction:
    """Имитирует функцию C ABI и позволяет задавать её возвращаемое значение."""
    def __init__(self, value):
        self.value = value

    def __call__(self, *args):
        return self.value(*args) if callable(self.value) else self.value


class FakeLibrary:
    """Имитирует нативный трекер без загрузки внешних библиотек."""
    def __init__(self):
        self.v16_observe_new = NativeFunction(1)
        self.v16_observe_free = NativeFunction(None)
        self.v16_observe_init = NativeFunction(1)
        def update(*args):
            args[-1][:] = [12, 11, 20, 20]
            return 1
        self.v16_observe_update = NativeFunction(update)


class KcfTests(unittest.TestCase):
    """Контролирует привязку рамки, срок жизни и сброс при пропусках."""
    def setUp(self):
        self.image = np.zeros((50, 50, 3), np.uint8)
        self.box = Detection(0, 'drone', 0.8, 10, 10, 30, 30)
        self.result = DisplayResult(1, 1_000_000_000, 1_010_000_000, 10, (self.box,), (self.box,))
        with patch('observe_kcf.ctypes.CDLL', return_value=FakeLibrary()):
            self.tracker = KcfOverlay(__file__, 400)

    def frame(self, number, timestamp):
        return Frame(number, self.image, 't', timestamp, timestamp)

    def tearDown(self):
        self.tracker.reset()

    def test_exact_seed_and_consecutive_update(self):
        self.assertTrue(self.tracker.seed(self.frame(1, 1_000_000_000), self.result))
        boxes = self.tracker.update(self.frame(2, 1_033_000_000))
        self.assertEqual((boxes[0].left, boxes[0].top, boxes[0].right, boxes[0].bottom), (12, 11, 32, 31))
        self.assertEqual(boxes[0].label, '~drone')

    def test_seed_requires_matching_frame(self):
        self.assertFalse(self.tracker.seed(self.frame(2, 1_033_000_000), self.result))
        self.assertEqual(self.tracker.update(self.frame(3, 1_066_000_000)), ())

    def test_skipped_or_expired_frame_clears_tracker(self):
        self.assertTrue(self.tracker.seed(self.frame(1, 1_000_000_000), self.result))
        self.assertEqual(self.tracker.update(self.frame(3, 1_066_000_000)), ())
        self.assertTrue(self.tracker.seed(self.frame(1, 1_000_000_000), self.result))
        self.assertEqual(self.tracker.update(self.frame(2, 1_401_000_000)), ())
        self.assertIsNone(self.tracker.anchor)

    def test_missing_detection_does_not_seed(self):
        empty = DisplayResult(1, 1_000_000_000, 1_010_000_000, 10, (), ())
        self.assertFalse(self.tracker.seed(self.frame(1, 1_000_000_000), empty))


if __name__ == '__main__':
    unittest.main()
