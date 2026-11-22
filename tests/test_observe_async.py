"""Проверяет изоляцию экспериментального трекера от потока отображения."""

from pathlib import Path
import sys
import threading
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'video_cap_buf'))
from capture import DisplayResult, Frame
from detector import Detection
from observe_async import AsyncKcf


class FakeLog:
    """Запоминает сообщения о безопасном отключении эксперимента."""
    def __init__(self):
        self.events = []

    def write(self, event, **fields):
        self.events.append((event, fields))


class FakeTracker:
    """Даёт предсказуемые рамки, сохраняя контракт подтверждения."""
    def __init__(self, library, max_age_ms):
        self.anchor = None

    def reset(self):
        self.anchor = None

    def seed(self, frame, result):
        self.anchor = result if frame.number == result.number and result.detections else None
        return self.anchor is not None

    def update(self, frame):
        if self.anchor is None or frame.number <= self.anchor.number:
            return ()
        return (Detection(0, '~drone', 0.9, frame.number, 1, frame.number + 4, 5),)


class AsyncTests(unittest.TestCase):
    """Проверяет точное соответствие кадру, отзыв и переполнение очереди."""
    def setUp(self):
        self.image = np.zeros((8, 8, 3), np.uint8)
        self.log = FakeLog()
        self.worker = AsyncKcf('fake', 400, self.log, FakeTracker)
        self.worker.start()

    def tearDown(self):
        self.assertTrue(self.worker.close())

    def frame(self, number):
        return Frame(number, self.image, 't', number * 1000000, number * 1000000)

    def result(self, number, detected=True):
        box = Detection(0, 'drone', 0.9, 0, 0, 4, 4)
        boxes = (box,) if detected else ()
        return DisplayResult(number, number * 1000000, number * 1000000, 0, boxes, boxes)

    def test_delayed_result_replays_frame_history(self):
        self.worker.submit_frame(self.frame(1))
        self.worker.submit_frame(self.frame(2))
        self.worker.submit_result(self.result(1))
        deadline = time.monotonic() + 2
        found = None
        while found is None and time.monotonic() < deadline:
            found = self.worker.take(2)
            threading.Event().wait(0.005)
        self.assertIsNotNone(found)
        self.assertEqual(found[0].number, 1)
        self.assertEqual(found[1][0].left, 2)
        self.assertIsNone(self.worker.take(3))

    def test_new_negative_immediately_revokes_cached_output(self):
        self.worker.submit_frame(self.frame(1))
        self.worker.submit_frame(self.frame(2))
        self.worker.submit_result(self.result(1))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.worker.lock:
                ready = 2 in self.worker.outputs
            if ready:
                break
            threading.Event().wait(0.005)
        self.assertTrue(ready)
        self.worker.submit_result(self.result(2, detected=False))
        self.assertIsNone(self.worker.take(2))

    def test_future_result_preserves_earlier_buffered_frame(self):
        self.worker.submit_frame(self.frame(1))
        self.worker.submit_frame(self.frame(2))
        self.worker.submit_result(self.result(1))
        deadline = time.monotonic() + 2
        ready = False
        while time.monotonic() < deadline:
            with self.worker.lock:
                ready = 2 in self.worker.outputs
            if ready:
                break
            threading.Event().wait(0.005)
        self.assertTrue(ready)
        self.worker.submit_result(self.result(3, detected=False))
        self.assertIsNotNone(self.worker.take(2))

    def test_out_of_order_result_is_ignored(self):
        self.worker.submit_result(self.result(10))
        self.worker.submit_result(self.result(5))
        self.assertEqual(self.worker.last_result_number, 10)
        self.assertEqual(self.worker.generation, 1)


class QueueTests(unittest.TestCase):
    """При переполнении старая рамка не выдаётся как актуальная."""
    def test_full_queue_disables_without_blocking(self):
        log = FakeLog()
        worker = AsyncKcf('fake', 400, log, FakeTracker, queue_size=1)
        worker.submit_frame(object())
        worker.submit_frame(object())
        self.assertTrue(worker.disabled)
        self.assertIsNone(worker.take(1))
        self.assertEqual(log.events[0][1]['reason'], 'queue_full')
        self.assertTrue(worker.close())

    def test_display_does_not_wait_for_busy_tracker(self):
        entered, release = threading.Event(), threading.Event()

        class SlowTracker(FakeTracker):
            def update(self, frame):
                entered.set()
                release.wait(timeout=2)
                return ()

        worker = AsyncKcf('fake', 400, FakeLog(), SlowTracker)
        worker.start()
        try:
            worker.submit_frame(object())
            self.assertTrue(entered.wait(timeout=1))
            started = time.monotonic()
            self.assertIsNone(worker.take(1))
            self.assertLess(time.monotonic() - started, 0.1)
        finally:
            release.set()
            self.assertTrue(worker.close())


if __name__ == '__main__':
    unittest.main()
