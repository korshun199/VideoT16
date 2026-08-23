import threading
import time
import unittest

from src.realtime import Detection, LatestInferenceWorker


def wait_for_sequence(worker, expected, timeout=1.0):
    """Ожидает результат фонового потока с ограничением времени."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = worker.latest()
        if snapshot.sequence >= expected:
            return snapshot
        time.sleep(0.005)
    raise AssertionError(f"Не получен результат с номером {expected}")


class LatestInferenceWorkerTests(unittest.TestCase):
    def test_pending_frame_is_replaced_by_newest(self):
        started = threading.Event()
        release = threading.Event()

        def predict(value):
            if value == 1:
                started.set()
                release.wait(1.0)
            return (Detection(value, 0, value + 1, 1, "OBJECT", 0.9),)

        worker = LatestInferenceWorker(predict)
        try:
            worker.submit(1)
            self.assertTrue(started.wait(1.0))
            worker.submit(2)
            worker.submit(3)
            release.set()
            snapshot = wait_for_sequence(worker, 2)
            self.assertEqual(snapshot.detections[0].x1, 3)
        finally:
            release.set()
            worker.close()

    def test_worker_reports_prediction_error(self):
        def predict(_frame):
            raise ValueError("тестовая ошибка")

        worker = LatestInferenceWorker(predict)
        try:
            worker.submit(object())
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    worker.latest()
                except RuntimeError as error:
                    self.assertIn("тестовая ошибка", str(error))
                    break
                time.sleep(0.005)
            else:
                self.fail("Ошибка фонового потока не передана главному циклу")
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
