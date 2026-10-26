"""Проверяет фоновые работники без камеры, модели и окна OpenCV."""

from __future__ import annotations

import threading
import time
import unittest

from src.realtime import LatestValueWorker


class LatestValueWorkerTests(unittest.TestCase):
    """Проверяет отсутствие очереди и передачу ошибок лёгкого фонового этапа."""

    def test_worker_discards_stale_pending_value(self) -> None:
        """Пока обработка первого значения занята, второе заменяется третьим."""
        first_started = threading.Event()
        release_first = threading.Event()
        processed: list[str] = []

        def process(value: str) -> str:
            """Удерживает первый вход, чтобы детерминированно заполнить ожидание."""
            processed.append(value)
            if value == "first":
                first_started.set()
                release_first.wait(timeout=1.0)
            return value.upper()

        worker = LatestValueWorker(process, name="test-latest-value")
        try:
            worker.submit("first")
            self.assertTrue(first_started.wait(timeout=1.0))
            worker.submit("stale")
            worker.submit("fresh")
            release_first.set()
            deadline = time.monotonic() + 1.0
            snapshot = worker.latest()
            while snapshot.sequence < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
                snapshot = worker.latest()

            self.assertEqual(processed, ["first", "fresh"])
            self.assertEqual(snapshot.sequence, 2)
            self.assertEqual(snapshot.value, "FRESH")
        finally:
            worker.close()

    def test_worker_surfaces_background_error(self) -> None:
        """Сбой быстрого анализа не остаётся незаметным в фоновом потоке."""
        worker = LatestValueWorker(lambda value: 1 / value, name="test-value-error")
        try:
            worker.submit(0)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                try:
                    worker.latest()
                except RuntimeError as error:
                    self.assertIn("Ошибка фонового быстрого анализа", str(error))
                    break
                time.sleep(0.01)
            else:
                self.fail("Ошибка фонового работника не дошла до latest()")
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
