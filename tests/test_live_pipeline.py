"""Проверяет временную целостность нового живого конвейера без камеры и ONNX."""

from __future__ import annotations

import threading
import time
import unittest

from src.live_pipeline import (
    FrameHub,
    InferenceRequest,
    LatestDetectionWorker,
    OverlayPacket,
    overlay_for_frame,
)
from src.realtime import Detection


class LivePipelineTests(unittest.TestCase):
    """Проверяет номера кадров, mailbox и запрет устаревших оверлеев."""

    def test_overlay_is_not_returned_for_a_different_video_frame(self) -> None:
        """Результат первого кадра нельзя нарисовать поверх второго кадра."""
        hub = FrameHub(tracking_history_size=4)
        first = hub.publish("frame-1", "gray-1", captured_at_ns=100)
        second = hub.publish("frame-2", "gray-2", captured_at_ns=200)
        overlay = OverlayPacket(
            frame_sequence=first.sequence,
            source_sequence=first.sequence,
            detections=(Detection(1, 2, 3, 4, "FPV-DRON", 0.8),),
            status="verified",
        )

        self.assertIs(overlay_for_frame(first, overlay), overlay)
        self.assertIsNone(overlay_for_frame(second, overlay))

    def test_frame_hub_has_latest_full_frame_and_bounded_tracking_history(self) -> None:
        """Полный кадр не копится очередью, а история хранит только последние номера."""
        hub = FrameHub(tracking_history_size=3)
        packets = [
            hub.publish(f"bgr-{index}", f"gray-{index}", captured_at_ns=index)
            for index in range(1, 6)
        ]

        self.assertEqual(hub.latest(after_sequence=0), packets[-1])
        self.assertEqual(
            tuple(item.sequence for item in hub.tracking_after(0)),
            (3, 4, 5),
        )
        self.assertEqual(
            tuple(item.image for item in hub.tracking_after(3)),
            ("gray-4", "gray-5"),
        )

    def test_worker_result_keeps_the_source_sequence_after_delayed_inference(self) -> None:
        """Даже после задержки YOLO результат сохраняет номер исходного кадра."""
        hub = FrameHub()
        packet = hub.publish("bgr", "gray", captured_at_ns=123)
        started = threading.Event()
        release = threading.Event()

        def predict(request: InferenceRequest):
            """Имитирует долгую нейросеть и возвращает одну рамку."""
            started.set()
            release.wait(timeout=1.0)
            return (Detection(10, 20, 30, 40, "FPV-DRON", 0.75),), 0.75

        worker = LatestDetectionWorker(predict, name="test-delayed-yolo")
        try:
            worker.submit(InferenceRequest(packet, "full_scan", 130, packet.bgr))
            self.assertTrue(started.wait(timeout=1.0))
            release.set()
            deadline = time.monotonic() + 1.0
            result = worker.latest()
            while result is None and time.monotonic() < deadline:
                time.sleep(0.01)
                result = worker.latest()

            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.source_sequence, packet.sequence)
            self.assertEqual(result.source_captured_at_ns, 123)
            self.assertGreaterEqual(result.inference_ms, 0.0)
            self.assertGreaterEqual(result.queue_ms, 0.0)
        finally:
            worker.close()

    def test_worker_discards_only_waiting_stale_requests(self) -> None:
        """Занятый worker после первого кадра берёт третий, а второй отбрасывает."""
        hub = FrameHub()
        packets = [hub.publish(f"bgr-{index}", f"gray-{index}") for index in range(1, 4)]
        first_started = threading.Event()
        release_first = threading.Event()
        processed: list[int] = []

        def predict(request: InferenceRequest):
            """Удерживает первый запрос, чтобы детерминированно заполнить mailbox."""
            processed.append(request.source.sequence)
            if request.source.sequence == packets[0].sequence:
                first_started.set()
                release_first.wait(timeout=1.0)
            return (), 0.0

        worker = LatestDetectionWorker(predict, name="test-latest-yolo")
        try:
            worker.submit(InferenceRequest(packets[0], "full_scan", 1, packets[0].bgr))
            self.assertTrue(first_started.wait(timeout=1.0))
            worker.submit(InferenceRequest(packets[1], "full_scan", 2, packets[1].bgr))
            worker.submit(InferenceRequest(packets[2], "full_scan", 3, packets[2].bgr))
            release_first.set()

            deadline = time.monotonic() + 1.0
            result = worker.latest()
            while (result is None or result.source_sequence != packets[2].sequence) and time.monotonic() < deadline:
                time.sleep(0.01)
                result = worker.latest()

            self.assertEqual(processed, [packets[0].sequence, packets[2].sequence])
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.source_sequence, packets[2].sequence)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
