"""Проверяет измерения кадров без обращения к настоящему V4L2-устройству."""

from __future__ import annotations

import unittest

import numpy as np

from src.capture_diagnostics import RollingCaptureMetrics, coarse_frame_signature


class CaptureDiagnosticsTests(unittest.TestCase):
    """Проверяет вычисления задержки, FPS и безопасную сигнатуру кадра."""

    def test_metrics_report_read_interval_and_repeated_coarse_signatures(self) -> None:
        """Окно измерений отделяет длительность read от паузы между кадрами."""
        metrics = RollingCaptureMetrics(window_size=4)
        metrics.add(0, 10_000_000, 10)
        metrics.add(33_000_000, 43_000_000, 10)
        metrics.add(66_000_000, 76_000_000, 11)
        metrics.add(99_000_000, 109_000_000, 11)

        telemetry = metrics.snapshot()

        self.assertEqual(telemetry.samples, 4)
        self.assertAlmostEqual(telemetry.observed_fps, 30.30, places=1)
        self.assertEqual(telemetry.read_p50_ms, 10.0)
        self.assertEqual(telemetry.read_p90_ms, 10.0)
        self.assertEqual(telemetry.interval_p50_ms, 33.0)
        self.assertEqual(telemetry.interval_p90_ms, 33.0)
        self.assertAlmostEqual(telemetry.coarse_same_percent, 66.67, places=1)

    def test_coarse_signature_is_stable_for_same_frame_and_changes_for_grid_pixel(self) -> None:
        """Сигнатура не требует записи изображения и замечает изменение в сетке."""
        first = np.zeros((120, 160, 3), dtype=np.uint8)
        second = first.copy()
        second[60, 80] = (10, 20, 30)

        self.assertEqual(coarse_frame_signature(first), coarse_frame_signature(first.copy()))
        self.assertNotEqual(coarse_frame_signature(first), coarse_frame_signature(second))

    def test_metrics_rejects_time_reversal(self) -> None:
        """Повреждённые временные метки не дают выдать ложную частоту кадров."""
        metrics = RollingCaptureMetrics()
        metrics.add(10, 20, 1)

        with self.assertRaisesRegex(ValueError, "по порядку"):
            metrics.add(15, 19, 2)


if __name__ == "__main__":
    unittest.main()
