"""Проверяет расчёт режимов и сводки стенда замера без доступа к камере."""

from __future__ import annotations

import argparse
import unittest

import numpy as np

from scripts.benchmark_live_inference import (
    build_modes,
    default_thread_counts,
    parse_thread_counts,
    summarize,
)


class BenchmarkLiveInferenceTests(unittest.TestCase):
    """Проверяет безопасную конфигурацию короткого сравнения ONNX."""

    def test_default_threads_are_limited_to_available_logical_cpus(self) -> None:
        """Сетка содержит 12 потоков на 12-ядерной системе и не превышает CPU."""
        self.assertEqual(default_thread_counts(12), [1, 2, 4, 8, 12])
        self.assertEqual(default_thread_counts(6), [1, 2, 4, 6])
        self.assertEqual(default_thread_counts(1), [1])

    def test_thread_argument_rejects_invalid_or_duplicate_values(self) -> None:
        """Опечатка в ручном списке потоков останавливает замер до открытия камеры."""
        self.assertEqual(parse_thread_counts("1, 2,12"), [1, 2, 12])
        for value in ("0,2", "2,nope", "2,2", " , "):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_thread_counts(value)

    def test_modes_include_four_current_tiles_two_larger_tiles_and_full_frame(self) -> None:
        """При 1280x960 сравниваются текущие 4 плитки, 2 крупные и полный кадр."""
        frame = np.zeros((960, 1280, 3), dtype=np.uint8)
        modes = build_modes(frame, baseline_tile_size=640, overlap=0.2)

        self.assertEqual([mode["kind"] for mode in modes], ["tiles", "tiles", "full"])
        self.assertEqual(
            [mode["name"] for mode in modes],
            ["текущие плитки (4)", "крупные плитки (2)", "весь кадр (1)"],
        )
        self.assertEqual([mode["tile_size"] for mode in modes], [640, 960, None])

    def test_summary_ranks_median_speed_and_counts_visible_detections(self) -> None:
        """Итог показывает медиану времени и долю кадров с рамкой выше порога."""
        records = [
            {"threads": 2, "mode": "tiles", "inference_ms": 100, "raw_confidence": 0.8, "visible_detections": 1},
            {"threads": 2, "mode": "tiles", "inference_ms": 300, "raw_confidence": 0.6, "visible_detections": 0},
        ]

        result = summarize(records, confidence_threshold=0.4)[0]

        self.assertEqual(result["median_ms"], 200)
        self.assertEqual(result["equivalent_fps"], 5)
        self.assertEqual(result["samples_with_box_above_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
