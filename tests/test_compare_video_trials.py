"""Проверки сравнения моделей на вручную выбранных кадрах."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from unittest.mock import patch

from scripts.compare_video_trials import (
    compare_samples,
    draw_overlay,
    generate_tiles,
    load_overlay_settings,
    merge_tile_detections,
)
from src.realtime import Detection


class CompareVideoTrialsTests(unittest.TestCase):
    """Проверяет локальное сравнение моделей без обучения и эталонной разметки."""

    def test_overlay_uses_configured_box_geometry_and_style(self):
        """Рамка и подпись берут геометрию и оформление из параметров профиля."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detection = Detection(10, 20, 30, 40, "drone", 0.80)
        style = {
            "box": {
                "scale_x": 2.0,
                "scale_y": 0.5,
                "thickness": 1,
                "color_rgb": [255, 0, 0],
            },
            "label": {
                "font": "FONT_HERSHEY_PLAIN",
                "font_scale": 0.7,
                "thickness": 1,
                "offset_y": 3,
                "min_y": 10,
                "color_rgb": [0, 255, 0],
            },
        }

        with (
            patch("scripts.compare_video_trials.cv2.rectangle") as rectangle,
            patch("scripts.compare_video_trials.cv2.putText") as put_text,
        ):
            draw_overlay(frame, (detection,), threshold=0.5, style=style)

        rectangle_args = rectangle.call_args.args
        self.assertEqual(rectangle_args[1], (0, 25))
        self.assertEqual(rectangle_args[2], (40, 35))
        self.assertEqual(rectangle_args[3], (0, 0, 255))
        self.assertEqual(rectangle_args[4], 1)
        text_args = put_text.call_args.args
        self.assertEqual(text_args[1], "drone 80%")
        self.assertEqual(text_args[2], (0, 22))
        self.assertEqual(text_args[3], cv2.FONT_HERSHEY_PLAIN)
        self.assertEqual(text_args[4], 0.7)
        self.assertEqual(text_args[5], (0, 255, 0))
        self.assertEqual(text_args[6], 1)

    def test_overlay_clamps_box_fully_outside_frame_to_valid_pixels(self):
        """Рамка у правого нижнего края остаётся в пределах размера кадра."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detection = Detection(100, 100, 120, 120, "drone", 0.80)

        with patch("scripts.compare_video_trials.cv2.rectangle") as rectangle:
            draw_overlay(frame, (detection,), threshold=0.5, style={
                "box": {
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "thickness": 2,
                    "color_rgb": [80, 230, 40],
                },
                "label": {
                    "font": "FONT_HERSHEY_SIMPLEX",
                    "font_scale": 0.55,
                    "thickness": 2,
                    "offset_y": 7,
                    "min_y": 18,
                    "color_rgb": [80, 230, 40],
                },
            })

        first, second = rectangle.call_args.args[1:3]
        self.assertLessEqual(first[0], second[0])
        self.assertLessEqual(first[1], second[1])
        self.assertLess(second[0], frame.shape[1])
        self.assertLess(second[1], frame.shape[0])

    def test_overlay_uses_configured_near_threshold_color_band(self):
        """Основной порог зелёный, полоса ниже оранжевая, слабые кандидаты скрыты."""
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        detections = (
            Detection(1, 1, 10, 10, "drone", 0.80),
            Detection(15, 15, 25, 25, "drone", 0.65),
            Detection(30, 30, 40, 40, "drone", 0.59),
        )
        style = load_overlay_settings()

        with (
            patch("scripts.compare_video_trials.cv2.rectangle") as rectangle,
            patch("scripts.compare_video_trials.cv2.putText") as put_text,
        ):
            draw_overlay(frame, detections, threshold=0.70, style=style)

        self.assertEqual(rectangle.call_count, 2)
        self.assertEqual(rectangle.call_args_list[0].args[3], (40, 230, 80))
        self.assertEqual(rectangle.call_args_list[1].args[3], (0, 165, 255))
        self.assertEqual(put_text.call_count, 2)

    def test_writes_per_model_overlays_and_threshold_report(self):
        """Один и тот же кадр получает результаты и пороговую сводку обеих моделей."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "samples"
            sample_dir.mkdir()
            frame_path = sample_dir / "selected.png"
            self.assertTrue(cv2.imwrite(str(frame_path), np.zeros((64, 96, 3), dtype=np.uint8)))
            models = (root / "first.onnx", root / "second.onnx")
            for model in models:
                model.touch()

            class FakeDetector:
                """Возвращает заранее заданные выходы для тестирования отчёта."""

                def __init__(self, detections, raw_confidence):
                    self.detections = detections
                    self.last_best_confidence = raw_confidence

                def __call__(self, _frame):
                    """Возвращает детекции без изменения входного изображения."""
                    return self.detections

            detections_by_model = {
                "first.onnx": FakeDetector(
                    (
                        Detection(10, 12, 42, 44, "quadcopter", 0.80),
                        Detection(50, 15, 80, 45, "noise", 0.12),
                    ),
                    0.87,
                ),
                "second.onnx": FakeDetector(
                    (Detection(20, 10, 45, 40, "copter_drone", 0.28),),
                    0.42,
                ),
            }

            def detector_factory(path, confidence, generic_label, size, object_label=None, threads=1):
                """Подставляет тестовые результаты, сохраняя интерфейс ONNX-детектора."""
                self.assertEqual(confidence, 0.01)
                self.assertFalse(generic_label)
                self.assertIsNone(object_label)
                self.assertEqual(threads, 2)
                return detections_by_model[path.name]

            output_dir = root / "comparison"
            summary = compare_samples(
                sample_dir,
                models,
                output_dir,
                minimum_confidence=0.01,
                draw_threshold=0.15,
                threads=2,
                input_size_reader=lambda path: 640 if path.name == "first.onnx" else 320,
                detector_factory=detector_factory,
            )

            self.assertEqual(summary["frames"], 1)
            self.assertEqual(summary["models"], 2)
            report = output_dir / "results.csv"
            with report.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["input_size"], "640")
            self.assertEqual(rows[0]["count_at_10_percent"], "2")
            self.assertEqual(rows[0]["count_at_20_percent"], "1")
            self.assertEqual(rows[1]["count_at_20_percent"], "1")
            self.assertEqual(rows[1]["count_at_30_percent"], "0")
            self.assertEqual(len(list(output_dir.glob("*/frame_*.png"))), 2)

    def test_rejects_empty_sample_directory(self):
        """Пустая папка завершается понятной ошибкой до загрузки моделей."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "samples"
            sample_dir.mkdir()
            model = root / "model.onnx"
            model.touch()

            with self.assertRaisesRegex(ValueError, "Нет изображений"):
                compare_samples(sample_dir, (model,), root / "comparison")

    def test_tiles_cover_landscape_frame_with_overlap(self):
        """Фрагменты покрывают кадр полностью, перекрывая границы для рамки дрона."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        tiles = generate_tiles(frame, tile_size=640, overlap=0.20)

        self.assertEqual(len(tiles), 4)
        self.assertEqual([tile[:2] for tile in tiles], [(0, 0), (569, 0), (0, 320), (569, 320)])
        self.assertEqual(tiles[0][2].shape[:2], (400, 711))
        self.assertEqual(tiles[-1][0] + tiles[-1][2].shape[1], 1280)
        self.assertEqual(tiles[-1][1] + tiles[-1][2].shape[0], 720)

    def test_tiled_comparison_maps_boxes_back_and_writes_separate_report_rows(self):
        """Режим фрагментов записывает рамки в координатах исходника отдельно от целого кадра."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "samples"
            sample_dir.mkdir()
            self.assertTrue(
                cv2.imwrite(
                    str(sample_dir / "selected.png"),
                    np.zeros((720, 1280, 3), dtype=np.uint8),
                )
            )
            model = root / "model.onnx"
            model.touch()
            second_model = root / "second_model.onnx"
            second_model.touch()

            class FakeDetector:
                """Возвращает одну рамку в координатах переданного фрагмента."""

                def __init__(self):
                    self.calls = 0
                    self.frame_shapes = []
                    self.last_best_confidence = 0.91

                def __call__(self, _frame):
                    """Считает вызовы и возвращает тестовую детекцию."""
                    self.calls += 1
                    self.frame_shapes.append(_frame.shape)
                    return (Detection(20, 30, 80, 90, "quadcopter", 0.85),)

            detectors = {"model.onnx": FakeDetector(), "second_model.onnx": FakeDetector()}

            def detector_factory(path, _confidence, _generic_label, size, **_kwargs):
                """Возвращает детектор и проверяет, что у моделей разные входы."""
                self.assertEqual(size, 640 if path.name == "model.onnx" else 320)
                return detectors[path.name]

            result = compare_samples(
                sample_dir,
                (model, second_model),
                root / "comparison",
                draw_threshold=0.15,
                input_size_reader=lambda path: 640 if path.name == "model.onnx" else 320,
                detector_factory=detector_factory,
                include_tiles=True,
            )

            for detector in detectors.values():
                self.assertEqual(detector.calls, 5)  # целый кадр + четыре фрагмента
            self.assertEqual(detectors["model.onnx"].frame_shapes[1:], detectors["second_model.onnx"].frame_shapes[1:])
            self.assertEqual(result["rows"], 4)
            with (root / "comparison" / "results.csv").open(
                encoding="utf-8-sig", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(
                [row["inference_mode"] for row in rows],
                ["full", "tiles", "full", "tiles"],
            )
            tile_detections = json.loads(rows[1]["detections_json"])
            self.assertEqual(len(tile_detections), 4)
            self.assertEqual(tile_detections[1]["box_xyxy"], [589, 30, 649, 90])
            self.assertEqual(tile_detections[2]["box_xyxy"], [20, 350, 80, 410])

    def test_tile_merge_removes_cross_crop_duplicates_not_other_classes(self):
        """Склеиваются повторные рамки из перекрытия, но иные классы сохраняются."""
        merged = merge_tile_detections(
            (
                Detection(100, 100, 180, 180, "quadcopter", 0.80),
                Detection(105, 103, 185, 183, "quadcopter", 0.90),
                Detection(100, 100, 180, 180, "fixed-wing", 0.70),
            )
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].confidence, 0.90)
        self.assertEqual({item.name for item in merged}, {"quadcopter", "fixed-wing"})


if __name__ == "__main__":
    unittest.main()
