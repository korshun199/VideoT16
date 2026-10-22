"""Проверки параметров отдельного живого испытания модели."""

import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts.live_drone_trial import (
    CameraInfo,
    build_parser,
    choose_camera,
    console_status,
    main,
    parse_source,
    resolve_run_args,
    tee_output,
    validate_args,
)


class LiveDroneTrialTests(unittest.TestCase):
    """Проверяет выбор камеры и валидацию CLI без физического оборудования."""

    def test_sky_profile_restores_owner_confirmed_settings(self):
        """Профиль sky загружает закреплённую рабочую модель и параметры."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))

        self.assertEqual(args.source, "auto")
        self.assertEqual(args.model, Path(__file__).resolve().parents[1] / "models/anti_uav_fast_320_int8_pc.onnx")
        self.assertEqual(args.confidence, 0.40)
        self.assertEqual(args.min_confidence, 0.05)
        self.assertEqual(args.tile_size, 640)
        self.assertEqual(args.tile_overlap, 0.20)
        self.assertEqual(args.threads, 2)
        self.assertFalse(args.full_frame)

    def test_command_line_values_override_sky_profile(self):
        """Явный порог и размер фрагмента заменяют значения профиля."""
        args = resolve_run_args(
            build_parser().parse_args(
                ["--profile", "sky", "--confidence", "0.25", "--tile-size", "480"]
            )
        )

        self.assertEqual(args.confidence, 0.25)
        self.assertEqual(args.tile_size, 480)

    def test_numeric_camera_source_becomes_index(self):
        """Числовой индекс передаётся OpenCV как номер, а не путь."""
        self.assertEqual(parse_source("4"), 4)

    def test_device_path_remains_path(self):
        """Явный V4L2-путь сохраняет точное имя устройства."""
        self.assertEqual(parse_source("/dev/video4"), "/dev/video4")

    def test_single_camera_is_selected_without_prompt(self):
        """Единственная проверенная камера выбирается автоматически."""
        camera = CameraInfo("/dev/video4", 1280, 720, 60.0)
        self.assertEqual(choose_camera([camera]), "/dev/video4")

    def test_multiple_cameras_are_not_guessed(self):
        """Без интерактивного ввода скрипт не угадывает встроенную камеру."""
        cameras = [
            CameraInfo("/dev/video0", 640, 480, 30.0),
            CameraInfo("/dev/video4", 1280, 720, 60.0),
        ]
        with patch("builtins.input", side_effect=EOFError):
            with self.assertRaisesRegex(RuntimeError, "номер камеры"):
                choose_camera(cameras)

    def test_multiple_cameras_accept_explicit_choice(self):
        """Пользователь может выбрать видеозахват среди нескольких устройств."""
        cameras = [
            CameraInfo("/dev/video0", 640, 480, 30.0),
            CameraInfo("/dev/video4", 1280, 720, 60.0),
        ]
        with patch("builtins.input", return_value="2"):
            self.assertEqual(choose_camera(cameras), "/dev/video4")

    def test_confidence_thresholds_must_be_consistent(self):
        """Порог кандидатов не может быть выше порога показа рамки."""
        args = Namespace(confidence=0.70, min_confidence=0.80, tile_overlap=0.20, tile_size=640, threads=2)
        with self.assertRaisesRegex(ValueError, "не должен быть выше"):
            validate_args(args)

    def test_accepts_candidate_settings(self):
        """Стартовый профиль B проходит проверку параметров."""
        args = Namespace(confidence=0.70, min_confidence=0.05, tile_overlap=0.20, tile_size=640, threads=2)
        validate_args(args)

    def test_console_status_reports_raw_confidence_and_thresholded_boxes(self):
        """Терминал различает сырую уверенность модели и рамки выше порога."""
        self.assertEqual(
            console_status(0.846, 1, 0.70, 351.5),
            "confidence=85% | рамок выше 70%: 1 | инференс=352 мс",
        )

    def test_tee_output_records_stdout_and_stderr_and_keeps_console_output(self):
        """Обычные сообщения и ошибки одновременно остаются в консоли и журнале."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "logs" / "trial.log"
            console_out = io.StringIO()
            console_err = io.StringIO()
            with redirect_stdout(console_out), redirect_stderr(console_err):
                with tee_output(log_path):
                    print("confidence=85%")
                    print("Ошибка камеры", file=sys.stderr)
                    os.write(2, b"native camera diagnostic\n")

            self.assertIn("confidence=85%", console_out.getvalue())
            self.assertIn("Ошибка камеры", console_err.getvalue())
            saved_log = log_path.read_text(encoding="utf-8")
            self.assertIn("confidence=85%", saved_log)
            self.assertIn("Ошибка камеры", saved_log)
            self.assertIn("native camera diagnostic", saved_log)

    def test_startup_error_is_written_to_the_selected_log(self):
        """Ошибка настройки записывается, даже если камера ещё не открывалась."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "trial.log"
            console_out = io.StringIO()
            console_err = io.StringIO()
            with redirect_stdout(console_out), redirect_stderr(console_err):
                with patch("scripts.live_drone_trial.default_log_path", return_value=log_path):
                    result = main(["--model", "/missing/model.onnx", "--source", "0"])

            self.assertEqual(result, 2)
            self.assertIn("Журнал сохранён:", console_out.getvalue())
            self.assertIn("ONNX-модель не найдена", console_err.getvalue())
            self.assertIn("ONNX-модель не найдена", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
