"""Проверки параметров отдельного живого испытания модели."""

import io
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.compare_video_trials import generate_tiles
from scripts.live_drone_trial import (
    CameraInfo,
    build_parser,
    candidate_inference_request,
    choose_camera,
    compose_display_frame,
    console_status,
    configure_capture_device,
    display_poll_delay_ms,
    draw_status,
    fit_window,
    main,
    move_detections_to_frame,
    parse_source,
    resolve_run_args,
    run_viewer,
    should_emit_console_status,
    TeeOutput,
    tee_output,
    validate_args,
)
from src.motion_compensated_detection import MotionBox, MotionResult
from src.realtime import Detection


class LiveDroneTrialTests(unittest.TestCase):
    """Проверяет выбор камеры и валидацию CLI без физического оборудования."""

    def test_sky_profile_restores_owner_confirmed_settings(self):
        """Профиль sky загружает закреплённую рабочую модель и параметры."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))

        self.assertEqual(
            args.source,
            "/dev/v4l/by-id/usb-UltraSemi_USB3_Video_73908951-video-index0",
        )
        self.assertEqual(args.model, Path(__file__).resolve().parents[1] / "models/anti_uav_fast_320_int8_pc.onnx")
        self.assertEqual(args.confidence, 0.40)
        self.assertEqual(args.min_confidence, 0.05)
        self.assertEqual(args.tile_size, 640)
        self.assertEqual(args.tile_overlap, 0.20)
        self.assertEqual(args.threads, 2)
        self.assertFalse(args.full_frame)
        self.assertEqual(args.overlay["box"]["thickness"], 1)
        self.assertEqual(args.overlay["box"]["scale_x"], 1.0)
        self.assertEqual(args.overlay["box"]["scale_y"], 1.0)
        self.assertEqual(args.overlay["near_threshold_margin"], 0.10)
        self.assertEqual(args.overlay["near_threshold_color_rgb"], [255, 165, 0])
        self.assertEqual(args.overlay["label"]["font_scale"], 0.55)
        self.assertEqual(args.capture, {"width": 1280, "height": 960, "fps": 60, "buffer_size": 1})
        self.assertEqual(args.display["window_width"], 1480)
        self.assertEqual(args.display["window_height"], 1120)
        self.assertEqual(args.display["max_frame_width"], 1440)
        self.assertEqual(args.display["max_frame_height"], 1080)
        self.assertTrue(args.display["scale_up"])
        self.assertEqual(args.diagnostics["console_status_interval_ms"], 1000)
        self.assertTrue(args.two_stage["enabled"])
        self.assertEqual(args.two_stage["candidate_crop_size"], 640)
        self.assertEqual(args.two_stage["full_scan_interval_ms"], 1500)
        self.assertEqual(args.two_stage["max_detection_age_ms"], 300)
        self.assertEqual(args.status_panel["position"], "bottom")
        self.assertEqual(args.status_panel["height"], 28)
        self.assertEqual(args.status_panel["font_scale"], 0.42)

    def test_profile_rejects_zero_frame_thickness(self):
        """Нулевая толщина рамки отклоняется до открытия камеры."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        args.overlay["box"]["thickness"] = 0

        with self.assertRaisesRegex(ValueError, "overlay.box.thickness"):
            validate_args(args)

    def test_console_status_is_throttled_and_can_be_disabled(self):
        """Статистика не пишет чаще интервала, а ноль отключает её полностью."""
        self.assertTrue(should_emit_console_status(10.0, None, 1000))
        self.assertFalse(should_emit_console_status(10.5, 10.0, 1000))
        self.assertTrue(should_emit_console_status(11.0, 10.0, 1000))
        self.assertFalse(should_emit_console_status(11.0, None, 0))

    def test_window_polling_matches_capture_rate_and_caps_at_60_fps(self):
        """Опрос окна не крутится быстрее видеовхода и не превышает 60 Гц."""
        self.assertEqual(display_poll_delay_ms(1, 60), 17)
        self.assertEqual(display_poll_delay_ms(1, 30), 34)
        self.assertEqual(display_poll_delay_ms(1, 0), 17)
        self.assertEqual(display_poll_delay_ms(25, 60), 25)

    def test_window_keeps_rendering_while_inference_is_blocked(self):
        """Окно продолжает обновляться, пока отдельный worker занят распознаванием."""
        inference_started = threading.Event()
        allow_inference_to_finish = threading.Event()
        blocked_render_count = 0
        wait_count = 0
        frame = np.zeros((32, 48, 3), dtype=np.uint8)

        class FakeCapture:
            """Имитирует открытый USB-захват без изменения его параметров."""

            def isOpened(self):
                """Сообщает, что тестовый источник открыт."""
                return True

            def release(self):
                """Заглушка освобождения тестового устройства."""
                return None

        class FakeCamera:
            """Возвращает новый тестовый кадр при каждом запросе GUI."""

            def __init__(self, capture):
                self.capture = capture

            def latest(self, timeout=2.0):
                """Возвращает новый объект кадра, как будто захват идёт непрерывно."""
                return frame.copy()

            def close(self):
                """Заглушка остановки потока захвата."""
                return None

        def delayed_inference(detector, image, tile_size, overlap):
            """Удерживает инференс, пока тест проверяет независимую отрисовку."""
            inference_started.set()
            if not allow_inference_to_finish.wait(timeout=2.0):
                raise TimeoutError("Тест не освободил фоновое распознавание")
            return (), 100.0, 0.75

        def pump_window(delay_ms):
            """Ждёт старта распознавания, затем разрешает выводить кадры."""
            nonlocal blocked_render_count, wait_count
            wait_count += 1
            if wait_count == 1:
                self.assertTrue(inference_started.wait(timeout=1.0))
            if not allow_inference_to_finish.is_set():
                blocked_render_count += 1
                if blocked_render_count >= 5:
                    allow_inference_to_finish.set()
            return ord("q") if wait_count >= 8 else -1

        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        args.model = Path(__file__)
        with (
            patch("scripts.live_drone_trial.load_model_dependencies"),
            patch("scripts.live_drone_trial.model_input_size", return_value=320),
            patch("scripts.live_drone_trial.OnnxDetector", return_value=object()),
            patch("scripts.live_drone_trial.cv2", cv2),
            patch.object(cv2, "VideoCapture", return_value=FakeCapture()),
            patch.object(cv2, "namedWindow"),
            patch.object(cv2, "resizeWindow"),
            patch.object(cv2, "imshow") as show_frame,
            patch.object(cv2, "waitKey", side_effect=pump_window),
            patch.object(cv2, "destroyAllWindows"),
            patch("scripts.live_drone_trial.camera_api", return_value=cv2.CAP_V4L2),
            patch("scripts.live_drone_trial.configure_capture_device", return_value={"fps": 60.0}),
            patch("scripts.live_drone_trial.LatestFrameCapture", FakeCamera),
            patch("scripts.live_drone_trial.infer_tiles", side_effect=delayed_inference),
            patch("scripts.live_drone_trial.draw_overlay", side_effect=lambda image, *args: image),
            patch("scripts.live_drone_trial.compose_display_frame", side_effect=lambda image, *args: image),
            patch("scripts.live_drone_trial.log_diagnostic"),
        ):
            self.assertEqual(run_viewer(args, 0), 0)

        self.assertGreaterEqual(blocked_render_count, 5)
        self.assertEqual(show_frame.call_count, 8)

    def test_diagnostics_can_be_written_to_log_without_console_output(self):
        """Подробная диагностика уходит в файл, не попадая в консоль."""
        console = io.StringIO()
        log_file = io.StringIO()
        output = TeeOutput(console, log_file, threading.RLock())

        output.write_log_only("Параметры: capture=1280x960@60\n")

        self.assertEqual(console.getvalue(), "")
        self.assertEqual(log_file.getvalue(), "Параметры: capture=1280x960@60\n")

    def test_sky_profile_uses_four_tiles_for_small_drone_detail(self):
        """Профиль sky делит HD-кадр на четыре фрагмента, не уменьшая маленький дрон."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        frame = np.zeros((args.capture["height"], args.capture["width"], 3), dtype=np.uint8)

        tiles = generate_tiles(frame, args.tile_size, args.tile_overlap)

        self.assertEqual(len(tiles), 4)
        self.assertLess(tiles[0][2].shape[0], args.capture["height"])
        self.assertLess(tiles[0][2].shape[1], args.capture["width"])

    def test_profile_rejects_invalid_capture_fps_and_font_with_clear_error(self):
        """Ошибочные значения JSON получают понятную ошибку до запуска камеры."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        args.capture["fps"] = "60"
        with self.assertRaisesRegex(ValueError, "capture.fps"):
            validate_args(args)

        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        args.overlay["label"]["font"] = "NOT_AN_OPENCV_FONT"
        with self.assertRaisesRegex(ValueError, "overlay.label.font"):
            validate_args(args)

    def test_profile_rejects_inconsistent_two_stage_background_settings(self):
        """Профиль не запускается, если RANSAC требует больше точек, чем разрешено."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        args.two_stage["motion_min_background_features"] = 700

        with self.assertRaisesRegex(ValueError, "motion_min_background_features"):
            validate_args(args)

    def test_candidate_crop_preserves_full_frame_coordinates(self):
        """Быстрый вырез вокруг кандидата не теряет его смещение в исходном кадре."""
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        result = MotionResult(
            roi=(20, 10, 60, 50),
            mask=np.zeros((50, 60), dtype=np.uint8),
            boxes=(),
            background_shift=(0.0, 0.0),
            background_inliers=8,
            compensated=True,
            status="Фон скомпенсирован",
            luma_pixels=0,
            chroma_pixels=0,
        )
        candidate = MotionBox(10, 20, 20, 30, 100, 20.0, 4.0, 8.0)
        settings = dict(args.two_stage)
        settings["candidate_crop_size"] = 50

        request = candidate_inference_request(frame, result, candidate, settings, 12.5)

        self.assertEqual(request.mode, "candidate")
        self.assertEqual(request.frame.shape, (50, 50, 3))
        self.assertEqual((request.origin_x, request.origin_y), (10, 10))
        self.assertEqual(request.frame_shape, frame.shape)
        moved = move_detections_to_frame(
            (Detection(12, 18, 20, 26, "FPV-DRON", 0.8),),
            request.origin_x,
            request.origin_y,
            frame.shape,
        )
        self.assertEqual(moved[0], Detection(22, 28, 30, 36, "FPV-DRON", 0.8))

    def test_capture_dimensions_and_fps_are_requested_from_config(self):
        """Размер, частота кадров и буфер применяются к устройству захвата."""
        class FakeCapture:
            """Имитирует VideoCapture для проверки запрошенных режимов."""

            def __init__(self):
                self.values = {}

            def set(self, prop, value):
                """Сохраняет свойство и сообщает, что драйвер принял запрос."""
                self.values[prop] = value
                return True

            def get(self, prop):
                """Возвращает ранее сохранённое значение свойства."""
                return self.values.get(prop, 0)

        capture = FakeCapture()
        settings = {"width": 640, "height": 480, "fps": 30, "buffer_size": 1}
        with patch("scripts.live_drone_trial.cv2", cv2):
            actual = configure_capture_device(capture, settings)

        self.assertEqual(capture.values[cv2.CAP_PROP_FRAME_WIDTH], 640)
        self.assertEqual(capture.values[cv2.CAP_PROP_FRAME_HEIGHT], 480)
        self.assertEqual(capture.values[cv2.CAP_PROP_FPS], 30)
        self.assertEqual(capture.values[cv2.CAP_PROP_BUFFERSIZE], 1)
        self.assertEqual(actual, {"width": 640, "height": 480, "fps": 30.0})

    def test_status_panel_shows_only_cpu_temperature_and_confidence_threshold(self):
        """Нижняя компактная полоса содержит все четыре запрошенных показателя."""
        class FakeFrame:
            """Минимальный кадр для проверки параметров отрисовки."""

            shape = (480, 640, 3)

            def copy(self):
                """Возвращает сам объект, поскольку пиксели в этом тесте не нужны."""
                return self

        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))
        with (
            patch("scripts.live_drone_trial.cv2", cv2),
            patch.object(cv2, "rectangle") as rectangle,
            patch.object(cv2, "putText") as put_text,
        ):
            draw_status(FakeFrame(), args, "TEMP 51.2C CPU 24.0%", 0.73)

        self.assertEqual(rectangle.call_args.args[1], (0, 452))
        self.assertEqual(rectangle.call_args.args[2], (639, 479))
        self.assertEqual(rectangle.call_args.args[3], (35, 35, 35))
        self.assertEqual(put_text.call_count, 1)
        status_text = put_text.call_args.args
        self.assertEqual(status_text[1], "CPU 24.0% | TEMP 51.2C | CONF 73% | THR 40%")
        self.assertEqual(status_text[2], (12, 471))
        self.assertEqual(status_text[3], cv2.FONT_HERSHEY_SIMPLEX)
        self.assertEqual(status_text[4], 0.42)
        self.assertEqual(status_text[5], (230, 230, 230))
        self.assertEqual(status_text[6], 1)

    def test_display_scales_camera_frame_up_with_suitable_interpolation(self):
        """Увеличение кадра сохраняет пропорции и использует сглаженную интерполяцию."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        unchanged = fit_window(frame, max_width=1280, max_height=720)
        self.assertIs(unchanged, frame)

        with (
            patch("scripts.live_drone_trial.cv2", cv2),
            patch.object(cv2, "resize") as resize,
        ):
            fit_window(frame, max_width=1440, max_height=1080, scale_up=True)

        self.assertEqual(resize.call_args.args[1], (1440, 1080))
        self.assertEqual(resize.call_args.kwargs["interpolation"], cv2.INTER_LINEAR)

    def test_wide_status_canvas_keeps_square_video_square_and_centered(self):
        """Широкая строка статуса не растягивает квадратный видеокадр."""
        frame = np.full((640, 640, 3), 7, dtype=np.uint8)
        args = resolve_run_args(build_parser().parse_args(["--profile", "sky"]))

        with (
            patch("scripts.live_drone_trial.cv2", cv2),
            patch.object(cv2, "rectangle") as rectangle,
            patch.object(cv2, "putText"),
        ):
            result = compose_display_frame(frame, args, "TEMP 51.2C CPU 24.0%", 0.73)

        self.assertEqual(result.shape, (1120, 1480, 3))
        self.assertTrue(np.all(result[6:1086, 200:1280] == 7))
        self.assertEqual(rectangle.call_args.args[1], (0, 1092))
        self.assertEqual(rectangle.call_args.args[2], (1479, 1119))

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
            self.assertIn("Журнал:", console_out.getvalue())
            self.assertNotIn("Параметры:", console_out.getvalue())
            self.assertIn("ONNX-модель не найдена", console_err.getvalue())
            self.assertIn("ONNX-модель не найдена", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
