"""Проверки нативного Qt-редактора конфигураций VideoT16."""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QComboBox, QGroupBox, QLabel, QMessageBox, QSlider

from config.editor import CONFIG_DIR, PROJECT_DIR, ConfigEditorWindow, StaleConfigError, apply_defaults, available_capture_modes, build_options, launcher_command, list_config_files, numeric_spec, read_config, save_config
from config.runtime_settings import DEFAULT_SETTINGS


class ConfigEditorTests(unittest.TestCase):
    """Проверяет безопасное чтение, изменение и сохранение JSON-настроек."""

    @classmethod
    def setUpClass(cls) -> None:
        """Создаёт одно Qt-приложение для всех тестов без физического экрана."""
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        """Создаёт тестовый проект с одной моделью и одной конфигурацией."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        self.config_dir = self.project_dir / "config"
        self.models_dir = self.project_dir / "models"
        self.config_dir.mkdir()
        self.models_dir.mkdir()
        (self.models_dir / "drone.onnx").write_bytes(b"test model")
        self.config_path = self.config_dir / "settings.json"
        self.original_data = {
            "version": 1,
            "capture": {"width": 1280, "height": 720, "fps": 60, "buffer_size": 1},
            "status_panel": {"height": 28},
            "detection": {
                "model_path": "models/drone.onnx",
                "camera_source": "usb",
                "confidence": 0.4,
                "inference_size": 320,
                "enabled": True,
                "color_rgb": [0, 255, 0],
                "unknown_value": "только чтение",
            }
        }
        self.original = json.dumps(self.original_data, ensure_ascii=False, indent=2) + "\n"
        self.config_path.write_text(self.original, encoding="utf-8")
        self.window: ConfigEditorWindow | None = None

    def tearDown(self) -> None:
        """Закрывает временное Qt-окно и удаляет тестовые файлы."""
        if self.window is not None:
            self.window.close()
            self.application.processEvents()
        self.temp_dir.cleanup()

    def test_lists_only_plain_top_level_json(self) -> None:
        """Скрывает вложенные, скрытые и не-JSON файлы из списка выбора."""
        (self.config_dir / "settings.json.bak").write_text("backup", encoding="utf-8")
        nested = self.config_dir / "nested"
        nested.mkdir()
        (nested / "other.json").write_text("{}", encoding="utf-8")
        (self.config_dir / ".hidden.json").write_text("{}", encoding="utf-8")
        try:
            (self.config_dir / "link.json").symlink_to(self.config_path)
        except OSError:
            pass
        self.assertEqual(list_config_files(self.config_dir), ["settings.json"])

    def test_options_include_model_camera_and_standard_sizes(self) -> None:
        """Готовит видимые мышью варианты модели, камеры и размера инференса."""
        options = build_options(self.project_dir, {"model_path": "models/drone.onnx", "camera_device": "auto"})
        self.assertIn({"value": "models/drone.onnx", "label": "models/drone.onnx"}, options["models"])
        self.assertIn({"value": "auto", "label": "Автовыбор"}, options["cameras"])
        self.assertIn(320, options["sizes"])

    def test_motion_trial_config_routes_to_separate_launcher_only_on_linux(self) -> None:
        """Новый Linux JSON запускает lcam3, а штатные файлы и Windows не меняют маршрут."""
        launcher, command = launcher_command(self.project_dir, "motion_trial.json", platform="posix")
        self.assertEqual(launcher, self.project_dir / "lcam3")
        self.assertEqual(command, [sys.executable, str(launcher)])

        launcher, command = launcher_command(self.project_dir, "live_drone_profiles.json", platform="posix")
        self.assertEqual(launcher, self.project_dir / "lcam")
        self.assertEqual(command, [str(launcher), "--v2", "--run"])

        launcher, command = launcher_command(self.project_dir, "runtime_settings.json", platform="posix")
        self.assertEqual(launcher, self.project_dir / "lcam")
        self.assertEqual(command, [str(launcher)])

        launcher, command = launcher_command(self.project_dir, "motion_trial.json", platform="nt")
        self.assertEqual(launcher, self.project_dir / "start.py")
        self.assertEqual(command, [sys.executable, str(launcher)])

    def test_motion_trial_menu_launches_python_entrypoint(self) -> None:
        """Кнопка запуска выбранного JSON вызывает отдельный прототип через Python."""
        (self.config_dir / "motion_trial.json").write_text(
            json.dumps({"capture": {"camera_device": "auto"}}), encoding="utf-8"
        )
        launcher = self.project_dir / "lcam3"
        launcher.write_text("# test entrypoint\n", encoding="utf-8")
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.window.file_combo.setCurrentText("motion_trial.json")
        self.application.processEvents()
        fake_process = Mock()
        fake_process.poll.return_value = None

        with patch("config.editor.subprocess.Popen", return_value=fake_process) as start_process:
            self.window.launch_button.click()

        self.assertEqual(start_process.call_args.args[0], [sys.executable, str(launcher)])

    def test_v2_menu_opens_selected_profile_and_model_change_keeps_camera_mode(self) -> None:
        """Открывает V2-профиль сразу и при смене модели сохраняет выбранный режим камеры."""
        v2_path = self.config_dir / "live_drone_profiles.json"
        v2_path.write_text(json.dumps({
            "version": 1,
            "profiles": {"sky": {
                "settings": {"model": "models/first.onnx", "tile_size": 640},
                "capture": {"width": 1280, "height": 960, "fps": 60, "buffer_size": 1},
            }},
        }), encoding="utf-8")
        options = {
            "models": [
                {"value": "models/first.onnx", "label": "first — вход 320×320"},
                {"value": "models/second.onnx", "label": "second — вход 640×640"},
            ],
            "model_input_sizes": {"models/first.onnx": 320, "models/second.onnx": 640},
            "capture_modes": [{"width": 1280, "height": 960, "fps": 60, "label": "1280×960 @ 60 кадр/с"}],
            "cameras": [{"value": "auto", "label": "Автовыбор"}],
            "camera_sources": [], "pixel_formats": [], "output_devices": [],
            "fonts": [], "positions": [], "sizes": [320, 640],
        }
        with patch("config.editor.build_options", return_value=options):
            self.window = ConfigEditorWindow(self.project_dir, self.config_dir, "live_drone_profiles.json")
            self.application.processEvents()

            self.assertEqual(self.window.file_combo.currentText(), "live_drone_profiles.json")
            model_combo = next(
                combo for combo in self.window.findChildren(QComboBox)
                if combo.findData("models/first.onnx") >= 0
            )
            model_combo.setCurrentIndex(model_combo.findData("models/second.onnx"))
            self.application.processEvents()

        profile = self.window.current_data["profiles"]["sky"]
        self.assertEqual(profile["settings"]["model"], "models/second.onnx")
        self.assertEqual(profile["settings"]["tile_size"], 640)
        self.assertEqual((profile["capture"]["width"], profile["capture"]["height"], profile["capture"]["fps"]), (1280, 960, 60))

    def test_v2_model_selector_does_not_offer_incompatible_current_model(self) -> None:
        """Скрывает PT/неподтверждённые модели из V2-списка, оставляя выбор валидной ONNX."""
        v2_path = self.config_dir / "live_drone_profiles.json"
        v2_path.write_text(json.dumps({
            "version": 1,
            "profiles": {"sky": {
                "settings": {"model": "models/legacy.pt", "tile_size": 640},
                "capture": {"width": 1280, "height": 960, "fps": 60, "buffer_size": 1},
            }},
        }), encoding="utf-8")
        options = {
            "models": [{"value": "models/valid.onnx", "label": "valid — вход 320×320"}],
            "model_input_sizes": {"models/valid.onnx": 320},
            "capture_modes": [{"width": 1280, "height": 960, "fps": 60, "label": "1280×960 @ 60 кадр/с"}],
            "cameras": [], "camera_sources": [], "pixel_formats": [], "output_devices": [],
            "fonts": [], "positions": [], "sizes": [320, 640],
        }
        with patch("config.editor.build_options", return_value=options):
            self.window = ConfigEditorWindow(self.project_dir, self.config_dir, "live_drone_profiles.json")
            self.application.processEvents()

        model_combo = next(
            combo for combo in self.window.findChildren(QComboBox)
            if combo.count() and combo.itemData(combo.count() - 1) == "models/valid.onnx"
        )
        self.assertIsNone(model_combo.currentData())
        self.assertNotIn("models/legacy.pt", [model_combo.itemData(index) for index in range(model_combo.count())])

    def test_v2_capture_mode_list_is_read_from_selected_device(self) -> None:
        """Разбирает только форматы и FPS, которые V4L2 перечислил у камеры."""
        output = """[0]: 'MJPG'\n  Size: Discrete 1280x720\n    Interval: Discrete 0.017s (60.000 fps)\n    Interval: Discrete 0.033s (30.000 fps)\n"""
        with patch("config.editor.subprocess.run", return_value=Mock(returncode=0, stdout=output)) as run:
            modes = available_capture_modes("/dev/video4")

        run.assert_called_once()
        self.assertEqual(
            modes,
            [
                {"width": 1280, "height": 720, "fps": 60, "label": "1280×720 @ 60 кадр/с"},
                {"width": 1280, "height": 720, "fps": 30, "label": "1280×720 @ 30 кадр/с"},
            ],
        )

    def test_v2_numeric_parameters_have_safe_ranges(self) -> None:
        """Показывает основные параметры буферизованного вывода V2 мышиными ползунками."""
        self.assertEqual(numeric_spec("output_delay_ms", 2200), (100, 5000, 100, 1.0, " мс"))
        self.assertEqual(numeric_spec("output_buffer_max_mb", 512), (64, 2048, 32, 1.0, " МиБ"))
        self.assertEqual(numeric_spec("box_hold_ms", 1000), (0, 5000, 100, 1.0, " мс"))
        self.assertEqual(numeric_spec("submission_interval_frames", 1), (1, 120, 1, 1.0, " кадр."))

    def test_motion_trial_json_exposes_algorithm_values_as_mouse_controls(self) -> None:
        """Все числовые настройки алгоритма появляются ползунками в том же окне Qt."""
        motion_path = self.config_dir / "motion_trial.json"
        motion_path.write_text(
            (CONFIG_DIR / "motion_trial.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        data = json.loads(motion_path.read_text(encoding="utf-8"))

        def count_sliders(value: object, path: tuple[str, ...] = ()) -> int:
            """Считает числовые настройки, которым редактор обязан дать ползунок."""
            if isinstance(value, dict):
                return sum(count_sliders(child, (*path, key)) for key, child in value.items())
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(numeric_spec(path[-1], value, path) is not None)
            return 0

        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()

        self.assertEqual(self.window.current_name, "motion_trial.json")
        self.assertEqual(len(self.window.findChildren(QSlider)), count_sliders(data))
        self.assertGreaterEqual(len(self.window.findChildren(QGroupBox)), 6)
        self.assertFalse(self.window._is_dirty())

    def test_nested_sections_render_without_nested_frames(self) -> None:
        """Сохраняет структуру настроек заголовками, не вкладывая рамки друг в друга."""
        nested_data = {
            "profiles": {
                "sky": {
                    "settings": {"model": "models/drone.onnx", "confidence": 0.4},
                    "capture": {"width": 1280, "height": 720},
                }
            },
            "version": 1,
        }
        self.config_path.write_text(json.dumps(nested_data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()

        groups = self.window.findChildren(QGroupBox)
        self.assertEqual([group.title() for group in groups], ["Профили"])
        labels = [label.text() for label in self.window.findChildren(QLabel)]
        self.assertIn("Профиль: sky", labels)
        self.assertIn("Настройки распознавания", labels)
        self.assertIn("Захват видео", labels)

    def test_launch_button_starts_lcam_without_opening_camera_in_test(self) -> None:
        """Проверяет запуск через кнопку, закрытие настроек и отсутствие доступа к камере в тесте."""
        launcher = self.project_dir / "lcam"
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.window.show()
        self.application.processEvents()
        self.assertTrue(self.window.isVisible())
        fake_process = Mock()
        fake_process.poll.return_value = None

        with patch("config.editor.subprocess.Popen", return_value=fake_process) as start_process:
            self.window.launch_button.click()

        start_process.assert_called_once()
        command = start_process.call_args.args[0]
        self.assertEqual(command, [str(launcher)])
        self.assertEqual(start_process.call_args.kwargs["cwd"], self.project_dir)
        self.assertTrue(start_process.call_args.kwargs["start_new_session"])
        self.application.processEvents()
        self.assertFalse(self.window.isVisible())

    def test_launch_failure_keeps_settings_open(self) -> None:
        """Не закрывает настройки и сообщает об ошибке, если процесс не стартовал."""
        launcher = self.project_dir / "lcam"
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.window.show()
        self.application.processEvents()

        with patch("config.editor.subprocess.Popen", side_effect=OSError("ошибка запуска")) as start_process:
            with patch("config.editor.QMessageBox.critical") as show_error:
                self.window.launch_button.click()
                self.application.processEvents()

        start_process.assert_called_once()
        show_error.assert_called_once()
        self.assertTrue(self.window.isVisible())

    def test_save_atomically_updates_file_without_backup(self) -> None:
        """Сохраняет корректный JSON и не создаёт резервных файлов."""
        _, revision = read_config(self.config_dir, "settings.json")
        updated = json.dumps({"detection": {"confidence": 0.55}}, indent=2) + "\n"
        new_revision = save_config(self.config_dir, "settings.json", updated, revision)
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), updated)
        self.assertNotEqual(new_revision, revision)
        self.assertEqual(list(self.config_dir.glob("*")), [self.config_path])

    def test_invalid_json_and_stale_revision_do_not_overwrite(self) -> None:
        """Отклоняет ошибочные и устаревшие данные до замены исходного файла."""
        _, revision = read_config(self.config_dir, "settings.json")
        with self.assertRaises(ValueError):
            save_config(self.config_dir, "settings.json", '{"bad": }', revision)
        self.config_path.write_text('{"detection": {"confidence": 0.45}}\n', encoding="utf-8")
        with self.assertRaises(StaleConfigError):
            save_config(self.config_dir, "settings.json", '{"detection": {"confidence": 0.9}}\n', revision)
        self.assertEqual(json.loads(self.config_path.read_text(encoding="utf-8"))["detection"]["confidence"], 0.45)

    def test_path_traversal_and_non_json_are_rejected(self) -> None:
        """Запрещает выбирать не-JSON-файлы и выходить за пределы config/."""
        for name in ("../outside.json", "settings.json.bak", ".hidden.json", "nested/other.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_config(self.config_dir, name)

    def test_window_is_native_and_renders_mouse_controls(self) -> None:
        """Открывает нативное Qt-окно и проверяет список файла и ползунки."""
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.window.show()
        self.application.processEvents()
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.window.windowTitle(), "VideoT16 — настройки")
        self.assertEqual(self.window.file_combo.currentText(), "settings.json")
        self.assertGreaterEqual(len(self.window.findChildren(QSlider)), 1)
        self.assertFalse(self.window.save_button.isEnabled())

    def test_opening_config_does_not_change_any_value(self) -> None:
        """Открытие формы сохраняет исходные типы чисел и малые размеры панелей."""
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()
        self.assertFalse(self.window._is_dirty())
        self.assertEqual(self.config_path.read_text(encoding="utf-8"), self.original)
        self.assertEqual(self.window.current_data["status_panel"]["height"], 28)

    def test_opening_config_preserves_slider_precision_without_marking_dirty(self) -> None:
        """Построение ползунка не округляет точное JSON-значение до действия пользователя."""
        self.original_data["detection"]["confidence"] = 0.3333
        self.original = json.dumps(self.original_data, ensure_ascii=False, indent=2) + "\n"
        self.config_path.write_text(self.original, encoding="utf-8")
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()

        self.assertFalse(self.window._is_dirty())
        self.assertEqual(self.window.current_data["detection"]["confidence"], 0.3333)

    def test_default_button_uses_program_defaults_and_waits_for_save(self) -> None:
        """Кнопка подставляет программные значения, но не записывает файл до сохранения."""
        initial_data = copy.deepcopy(DEFAULT_SETTINGS)
        initial_data["desktop"]["confidence_percent"] = 30
        initial_data["desktop"]["camera_fps"] = 60
        initial_data["custom_extension"] = {"preserve": True}
        initial_content = json.dumps(initial_data, ensure_ascii=False, indent=2) + "\n"
        runtime_path = self.config_dir / "runtime_settings.json"
        runtime_path.write_text(initial_content, encoding="utf-8")

        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()
        self.assertEqual(self.window.file_combo.currentText(), "runtime_settings.json")
        self.assertTrue(self.window.defaults_button.isEnabled())

        self.window.defaults_button.click()
        self.application.processEvents()

        self.assertEqual(self.window.current_data["desktop"]["confidence_percent"], 50)
        self.assertEqual(self.window.current_data["desktop"]["camera_fps"], 55)
        self.assertEqual(self.window.current_data["custom_extension"], {"preserve": True})
        self.assertTrue(self.window.save_button.isEnabled())
        self.assertEqual(runtime_path.read_text(encoding="utf-8"), initial_content)

        self.window.save_button.click()
        self.application.processEvents()
        saved = json.loads(runtime_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["desktop"]["confidence_percent"], 50)
        self.assertEqual(saved["desktop"]["camera_fps"], 55)
        self.assertEqual(saved["custom_extension"], {"preserve": True})

    def test_runtime_settings_opens_first_and_profiles_have_no_default_button(self) -> None:
        """При наличии главной конфигурации сразу открывает её, а для профилей отключает сброс."""
        (self.config_dir / "runtime_settings.json").write_text(
            json.dumps(DEFAULT_SETTINGS, ensure_ascii=False), encoding="utf-8"
        )
        (self.config_dir / "live_drone_profiles.json").write_text(
            json.dumps({"version": 1, "profiles": {}}, ensure_ascii=False), encoding="utf-8"
        )
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()

        self.assertEqual(self.window.file_combo.currentText(), "runtime_settings.json")
        self.assertTrue(self.window.defaults_button.isEnabled())
        self.window.file_combo.setCurrentText("live_drone_profiles.json")
        self.application.processEvents()
        self.assertFalse(self.window.defaults_button.isEnabled())

    def test_default_button_warns_before_enabling_video_outputs(self) -> None:
        """Предупреждает об изменении выключенных видеовыходов и уважает отказ владельца."""
        initial_data = copy.deepcopy(DEFAULT_SETTINGS)
        initial_data["video_output"]["j7_enabled"] = False
        initial_data["video_output"]["j7_overlay_enabled"] = False
        initial_data["video_output"]["flight_controller_osd_enabled"] = False
        runtime_path = self.config_dir / "runtime_settings.json"
        runtime_path.write_text(json.dumps(initial_data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()

        with patch("config.editor.QMessageBox.warning", return_value=QMessageBox.StandardButton.Cancel) as warn:
            self.window.defaults_button.click()

        warn.assert_called_once()
        self.assertFalse(self.window._is_dirty())
        self.assertEqual(self.window.current_data["video_output"]["j7_enabled"], False)
        self.assertEqual(runtime_path.read_text(encoding="utf-8"), json.dumps(initial_data, ensure_ascii=False, indent=2))

        with patch("config.editor.QMessageBox.warning", return_value=QMessageBox.StandardButton.Yes):
            self.window.defaults_button.click()

        self.assertEqual(self.window.current_data["video_output"]["j7_enabled"], True)
        self.assertEqual(self.window.current_data["video_output"]["j7_overlay_enabled"], True)
        self.assertEqual(self.window.current_data["video_output"]["flight_controller_osd_enabled"], True)
        self.assertTrue(self.window.save_button.isEnabled())

    def test_apply_defaults_preserves_unknown_fields_and_does_not_mutate_source(self) -> None:
        """Возвращает копию с дефолтами, не меняя исходные данные и неизвестные параметры."""
        source = {"desktop": {"confidence_percent": 30}, "extra": "оставить"}
        defaults = {"desktop": {"confidence_percent": 50, "camera_fps": 55}, "new_section": {"enabled": True}}

        restored = apply_defaults(source, defaults)

        self.assertEqual(restored, {
            "desktop": {"confidence_percent": 50, "camera_fps": 55},
            "extra": "оставить",
            "new_section": {"enabled": True},
        })
        self.assertEqual(source, {"desktop": {"confidence_percent": 30}, "extra": "оставить"})

    def test_close_saves_unsaved_mouse_changes_and_closes_without_modal_dialog(self) -> None:
        """Крестик закрывает окно и сохраняет внесённые мышью изменения без лишнего окна."""
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.window.show()
        self.application.processEvents()
        detection_group = next(group for group in self.window.findChildren(QGroupBox) if group.title() == "Распознавание")
        slider = detection_group.findChild(QSlider)
        self.assertIsNotNone(slider)
        slider.setValue(55)
        self.application.processEvents()

        with patch("config.editor.QMessageBox.question") as ask_question:
            ask_question.return_value = QMessageBox.StandardButton.Save
            closed = self.window.close()
            self.application.processEvents()

        self.assertTrue(closed)
        self.assertFalse(self.window.isVisible())
        ask_question.assert_not_called()
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["detection"]["confidence"], 0.55)

    def test_mouse_slider_changes_and_saves_threshold(self) -> None:
        """Проверяет полный путь от движения ползунка до изменения файла."""
        self.window = ConfigEditorWindow(self.project_dir, self.config_dir)
        self.application.processEvents()
        detection_group = next(group for group in self.window.findChildren(QGroupBox) if group.title() == "Распознавание")
        slider = detection_group.findChild(QSlider)
        self.assertIsNotNone(slider)
        slider.setValue(55)
        self.application.processEvents()
        self.assertTrue(self.window.save_button.isEnabled())
        self.window.save_button.click()
        self.application.processEvents()
        saved = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["detection"]["confidence"], 0.55)
        self.assertFalse(self.window.save_button.isEnabled())
        self.assertEqual(self.window.original_data, copy.deepcopy(saved))
        self.assertEqual(list(self.config_dir.glob("*")), [self.config_path])


if __name__ == "__main__":
    unittest.main()
