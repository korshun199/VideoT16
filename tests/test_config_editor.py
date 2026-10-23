"""Проверки нативного Qt-редактора конфигураций VideoT16."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QGroupBox, QLabel, QMessageBox, QSlider

from config.editor import ConfigEditorWindow, StaleConfigError, build_options, list_config_files, read_config, save_config


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
