"""Проверяет рекомендуемый профиль в меню нового конвейера."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video_cap_buf"))

from PySide6.QtWidgets import QApplication

from capture import validate_settings
from editor import DEFAULT_SETTINGS, FIELD_SPECS, SettingsEditor


class DefaultSettingsButtonTests(unittest.TestCase):
    """Проверяет применение профиля, сохранение источника и отсутствие автозаписи."""

    @classmethod
    def setUpClass(cls) -> None:
        """Создаёт одно Qt-приложение для тестов окна в безэкранном режиме."""
        cls.application = QApplication.instance() or QApplication([])

    def test_profile_covers_all_editable_settings_and_is_valid(self) -> None:
        """Проверяет, что профиль задаёт каждое поле меню, кроме USB-пути."""
        editable_keys = {spec.key for spec in FIELD_SPECS}
        self.assertEqual(set(DEFAULT_SETTINGS), editable_keys - {"source"})
        profile = {"source": "/dev/video-test", **deepcopy(DEFAULT_SETTINGS)}

        validate_settings(profile)

    def test_button_applies_profile_without_changing_source_or_store(self) -> None:
        """Кнопка меняет форму, сохраняя источник и исходный документ магазина."""
        settings_path = Path(__file__).resolve().parents[1] / "video_cap_buf" / "settings.json"
        original = json.loads(settings_path.read_text(encoding="utf-8"))
        original["source"] = "/dev/video-custom"
        original["width"] = 640
        original["height"] = 480
        original["fps"] = 20
        original["custom_note"] = {"keep": True}
        stored_before = deepcopy(original)
        store = SimpleNamespace(document=stored_before)
        editor = SettingsEditor(store, [])

        editor.defaults_button.click()

        expected = deepcopy(original)
        expected.update(deepcopy(DEFAULT_SETTINGS))
        expected["source"] = original["source"]
        self.assertEqual(editor.defaults_button.text(), "По умолчанию")
        self.assertEqual(editor.working_document, expected)
        self.assertEqual(store.document, stored_before)
        self.assertTrue(editor.save_button.isEnabled())
        self.assertIn("Сохранить", editor.status_label.text())

    def test_applying_profile_when_already_active_does_not_mark_dirty(self) -> None:
        """Повторное применение тех же значений не включает сохранение."""
        settings_path = Path(__file__).resolve().parents[1] / "video_cap_buf" / "settings.json"
        current = json.loads(settings_path.read_text(encoding="utf-8"))
        current.update(deepcopy(DEFAULT_SETTINGS))
        store = SimpleNamespace(document=deepcopy(current))
        editor = SettingsEditor(store, [])

        editor.defaults_button.click()

        self.assertFalse(editor.save_button.isEnabled())
        self.assertIn("уже установлен", editor.status_label.text())


if __name__ == "__main__":
    unittest.main()
