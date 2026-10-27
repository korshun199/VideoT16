"""Проверяет схему Qt-редактора и безопасное сохранение JSON VideoT16."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtWidgets import QApplication, QLabel

    from config.editor import (
        FIELD_SPECS,
        HelpDialog,
        SettingsDocument,
        SettingsEditor,
        leaf_paths,
    )
except ImportError:
    QApplication = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(QApplication is None, "Для тестов редактора нужна необязательная PySide6")
class SettingsDocumentTests(unittest.TestCase):
    """Проверяет покрытие полей, проверку диапазонов и сохранение с откатом."""

    def setUp(self) -> None:
        """Создаёт отдельный временный JSON, не затрагивая рабочую конфигурацию."""
        self.temp_root = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_root.name)
        self.path = self.root / "settings.json"
        self.temp_dir = self.root / "tmp"
        self.path.write_bytes((PROJECT_ROOT / "config" / "settings.json").read_bytes())
        self.store = SettingsDocument(self.path, self.temp_dir)

    def tearDown(self) -> None:
        """Удаляет только изолированный каталог текущего теста."""
        self.temp_root.cleanup()

    def test_every_current_profile_value_has_a_field_description(self) -> None:
        """У каждого конечного значения профиля есть редактируемая строка со справкой."""
        profile = json.loads((PROJECT_ROOT / "config" / "settings.json").read_text(encoding="utf-8"))["profiles"]["sky"]
        described_paths = {spec.path for spec in FIELD_SPECS}

        self.assertEqual(leaf_paths(profile), described_paths)

    def test_save_creates_one_backup_and_preserves_unknown_json_keys(self) -> None:
        """Сохранение атомарно обновляет файл, хранит исходник и не теряет соседние поля."""
        original = self.path.read_bytes()
        changed = deepcopy(self.store.document)
        changed["profiles"]["sky"]["settings"]["confidence"] = 0.41
        changed["profiles"]["sky"]["custom_owner_note"] = {"keep": "yes"}

        backup_path = self.store.save(changed)

        self.assertEqual(backup_path, self.temp_dir / "settings.json.bak")
        self.assertEqual(backup_path.read_bytes(), original)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(saved["profiles"]["sky"]["settings"]["confidence"], 0.41)
        self.assertEqual(saved["profiles"]["sky"]["custom_owner_note"], {"keep": "yes"})
        self.assertEqual(list(self.temp_dir.glob("settings-*.tmp")), [])

    def test_invalid_profile_is_rejected_without_replacing_settings(self) -> None:
        """Плохой порог отклоняется до резервирования и записи рабочего файла."""
        original = self.path.read_bytes()
        changed = deepcopy(self.store.document)
        changed["profiles"]["sky"]["settings"]["min_confidence"] = 0.90

        with self.assertRaisesRegex(ValueError, "не выше confidence"):
            self.store.save(changed)

        self.assertEqual(self.path.read_bytes(), original)
        self.assertFalse(self.store.backup_path.exists())

    def test_editor_only_ranges_are_rejected(self) -> None:
        """Диапазоны, показанные в виджетах, также действуют при проверке файла."""
        changed = deepcopy(self.store.document)
        changed["profiles"]["sky"]["overlay"]["box"]["scale_x"] = 4.0

        with self.assertRaisesRegex(ValueError, "допустимый максимум 3"):
            self.store.validate(changed)

    def test_external_edit_is_never_overwritten(self) -> None:
        """Проверка ревизии сохраняет внешнее изменение и просит перечитать файл."""
        changed = deepcopy(self.store.document)
        changed["profiles"]["sky"]["settings"]["confidence"] = 0.40
        self.path.write_text('{"external": true}\n', encoding="utf-8")

        with self.assertRaisesRegex(RuntimeError, "изменён извне"):
            self.store.save(changed)

        self.assertEqual(self.path.read_text(encoding="utf-8"), '{"external": true}\n')


@unittest.skipIf(QApplication is None, "Для тестов редактора нужна необязательная PySide6")
class SettingsEditorTests(unittest.TestCase):
    """Проверяет работу формы и отдельных окон книжной справки без камеры."""

    @classmethod
    def setUpClass(cls) -> None:
        """Создаёт одно невидимое Qt-приложение для всех проверок интерфейса."""
        cls.app = QApplication.instance() or QApplication([])

    def test_form_round_trips_all_values_without_implicit_changes(self) -> None:
        """Открытие и чтение формы не меняет исходное дерево JSON."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = PROJECT_ROOT / "config" / "settings.json"
            copied = Path(temporary_directory) / "settings.json"
            copied.write_bytes(source.read_bytes())
            editor = SettingsEditor(SettingsDocument(copied, Path(temporary_directory) / "tmp"))
            editor._collect_current_profile()

            self.assertEqual(editor.working_document, editor.store.document)
            self.assertEqual(len(editor.help_buttons), len(FIELD_SPECS))
            editor.close()

    def test_book_button_opens_detailed_help_for_its_field(self) -> None:
        """Нажатие книжки показывает назначение, диапазон и текущее значение."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            copied = Path(temporary_directory) / "settings.json"
            copied.write_bytes((PROJECT_ROOT / "config" / "settings.json").read_bytes())
            editor = SettingsEditor(SettingsDocument(copied, Path(temporary_directory) / "tmp"))
            threshold_path = ("settings", "confidence")
            spec = next(item for item in FIELD_SPECS if item.path == threshold_path)
            button = editor.help_buttons[threshold_path]

            with patch.object(HelpDialog, "exec", return_value=0):
                button.click()

            self.assertIsNotNone(editor.last_help_dialog)
            text = "\n".join(label.text() for label in editor.last_help_dialog.findChildren(QLabel))
            self.assertIn(spec.description, text)
            self.assertIn(spec.allowed, text)
            self.assertIn("32%", text)
            editor.close()


if __name__ == "__main__":
    unittest.main()
