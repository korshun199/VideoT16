"""Проверяет продолжение запуска после ввода кода в окне активации."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton
from WinT16.app.activation_ui import show_activation_request
from runtime.activation import (
    DeviceIdentity,
    _short_model_key,
    activation_key,
    ensure_short_request,
    issue_short_activation_code,
)


class ActivationUiTests(unittest.TestCase):
    """Проверяет успешное закрытие окна только после приёма кода."""

    def test_accepting_response_returns_success(self) -> None:
        """Имитирует вставку и нажатие кнопки без настоящих ключей и оборудования."""
        application = QApplication.instance() or QApplication([])

        def enter_response() -> None:
            """Находит окно и выполняет действия пользователя."""
            dialog = next(item for item in application.topLevelWidgets() if isinstance(item, QDialog))
            fields = dialog.findChildren(QLineEdit)
            fields[1].setText("A" * 12)
            button = next(item for item in dialog.findChildren(QPushButton) if item.text() == "Активировать")
            button.click()

        with TemporaryDirectory() as temporary:
            app_root = Path(temporary)

            def save_auth(root: Path, *_arguments: object) -> None:
                """Имитирует успешное сохранение постоянного файла активации."""
                (root / "auth.dat").write_text("test activation", encoding="utf-8")

            with patch("WinT16.app.activation_ui.ensure_short_request", return_value="B" * 12), \
                    patch("WinT16.app.activation_ui.accept_short_activation_code",
                          side_effect=save_auth) as accept, \
                    patch("WinT16.app.activation_ui.QMessageBox.information"):
                QTimer.singleShot(0, enter_response)
                status = show_activation_request(app_root, "Тест", DeviceIdentity("test", ()))
        self.assertEqual(status, 0)
        accept.assert_called_once()

    def test_real_code_saves_auth_and_survives_restart(self) -> None:
        """Окно принимает настоящий код и файл активации проходит повторную проверку."""
        application = QApplication.instance() or QApplication([])
        temporary_root = Path(__file__).resolve().parents[2] / "tmp" / "WinT16"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            owner_key = bytes(range(32))
            owner_path = root / "owner.key"
            owner_path.write_bytes(owner_key)
            app_root = root / "client"
            app_root.mkdir()
            identity = DeviceIdentity("a" * 64, ("baseboard_serial", "system_uuid"))
            request = ensure_short_request(app_root, identity)
            response = issue_short_activation_code(request, owner_path)
            embedded_key = _short_model_key(owner_key)

            def enter_response() -> None:
                """Вводит выданный владельцем ответ и нажимает кнопку активации."""
                dialog = next(item for item in application.topLevelWidgets() if isinstance(item, QDialog))
                QTimer.singleShot(2000, dialog.reject)
                dialog.findChildren(QLineEdit)[1].setText(response)
                next(item for item in dialog.findChildren(QPushButton) if item.text() == "Активировать").click()

            with patch("runtime.activation._embedded_model_key", return_value=embedded_key), \
                    patch("WinT16.app.activation_ui.QMessageBox.information"), \
                    patch("WinT16.app.activation_ui.QMessageBox.critical"):
                QTimer.singleShot(0, enter_response)
                status = show_activation_request(app_root, "Тест", identity)
                self.assertEqual(status, 0)
                self.assertTrue((app_root / "auth.dat").is_file())
                self.assertEqual(activation_key(app_root, identity), embedded_key)
                self.assertEqual(activation_key(app_root, identity), embedded_key)


if __name__ == "__main__":
    unittest.main()
