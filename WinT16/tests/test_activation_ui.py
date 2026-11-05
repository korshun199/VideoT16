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
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QPushButton
from WinT16.app.activation_ui import show_activation_request
from runtime.activation import DeviceIdentity


class ActivationUiTests(unittest.TestCase):
    """Проверяет успешное закрытие окна только после приёма кода."""

    def test_accepting_response_returns_success(self) -> None:
        """Имитирует вставку и нажатие кнопки без настоящих ключей и оборудования."""
        application = QApplication.instance() or QApplication([])

        def enter_response() -> None:
            """Находит окно и выполняет действия пользователя."""
            dialog = next(item for item in application.topLevelWidgets() if isinstance(item, QDialog))
            fields = dialog.findChildren(QPlainTextEdit)
            fields[1].setPlainText("A" * 32)
            button = next(item for item in dialog.findChildren(QPushButton) if item.text() == "Активировать")
            button.click()

        with TemporaryDirectory() as temporary:
            app_root = Path(temporary)

            def save_auth(root: Path, *_arguments: object) -> None:
                """Имитирует успешное сохранение постоянного файла активации."""
                (root / "auth.dat").write_text("test activation", encoding="utf-8")

            with patch("WinT16.app.activation_ui.ensure_short_request", return_value="B" * 32), \
                    patch("WinT16.app.activation_ui.accept_short_activation_code",
                          side_effect=save_auth) as accept, \
                    patch("WinT16.app.activation_ui.QMessageBox.information"):
                QTimer.singleShot(0, enter_response)
                status = show_activation_request(app_root, "Тест", DeviceIdentity("test", ()))
        self.assertEqual(status, 0)
        accept.assert_called_once()


if __name__ == "__main__":
    unittest.main()
