"""Показывает простое окно активации с копированием и вставкой кодов."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from runtime.activation import (
    AUTH_FILE,
    ActivationError,
    DeviceIdentity,
    accept_short_activation_code,
    ensure_short_request,
)

_application: QApplication | None = None


def show_activation_request(
    app_root: Path,
    message: str,
    identity: DeviceIdentity | None = None,
) -> int:
    """Показывает код запроса и принимает ответ владельца прямо в окне."""
    global _application
    application = QApplication.instance() or QApplication([])
    _application = application
    dialog = QDialog()
    dialog.setWindowTitle("VideoT16 — активация компьютера")
    dialog.setMinimumSize(700, 500)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(4)
    code_font = QFont("Consolas", 27, QFont.Weight.Bold)
    code_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3.0)

    heading = QLabel(
        "Сообщите владельцу VideoT16 код из первого поля — можно продиктовать по телефону. "
        "Ответ введите во второе поле и нажмите «Активировать»."
    )
    heading.setWordWrap(True)
    layout.addWidget(heading)

    details = QLabel(f"Состояние: {message}")
    details.setWordWrap(True)
    layout.addWidget(details)

    layout.addWidget(QLabel("1. Код этого компьютера — отправьте его владельцу:"))
    request_box = QLineEdit()
    request_box.setReadOnly(True)
    request_box.setPlaceholderText("12 знаков")
    request_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    request_box.setFont(code_font)
    request_box.setMaxLength(12)
    request_box.setFixedHeight(68)
    request_box.setStyleSheet("QLineEdit { padding: 4px; }")
    try:
        if identity is None:
            raise ActivationError("Не удалось определить оборудование этого компьютера.")
        request_box.setText(ensure_short_request(app_root, identity))
    except (ActivationError, OSError) as error:
        details.setText(f"Состояние: {error}")
        request_box.clear()
    layout.addWidget(request_box)

    request_buttons = QHBoxLayout()
    copy_button = QPushButton("Скопировать код одной строкой")
    request_buttons.addWidget(copy_button)
    request_buttons.addStretch(1)
    layout.addLayout(request_buttons)

    layout.addWidget(QLabel("2. Введите ответный код владельца — можно вставить или набрать:"))
    response_box = QLineEdit()
    response_box.setPlaceholderText("12 заглавных букв/цифр; можно вставить с дефисами")
    response_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    response_box.setFont(code_font)
    response_box.setMaxLength(20)
    response_box.setFixedHeight(68)
    response_box.setStyleSheet("QLineEdit { padding: 4px; }")
    layout.addWidget(response_box)

    buttons = QHBoxLayout()
    activate_button = QPushButton("Активировать")
    activate_button.setEnabled(identity is not None and len(request_box.text()) == 12)
    close_button = QPushButton("Закрыть")
    buttons.addStretch(1)
    buttons.addWidget(activate_button)
    buttons.addWidget(close_button)
    layout.addLayout(buttons)

    def copy_request() -> None:
        """Копирует полный код запроса в системный буфер обмена."""
        code = request_box.text().strip()
        if len(code) != 12:
            QMessageBox.warning(dialog, "Запрос не готов", "Код этого компьютера ещё не создан.")
            return
        application.clipboard().setText(code)
        QMessageBox.information(dialog, "Скопировано", "Код запроса скопирован. Отправьте его владельцу.")

    def activate() -> None:
        """Проверяет вставленный ответный код и создаёт auth.dat после совпадения."""
        if identity is None:
            QMessageBox.critical(dialog, "Нет данных компьютера", "Перезапустите программу или обратитесь к владельцу.")
            return
        code = response_box.text().strip()
        if not code:
            QMessageBox.information(dialog, "Нужен ответный код", "Вставьте код из сообщения владельца.")
            response_box.setFocus()
            return
        activate_button.setEnabled(False)
        try:
            accept_short_activation_code(app_root, identity, code)
            auth_path = app_root / AUTH_FILE
            if not auth_path.is_file():
                raise ActivationError(f"Файл активации не найден после сохранения: {auth_path}")
        except (ActivationError, OSError) as error:
            QMessageBox.critical(dialog, "Не удалось активировать", str(error))
            activate_button.setEnabled(True)
            response_box.setFocus()
            return
        QMessageBox.information(
            dialog,
            "Активация завершена",
            f"Компьютер активирован. Файл сохранён здесь:\n{auth_path}\n\n"
            "При следующих запусках повторный код не потребуется.",
        )
        dialog.accept()

    def uppercase_response(text: str) -> None:
        """Показывает ввод заглавными, не меняя позицию курсора пользователя."""
        normalized = text.upper()
        if normalized != text:
            cursor = response_box.cursorPosition()
            response_box.setText(normalized)
            response_box.setCursorPosition(min(cursor, len(normalized)))

    copy_button.clicked.connect(copy_request)
    activate_button.clicked.connect(activate)
    close_button.clicked.connect(dialog.reject)
    response_box.textEdited.connect(uppercase_response)
    return 0 if dialog.exec() == QDialog.DialogCode.Accepted else 2
