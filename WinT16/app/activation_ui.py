"""Показывает простое окно активации с копированием и вставкой кодов."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
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

    heading = QLabel(
        "Нажмите «Скопировать код», отправьте одну строку владельцу VideoT16. "
        "Ответ вставьте во второе поле и нажмите «Активировать»."
    )
    heading.setWordWrap(True)
    layout.addWidget(heading)

    details = QLabel(f"Состояние: {message}")
    details.setWordWrap(True)
    layout.addWidget(details)

    layout.addWidget(QLabel("1. Код этого компьютера — отправьте его владельцу:"))
    request_box = QPlainTextEdit()
    request_box.setReadOnly(True)
    request_box.setPlaceholderText("Код запроса появится здесь")
    request_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    request_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    request_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    request_box.setFixedHeight(40)
    request_box.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse
        | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )
    try:
        if identity is None:
            raise ActivationError("Не удалось определить оборудование этого компьютера.")
        request_box.setPlainText(ensure_short_request(app_root, identity))
    except (ActivationError, OSError) as error:
        details.setText(f"Состояние: {error}")
        request_box.setPlainText("Код запроса не создан. Обратитесь к владельцу программы.")
    layout.addWidget(request_box)

    request_buttons = QHBoxLayout()
    copy_button = QPushButton("Скопировать код одной строкой")
    request_buttons.addWidget(copy_button)
    request_buttons.addStretch(1)
    layout.addLayout(request_buttons)

    layout.addWidget(QLabel("2. Вставьте сюда ответный код из сообщения владельца:"))
    response_box = QPlainTextEdit()
    response_box.setPlaceholderText("Вставьте ответ из 32 заглавных символов")
    response_box.setFixedHeight(40)
    response_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    response_box.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    response_box.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    layout.addWidget(response_box)

    buttons = QHBoxLayout()
    activate_button = QPushButton("Активировать")
    activate_button.setEnabled(identity is not None and len(request_box.toPlainText()) == 32)
    close_button = QPushButton("Закрыть")
    buttons.addStretch(1)
    buttons.addWidget(activate_button)
    buttons.addWidget(close_button)
    layout.addLayout(buttons)

    def copy_request() -> None:
        """Копирует полный код запроса в системный буфер обмена."""
        code = request_box.toPlainText().strip()
        if len(code) != 32:
            QMessageBox.warning(dialog, "Запрос не готов", "Код этого компьютера ещё не создан.")
            return
        application.clipboard().setText(code)
        QMessageBox.information(dialog, "Скопировано", "Код запроса скопирован. Отправьте его владельцу.")

    def activate() -> None:
        """Проверяет вставленный ответ и создаёт auth.dat после проверки подписи."""
        if identity is None:
            QMessageBox.critical(dialog, "Нет данных компьютера", "Перезапустите программу или обратитесь к владельцу.")
            return
        code = response_box.toPlainText().strip()
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

    copy_button.clicked.connect(copy_request)
    activate_button.clicked.connect(activate)
    close_button.clicked.connect(dialog.reject)
    return 0 if dialog.exec() == QDialog.DialogCode.Accepted else 2
