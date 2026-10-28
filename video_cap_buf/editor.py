#!/usr/bin/env python3
"""Графический редактор настроек нового конвейера VideoT16."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from capture import resolve_model_path, validate_settings


PROGRAM_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PROGRAM_DIR.parent
SETTINGS_PATH = PROGRAM_DIR / "settings.json"
TMP_DIR = PROJECT_DIR / "tmp" / "video_cap_buf"


@dataclass(frozen=True)
class FieldSpec:
    """Описывает ключ JSON, виджет ввода, диапазон и подсказку поля."""

    key: str
    group: str
    label: str
    kind: str
    description: str
    allowed: str
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 0
    step: float = 1
    choices: tuple[tuple[str, str], ...] = ()


FIELD_SPECS = (
    FieldSpec("source", "Видео", "Источник USB-видео", "text",
              "Путь к камере или USB-оцифровщику. В Linux обычно /dev/videoN или /dev/v4l/by-id/… .",
              "Непустой путь к подключённому устройству."),
    FieldSpec("pixel_format", "Видео", "Формат пикселей", "enum",
              "Формат, который программа запрашивает у устройства. Фактический режим записывается в журнал.",
              "Выберите формат, поддерживаемый оцифровщиком.",
              choices=(("YUYV", "YUYV — без сжатия"), ("MJPG", "MJPG — сжатый поток"))),
    FieldSpec("width", "Видео", "Ширина захвата", "int",
              "Запрашиваемая ширина одного кадра.", "32–7680 пикселей.", 32, 7680, step=16),
    FieldSpec("height", "Видео", "Высота захвата", "int",
              "Запрашиваемая высота одного кадра.", "32–7680 пикселей.", 32, 7680, step=16),
    FieldSpec("fps", "Видео", "Частота захвата", "int",
              "Запрашиваемая частота кадров. Устройство может выбрать другой режим.", "1–240 кадров/с.", 1, 240),
    FieldSpec("buffer_max_mb", "Видео", "Предел памяти кадров", "int",
              "Ограничивает объём исходных изображений в буфере захвата.", "1–16384 МиБ.", 1, 16384, step=64),
    FieldSpec("model_path", "Распознавание", "Файл модели ONNX", "model",
              "ONNX-модель для поиска объектов. Для этой программы используется модель с фиксированным входом.",
              "Существующий файл .onnx; размер входа должен совпадать с полем ниже."),
    FieldSpec("inference_size", "Распознавание", "Размер входа модели", "int",
              "Сторона квадратного изображения, которое получает модель.", "Положительное число; должно совпадать с моделью.", 32, 4096, step=32),
    FieldSpec("inference_backend", "Распознавание", "Движок", "enum",
              "Библиотека, которая выполняет ONNX-модель на CPU.", "Движок должен быть установлен в общем .venv.",
              choices=(("onnxruntime", "ONNX Runtime"), ("opencv", "OpenCV DNN"))),
    FieldSpec("recognition_frame_policy", "Распознавание", "Обработка кадров", "enum",
              "latest пропускает старые кадры ради свежего результата; all обрабатывает очередь FIFO.",
              "latest или all.", choices=(("latest", "latest — минимальная задержка"), ("all", "all — каждый кадр"))),
    FieldSpec("recognition_workers", "Распознавание", "Число распознавателей", "int",
              "Число независимых копий модели. Несколько копий могут замедлить работу.", "1–32.", 1, 32),
    FieldSpec("inference_threads_per_worker", "Распознавание", "Потоков на модель", "int",
              "Внутренние потоки ONNX Runtime для одной сессии.", "1–128.", 1, 128),
    FieldSpec("confidence_percent", "Распознавание", "Порог зелёной рамки", "float",
              "Минимальная уверенность для обычной зелёной рамки.", "0–100%.", 0, 100, 1, 1),
    FieldSpec("presence_confidence_percent", "Распознавание", "Порог пурпурного кандидата", "float",
              "Нижний порог сигнала модели. Между ним и порогом рамки рисуется пурпурный кандидат с вопросительным знаком.",
              "0–порог зелёной рамки.", 0, 100, 1, 1),
    FieldSpec("nms_iou_threshold", "Распознавание", "Подавление дублей", "float",
              "Перекрытие, при котором похожие рамки модели считаются дублями.", "0–1.", 0, 1, 2, 0.01),
    FieldSpec("class_names", "Распознавание", "Названия классов", "list",
              "Названия классов в том же порядке, в котором их выдаёт модель.", "Разделяйте запятыми; порядок важен."),
    FieldSpec("result_buffer_max_items", "Распознавание", "Лимит результатов", "int",
              "Максимум компактных записей обнаружения во втором буфере.", "1–100000.", 1, 100000, step=100),
    FieldSpec("box_color_bgr", "Рамка и подпись", "Цвет зелёной рамки", "color",
              "Цвет уверенного обнаружения. В JSON OpenCV хранится порядок BGR.", "Выберите цвет в диалоге."),
    FieldSpec("text_color_bgr", "Рамка и подпись", "Цвет подписи", "color",
              "Цвет текста класса и уверенности. В JSON хранится порядок BGR.", "Выберите цвет в диалоге."),
    FieldSpec("weak_candidate_color_bgr", "Рамка и подпись", "Цвет слабого кандидата", "color",
              "Цвет рамки кандидата ниже зелёного порога. В JSON хранится порядок BGR.", "Выберите цвет в диалоге."),
    FieldSpec("box_thickness", "Рамка и подпись", "Толщина рамки", "int",
              "Толщина линий рамок на исходном кадре.", "1–20 пикселей.", 1, 20),
    FieldSpec("font_scale", "Рамка и подпись", "Размер подписи", "float",
              "Масштаб шрифта подписи OpenCV.", "Больше 0 и не более 10.", 0.05, 10, 2, 0.05),
)

GROUP_ORDER = ("Видео", "Распознавание", "Рамка и подпись")


class SettingsDocument:
    """Загружает JSON, замечает внешние правки и безопасно сохраняет профиль."""

    def __init__(self, path: Path = SETTINGS_PATH, temp_dir: Path = TMP_DIR) -> None:
        """Запоминает исходные байты, чтобы не затереть чужое изменение файла."""
        self.path = path
        self.temp_dir = temp_dir
        self.original_bytes = path.read_bytes()
        self.document = json.loads(self.original_bytes.decode("utf-8"))
        validate_settings(self.document)

    def ensure_current(self) -> None:
        """Отменяет запись, если настройки изменились после открытия редактора."""
        if self.path.read_bytes() != self.original_bytes:
            raise RuntimeError("settings.json изменён извне. Закрой редактор и открой его снова.")

    def validate(self, document: dict[str, Any]) -> None:
        """Проверяет общие ограничения и наличие выбранной ONNX-модели."""
        validate_settings(document)
        model_path = resolve_model_path(document, self.path)
        if model_path.suffix.lower() != ".onnx":
            raise ValueError("Новая программа принимает ONNX-модель (.onnx), не .pt")
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

    def save(self, document: dict[str, Any]) -> Path:
        """Делает резервную копию и атомарно заменяет файл настроек."""
        self.validate(document)
        self.ensure_current()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = self.temp_dir / f"settings_before_editor_{stamp}.json"
        staged_path: Path | None = None
        mode = stat.S_IMODE(self.path.stat().st_mode)
        serialized = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        try:
            shutil.copy2(self.path, backup_path)
            with tempfile.NamedTemporaryFile(
                "wb", dir=self.temp_dir, prefix="settings-", suffix=".tmp", delete=False
            ) as staged_file:
                staged_file.write(serialized)
                staged_file.flush()
                os.fsync(staged_file.fileno())
                staged_path = Path(staged_file.name)
            staged_path.chmod(mode)
            os.replace(staged_path, self.path)
            self.original_bytes = serialized
            self.document = deepcopy(document)
            return backup_path
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)


class HelpDialog(QDialog):
    """Показывает назначение параметра и его допустимые значения."""

    def __init__(self, spec: FieldSpec, value: Any, parent: QWidget | None = None) -> None:
        """Создаёт простую справку по выбранной настройке."""
        super().__init__(parent)
        self.setWindowTitle(f"Описание: {spec.label}")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        title = QLabel(spec.label)
        title.setObjectName("helpTitle")
        layout.addWidget(title)
        description = QLabel(spec.description)
        description.setWordWrap(True)
        layout.addWidget(description)
        allowed = QLabel(f"<b>Допустимые значения</b><br>{spec.allowed}")
        allowed.setWordWrap(True)
        allowed.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(allowed)
        current = QLabel(f"<b>Сейчас</b><br>{self._format_value(spec, value)}")
        current.setWordWrap(True)
        current.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(current)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)

    @staticmethod
    def _format_value(spec: FieldSpec, value: Any) -> str:
        """Представляет текущий JSON-тип в читаемой форме."""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        if spec.key.endswith("_percent"):
            return f"{value}%"
        return str(value)


class SettingsEditor(QMainWindow):
    """Редактирует только параметры конвейера и запускает его по кнопке."""

    def __init__(self, store: SettingsDocument, runtime_args: list[str]) -> None:
        """Строит вкладки и загружает значения из settings.json."""
        super().__init__()
        self.store = store
        self.runtime_args = runtime_args
        self.working_document = deepcopy(store.document)
        self.field_widgets: dict[str, QWidget] = {}
        self._refreshing = False
        self.run_requested = False
        self.setWindowTitle("VideoT16 — настройки испытательного конвейера")
        self.resize(880, 760)
        self._build_window()

    def _build_window(self) -> None:
        """Размещает редактор, подсказки и кнопки сохранения/запуска."""
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 14)
        title = QLabel("Настройки распознавания")
        title.setObjectName("mainTitle")
        outer.addWidget(title)
        note = QLabel("Поля загружены из video_cap_buf/settings.json. Обычный ./run эти настройки использует без этого окна.")
        note.setWordWrap(True)
        outer.addWidget(note)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._render_fields()

        self.status_label = QLabel("Настройки загружены")
        self.status_label.setObjectName("statusLabel")
        outer.addWidget(self.status_label)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.save_button = QPushButton("Сохранить")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        actions.addWidget(self.save_button)
        self.run_button = QPushButton("Запуск")
        self.run_button.setObjectName("runButton")
        self.run_button.setToolTip("Сохранить проверенные настройки и запустить распознавание")
        self.run_button.clicked.connect(self._save_and_run)
        actions.addWidget(self.run_button)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        outer.addLayout(actions)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QMainWindow,QWidget { background:#20242a; color:#e6e9ed; font-size:14px; }"
            "QTabWidget::pane { border:1px solid #454c55; top:-1px; }"
            "QTabBar::tab { background:#2b3038; padding:9px 14px; margin-right:2px; }"
            "QTabBar::tab:selected { background:#3a4654; border-bottom:2px solid #69b7ff; }"
            "QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox { background:#171a1f; border:1px solid #505862; border-radius:4px; padding:6px; min-height:24px; }"
            "QPushButton,QToolButton { background:#343b44; border:1px solid #59636f; border-radius:4px; padding:7px 11px; }"
            "QPushButton:hover,QToolButton:hover { background:#46515d; }"
            "QPushButton#runButton { background:#245e8c; border-color:#4d94cc; font-weight:600; }"
            "QLabel#mainTitle { font-size:20px; font-weight:600; }"
            "QLabel#helpTitle { font-size:18px; font-weight:600; color:#8bc8ff; }"
            "QLabel#statusLabel { color:#b6c4d0; padding:4px; }"
            "QScrollArea { border:0; }"
        )

    def _render_fields(self) -> None:
        """Строит вкладки только из полей действующего плоского JSON-конфига."""
        self._refreshing = True
        self.tabs.clear()
        self.field_widgets.clear()
        for group in GROUP_ORDER:
            specs = [spec for spec in FIELD_SPECS if spec.group == group]
            content = QWidget()
            form = QGridLayout(content)
            form.setColumnStretch(1, 1)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            form.setContentsMargins(14, 14, 14, 14)
            for row, spec in enumerate(specs):
                label = QLabel(spec.label)
                label.setMinimumWidth(220)
                label.setWordWrap(True)
                editor = self._create_editor(spec, self.working_document[spec.key])
                self._connect_change(spec, editor)
                help_button = QToolButton()
                help_button.setText("?")
                help_button.setToolTip("Описание и диапазон значения")
                help_button.clicked.connect(
                    lambda checked=False, item=spec: self._show_help(item)
                )
                form.addWidget(label, row, 0)
                form.addWidget(editor, row, 1)
                form.addWidget(help_button, row, 2)
                self.field_widgets[spec.key] = editor
            form.setRowStretch(len(specs), 1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, group)
        self._refreshing = False

    def _create_editor(self, spec: FieldSpec, value: Any) -> QWidget:
        """Создаёт виджет подходящего типа и ограничивает число диапазоном."""
        if spec.kind == "int":
            editor = QSpinBox()
            editor.setRange(int(spec.minimum or 0), int(spec.maximum or 999999))
            editor.setSingleStep(int(spec.step))
            editor.setValue(int(value))
            return editor
        if spec.kind == "float":
            editor = QDoubleSpinBox()
            editor.setDecimals(spec.decimals)
            editor.setRange(float(spec.minimum or 0), float(spec.maximum or 999999))
            editor.setSingleStep(spec.step)
            editor.setKeyboardTracking(False)
            if spec.key.endswith("_percent"):
                editor.setSuffix(" %")
            editor.setValue(float(value))
            return editor
        if spec.kind == "enum":
            editor = QComboBox()
            for saved, label in spec.choices:
                editor.addItem(label, saved)
            index = editor.findData(value)
            if index < 0:
                editor.addItem(f"Неизвестно: {value}", value)
                index = editor.count() - 1
            editor.setCurrentIndex(index)
            return editor
        if spec.kind == "color":
            button = QPushButton()
            self._paint_color_button(button, value)
            button.clicked.connect(lambda checked=False, target=button: self._choose_color(target))
            return button
        line = QLineEdit(", ".join(value) if spec.kind == "list" else str(value))
        line.setClearButtonEnabled(True)
        if spec.kind != "model":
            return line
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(line, 1)
        browse = QPushButton("Выбрать…")
        browse.clicked.connect(lambda checked=False, target=line: self._browse_model(target))
        layout.addWidget(browse)
        row.setProperty("lineEditor", line)
        return row

    @staticmethod
    def _line_editor(widget: QWidget) -> QLineEdit:
        """Возвращает поле пути модели из составного виджета."""
        line = widget.property("lineEditor")
        if not isinstance(line, QLineEdit):
            raise TypeError("Не найдено поле пути к модели")
        return line

    def _connect_change(self, spec: FieldSpec, widget: QWidget) -> None:
        """Помечает форму изменённой после правки любого поля."""
        if spec.kind == "int":
            widget.valueChanged.connect(self._mark_modified)
        elif spec.kind == "float":
            widget.valueChanged.connect(self._mark_modified)
        elif spec.kind == "enum":
            widget.currentIndexChanged.connect(self._mark_modified)
        elif spec.kind == "color":
            return
        elif spec.kind == "model":
            self._line_editor(widget).textChanged.connect(self._mark_modified)
        else:
            widget.textChanged.connect(self._mark_modified)

    def _read_value(self, spec: FieldSpec, widget: QWidget) -> Any:
        """Преобразует введённое значение в ожидаемый тип JSON."""
        if spec.kind == "int":
            return int(widget.value())
        if spec.kind == "float":
            return round(float(widget.value()), spec.decimals + 2)
        if spec.kind == "enum":
            return widget.currentData()
        if spec.kind == "color":
            return widget.property("bgrValue")
        if spec.kind == "model":
            return self._line_editor(widget).text().strip()
        text = widget.text().strip()
        if spec.kind == "list":
            return [part.strip() for part in text.split(",") if part.strip()]
        return text

    def _collect(self) -> None:
        """Обновляет рабочую копию JSON по всем редактируемым полям."""
        if self._refreshing:
            return
        for spec in FIELD_SPECS:
            self.working_document[spec.key] = self._read_value(
                spec, self.field_widgets[spec.key]
            )

    def _mark_modified(self, *_values: Any) -> None:
        """Показывает несохранённое состояние и включает кнопку сохранения."""
        if self._refreshing:
            return
        self._collect()
        changed = self.working_document != self.store.document
        self.status_label.setText("Есть несохранённые изменения" if changed else "Изменений нет")
        self.save_button.setEnabled(changed)

    def _show_help(self, spec: FieldSpec) -> None:
        """Открывает подсказку со смыслом поля и текущим значением."""
        self._collect()
        HelpDialog(spec, self.working_document[spec.key], self).exec()

    @staticmethod
    def _paint_color_button(button: QPushButton, bgr: list[int]) -> None:
        """Показывает цветовой образец, сохраняя порядок BGR конфигурации."""
        blue, green, red = (int(channel) for channel in bgr)
        button.setProperty("bgrValue", [blue, green, red])
        button.setText(f"Цвет  RGB {red}, {green}, {blue}")
        button.setStyleSheet(
            "QPushButton { text-align:left; border:1px solid #59636f; border-radius:4px; padding:6px; "
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgb({red},{green},{blue}),stop:0.22 #171a1f); }}"
        )

    def _choose_color(self, button: QPushButton) -> None:
        """Открывает выбор RGB-цвета и переводит его в BGR для OpenCV."""
        blue, green, red = button.property("bgrValue")
        chosen = QColorDialog.getColor(QColor(red, green, blue), self, "Выберите цвет")
        if chosen.isValid():
            self._paint_color_button(
                button, [chosen.blue(), chosen.green(), chosen.red()]
            )
            self._mark_modified()

    def _browse_model(self, editor: QLineEdit) -> None:
        """Выбирает ONNX-файл и сохраняет путь относительно каталога настроек."""
        selected, _filter = QFileDialog.getOpenFileName(
            self, "Выберите ONNX-модель", str(PROJECT_DIR / "models"), "ONNX-модели (*.onnx)"
        )
        if not selected:
            return
        model_path = Path(selected).resolve()
        stored_path = Path(os.path.relpath(model_path, PROGRAM_DIR)).as_posix()
        editor.setText(stored_path)

    def _save(self) -> bool:
        """Проверяет, резервирует и сохраняет JSON без изменения иных ключей."""
        self._collect()
        if self.working_document == self.store.document:
            try:
                self.store.ensure_current()
            except (OSError, RuntimeError) as error:
                QMessageBox.critical(self, "Настройки изменены", str(error))
                return False
            self.status_label.setText("Изменений нет — файл не перезаписывался")
            self.save_button.setEnabled(False)
            return True
        try:
            backup = self.store.save(self.working_document)
        except (OSError, UnicodeError, ValueError, RuntimeError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Не удалось сохранить настройки", str(error))
            self.status_label.setText("Сохранение не выполнено; исходный файл не заменён")
            return False
        self.status_label.setText(f"Сохранено. Копия исходника: {backup}")
        self.save_button.setEnabled(False)
        return True

    def _save_and_run(self) -> None:
        """Сначала проверяет конфиг, затем закрывает окно и запускает распознавание."""
        self._collect()
        try:
            self.store.ensure_current()
            self.store.validate(self.working_document)
        except (OSError, UnicodeError, ValueError, RuntimeError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Настройки не прошли проверку", str(error))
            return
        if self.working_document != self.store.document and not self._save():
            return
        self.run_requested = True
        self.close()

    def closeEvent(self, event: Any) -> None:
        """Спрашивает о несохранённых настройках при закрытии окна."""
        self._collect()
        if self.working_document == self.store.document:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Несохранённые изменения",
            "Сохранить настройки перед закрытием?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            event.accept() if self._save() else event.ignore()
        elif answer == QMessageBox.StandardButton.Discard:
            event.accept()
        else:
            event.ignore()


def main() -> int:
    """Открывает настройки и после подтверждения запускает штатную оболочку run."""
    runtime_args = sys.argv[1:]
    application = QApplication([sys.argv[0]])
    application.setStyle("Fusion")
    application.setQuitOnLastWindowClosed(True)
    try:
        store = SettingsDocument()
        editor = SettingsEditor(store, runtime_args)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        QMessageBox.critical(None, "Не удалось открыть настройки", str(error))
        return 1
    editor.show()
    application.exec()
    if not editor.run_requested:
        return 0
    command = [str(PROGRAM_DIR / "run"), *runtime_args]
    try:
        return subprocess.run(command, cwd=PROJECT_DIR, check=False).returncode
    except OSError as error:
        QMessageBox.critical(None, "Не удалось запустить распознавание", str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
