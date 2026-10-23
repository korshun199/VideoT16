#!/usr/bin/env python3
"""Нативное Qt-окно для мышиного редактирования настроек VideoT16."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# Пути проекта и предел размера одного редактируемого файла.
PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_DIR / "config"
MAX_CONFIG_BYTES = 1_048_576

# Подписи параметров, которые могут встретиться в JSON проекта.
FIELD_LABELS = {
    "model": "Модель распознавания",
    "model_path": "Модель распознавания",
    "source": "Источник видео",
    "camera_source": "Тип камеры",
    "camera_device": "Устройство захвата",
    "digital_camera_device": "Цифровая камера",
    "confidence": "Порог уверенности",
    "confidence_percent": "Порог уверенности",
    "min_confidence": "Минимальная уверенность",
    "inference_size": "Размер для распознавания",
    "inference_interval": "Обрабатывать каждый N-й кадр",
    "tile_size": "Размер области поиска",
    "tile_overlap": "Перекрытие областей",
    "threads": "Потоки распознавания",
    "full_frame": "Распознавать весь кадр",
    "width": "Ширина захвата",
    "height": "Высота захвата",
    "capture_width": "Ширина захвата",
    "capture_height": "Высота захвата",
    "fps": "Частота кадров",
    "camera_fps": "Частота кадров камеры",
    "buffer_size": "Буфер кадров",
    "window_width": "Ширина окна",
    "window_height": "Высота окна",
    "max_frame_width": "Ширина изображения в окне",
    "max_frame_height": "Высота изображения в окне",
    "scale_up": "Увеличивать изображение",
    "stretch_to_fill": "Растягивать изображение на окно",
    "poll_delay_ms": "Интервал обновления окна",
    "near_threshold_margin": "Запас для оранжевой рамки",
    "thickness": "Толщина линии",
    "box_thickness": "Толщина рамки объекта",
    "text_thickness": "Толщина подписи",
    "font_scale": "Размер шрифта",
    "scale_x": "Масштаб рамки по ширине",
    "scale_y": "Масштаб рамки по высоте",
    "font": "Шрифт",
    "preview_enabled": "Веб-просмотр включён",
    "preview_fps": "Частота веб-просмотра",
    "displayport_osd_fps": "Частота обновления OSD",
    "j7_enabled": "Аналоговый выход J7",
    "j7_overlay_enabled": "Наложение на J7",
    "flight_controller_osd_enabled": "OSD полётного контроллера",
    "displayport_proxy_enabled": "Цифровой OSD-прокси",
    "clean_window": "Чистое окно просмотра",
    "color_rgb": "Цвет",
    "text_color_rgb": "Цвет текста",
    "background_color_rgb": "Цвет фона",
    "near_threshold_color_rgb": "Цвет рамки ниже порога",
}

# Подписи групп JSON-конфигураций.
SECTION_LABELS = {
    "profiles": "Профили",
    "settings": "Настройки распознавания",
    "capture": "Захват видео",
    "display": "Окно просмотра",
    "overlay": "Рамка и подпись",
    "box": "Рамка объекта",
    "label": "Подпись объекта",
    "status_panel": "Панель состояния",
    "diagnostics": "Диагностика",
    "detection": "Распознавание",
    "desktop": "Настройки компьютера",
    "video_output": "Видеовыходы",
    "object": "Оформление объекта",
    "osd_text": "Текст OSD",
    "system_status": "Показатели системы",
    "horizon": "Линия горизонта",
    "axis": "Прицельные оси",
    "flight_status": "Показатели полёта",
    "arm_banner": "Статус вооружения",
}

# Допустимые варианты шрифта OpenCV для мышиного списка.
FONT_OPTIONS = [
    ("FONT_HERSHEY_SIMPLEX", "Простой"),
    ("FONT_HERSHEY_PLAIN", "Узкий"),
    ("FONT_HERSHEY_DUPLEX", "Двойной"),
    ("FONT_HERSHEY_COMPLEX", "Комплексный"),
    ("FONT_HERSHEY_TRIPLEX", "Тройной"),
    ("FONT_HERSHEY_COMPLEX_SMALL", "Комплексный узкий"),
    ("FONT_HERSHEY_SCRIPT_SIMPLEX", "Рукописный простой"),
    ("FONT_HERSHEY_SCRIPT_COMPLEX", "Рукописный сложный"),
]

# Стандартные размеры сетевого входа и областей поиска.
SIZE_OPTIONS = [160, 192, 224, 256, 320, 416, 512, 640, 960, 1280, 1536]

# Стандартное оформление Qt-окна с контрастными полями и крупными кнопками.
WINDOW_STYLE = """
QMainWindow, QWidget { background: #111720; color: #e8edf6; font-size: 14px; }
QGroupBox { border: 1px solid #354257; border-radius: 9px; margin-top: 12px; padding: 12px 10px 8px; background: #171f2b; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #79d3e5; font-weight: 600; }
QLabel#sectionHeading { font-size: 16px; font-weight: 700; color: #83d7e8; padding-top: 8px; padding-bottom: 4px; border-bottom: 1px solid #354257; }
QLabel#subsectionHeading { font-size: 14px; font-weight: 600; color: #b8c5d9; padding-top: 5px; padding-bottom: 2px; }
QComboBox, QPushButton { min-height: 38px; border: 1px solid #42516a; border-radius: 7px; background: #222d3c; color: #f0f4fa; padding: 5px 10px; }
QComboBox:hover, QPushButton:hover { border-color: #65bdd0; }
QComboBox QAbstractItemView { background: #202a38; color: #f0f4fa; selection-background-color: #176b5b; }
QPushButton#saveButton { background: #176b5b; border-color: #258d77; font-weight: 650; }
QPushButton#launchButton { background: #245d82; border-color: #3488b3; font-weight: 700; }
QPushButton#launchButton:hover { background: #2d739e; }
QPushButton:disabled { color: #78869a; background: #1b222d; }
QCheckBox { spacing: 9px; }
QCheckBox::indicator { width: 22px; height: 22px; }
QSlider::groove:horizontal { height: 6px; background: #354257; border-radius: 3px; }
QSlider::sub-page:horizontal { background: #40a9ba; border-radius: 3px; }
QSlider::handle:horizontal { width: 18px; margin: -7px 0; border-radius: 9px; background: #73d5e4; }
QScrollArea { border: 0; }
QScrollBar:vertical { width: 20px; background: #29374a; margin: 2px; border-radius: 8px; }
QScrollBar::handle:vertical { min-height: 48px; background: #56bfd2; border: 3px solid #29374a; border-radius: 8px; }
QScrollBar::handle:vertical:hover { background: #8ae5f1; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; border: 0; background: transparent; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: #29374a; border-radius: 8px; }
QStatusBar { background: #151c26; color: #b8c5d9; }
"""


class StaleConfigError(Exception):
    """Файл успели изменить после загрузки, поэтому сохранение остановлено."""


def list_config_files(config_dir: Path = CONFIG_DIR) -> list[str]:
    """Возвращает обычные JSON-файлы прямо из каталога настроек."""
    root = Path(config_dir).resolve(strict=True)
    names: list[str] = []
    for candidate in root.glob("*.json"):
        try:
            if candidate.name.startswith("."):
                continue
            if stat.S_ISREG(candidate.lstat().st_mode) and not candidate.is_symlink():
                names.append(candidate.name)
        except FileNotFoundError:
            continue
    return sorted(names)


def resolve_config_file(config_dir: Path, name: str) -> Path:
    """Проверяет, что выбран обычный JSON непосредственно в config/."""
    root = Path(config_dir).resolve(strict=True)
    if not isinstance(name, str) or not name.endswith(".json") or name.startswith(".") or Path(name).name != name:
        raise ValueError("Разрешены только JSON-файлы непосредственно в каталоге config/.")
    candidate = root / name
    try:
        info = candidate.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError("Файл конфигурации не найден.") from error
    if not stat.S_ISREG(info.st_mode) or candidate.is_symlink():
        raise ValueError("Можно редактировать только обычные файлы, не символические ссылки.")
    if candidate.resolve(strict=True).parent != root:
        raise ValueError("Файл должен находиться непосредственно в каталоге config/.")
    return candidate


def read_config(config_dir: Path, name: str) -> tuple[str, str]:
    """Читает UTF-8 JSON и возвращает его контрольную сумму SHA-256."""
    raw = resolve_config_file(config_dir, name).read_bytes()
    if len(raw) > MAX_CONFIG_BYTES:
        raise ValueError("Файл превышает допустимый размер 1 МиБ.")
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Конфигурация должна быть в кодировке UTF-8.") from error
    return content, hashlib.sha256(raw).hexdigest()


def reject_json_constant(value: str) -> None:
    """Отклоняет нестандартные JSON-значения NaN и Infinity."""
    raise ValueError(f"Недопустимое значение JSON: {value}")


def build_options(project_dir: Path, current: dict[str, Any]) -> dict[str, Any]:
    """Собирает модели, камеры, шрифты и другие варианты списков."""
    models: list[dict[str, str]] = []
    model_dir = Path(project_dir) / "models"
    if model_dir.is_dir():
        for path in sorted(model_dir.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix.lower() in {".onnx", ".pt", ".engine"}:
                relative = path.relative_to(project_dir).as_posix()
                models.append({"value": relative, "label": relative})

    cameras: dict[str, str] = {"auto": "Автовыбор"}
    for index in range(10):
        cameras[str(index)] = f"Камера {index}"
    device_dir = Path("/dev")
    if device_dir.is_dir():
        for path in sorted(device_dir.glob("video*")):
            if path.exists():
                cameras[str(path)] = str(path)
        by_id = device_dir / "v4l" / "by-id"
        if by_id.is_dir():
            for path in sorted(by_id.iterdir()):
                if path.exists():
                    cameras[str(path)] = path.name

    current_strings: set[str] = set()

    def collect_strings(value: Any) -> None:
        """Собирает пути из текущей конфигурации, чтобы не терять их в списке."""
        if isinstance(value, dict):
            for child in value.values():
                collect_strings(child)
        elif isinstance(value, str):
            current_strings.add(value)

    collect_strings(current)
    for value in current_strings:
        if value.lower().endswith((".onnx", ".pt", ".engine")):
            if not any(item["value"] == value for item in models):
                models.append({"value": value, "label": f"Текущая: {value}"})
        if value.startswith("/dev/") or value.isdigit() or value == "auto":
            cameras.setdefault(value, f"Текущее: {value}")

    return {
        "models": models,
        "cameras": [
            {"value": value, "label": label}
            for value, label in sorted(cameras.items(), key=lambda item: (item[0] != "auto", item[0]))
        ],
        "camera_sources": [
            {"value": value, "label": label}
            for value, label in [
                ("easycap", "USB-видеозахват (EasyCap)"),
                ("digital", "Цифровая камера"),
                ("usb", "USB-камера"),
                ("auto", "Автовыбор"),
            ]
        ],
        "positions": [
            {"value": value, "label": label}
            for value, label in [("top", "Сверху"), ("bottom", "Снизу"), ("left", "Слева"), ("right", "Справа")]
        ],
        "fonts": [{"value": value, "label": label} for value, label in FONT_OPTIONS],
        "sizes": SIZE_OPTIONS,
    }


def save_config(config_dir: Path, name: str, content: str, expected_revision: str) -> str:
    """Проверяет JSON и атомарно сохраняет его, без создания резервной копии."""
    path = resolve_config_file(config_dir, name)
    if not isinstance(content, str):
        raise ValueError("Содержимое должно быть текстом.")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_CONFIG_BYTES:
        raise ValueError("Файл превышает допустимый размер 1 МиБ.")
    try:
        parsed = json.loads(content, parse_constant=reject_json_constant)
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Некорректный JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("Корнем конфигурации должен быть JSON-объект.")

    original = path.read_bytes()
    if len(original) > MAX_CONFIG_BYTES:
        raise ValueError("Текущий файл превышает допустимый размер 1 МиБ.")
    actual_revision = hashlib.sha256(original).hexdigest()
    if not isinstance(expected_revision, str) or not hmac.compare_digest(actual_revision, expected_revision):
        raise StaleConfigError("Файл изменился после загрузки. Перечитай его перед сохранением.")
    if encoded == original:
        return actual_revision

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".config_editor_", suffix=".tmp", dir=path.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(encoded)
            temporary.flush()
            os.fsync(temporary.fileno())
        if os.name != "nt":
            os.chmod(temp_path, stat.S_IMODE(path.stat().st_mode))
        os.replace(temp_path, path)
        temp_path = None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    return hashlib.sha256(encoded).hexdigest()


def field_label(key: str) -> str:
    """Показывает русскую подпись, а для неизвестного поля — имя без подчёркиваний."""
    return FIELD_LABELS.get(key, key.replace("_", " ").capitalize())


def section_label(key: str) -> str:
    """Возвращает русское название группы настроек."""
    return SECTION_LABELS.get(key, key.replace("_", " ").capitalize())


def numeric_spec(key: str, value: int | float, path: tuple[str, ...] = ()) -> tuple[int, int, int, float, str] | None:
    """Определяет безопасный диапазон ползунка и формат отображения числа."""
    lower = key.lower()
    if lower == "confidence_percent":
        return (0, 100, 1, 1.0, "%")
    if lower in {"confidence", "min_confidence", "tile_overlap", "near_threshold_margin"} and 0 <= value <= 1:
        return (0, 100, 1, 0.01, "%")
    if "fps" in lower:
        return (1, 120, 1, 1.0, " кадр/с")
    if lower == "threads":
        return (1, 16, 1, 1.0, "")
    if "interval" in lower and any(word in lower for word in ("inference", "poll", "status")):
        return (1, 5000, 10 if lower.endswith("ms") else 1, 1.0, " мс" if lower.endswith("ms") else " кадр")
    if lower in {"inference_size", "tile_size"}:
        return None
    if lower in {"scale_x", "scale_y"}:
        return (10, 300, 5, 0.01, "×")
    if lower in {"bar_width", "bar_height", "buffer_size"} and isinstance(value, int):
        return (1, 240, 1, 1.0, "")
    if lower == "height" and len(path) > 1 and path[-2] == "status_panel":
        return (1, 240, 1, 1.0, " пикс.")
    if lower in {"width", "capture_width", "window_width", "max_frame_width"}:
        return (320, 3840, 16, 1.0, " пикс.")
    if lower in {"height", "capture_height", "window_height", "max_frame_height"}:
        return (240, 2160, 16, 1.0, " пикс.")
    if "thickness" in lower:
        return (1, 10, 1, 1.0, " пикс.")
    if "font_scale" in lower:
        return (10, 300, 5, 0.01, "")
    if any(word in lower for word in ("offset", "left", "top", "right", "bottom")) or lower.startswith("min_"):
        return (-200, 2160, 1, 1.0, " пикс.")
    return None


def config_value(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Находит значение вложенного поля по его JSON-пути."""
    value: Any = data
    for segment in path:
        value = value[segment]
    return value


def set_config_value(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Обновляет вложенное поле конфигурации по JSON-пути."""
    target: Any = data
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = value


class ConfigEditorWindow(QMainWindow):
    """Главное нативное окно выбора, изменения и сохранения конфигураций."""

    def __init__(self, project_dir: Path = PROJECT_DIR, config_dir: Path = CONFIG_DIR) -> None:
        """Создаёт окно управления настройками без браузера и сетевого сервера."""
        super().__init__()
        self.project_dir = Path(project_dir).resolve()
        self.config_dir = Path(config_dir).resolve(strict=True)
        self.current_name: str | None = None
        self.current_revision: str | None = None
        self.current_data: dict[str, Any] | None = None
        self.original_data: dict[str, Any] | None = None
        self.options: dict[str, Any] = {}
        self.recognition_process: subprocess.Popen[Any] | None = None

        self.setWindowTitle("VideoT16 — настройки")
        self.resize(1050, 820)
        self.setMinimumSize(780, 560)
        self.setStyleSheet(WINDOW_STYLE)
        self._build_window()
        self._load_file_list()

    def _build_window(self) -> None:
        """Создаёт список файлов, прокручиваемую форму, кнопки и строку статуса."""
        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 16, 18, 12)
        outer.setSpacing(12)

        heading = QLabel("VideoT16 — настройки")
        heading.setStyleSheet("font-size: 22px; font-weight: 700; color: #83d7e8;")
        hint = QLabel("Выбирай параметры мышью. Изменения попадут в систему после сохранения и следующего запуска.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #a8b4c8;")
        outer.addWidget(heading)
        outer.addWidget(hint)

        file_row = QHBoxLayout()
        file_label = QLabel("Файл настроек:")
        file_label.setMinimumWidth(125)
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(300)
        self.reload_button = QPushButton("Перечитать")
        self.reload_button.clicked.connect(self._reload_selected)
        file_row.addWidget(file_label)
        file_row.addWidget(self.file_combo, 1)
        file_row.addWidget(self.reload_button)
        outer.addLayout(file_row)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.form_root = QWidget()
        self.form_layout = QVBoxLayout(self.form_root)
        self.form_layout.setContentsMargins(4, 2, 4, 8)
        self.form_layout.setSpacing(9)
        self.scroll_area.setWidget(self.form_root)
        outer.addWidget(self.scroll_area, 1)

        button_row = QHBoxLayout()
        self.launch_button = QPushButton("Запуск распознавания")
        self.launch_button.setObjectName("launchButton")
        self.launch_button.clicked.connect(self._launch_recognition)
        self.save_button = QPushButton("Сохранить настройки")
        self.save_button.setObjectName("saveButton")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_current)
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.close)
        button_row.addWidget(self.launch_button)
        button_row.addStretch(1)
        button_row.addWidget(self.save_button)
        button_row.addWidget(close_button)
        outer.addLayout(button_row)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Готово к загрузке настроек")
        self.file_combo.currentTextChanged.connect(self._select_file)

    def _load_file_list(self) -> None:
        """Заполняет список JSON-файлов из config/ и загружает первый файл."""
        try:
            names = list_config_files(self.config_dir)
        except (OSError, ValueError) as error:
            self._show_error("Не удалось прочитать список настроек", error)
            return
        blocker = QSignalBlocker(self.file_combo)
        self.file_combo.clear()
        self.file_combo.addItems(names)
        del blocker
        if not names:
            self.statusBar().showMessage("В каталоге config/ нет JSON-файлов")
            self._show_empty_message("В каталоге config/ не найдено файлов настроек.")
            return
        self._load_config(names[0])

    def _show_empty_message(self, message: str) -> None:
        """Показывает объяснение, если конфигурации отсутствуют или не читаются."""
        self._clear_form()
        self.form_layout.addWidget(QLabel(message))
        self.form_layout.addStretch(1)

    def _clear_form(self) -> None:
        """Удаляет предыдущие элементы формы перед построением новой конфигурации."""
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _select_file(self, name: str) -> None:
        """Переключает файл и возвращает выбор, если владелец отменил смену."""
        if not name or name == self.current_name:
            return
        if self._is_dirty() and not self._confirm_discard("переключить файл"):
            blocker = QSignalBlocker(self.file_combo)
            self.file_combo.setCurrentText(self.current_name or "")
            del blocker
            return
        self._load_config(name)

    def _reload_selected(self) -> None:
        """Повторно читает выбранный JSON после подтверждения потери несохранённых правок."""
        if self._is_dirty() and not self._confirm_discard("перечитать файл"):
            return
        name = self.file_combo.currentText()
        if name:
            self._load_config(name)

    def _load_config(self, name: str) -> None:
        """Читает конфигурацию, её ревизию и перерисовывает нативные элементы управления."""
        try:
            content, revision = read_config(self.config_dir, name)
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("Корнем настроек должен быть JSON-объект.")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            self._show_error("Не удалось загрузить настройки", error)
            self.statusBar().showMessage(f"Ошибка загрузки: {error}")
            self._show_empty_message("Файл не удалось прочитать. Выбери другой файл или нажми «Перечитать».")
            return

        self.current_name = name
        self.current_revision = revision
        self.current_data = data
        self.original_data = copy.deepcopy(data)
        self.options = build_options(self.project_dir, data)
        self._render_config()
        self.statusBar().showMessage(f"Загружено: {name}")
        self.save_button.setEnabled(False)

    def _render_config(self) -> None:
        """Строит форму по всем секциям, показывая непонятные значения только для чтения."""
        self._clear_form()
        if self.current_data is None:
            self.form_layout.addStretch(1)
            return
        self._render_mapping(self.form_layout, self.current_data, ())
        self.form_layout.addStretch(1)

    def _render_mapping(self, parent_layout: QVBoxLayout | QFormLayout, values: dict[str, Any], path: tuple[str, ...]) -> None:
        """Рисует верхние секции в рамках, а вложенные разделы — плоскими заголовками."""
        for key, value in values.items():
            current_path = (*path, key)
            if isinstance(value, dict):
                if not path:
                    group = QGroupBox(section_label(key))
                    group_layout = QFormLayout(group)
                    group_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
                    group_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                    self._render_mapping(group_layout, value, current_path)
                    parent_layout.addWidget(group)
                else:
                    if len(path) == 1 and path[0] == "profiles":
                        title = f"Профиль: {key}"
                    else:
                        title = section_label(key)
                    heading = QLabel(title)
                    heading.setObjectName("sectionHeading" if len(path) == 1 else "subsectionHeading")
                    if isinstance(parent_layout, QFormLayout):
                        parent_layout.addRow(heading)
                    else:
                        parent_layout.addWidget(heading)
                    self._render_mapping(parent_layout, value, current_path)
            else:
                control = self._make_control(key, value, current_path)
                if control is None:
                    control = QLabel(self._display_value(value))
                    control.setWordWrap(True)
                    control.setStyleSheet("color: #b8c5d9;")
                if isinstance(parent_layout, QFormLayout):
                    parent_layout.addRow(field_label(key) + ":", control)
                else:
                    row = QWidget()
                    row_layout = QHBoxLayout(row)
                    row_layout.setContentsMargins(4, 2, 4, 2)
                    label = QLabel(field_label(key) + ":")
                    label.setMinimumWidth(250)
                    row_layout.addWidget(label)
                    row_layout.addWidget(control, 1)
                    parent_layout.addWidget(row)

    def _make_control(self, key: str, value: Any, path: tuple[str, ...]) -> QWidget | None:
        """Создаёт переключатель, список, ползунок или выбор цвета для значения."""
        if isinstance(value, bool):
            checkbox = QCheckBox("Включено" if value else "Выключено")
            checkbox.setChecked(value)
            checkbox.toggled.connect(lambda enabled, field=checkbox, field_path=path: self._set_value(field_path, enabled, field.setText))
            return checkbox

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key.lower() in {"inference_size", "tile_size"}:
                choices = self.options.get("sizes", SIZE_OPTIONS)
                return self._make_combo(key, value, [(size, f"{size} пикс.") for size in choices], path)
            spec = numeric_spec(key, value, path)
            if spec is None:
                return None
            minimum, maximum, step, scale, suffix = spec
            slider = QSlider(Qt.Orientation.Horizontal)
            stored_value = round(float(value) / scale)
            slider.setRange(min(minimum, stored_value), max(maximum, stored_value))
            slider.setSingleStep(step)
            slider.setPageStep(max(step, (maximum - minimum) // 20))
            slider.setValue(stored_value)
            display = QLabel()
            display.setMinimumWidth(86)
            display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            keep_integer = isinstance(value, int)

            def format_number(raw: int, multiplier: float = scale, unit: str = suffix, as_integer: bool = keep_integer) -> tuple[int | float, str]:
                """Форматирует значение ползунка, не меняя конфигурацию при показе формы."""
                scaled = raw * multiplier
                number = int(round(scaled)) if as_integer else round(scaled, 4)
                shown = f"{raw:g}{unit}" if unit != "×" else f"{number:.2f}×"
                if unit == "" and multiplier != 1.0:
                    shown = f"{number:.2f}"
                return number, shown

            def update_number(raw: int, field_path: tuple[str, ...] = path, multiplier: float = scale, unit: str = suffix, label: QLabel = display, as_integer: bool = keep_integer) -> None:
                """Сохраняет выбранное значение и обновляет подпись после движения ползунка."""
                number, shown = format_number(raw, multiplier, unit, as_integer)
                self._set_value(field_path, number)
                label.setText(shown)

            # Первичный показ текста не должен округлять исходный JSON или включать режим изменений.
            display.setText(format_number(slider.value())[1])
            slider.valueChanged.connect(update_number)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(display)
            return row

        if isinstance(value, list) and len(value) == 3 and all(isinstance(part, int) and 0 <= part <= 255 for part in value) and key.endswith("_rgb"):
            color_button = QPushButton()
            self._set_color_button(color_button, value)
            color_button.clicked.connect(lambda checked=False, field=color_button, field_path=path: self._choose_color(field, field_path))
            return color_button

        if isinstance(value, str):
            lower = key.lower()
            choices: list[tuple[Any, str]] | None = None
            if lower in {"model", "model_path"}:
                choices = [(item["value"], item["label"]) for item in self.options["models"]]
            elif lower in {"source", "camera_device", "digital_camera_device"}:
                choices = [(item["value"], item["label"]) for item in self.options["cameras"]]
            elif lower == "camera_source":
                choices = [(item["value"], item["label"]) for item in self.options["camera_sources"]]
            elif lower == "font":
                choices = [(item["value"], item["label"]) for item in self.options["fonts"]]
            elif lower == "position":
                choices = [(item["value"], item["label"]) for item in self.options["positions"]]
            if choices:
                return self._make_combo(key, value, choices, path)
        return None

    def _make_combo(self, key: str, value: Any, choices: list[tuple[Any, str]], path: tuple[str, ...]) -> QComboBox:
        """Строит список значений и сохраняет фактический тип JSON-параметра."""
        combo = QComboBox()
        present_values = [choice[0] for choice in choices]
        if value not in present_values:
            choices.insert(0, (value, f"Текущее значение: {value}"))
        for option_value, label in choices:
            combo.addItem(label, option_value)
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

        def update_selection(index: int, field: QComboBox = combo, field_path: tuple[str, ...] = path) -> None:
            """Применяет выбранное мышью значение к редактируемому JSON."""
            if index >= 0:
                self._set_value(field_path, field.currentData())

        combo.currentIndexChanged.connect(update_selection)
        return combo

    def _set_value(self, path: tuple[str, ...], value: Any, callback: Any = None) -> None:
        """Обновляет значение JSON и включает кнопку сохранения при изменениях."""
        if self.current_data is None:
            return
        set_config_value(self.current_data, path, value)
        if callback is not None and path[-1] in {"preview_enabled", "j7_enabled", "full_frame"}:
            callback("Включено" if value else "Выключено")
        self.save_button.setEnabled(self._is_dirty())
        if self._is_dirty():
            self.statusBar().showMessage("Есть несохранённые изменения")
        else:
            self.statusBar().showMessage(f"Загружено: {self.current_name}")

    def _choose_color(self, button: QPushButton, path: tuple[str, ...]) -> None:
        """Открывает палитру Qt и записывает выбранный цвет в формате RGB."""
        original = config_value(self.current_data or {}, path)
        color = QColorDialog.getColor(QColor(*original), self, field_label(path[-1]))
        if not color.isValid():
            return
        rgb = [color.red(), color.green(), color.blue()]
        self._set_color_button(button, rgb)
        self._set_value(path, rgb)

    @staticmethod
    def _set_color_button(button: QPushButton, rgb: list[int]) -> None:
        """Показывает цвет и его RGB-код на кнопке выбора палитры."""
        color = QColor(*rgb)
        button.setText(f"{color.name().upper()}  (нажми, чтобы выбрать)")
        button.setStyleSheet(
            f"QPushButton {{ background: {color.name()}; color: {'#111720' if color.lightness() > 145 else '#ffffff'}; "
            "border: 1px solid #65738a; border-radius: 7px; min-height: 36px; padding: 4px 9px; }"
        )

    @staticmethod
    def _display_value(value: Any) -> str:
        """Форматирует справочные и неподдерживаемые значения без возможности правки."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _is_dirty(self) -> bool:
        """Проверяет, отличается ли открытый объект настроек от исходного."""
        return self.current_data is not None and self.original_data is not None and self.current_data != self.original_data

    def _confirm_discard(self, action: str) -> bool:
        """Спрашивает мышиное подтверждение перед отменой несохранённых изменений."""
        answer = QMessageBox.question(
            self,
            "Несохранённые настройки",
            f"Сначала сохранить изменения или отменить их и {action}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self._save_current()
        return answer == QMessageBox.StandardButton.Discard

    def _save_current(self) -> bool:
        """Сохраняет только валидный JSON и не создаёт резервную копию."""
        if self.current_name is None or self.current_revision is None or self.current_data is None:
            return False
        content = json.dumps(self.current_data, ensure_ascii=False, indent=2) + "\n"
        try:
            revision = save_config(self.config_dir, self.current_name, content, self.current_revision)
        except StaleConfigError as error:
            self.save_button.setEnabled(True)
            self.statusBar().showMessage(str(error))
            self._show_error("Настройки не сохранены", error)
            return False
        except (OSError, ValueError) as error:
            self.save_button.setEnabled(True)
            self.statusBar().showMessage(f"Ошибка сохранения: {error}")
            self._show_error("Настройки не сохранены", error)
            return False

        self.current_revision = revision
        self.original_data = copy.deepcopy(self.current_data)
        self.save_button.setEnabled(False)
        self.statusBar().showMessage(f"Сохранено: {self.current_name}. Применится при следующем запуске.")
        return True

    def _launch_recognition(self) -> None:
        """Сохраняет правки и запускает живое распознавание отдельным процессом."""
        if self.current_data is None:
            self.statusBar().showMessage("Сначала выбери файл настроек.")
            return
        if self._is_dirty():
            answer = QMessageBox.question(
                self,
                "Запуск VideoT16",
                "Перед запуском сохранить изменённые настройки?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer != QMessageBox.StandardButton.Save or not self._save_current():
                return

        if self.recognition_process is not None and self.recognition_process.poll() is None:
            self.statusBar().showMessage("Распознавание уже запущено. Нажми Esc в окне видео, чтобы остановить.")
            return

        if os.name == "nt":
            launcher = self.project_dir / "start.py"
            command = [sys.executable, str(launcher)]
        else:
            launcher = self.project_dir / "lcam"
            command = [str(launcher)]
        if not launcher.is_file():
            self._show_error("Не удалось запустить распознавание", FileNotFoundError(f"Файл запуска не найден: {launcher}"))
            return
        if os.name != "nt" and not os.access(launcher, os.X_OK):
            self._show_error("Не удалось запустить распознавание", PermissionError(f"Файл не исполняемый: {launcher}"))
            return

        try:
            self.recognition_process = subprocess.Popen(
                command,
                cwd=self.project_dir,
                stdin=subprocess.DEVNULL,
                start_new_session=os.name != "nt",
            )
        except OSError as error:
            self._show_error("Не удалось запустить распознавание", error)
            return
        self.statusBar().showMessage("Распознавание запущено. Нажми Esc в окне видео, чтобы остановить.")
        self.close()

    def _show_error(self, title: str, error: Exception) -> None:
        """Показывает ошибку в самом приложении, а не в скрытом терминале."""
        QMessageBox.critical(self, title, str(error))

    def closeEvent(self, event: Any) -> None:
        """Сохраняет несохранённые изменения и закрывает окно без блокирующего диалога."""
        if self._is_dirty():
            if not self._save_current():
                event.ignore()
                return
        event.accept()


def main() -> int:
    """Запускает нативное приложение Qt и сообщает ошибки в окне."""
    application = QApplication(sys.argv)
    application.setApplicationName("VideoT16 — настройки")
    application.setOrganizationName("VideoT16")
    window = ConfigEditorWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
