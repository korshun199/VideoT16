#!/usr/bin/env python3
"""Оконный редактор основного JSON-профиля VideoT16."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.json"
TEMP_PATH = PROJECT_ROOT / "tmp"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_pipeline_trial import load_profile, validate_profile  # noqa: E402

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class FieldSpec:
    """Описывает поле формы, его тип и встроенную справку."""

    path: tuple[str, ...]
    group: str
    label: str
    kind: str
    description: str
    allowed: str
    minimum: float | None = None
    maximum: float | None = None
    decimals: int = 2
    step: float = 0.01
    choices: tuple[tuple[str, str], ...] = ()
    enabled: bool = True


def _spec(
    path: str,
    group: str,
    label: str,
    kind: str,
    description: str,
    allowed: str,
    minimum: float | None = None,
    maximum: float | None = None,
    decimals: int = 2,
    step: float = 0.01,
    choices: tuple[tuple[str, str], ...] = (),
    enabled: bool = True,
) -> FieldSpec:
    """Создаёт компактное описание поля по пути через точки."""
    return FieldSpec(tuple(path.split(".")), group, label, kind, description, allowed, minimum, maximum, decimals, step, choices, enabled)


FIELD_SPECS = (
    _spec("description", "Распознавание", "Описание профиля", "text", "Короткая заметка для человека. На запуск распознавания и параметры камеры не влияет.", "Любой короткий текст."),
    _spec("settings.source", "Распознавание", "Источник видео", "text", "Номер камеры или путь к USB-видеоустройству. Для Linux обычно указывают /dev/videoN или путь /dev/v4l/by-id/…; строка передаётся OpenCV.", "Непустой номер или путь. Само устройство должно быть подключено."),
    _spec("settings.model", "Распознавание", "Файл модели", "model", "Путь к модели распознавания. Относительный путь считается от корня VideoT16. Кнопка выбора файла помогает найти .onnx или .pt.", "Существующий файл модели .onnx или .pt."),
    _spec("settings.confidence", "Распознавание", "Порог рамки", "percent", "Минимальная уверенность детектора, при которой основная рамка считается обнаружением.", "Больше 0% и до 100% включительно.", 0.1, 100.0, 1, 1.0),
    _spec("settings.min_confidence", "Распознавание", "Нижний порог кандидата", "percent", "Нижний порог для обработки кандидатов до окончательной отрисовки. Он должен быть не выше порога рамки.", "Больше 0%; не выше поля «Порог рамки».", 0.1, 100.0, 1, 1.0),
    _spec("settings.tile_size", "Распознавание", "Размер фрагмента", "int", "Размер квадратного участка изображения, который отправляется модели за один раз.", "Целое число от 32 до 4096 пикселей.", 32, 4096, 0, 32),
    _spec("settings.tile_overlap", "Распознавание", "Перекрытие фрагментов", "percent", "Насколько соседние участки изображения заходят друг на друга. Большое перекрытие уменьшает пропуски на границах, но повышает нагрузку.", "От 0% до значения меньше 50%.", 0, 49.9, 1, 1.0),
    _spec("settings.full_frame", "Распознавание", "Обрабатывать целый кадр", "bool", "Включает распознавание полного кадра вместо обычного прохода по фрагментам.", "Включено / выключено."),
    _spec("capture.pixel_format", "Захват", "Формат USB-видео", "enum", "Формат пикселей, который программа просит у видеозахвата. Устройство может не принять запрос; фактический режим записывается в журнал.", "MJPG — сжатый видеопоток; YUYV — несжатый.", choices=(("MJPG", "MJPG — сжатый"), ("YUYV", "YUYV — несжатый"))),
    _spec("capture.width", "Захват", "Ширина кадра", "int", "Запрашиваемая ширина входного кадра. Это не гарантирует, что оцифровщик поддерживает такую комбинацию с высотой, форматом и FPS.", "Целое число от 32 до 7680 пикселей.", 32, 7680, 0, 16),
    _spec("capture.height", "Захват", "Высота кадра", "int", "Запрашиваемая высота входного кадра. Проверьте журнал: драйвер может выбрать ближайший поддерживаемый режим.", "Целое число от 32 до 7680 пикселей.", 32, 7680, 0, 16),
    _spec("capture.fps", "Захват", "Частота кадров", "float", "Запрашиваемая частота захвата в кадрах за секунду. Реальное значение зависит от USB-устройства, разрешения и формата.", "Больше 0 и не выше 240 кадров/с.", 0.1, 240, 1, 1.0),
    _spec("capture.buffer_size", "Захват", "Буфер захвата", "int", "Количество кадров в буфере драйвера. Небольшой буфер обычно уменьшает задержку, если устройство справляется с потоком.", "Целое число от 1 до 32.", 1, 32, 0, 1),
    _spec("display.window_title", "Окно", "Заголовок окна", "text", "Текст в заголовке окна видеовывода.", "Любой короткий текст."),
    _spec("display.window_width", "Окно", "Ширина окна", "int", "Запрашиваемая ширина окна вывода. Полноэкранный режим может переопределить размер.", "Целое число от 320 до 7680 пикселей.", 320, 7680, 0, 80),
    _spec("display.window_height", "Окно", "Высота окна", "int", "Запрашиваемая высота окна вывода. Полноэкранный режим может переопределить размер.", "Целое число от 240 до 4320 пикселей.", 240, 4320, 0, 80),
    _spec("display.max_frame_width", "Окно", "Максимальная ширина видео", "int", "Максимальная ширина изображения внутри окна. Соотношение сторон исходного кадра сохраняется.", "Целое число от 1 до 7680 пикселей.", 1, 7680, 0, 80),
    _spec("display.max_frame_height", "Окно", "Максимальная высота видео", "int", "Максимальная высота изображения внутри окна, без полосы показателей.", "Целое число от 1 до 4320 пикселей.", 1, 4320, 0, 80),
    _spec("display.scale_up", "Окно", "Увеличивать маленькое видео", "bool", "Разрешает растянуть источник с меньшим разрешением до размеров окна. Увеличение может сделать изображение менее чётким.", "Включено / выключено."),
    _spec("display.fullscreen", "Окно", "Полноэкранный вывод", "bool", "Запускать окно видеовывода на весь экран.", "Включено / выключено."),
    _spec("display.poll_delay_ms", "Окно", "Пауза цикла окна", "int", "Короткая пауза цикла обработки оконных событий в миллисекундах.", "Целое число от 0 до 1000 мс.", 0, 1000, 0, 1),
    _spec("diagnostics.console_status_interval_ms", "Потоки и задержка", "Интервал сообщений в консоли", "int", "Как часто печатать краткую диагностическую строку. Ноль отключает периодический вывод в консоль.", "Целое число от 0 до 60000 мс.", 0, 60000, 0, 100),
    _spec("pipeline_v2.threads", "Потоки и задержка", "Потоки распознавания", "int", "Число потоков ONNX Runtime. Больше потоков не всегда ускоряет распознавание и может занять весь процессор.", "Целое число от 1 до 64.", 1, 64, 0, 1),
    _spec("pipeline_v2.submission_interval_frames", "Потоки и задержка", "Распознавать каждый N-й кадр", "int", "Передавать на распознавание каждый N-й кадр. Значение 1 означает каждый кадр.", "Целое число от 1 до 120.", 1, 120, 0, 1),
    _spec("pipeline_v2.display_fps", "Потоки и задержка", "Частота обновления окна", "int", "Целевая частота выдачи кадров в окно. Она ограничивает скорость вывода, но не может ускорить сам источник.", "Целое число от 1 до 120 кадров/с.", 1, 120, 0, 1),
    _spec("pipeline_v2.output_delay_ms", "Потоки и задержка", "Задержка видеовывода", "int", "Задержка выхода в миллисекундах, чтобы приложить рамку к соответствующему кадру. Для этой задержки нужен буфер памяти.", "Целое число от 100 до 5000 мс; вместимость буфера проверяется перед сохранением.", 100, 5000, 0, 100),
    _spec("pipeline_v2.output_buffer_max_mb", "Потоки и задержка", "Предел памяти буфера", "int", "Максимальный объём памяти для очереди выходного видео. Недостаточный объём для выбранных FPS, разрешения и задержки не даст сохранить профиль.", "Целое число от 64 до 2048 МиБ.", 64, 2048, 0, 64),
    _spec("pipeline_v2.box_hold_ms", "Потоки и задержка", "Удержание рамки", "int", "Сколько миллисекунд сохранять последнюю рамку после пропуска детектора.", "Целое число от 0 до 5000 мс.", 0, 5000, 0, 100),
    _spec("pipeline_v2.max_match_distance_fraction", "Потоки и задержка", "Допустимое смещение рамки", "float", "Максимальное относительное расстояние для сопоставления результатов рамки с соседними кадрами.", "Больше 0 и не выше 1.00.", 0.01, 1.0, 2, 0.01),
    _spec("fast_scan.enabled", "Быстрый поиск", "Ускоренный поиск области", "bool", "После обнаружения проверять область рядом с объектом, чередуя её с обзором всего кадра.", "Включено / выключено."),
    _spec("fast_scan.focus_min_confidence", "Быстрый поиск", "Порог фокусировки", "percent", "Минимальная уверенность кандидата, чтобы временно направить быстрый поиск на его область.", "Больше 0% и не выше 100%.", 0.1, 100, 1, 1.0),
    _spec("fast_scan.focus_crop_width", "Быстрый поиск", "Ширина области поиска", "int", "Ширина выреза вокруг последнего кандидата в пикселях.", "Целое число от 32 до 7680 пикселей.", 32, 7680, 0, 16),
    _spec("fast_scan.focus_crop_height", "Быстрый поиск", "Высота области поиска", "int", "Высота выреза вокруг последнего кандидата в пикселях.", "Целое число от 32 до 7680 пикселей.", 32, 7680, 0, 16),
    _spec("fast_scan.focus_lost_after_results", "Быстрый поиск", "Возврат к полному обзору", "int", "Сколько результатов без подходящего кандидата переждать, прежде чем снова сканировать весь кадр.", "Целое число от 1 до 120 результатов.", 1, 120, 0, 1),
    _spec("overlay.near_threshold_margin", "Рамка и подпись", "Предупредительный диапазон", "percent", "Диапазон ниже основного порога, в котором рамка получает предупредительный цвет.", "От 0% до значения меньше 50%.", 0, 49.9, 1, 1.0),
    _spec("overlay.near_threshold_color_rgb", "Рамка и подпись", "Предупредительный цвет", "color", "Цвет рамки кандидата чуть ниже основного порога. В файле хранится как RGB-массив.", "Каждый компонент красный/зелёный/синий: целое число 0..255."),
    _spec("overlay.box.scale_x", "Рамка и подпись", "Ширина рамки", "float", "Масштаб ширины рамки относительно найденного прямоугольника.", "Число от 0.1 до 3.0.", 0.1, 3, 2, 0.1),
    _spec("overlay.box.scale_y", "Рамка и подпись", "Высота рамки", "float", "Масштаб высоты рамки относительно найденного прямоугольника.", "Число от 0.1 до 3.0.", 0.1, 3, 2, 0.1),
    _spec("overlay.box.thickness", "Рамка и подпись", "Толщина линии рамки", "int", "Толщина цветной рамки в пикселях.", "Целое число от 1 до 20 пикселей.", 1, 20, 0, 1),
    _spec("overlay.box.color_rgb", "Рамка и подпись", "Цвет рамки", "color", "Основной цвет рамки. В JSON значения перечислены в порядке красный, зелёный, синий.", "Каждый компонент RGB: целое число 0..255."),
    _spec("overlay.label.font", "Рамка и подпись", "Шрифт подписи", "enum", "Встроенный шрифт OpenCV для названия объекта и процента уверенности.", "Выберите один из встроенных шрифтов OpenCV.", choices=(("FONT_HERSHEY_SIMPLEX", "Простой"), ("FONT_HERSHEY_PLAIN", "Тонкий простой"), ("FONT_HERSHEY_DUPLEX", "Двойной"), ("FONT_HERSHEY_COMPLEX", "Сложный"), ("FONT_HERSHEY_TRIPLEX", "Тройной"), ("FONT_HERSHEY_COMPLEX_SMALL", "Сложный мелкий"), ("FONT_HERSHEY_SCRIPT_SIMPLEX", "Рукописный простой"), ("FONT_HERSHEY_SCRIPT_COMPLEX", "Рукописный сложный"))),
    _spec("overlay.label.font_scale", "Рамка и подпись", "Размер подписи", "float", "Масштаб размера подписи над рамкой.", "Больше 0 и не выше 5.0.", 0.05, 5, 2, 0.05),
    _spec("overlay.label.thickness", "Рамка и подпись", "Толщина подписи", "int", "Толщина линий букв подписи.", "Целое число от 1 до 10.", 1, 10, 0, 1),
    _spec("overlay.label.offset_y", "Рамка и подпись", "Отступ подписи", "int", "Вертикальный отступ подписи от верхнего края рамки в пикселях.", "Целое число от 0 до 300 пикселей.", 0, 300, 0, 2),
    _spec("overlay.label.min_y", "Рамка и подпись", "Минимальная высота подписи", "int", "Минимальная координата текста сверху кадра, чтобы подпись не ушла за край.", "Целое число от 0 до 2000 пикселей.", 0, 2000, 0, 10),
    _spec("overlay.label.color_rgb", "Рамка и подпись", "Цвет подписи", "color", "Цвет текста с названием объекта и уверенностью. В JSON записывается RGB-массив.", "Каждый компонент RGB: целое число 0..255."),
    _spec("status_panel.position", "Показатели", "Положение полосы", "readonly_enum", "В текущем видеоприложении строка показателей всегда рисуется снизу. Это поле сохранено для совместимости и сейчас не меняет расположение.", "Только bottom — поле заблокировано до появления поддержки других положений.", choices=(("bottom", "Снизу"),), enabled=False),
    _spec("status_panel.height", "Показатели", "Высота полосы", "int", "Высота нижней полосы технических показателей в пикселях.", "Целое число от 1 до 500 пикселей.", 1, 500, 0, 2),
    _spec("status_panel.background_color_rgb", "Показатели", "Фон полосы", "color", "Цвет фона нижней полосы показателей. В JSON записывается RGB-массив.", "Каждый компонент RGB: целое число 0..255."),
    _spec("status_panel.left", "Показатели", "Левый отступ текста", "int", "Отступ строки показателей слева в пикселях.", "Целое число от 0 до 2000 пикселей.", 0, 2000, 0, 4),
    _spec("status_panel.text_y", "Показатели", "Положение текста по высоте", "int", "Вертикальная позиция строки внутри нижней полосы в пикселях.", "Целое число от 0 до 500 пикселей.", 0, 500, 0, 2),
    _spec("status_panel.font", "Показатели", "Шрифт строки показателей", "enum", "Встроенный шрифт OpenCV для CPU, температуры, уверенности и диагностических значений.", "Выберите один из встроенных шрифтов OpenCV.", choices=(("FONT_HERSHEY_SIMPLEX", "Простой"), ("FONT_HERSHEY_PLAIN", "Тонкий простой"), ("FONT_HERSHEY_DUPLEX", "Двойной"), ("FONT_HERSHEY_COMPLEX", "Сложный"), ("FONT_HERSHEY_TRIPLEX", "Тройной"), ("FONT_HERSHEY_COMPLEX_SMALL", "Сложный мелкий"), ("FONT_HERSHEY_SCRIPT_SIMPLEX", "Рукописный простой"), ("FONT_HERSHEY_SCRIPT_COMPLEX", "Рукописный сложный"))),
    _spec("status_panel.font_scale", "Показатели", "Размер текста полосы", "float", "Масштаб размера текста показателей.", "Больше 0 и не выше 5.0.", 0.05, 5, 2, 0.05),
    _spec("status_panel.text_color_rgb", "Показатели", "Цвет текста полосы", "color", "Цвет текста технических показателей. В JSON записывается RGB-массив.", "Каждый компонент RGB: целое число 0..255."),
    _spec("status_panel.thickness", "Показатели", "Толщина текста полосы", "int", "Толщина линий букв в нижней полосе.", "Целое число от 1 до 10.", 1, 10, 0, 1),
    _spec("status_panel.max_text_chars", "Показатели", "Максимум символов", "int", "Ограничивает длину диагностической строки, чтобы она не выходила за экран.", "Целое число от 1 до 500 символов.", 1, 500, 0, 5),
)

GROUP_ORDER = ("Распознавание", "Захват", "Окно", "Потоки и задержка", "Быстрый поиск", "Рамка и подпись", "Показатели")


def get_path(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    """Возвращает значение JSON по пути из имён вложенных ключей."""
    value: Any = document
    for key in path:
        value = value[key]
    return value


def set_path(document: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Записывает значение JSON, не затрагивая соседние и неизвестные ключи."""
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def leaf_paths(value: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Собирает пути всех конечных значений JSON для контроля покрытия формы."""
    if isinstance(value, dict):
        return set().union(*(leaf_paths(child, prefix + (key,)) for key, child in value.items())) if value else set()
    return {prefix}


class SettingsDocument:
    """Читает, проверяет и атомарно сохраняет JSON с проверкой чужих изменений."""

    def __init__(self, path: Path = SETTINGS_PATH, temp_dir: Path = TEMP_PATH) -> None:
        """Сохраняет пути и загружает исходный JSON вместе с его ревизией."""
        self.path = path
        self.temp_dir = temp_dir
        self.backup_path = temp_dir / "settings.json.bak"
        self.original_bytes = path.read_bytes()
        self.document = json.loads(self.original_bytes.decode("utf-8"))
        if not isinstance(self.document, dict):
            raise ValueError("Корень config/settings.json должен быть JSON-объектом")

    def validate(self, document: dict[str, Any]) -> None:
        """Проверяет синтаксис JSON и каждый профиль действующим валидатором приложения."""
        profiles = document.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("В config/settings.json должен быть хотя бы один профиль")
        self._validate_editor_values(profiles)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=self.temp_dir, delete=False) as temporary_file:
                json.dump(document, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
                temporary_path = Path(temporary_file.name)
            for profile_name in profiles:
                validate_profile(load_profile(profile_name, temporary_path))
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _validate_editor_values(profiles: dict[str, Any]) -> None:
        """Проверяет диапазоны и списки выбора, показанные пользователю в форме."""
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                raise ValueError(f"Профиль {profile_name} должен быть объектом JSON")
            for spec in FIELD_SPECS:
                if not SettingsEditor._has_path(profile, spec.path):
                    continue
                value = get_path(profile, spec.path)
                field_name = f"profiles.{profile_name}.{'.'.join(spec.path)}"
                if spec.kind in ("int", "float", "percent"):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"{field_name}: нужно указать числовое значение")
                    compared_value = value * 100 if spec.kind == "percent" else value
                    if spec.minimum is not None and compared_value < spec.minimum:
                        raise ValueError(f"{field_name}: допустимый минимум {spec.minimum:g}")
                    if spec.maximum is not None and compared_value > spec.maximum:
                        raise ValueError(f"{field_name}: допустимый максимум {spec.maximum:g}")
                elif spec.kind in ("enum", "readonly_enum"):
                    allowed_values = {stored_value for stored_value, _label in spec.choices}
                    if value not in allowed_values:
                        raise ValueError(f"{field_name}: выберите одно из значений {', '.join(sorted(allowed_values))}")
                elif spec.kind == "color":
                    if not isinstance(value, list) or len(value) != 3 or any(
                        isinstance(component, bool) or not isinstance(component, int) or not 0 <= component <= 255
                        for component in value
                    ):
                        raise ValueError(f"{field_name}: укажите три целых компонента RGB в диапазоне 0..255")
                elif spec.kind == "bool" and not isinstance(value, bool):
                    raise ValueError(f"{field_name}: допустимо только true или false")
                elif spec.kind in ("text", "model") and not isinstance(value, str):
                    raise ValueError(f"{field_name}: нужно текстовое значение")
                elif spec.path in (("settings", "source"), ("settings", "model")) and not value.strip():
                    raise ValueError(f"{field_name}: поле не может быть пустым")
                if spec.kind == "model" and Path(value).suffix.lower() not in (".onnx", ".pt"):
                    raise ValueError(f"{field_name}: ожидается файл модели .onnx или .pt")

    def ensure_current(self) -> None:
        """Не разрешает запускать редактор с устаревшим представлением файла."""
        if self.path.read_bytes() != self.original_bytes:
            raise RuntimeError("Файл настроек изменён извне. Закройте редактор, проверьте новый файл и повторите.")

    def save(self, document: dict[str, Any]) -> Path:
        """Проверяет и сохраняет новые настройки, оставляя одну последнюю копию исходника."""
        self.validate(document)
        self.ensure_current()
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        file_mode = stat.S_IMODE(self.path.stat().st_mode)
        staged_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("wb", dir=self.temp_dir, prefix="settings-", suffix=".tmp", delete=False) as staged_file:
                staged_file.write(serialized)
                staged_file.flush()
                os.fsync(staged_file.fileno())
                staged_path = Path(staged_file.name)
            staged_path.chmod(file_mode)
            shutil.copy2(self.path, self.backup_path)
            os.replace(staged_path, self.path)
            self.original_bytes = serialized
            self.document = deepcopy(document)
            return self.backup_path
        finally:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)


def make_book_icon() -> QIcon:
    """Рисует простую книжную пиктограмму для контекстной справки."""
    pixmap = QPixmap(22, 22)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#69b7ff"), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    pages = QPainterPath()
    pages.moveTo(3, 5)
    pages.cubicTo(7, 3, 10, 5, 11, 7)
    pages.cubicTo(12, 5, 16, 3, 19, 5)
    pages.lineTo(19, 18)
    pages.cubicTo(16, 16, 13, 17, 11, 19)
    pages.cubicTo(9, 17, 6, 16, 3, 18)
    pages.closeSubpath()
    painter.drawPath(pages)
    painter.drawLine(11, 7, 11, 19)
    painter.drawLine(5, 8, 8, 8)
    painter.drawLine(14, 8, 17, 8)
    painter.drawLine(5, 11, 8, 11)
    painter.drawLine(14, 11, 17, 11)
    painter.end()
    return QIcon(pixmap)


class HelpDialog(QDialog):
    """Показывает в отдельном окне назначение поля и допустимые значения."""

    def __init__(self, spec: FieldSpec, value: Any, parent: QWidget | None = None) -> None:
        """Создаёт компактную справку по одному параметру конфигурации."""
        super().__init__(parent)
        self.setWindowTitle(f"Описание: {spec.label}")
        self.setMinimumWidth(440)
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
        current = QLabel(f"<b>Сейчас в настройках</b><br>{self._format_value(spec, value)}")
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
        """Преобразует значение поля в читаемый текст справки."""
        if isinstance(value, list):
            return "[" + ", ".join(str(component) for component in value) + "]"
        if isinstance(value, bool):
            return "включено" if value else "выключено"
        if spec.kind == "percent":
            return f"{float(value) * 100:g}%"
        return str(value)


class SettingsEditor(QMainWindow):
    """Предоставляет обычную форму с отдельной книжной справкой у каждого поля."""

    def __init__(self, store: SettingsDocument) -> None:
        """Строит вкладки редактора и загружает первый доступный профиль."""
        super().__init__()
        self.store = store
        self.store.validate(self.store.document)
        self.working_document = deepcopy(store.document)
        self.field_widgets: dict[tuple[str, ...], QWidget] = {}
        self.help_buttons: dict[tuple[str, ...], QToolButton] = {}
        self.last_help_dialog: HelpDialog | None = None
        self._refreshing = False
        self.current_profile_name = ""
        self.setWindowTitle("VideoT16 — настройки распознавания")
        self.resize(940, 820)
        self._build_window()

    def _build_window(self) -> None:
        """Размещает выбор профиля, простые вкладки, состояние и действия."""
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 14, 18, 14)
        heading = QHBoxLayout()
        heading_title = QLabel("Настройки VideoT16")
        heading_title.setObjectName("mainTitle")
        heading.addWidget(heading_title)
        heading.addStretch(1)
        heading.addWidget(QLabel("Профиль"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(self.working_document["profiles"].keys())
        self.profile_combo.currentTextChanged.connect(self._change_profile)
        heading.addWidget(self.profile_combo)
        outer.addLayout(heading)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self._render_profile(self.profile_combo.currentText())
        self.status_label = QLabel("Изменений нет")
        self.status_label.setObjectName("statusLabel")
        outer.addWidget(self.status_label)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.run_button = QPushButton("Сохранить и запустить")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self._save_and_run)
        actions.addWidget(self.run_button)
        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self._save)
        self.save_button.setEnabled(False)
        actions.addWidget(self.save_button)
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
            "QLineEdit:disabled,QComboBox:disabled { color:#89919b; }"
            "QPushButton { background:#343b44; border:1px solid #59636f; border-radius:4px; padding:7px 13px; }"
            "QPushButton:hover,QToolButton:hover { background:#46515d; }"
            "QPushButton#runButton { background:#245e8c; border-color:#4d94cc; font-weight:600; }"
            "QToolButton { background:#2a3037; border:1px solid #46505a; border-radius:4px; padding:4px; }"
            "QLabel#mainTitle { font-size:20px; font-weight:600; }"
            "QLabel#helpTitle { font-size:18px; font-weight:600; color:#8bc8ff; }"
            "QLabel#statusLabel { color:#b6c4d0; padding:4px; }"
            "QScrollArea { border:0; }"
        )

    def _render_profile(self, profile_name: str) -> None:
        """Строит поля выбранного профиля по их смысловым группам."""
        self._refreshing = True
        self.current_profile_name = profile_name
        self.tabs.clear()
        self.field_widgets.clear()
        self.help_buttons.clear()
        profile = self.working_document["profiles"][profile_name]
        profile_specs = [spec for spec in FIELD_SPECS if self._has_path(profile, spec.path)]
        for group in GROUP_ORDER:
            group_specs = [spec for spec in profile_specs if spec.group == group]
            if not group_specs:
                continue
            content = QWidget()
            form = QGridLayout(content)
            form.setColumnStretch(1, 1)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
            form.setContentsMargins(14, 14, 14, 14)
            for row, spec in enumerate(group_specs):
                label = QLabel(spec.label)
                label.setMinimumWidth(215)
                label.setWordWrap(True)
                value = get_path(profile, spec.path)
                editor = self._create_editor(spec, value)
                self._connect_editor_change(spec, editor)
                help_button = QToolButton()
                help_button.setIcon(make_book_icon())
                help_button.setIconSize(help_button.iconSize())
                help_button.setToolTip("Открыть подробное описание и допустимые значения")
                help_button.setAccessibleName(f"Описание: {spec.label}")
                help_button.clicked.connect(lambda checked=False, current=spec: self._show_help(current))
                form.addWidget(label, row, 0)
                form.addWidget(editor, row, 1)
                form.addWidget(help_button, row, 2)
                self.field_widgets[spec.path] = editor
                self.help_buttons[spec.path] = help_button
                if not spec.enabled and (spec.kind not in ("enum", "readonly_enum") or any(item[0] == value for item in spec.choices)):
                    editor.setEnabled(False)
                    label.setToolTip("Параметр пока не управляется текущим видеоприложением")
            form.setRowStretch(len(group_specs), 1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(content)
            self.tabs.addTab(scroll, group)
        self._refreshing = False

    def _connect_editor_change(self, spec: FieldSpec, editor: QWidget) -> None:
        """Обновляет признак несохранённых правок при изменении поля формы."""
        if spec.kind == "bool":
            editor.toggled.connect(self._mark_modified)
        elif spec.kind in ("int", "float", "percent"):
            editor.valueChanged.connect(self._mark_modified)
        elif spec.kind in ("enum", "readonly_enum"):
            editor.currentIndexChanged.connect(self._mark_modified)
        elif spec.kind == "color":
            return
        elif spec.kind == "model":
            editor.property("lineEditor").textChanged.connect(self._mark_modified)
        elif isinstance(editor, QLineEdit):
            editor.textChanged.connect(self._mark_modified)

    def _mark_modified(self, *_signal_values: Any) -> None:
        """Сверяет форму с исходным JSON и включает сохранение только при правках."""
        if self._refreshing:
            return
        self._collect_current_profile()
        changed = self.working_document != self.store.document
        self.status_label.setText("Есть несохранённые изменения" if changed else "Изменений нет")
        self.save_button.setEnabled(changed)

    @staticmethod
    def _has_path(document: dict[str, Any], path: tuple[str, ...]) -> bool:
        """Проверяет, существует ли путь поля в текущем профиле."""
        try:
            get_path(document, path)
            return True
        except (KeyError, TypeError):
            return False

    def _create_editor(self, spec: FieldSpec, value: Any) -> QWidget:
        """Создаёт стандартный виджет ввода подходящего типу параметра."""
        if spec.kind == "bool":
            editor = QCheckBox("Включено")
            editor.setChecked(bool(value))
            return editor
        if spec.kind in ("int", "float", "percent"):
            if spec.kind == "int":
                editor = QSpinBox()
                current_value = int(value)
                editor.setRange(min(int(spec.minimum or 0), current_value), max(int(spec.maximum or 999999), current_value))
                editor.setSingleStep(int(spec.step))
                editor.setValue(int(value))
                return editor
            editor = QDoubleSpinBox()
            editor.setDecimals(spec.decimals)
            current_value = float(value) * 100 if spec.kind == "percent" else float(value)
            editor.setRange(min(float(spec.minimum or 0), current_value), max(float(spec.maximum or 999999), current_value))
            editor.setSingleStep(spec.step)
            editor.setKeyboardTracking(False)
            if spec.kind == "percent":
                editor.setSuffix(" %")
                editor.setValue(float(value) * 100)
            else:
                editor.setValue(float(value))
            return editor
        if spec.kind in ("enum", "readonly_enum"):
            editor = QComboBox()
            if not any(stored_value == value for stored_value, _label in spec.choices):
                editor.addItem(f"Неизвестное значение: {value}", value)
            for stored_value, label in spec.choices:
                editor.addItem(label, stored_value)
            index = editor.findData(value)
            if index >= 0:
                editor.setCurrentIndex(index)
            return editor
        if spec.kind == "color":
            editor = QPushButton()
            self._paint_color_button(editor, value)
            editor.clicked.connect(lambda checked=False, button=editor: self._choose_color(button))
            return editor
        line = QLineEdit(str(value))
        line.setClearButtonEnabled(True)
        if spec.kind == "model":
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            layout.addWidget(line, 1)
            browse = QPushButton("…")
            browse.setFixedWidth(42)
            browse.setToolTip("Выбрать файл модели")
            browse.clicked.connect(lambda checked=False, target=line: self._browse_model(target))
            layout.addWidget(browse)
            row.setProperty("lineEditor", line)
            return row
        return line

    @staticmethod
    def _paint_color_button(button: QPushButton, rgb: list[int]) -> None:
        """Обновляет цветной образец и подпись кнопки выбора RGB."""
        if not isinstance(rgb, list) or len(rgb) != 3 or any(not isinstance(component, int) or not 0 <= component <= 255 for component in rgb):
            button.setProperty("rgbValue", rgb)
            button.setText("Некорректный RGB — нажмите для выбора")
            button.setStyleSheet("QPushButton { text-align:left; border:1px solid #c87562; padding:6px; background:#382b2a; }")
            return
        red, green, blue = (int(component) for component in rgb)
        button.setProperty("rgbValue", [red, green, blue])
        button.setText(f"  RGB  {red}, {green}, {blue}    ")
        button.setStyleSheet(
            "QPushButton { text-align:left; border:1px solid #59636f; border-radius:4px; padding:6px; "
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgb({red},{green},{blue}),stop:0.18 #171a1f); }}"
        )

    def _choose_color(self, button: QPushButton) -> None:
        """Открывает стандартный выбор цвета и сохраняет RGB только в форме."""
        current = button.property("rgbValue") or [255, 255, 255]
        if not isinstance(current, list) or len(current) != 3 or any(not isinstance(component, int) for component in current):
            current = [255, 255, 255]
        chosen = QColorDialog.getColor(QColor(*current), self, "Выберите цвет")
        if chosen.isValid():
            self._paint_color_button(button, [chosen.red(), chosen.green(), chosen.blue()])
            self._mark_modified()

    def _browse_model(self, editor: QLineEdit) -> None:
        """Предлагает выбрать модель и показывает относительный путь проекта."""
        selected, _ = QFileDialog.getOpenFileName(self, "Выберите модель распознавания", str(PROJECT_ROOT / "models"), "Модели (*.onnx *.pt);;Все файлы (*)")
        if not selected:
            return
        model_path = Path(selected).resolve()
        try:
            display_path = model_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            display_path = str(model_path)
        editor.setText(display_path)
        self._mark_modified()

    def _show_help(self, spec: FieldSpec) -> None:
        """Открывает отдельный диалог справки с текущим значением поля."""
        self._collect_current_profile()
        value = get_path(self.working_document["profiles"][self.current_profile_name], spec.path)
        self.last_help_dialog = HelpDialog(spec, value, self)
        self.last_help_dialog.exec()

    def _read_widget(self, spec: FieldSpec, widget: QWidget) -> Any:
        """Преобразует введённое значение виджета в соответствующий JSON-тип."""
        if spec.kind == "bool":
            return isinstance(widget, QCheckBox) and widget.isChecked()
        if spec.kind == "int":
            return int(widget.value())
        if spec.kind == "percent":
            return round(float(widget.value()) / 100, 6)
        if spec.kind == "float":
            return round(float(widget.value()), spec.decimals + 2)
        if spec.kind in ("enum", "readonly_enum"):
            return widget.currentData()
        if spec.kind == "color":
            return widget.property("rgbValue")
        if spec.kind == "model":
            return widget.property("lineEditor").text().strip()
        return widget.text().strip() if isinstance(widget, QLineEdit) else ""

    def _collect_current_profile(self) -> None:
        """Переносит все видимые поля формы в копию документа в памяти."""
        if self._refreshing:
            return
        profile_name = self.current_profile_name
        profile = self.working_document["profiles"][profile_name]
        for spec in FIELD_SPECS:
            widget = self.field_widgets.get(spec.path)
            if widget is not None:
                set_path(profile, spec.path, self._read_widget(spec, widget))

    def _change_profile(self, profile_name: str) -> None:
        """Сохраняет правки текущей формы в памяти и загружает выбранный профиль."""
        if self._refreshing or not profile_name:
            return
        self._collect_current_profile()
        self._render_profile(profile_name)
        self.status_label.setText("Есть несохранённые изменения" if self.working_document != self.store.document else "Изменений нет")
        self.save_button.setEnabled(self.working_document != self.store.document)

    def _save(self) -> bool:
        """Проверяет все профили и сохраняет JSON с одной резервной копией."""
        self._collect_current_profile()
        if self.working_document == self.store.document:
            self.status_label.setText("Изменений нет — файл не переписывался")
            self.save_button.setEnabled(False)
            return True
        try:
            backup_path = self.store.save(self.working_document)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Не удалось сохранить", str(error))
            self.status_label.setText("Сохранение не выполнено; файл не изменён")
            return False
        self.status_label.setText(f"Сохранено. Резервная копия: {backup_path}")
        self.save_button.setEnabled(False)
        self.run_button.setEnabled(True)
        return True

    def _save_and_run(self) -> None:
        """Сохраняет конфигурацию и только затем запускает основной видеопроцесс."""
        if self.working_document != self.store.document and not self._save():
            return
        self._collect_current_profile()
        try:
            self.store.ensure_current()
            self.store.validate(self.working_document)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Настройки не прошли проверку", str(error))
            return
        from PySide6.QtCore import QProcess

        process = QProcess(self)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProgram(str(PROJECT_ROOT / ".venv" / "bin" / "python3"))
        process.setArguments([str(PROJECT_ROOT / "scripts" / "live_pipeline_trial.py")])
        if not process.startDetached():
            QMessageBox.critical(self, "Не удалось запустить", "Не удалось запустить видеоприложение. Проверьте .venv/bin/python3 и журнал.")
            return
        self.close()

    def closeEvent(self, event: Any) -> None:
        """Предлагает сохранить правки перед закрытием окна редактора."""
        self._collect_current_profile()
        if self.working_document == self.store.document:
            event.accept()
            return
        answer = QMessageBox.question(
            self,
            "Несохранённые изменения",
            "Сохранить изменения перед закрытием?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Save:
            event.accept() if self._save() else event.ignore()
        elif answer == QMessageBox.StandardButton.Discard:
            event.accept()
        else:
            event.ignore()


def main() -> int:
    """Запускает нативный редактор и показывает понятную ошибку загрузки."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    try:
        editor = SettingsEditor(SettingsDocument())
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        QMessageBox.critical(None, "Не удалось открыть настройки", str(error))
        return 1
    editor.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
