"""Расчёт рамки цели для аналоговой сетки OSD Betaflight."""

from __future__ import annotations

from dataclasses import dataclass

from src.osd import set_crosshairs_position

OSD_COLUMNS = 30
OSD_ROWS = 16


@dataclass(frozen=True)
class TargetBox:
    """Четыре угловые позиции рамки цели в сетке OSD."""

    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_left: tuple[int, int]
    bottom_right: tuple[int, int]


def _grid_x(pixel_x: int, image_width: int) -> int:
    """Переводит координату X изображения в колонку OSD."""
    return max(0, min(OSD_COLUMNS - 1, int(pixel_x * OSD_COLUMNS / image_width)))


def _grid_y(pixel_y: int, image_height: int) -> int:
    """Переводит координату Y изображения в строку OSD."""
    return max(0, min(OSD_ROWS - 1, int(pixel_y * OSD_ROWS / image_height)))


def target_box_from_pixels(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> TargetBox:
    """Переводит рамку объекта из пикселей в четыре угла OSD."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Размер изображения должен быть больше нуля")
    if x2 < x1 or y2 < y1:
        raise ValueError("Правая и нижняя границы должны быть не меньше левой и верхней")

    left = _grid_x(x1, image_width)
    right = _grid_x(x2, image_width)
    top = _grid_y(y1, image_height)
    bottom = _grid_y(y2, image_height)
    return TargetBox((left, top), (right, top), (left, bottom), (right, bottom))


def update_pointer_from_detection(
    serial_port,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """Переводит центр обнаружения в OSD и перемещает штатный указатель."""
    box = target_box_from_pixels(x1, y1, x2, y2, image_width, image_height)
    center_x = (box.top_left[0] + box.top_right[0]) // 2
    center_y = (box.top_left[1] + box.bottom_left[1]) // 2
    set_crosshairs_position(serial_port, center_x, center_y)
    return center_x, center_y
