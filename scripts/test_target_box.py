#!/usr/bin/env python3
"""Показывает виртуальные координаты четырёх уголков рамки."""

from src.osd_target_box import target_box_from_pixels


def main() -> int:
    """Рассчитывает пример рамки без подключения камеры и полётника."""
    box = target_box_from_pixels(640, 360, 1280, 900, 1920, 1080)
    print("Виртуальная рамка цели:")
    print(f"Левый верх:   X={box.top_left[0]}, Y={box.top_left[1]}")
    print(f"Правый верх:  X={box.top_right[0]}, Y={box.top_right[1]}")
    print(f"Левый низ:    X={box.bottom_left[0]}, Y={box.bottom_left[1]}")
    print(f"Правый низ:   X={box.bottom_right[0]}, Y={box.bottom_right[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
