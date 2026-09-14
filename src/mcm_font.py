"""Рендеринг bitmap-шрифта Betaflight MCM для зеркала веб-OSD."""

from __future__ import annotations

from pathlib import Path


class McmFont:
    """Загружает MAX7456/MCM-шрифт и рисует его пиксели в кадре OpenCV."""

    def __init__(self, path: str | Path) -> None:
        """Читает 256 глифов MCM и сохраняет маски белых пикселей."""
        raw_lines = Path(path).read_text(encoding="ascii").splitlines()
        if not raw_lines or raw_lines[0].strip() != "MAX7456":
            raise ValueError("неподдерживаемый MCM-шрифт: отсутствует MAX7456")
        rows = [int(line, 2) for line in raw_lines[1:] if len(line) == 8]
        if len(rows) < 256 * 64:
            raise ValueError("неполный MCM-шрифт: ожидалось 256 глифов")
        self._glyphs = tuple(self._decode_glyph(rows[index * 64:index * 64 + 54]) for index in range(256))

    @staticmethod
    def _decode_glyph(data: list[int]):
        """Преобразует первые 54 байта глифа в маску 18×12."""
        import numpy as np

        mask = np.zeros((18, 12), dtype=np.uint8)
        for byte_index, value in enumerate(data):
            row, group = divmod(byte_index, 3)
            for pixel in range(4):
                # Формат MCM: 00 чёрный, 01 прозрачный, 10 белый, 11 прозрачный.
                pair = (value >> (6 - pixel * 2)) & 0b11
                if pair == 0b10:
                    mask[row, group * 4 + pixel] = 255
        return mask

    def draw(self, frame, lines: tuple[str, ...], columns: int, rows: int) -> None:
        """Рисует MCM-глифы в клетках Canvas поверх BGR-кадра."""
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        cell_width = max(1, round(width / columns))
        cell_height = max(1, round(height / rows))
        for row, line in enumerate(lines[:rows]):
            for column, character in enumerate(line[:columns]):
                code = ord(character) if ord(character) < 256 else ord("?")
                glyph = cv2.resize(
                    self._glyphs[code], (cell_width, cell_height), interpolation=cv2.INTER_NEAREST
                )
                y1, x1 = row * cell_height, column * cell_width
                y2, x2 = min(height, y1 + cell_height), min(width, x1 + cell_width)
                visible = glyph[:y2 - y1, :x2 - x1] > 0
                frame[y1:y2, x1:x2][visible] = np.array((255, 255, 255), dtype=np.uint8)
