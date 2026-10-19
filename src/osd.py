"""Минимальный обмен с аналоговым OSD Betaflight через MSP."""

from __future__ import annotations

import struct
import time


MSP_SET_OSD_CONFIG = 85
MSP_OSD_CHAR_WRITE = 87
OSD_CROSSHAIRS_INDEX = 2
OSD_PROFILE_1_FLAG = 1 << 11
OSD_MAX_X = 29
OSD_MAX_Y = 12
OSD_CHAR_WIDTH = 12
OSD_CHAR_HEIGHT = 18
OSD_CHAR_BLACK = 0
OSD_CHAR_TRANSPARENT = 1
OSD_CHAR_WHITE = 2
OSD_CROSSHAIR_GLYPHS = (0x72, 0x73, 0x74)


def encode_osd_position(x: int, y: int, visible: bool = True) -> int:
    """Кодирует координаты OSD в формат Betaflight."""
    if not 0 <= x <= OSD_MAX_X or not 0 <= y <= OSD_MAX_Y:
        raise ValueError("Координаты OSD должны быть X=0..30 и Y=0..15")
    position = x | (y << 5)
    if visible:
        position |= OSD_PROFILE_1_FLAG
    return position


def build_set_position_packet(x: int, y: int, visible: bool = True) -> bytes:
    """Формирует MSP-команду изменения позиции прицела."""
    position = encode_osd_position(x, y, visible)
    payload = bytes((OSD_CROSSHAIRS_INDEX,)) + struct.pack("<H", position) + bytes((1,))
    checksum = len(payload) ^ MSP_SET_OSD_CONFIG
    for byte in payload:
        checksum ^= byte
    return b"$M<" + bytes((len(payload), MSP_SET_OSD_CONFIG)) + payload + bytes((checksum,))


def set_crosshairs_position(serial_port, x: int, y: int, visible: bool = True) -> None:
    """Отправляет новую позицию прицела без сохранения во flash."""
    serial_port.write(build_set_position_packet(x, y, visible))
    serial_port.flush()


def build_box_glyphs() -> dict[int, bytes]:
    """Создаёт белую рамку с чёрной окантовкой для штатного OSD."""
    glyphs = {
        address: [[OSD_CHAR_TRANSPARENT] * OSD_CHAR_WIDTH for _ in range(OSD_CHAR_HEIGHT)]
        for address in OSD_CROSSHAIR_GLYPHS
    }
    # Горизонтальные стороны: чёрный контур и белая линия внутри.
    for address in OSD_CROSSHAIR_GLYPHS:
        for x in range(OSD_CHAR_WIDTH):
            glyphs[address][0][x] = OSD_CHAR_BLACK
            glyphs[address][1][x] = OSD_CHAR_WHITE
            glyphs[address][2][x] = OSD_CHAR_BLACK
            glyphs[address][OSD_CHAR_HEIGHT - 1][x] = OSD_CHAR_BLACK
            glyphs[address][OSD_CHAR_HEIGHT - 2][x] = OSD_CHAR_WHITE
            glyphs[address][OSD_CHAR_HEIGHT - 3][x] = OSD_CHAR_BLACK
    # Вертикальные стороны: чёрный контур по обеим сторонам белой линии.
    for y in range(3, OSD_CHAR_HEIGHT - 3):
        glyphs[OSD_CROSSHAIR_GLYPHS[0]][y][0] = OSD_CHAR_BLACK
        glyphs[OSD_CROSSHAIR_GLYPHS[0]][y][1] = OSD_CHAR_WHITE
        glyphs[OSD_CROSSHAIR_GLYPHS[0]][y][2] = OSD_CHAR_BLACK
        glyphs[OSD_CROSSHAIR_GLYPHS[2]][y][OSD_CHAR_WIDTH - 3] = OSD_CHAR_BLACK
        glyphs[OSD_CROSSHAIR_GLYPHS[2]][y][OSD_CHAR_WIDTH - 2] = OSD_CHAR_WHITE
        glyphs[OSD_CROSSHAIR_GLYPHS[2]][y][OSD_CHAR_WIDTH - 1] = OSD_CHAR_BLACK

    packed = {}
    for address, rows in glyphs.items():
        pixels = [pixel for row in rows for pixel in row]
        packed[address] = bytes(
            sum(pixels[index + offset] << (6 - offset * 2) for offset in range(4))
            for index in range(0, len(pixels), 4)
        )
    return packed


def build_char_write_packet(address: int, glyph: bytes) -> bytes:
    """Формирует MSP-команду загрузки одного символа OSD."""
    if not 0 <= address <= 255 or len(glyph) != 54:
        raise ValueError("Символ должен иметь адрес 0..255 и 54 байта")
    # Betaflight ожидает 64 байта символа: 54 видимых и 10 служебных.
    payload = bytes((address,)) + glyph + bytes(10)
    checksum = len(payload) ^ MSP_OSD_CHAR_WRITE
    for byte in payload:
        checksum ^= byte
    return b"$M<" + bytes((len(payload), MSP_OSD_CHAR_WRITE)) + payload + bytes((checksum,))


def upload_box_glyphs(serial_port) -> None:
    """Записывает три символа тёмной рамки во внутреннюю память MAX7456."""
    glyphs = build_box_glyphs()
    for address, glyph in glyphs.items():
        serial_port.write(build_char_write_packet(address, glyph))
        serial_port.flush()
        time.sleep(0.1)
    # Правый элемент повторяем отдельно: он мог остаться от прежнего шрифта.
    serial_port.write(build_char_write_packet(0x74, glyphs[0x74]))
    serial_port.flush()


def read_msp_response(serial_port, timeout: float = 0.3) -> bytes:
    """Читает ответ Betaflight после отправки команды MSP."""
    serial_port.timeout = timeout
    return serial_port.read(256)
