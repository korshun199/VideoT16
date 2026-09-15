"""Тесты формата цифровых OSD-команд без оборудования."""

import unittest

from src.displayport_proxy import (
    MSP_DISPLAYPORT,
    MSP_DP_WRITE_STRING,
    MspV1Parser,
    build_msp_frame,
    displayport_write_string,
    DisplayPortOverlay,
    CanvasMirror,
    DisplayPortProxy,
    displayport_draw_screen,
)


class RecordingProxy:
    """Запоминает исходящие команды overlay без настоящего UART."""

    def __init__(self) -> None:
        """Создаёт пустой список команд."""
        self.packets = []

    def send_displayport(self, packet: bytes) -> None:
        """Сохраняет команду, которую overlay отправил бы на VTX."""
        self.packets.append(packet)


class DisplayPortProxyTests(unittest.TestCase):
    """Проверяет безопасное формирование и разбор MSP DisplayPort."""

    def test_parser_accepts_fragmented_frame(self) -> None:
        """Кадр собирается корректно из нескольких порций UART."""
        packet = displayport_write_string(4, 5, "OBJECT 87%")
        parser = MspV1Parser()
        frames = parser.feed(packet[:3])
        frames += parser.feed(packet[3:8])
        frames += parser.feed(packet[8:])
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].command, MSP_DISPLAYPORT)
        self.assertEqual(packet[:3], b"$M>")
        self.assertEqual(frames[0].payload[:4], bytes((MSP_DP_WRITE_STRING, 5, 4, 0)))
        self.assertEqual(frames[0].payload[4:], b"OBJECT 87%")

    def test_string_is_limited_to_thirty_ascii_bytes(self) -> None:
        """Команда не выходит за ограничение DisplayPort по длине строки."""
        packet = displayport_write_string(0, 0, "x" * 100)
        self.assertEqual(packet[3], 34)

    def test_outgoing_displayport_uses_fc_to_device_header(self) -> None:
        """Исходящий пакет имеет направление FC к внешнему OSD-устройству."""
        self.assertEqual(displayport_write_string(0, 0, "FPV-DRON")[:3], b"$M>")

    def test_canvas_mirror_preserves_betaflight_glyph_code(self) -> None:
        """Штатный индекс символа до 255 не превращается в вопросительный знак."""
        mirror = CanvasMirror()
        packet = build_msp_frame(
            MSP_DISPLAYPORT, bytes((MSP_DP_WRITE_STRING, 2, 3, 0, 0xC1))
        )
        mirror.apply(packet)
        _columns, _rows, lines = mirror.snapshot()
        self.assertEqual(ord(lines[2][3]), 0xC1)

    def test_overlay_adds_ascii_system_status(self) -> None:
        """Системный статус формируется отдельной ASCII-строкой для VTX."""
        proxy = RecordingProxy()
        overlay = DisplayPortOverlay(proxy)
        proxy.packets.clear()
        overlay.update_status("TEMP 48.2C CPU 37%")
        self.assertTrue(any(b"TEMP 48.2C CPU 37%" in packet for packet in proxy.packets))

    def test_overlay_does_not_repeat_unchanged_status(self) -> None:
        """Неизменившийся статус не создаёт лишний поток OSD-команд."""
        proxy = RecordingProxy()
        overlay = DisplayPortOverlay(proxy)
        proxy.packets.clear()
        overlay.update_status("TEMP 48.2C CPU 37%")
        packet_count = len(proxy.packets)
        overlay.update_status("TEMP 48.2C CPU 37%")
        self.assertEqual(len(proxy.packets), packet_count)
        overlay._last_status_update -= 2.0
        overlay.update_status("TEMP 48.2C CPU 37%")
        self.assertEqual(len(proxy.packets), packet_count)

    def test_overlay_does_not_redraw_unchanged_frame(self) -> None:
        """Неизменившаяся рамка не мигает из-за повторной перерисовки."""
        proxy = RecordingProxy()
        overlay = DisplayPortOverlay(proxy)
        proxy.packets.clear()
        overlay.update(100, 80, 300, 260, 640, 480)
        packet_count = len(proxy.packets)
        overlay.update(100, 80, 300, 260, 640, 480)
        self.assertEqual(len(proxy.packets), packet_count)

    def test_overlay_filters_small_confidence_changes(self) -> None:
        """Малое изменение CONF не создаёт лишнее обновление строки."""
        proxy = RecordingProxy()
        overlay = DisplayPortOverlay(proxy)
        proxy.packets.clear()
        overlay.update_status("TEMP 48.2C CPU 37% CONF 50%")
        overlay._last_status_update -= 2.0
        overlay.update_status("TEMP 48.2C CPU 38% CONF 55%")
        self.assertTrue(any(b"CONF 50%" in packet for packet in proxy.packets))
        self.assertFalse(any(b"CONF 55%" in packet for packet in proxy.packets))

    def test_overlay_filters_small_system_status_changes(self) -> None:
        """Малые изменения температуры и CPU не мерцают в строке OSD."""
        proxy = RecordingProxy()
        overlay = DisplayPortOverlay(proxy)
        proxy.packets.clear()
        overlay.update_status("TEMP 48.2C CPU 37% CONF 50%")
        overlay._last_status_update -= 2.0
        overlay.update_status("TEMP 48.9C CPU 42% CONF 50%")
        self.assertTrue(any(b"TEMP 48.2C CPU 37% CONF 50%" in packet for packet in proxy.packets))
        self.assertFalse(any(b"TEMP 48.9C CPU 42%" in packet for packet in proxy.packets))

    def test_proxy_delays_fc_draw_screen(self) -> None:
        """DRAW_SCREEN FC не выводит промежуточный экран без нашей рамки."""
        proxy = object.__new__(DisplayPortProxy)
        proxy._fc_filter_buffer = bytearray()
        write_packet = displayport_write_string(1, 1, "FC")
        output = proxy._filter_fc_draw_commands(write_packet + displayport_draw_screen())
        self.assertIn(write_packet, output)
        self.assertNotIn(displayport_draw_screen(), output)

if __name__ == "__main__":
    unittest.main()
