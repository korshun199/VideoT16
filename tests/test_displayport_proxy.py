"""Тесты формата цифровых OSD-команд без оборудования."""

import unittest

from src.displayport_proxy import (
    MSP_DISPLAYPORT,
    MSP_DP_WRITE_STRING,
    MspV1Parser,
    build_msp_frame,
    displayport_write_string,
    DisplayPortOverlay,
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

if __name__ == "__main__":
    unittest.main()
