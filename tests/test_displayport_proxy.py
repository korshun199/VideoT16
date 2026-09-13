"""Тесты формата цифровых OSD-команд без оборудования."""

import unittest

from src.displayport_proxy import (
    MSP_DISPLAYPORT,
    MSP_DP_WRITE_STRING,
    MspV1Parser,
    build_msp_frame,
    displayport_write_string,
    LatinOnlyMspStream,
    latin_only_displayport,
)


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

    def test_non_latin_osd_bytes_are_replaced_with_spaces(self) -> None:
        """Русские и служебные байты не превращаются в кракозябры на VTX."""
        packet = build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_WRITE_STRING, 3, 2, 0, 65, 0xD0, 0x90)))
        filtered = latin_only_displayport(packet)
        self.assertEqual(filtered[5:-1], bytes((MSP_DP_WRITE_STRING, 3, 2, 0, 65, 32, 32)))

    def test_latin_filter_handles_fragmented_stream(self) -> None:
        """Фильтр обрабатывает пакет, разделённый на несколько чтений UART."""
        packet = build_msp_frame(MSP_DISPLAYPORT, bytes((MSP_DP_WRITE_STRING, 3, 2, 0, 65, 0xD0, 0x90)))
        stream = LatinOnlyMspStream()
        self.assertEqual(stream.feed(packet[:4]), b"")
        result = stream.feed(packet[4:])
        self.assertEqual(result[5:-1], bytes((MSP_DP_WRITE_STRING, 3, 2, 0, 65, 32, 32)))


if __name__ == "__main__":
    unittest.main()
