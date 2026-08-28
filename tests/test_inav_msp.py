import sys
import struct
import time
import types
import unittest
from collections import Counter
from unittest.mock import patch

from src.inav_msp import (
    MSP_ANALOG,
    MSP_ATTITUDE,
    MSP_BOXIDS,
    MSP_BOXNAMES,
    MSP_MODE_RANGES,
    MSP_RC,
    MSP_STATUS,
    InavMspReader,
)
from src.betaflight_osd import (
    build_char_write_packet,
    build_set_position_packet,
    build_box_glyphs,
    encode_osd_position,
)


class FakeSerialException(Exception):
    """Имитирует ошибку pyserial в тестах без оборудования."""


class FakeSerial:
    """Запоминает MSP-запросы вместо обращения к полётнику."""

    instances = []

    def __init__(self, port, baudrate, timeout):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.writes = []
        self.closed = False
        self.in_waiting = 0
        self.instances.append(self)

    def write(self, packet):
        self.writes.append(packet)
        return len(packet)

    def read(self, size):
        return b""

    def close(self):
        self.closed = True


class InavMspReaderTests(unittest.TestCase):
    def setUp(self):
        FakeSerial.instances.clear()
        self.serial_module = types.SimpleNamespace(
            Serial=FakeSerial,
            SerialException=FakeSerialException,
        )

    def test_attitude_is_requested_faster_than_battery(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                time.sleep(0.22)
                serial_port = FakeSerial.instances[0]
                commands = Counter(packet[4] for packet in serial_port.writes)
                self.assertGreaterEqual(commands[MSP_ATTITUDE], 4)
                self.assertGreater(commands[MSP_ATTITUDE], commands[MSP_ANALOG])
            finally:
                reader.close()

    def test_update_returns_independent_snapshot(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                with reader._telemetry_lock:
                    reader.telemetry.roll = 12.5
                first = reader.update()
                first.roll = 99.0
                second = reader.update()
                self.assertEqual(second.roll, 12.5)
            finally:
                reader.close()
            self.assertTrue(FakeSerial.instances[0].closed)


    def test_rc_decodes_fourth_channel_as_throttle(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                payload = struct.pack("<HHHH", 1500, 1500, 1500, 1750)
                with reader._telemetry_lock:
                    reader._decode(MSP_RC, payload)
                telemetry = reader.update()
                self.assertEqual(telemetry.throttle, 1750)
                self.assertEqual(telemetry.throttle_percent, 75)
            finally:
                reader.close()


    def test_arm_uses_box_id_position_in_status_mask(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                # ARM имеет permanent ID 0 и намеренно стоит третьим в тесте.
                status_payload = struct.pack("<HHHI", 1000, 0, 0, 1 << 2)
                with reader._telemetry_lock:
                    reader._decode(MSP_BOXIDS, bytes((1, 2, 0, 10)))
                    reader._decode(MSP_STATUS, status_payload)
                self.assertTrue(reader.update().armed)

                status_payload = struct.pack("<HHHI", 1000, 0, 0, 0)
                with reader._telemetry_lock:
                    reader._decode(MSP_STATUS, status_payload)
                self.assertFalse(reader.update().armed)
            finally:
                reader.close()

    def test_arm_switch_uses_configured_aux_range(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                # ARM настроен на AUX1 в диапазоне 1500–2100 мкс.
                mode_ranges = bytes((0, 0, 24, 48))
                channels_off = struct.pack("<HHHHH", 1500, 1500, 1500, 1000, 1475)
                channels_on = struct.pack("<HHHHH", 1500, 1500, 1500, 1000, 1800)
                with reader._telemetry_lock:
                    reader._decode(MSP_MODE_RANGES, mode_ranges)
                    reader._decode(MSP_RC, channels_off)
                telemetry = reader.update()
                self.assertEqual(telemetry.arm_aux_channel, 1)
                self.assertEqual(telemetry.arm_switch_value, 1475)
                self.assertFalse(telemetry.arm_switch)

                with reader._telemetry_lock:
                    reader._decode(MSP_RC, channels_on)
                telemetry = reader.update()
                self.assertEqual(telemetry.arm_switch_value, 1800)
                self.assertTrue(telemetry.arm_switch)
            finally:
                reader.close()

    def test_active_mode_is_decoded_from_status_mask(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                names = b"ARM;ANGLE;HORIZON;NAV RTH;AIR MODE;"
                status_payload = struct.pack("<HHHI", 1000, 0, 0, (1 << 0) | (1 << 3))
                with reader._telemetry_lock:
                    reader._decode(MSP_BOXNAMES, names)
                    reader._decode(MSP_BOXIDS, bytes((0, 1, 2, 3, 4)))
                    reader._decode(MSP_STATUS, status_payload)
                telemetry = reader.update()
                self.assertTrue(telemetry.armed)
                self.assertEqual(telemetry.active_modes, ("ARM", "NAV RTH"))
                self.assertEqual(telemetry.flight_mode, "RTH")
            finally:
                reader.close()

    def test_armed_without_stabilization_is_acro(self):
        with patch.dict(sys.modules, {"serial": self.serial_module}):
            reader = InavMspReader("/dev/fake")
            try:
                status_payload = struct.pack("<HHHI", 1000, 0, 0, 1)
                with reader._telemetry_lock:
                    reader._decode(MSP_BOXNAMES, b"ARM;ANGLE;AIR MODE;")
                    reader._decode(MSP_BOXIDS, bytes((0, 1, 2)))
                    reader._decode(MSP_STATUS, status_payload)
                self.assertEqual(reader.update().flight_mode, "ACRO")
            finally:
                reader.close()


class BetaflightOsdTests(unittest.TestCase):
    """Проверяет формат команды позиции штатного прицела."""

    def test_center_position_matches_betaflight_value(self):
        self.assertEqual(encode_osd_position(14, 8), 2318)

    def test_packet_contains_crosshairs_index_and_position(self):
        packet = build_set_position_packet(14, 8)
        self.assertEqual(packet[:3], b"$M<")
        self.assertEqual(packet[4], 85)
        self.assertEqual(packet[5], 2)
        self.assertEqual(packet[6:8], (2318).to_bytes(2, "little"))

    def test_box_glyphs_have_max7456_visible_size(self):
        glyphs = build_box_glyphs()
        self.assertEqual(set(glyphs), {0x72, 0x73, 0x74})
        self.assertTrue(all(len(glyph) == 54 for glyph in glyphs.values()))

    def test_char_write_packet_contains_glyph(self):
        glyph = build_box_glyphs()[0x72]
        packet = build_char_write_packet(0x72, glyph)
        self.assertEqual(packet[:3], b"$M<")
        self.assertEqual(packet[4], 87)
        self.assertEqual(packet[5], 0x72)
        self.assertEqual(len(packet), 71)


if __name__ == "__main__":
    unittest.main()
