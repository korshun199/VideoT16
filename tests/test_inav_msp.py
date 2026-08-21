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


if __name__ == "__main__":
    unittest.main()
