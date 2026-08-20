"""Безопасное чтение телеметрии INAV по протоколу MSP."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass

MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_STATUS = 101
MSP_RAW_GPS = 106
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_ANALOG = 110
MSP_SONAR_ALTITUDE = 58


@dataclass
class InavTelemetry:
    """Последние принятые значения телеметрии полётного контроллера."""

    variant: str = "INAV"
    version: str = ""
    api_version: str = ""
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    altitude: float | None = None
    surface_distance: float | None = None
    voltage: float | None = None
    battery_current: float | None = None
    battery_mah_drawn: int | None = None
    rssi: int | None = None
    gps_fix: int | None = None
    gps_satellites: int | None = None
    latitude: float | None = None
    longitude: float | None = None
    ground_speed: float | None = None
    ground_course: float | None = None
    mode_flags: int | None = None
    updated_at: float = 0.0


class InavMspReader:
    """Читает MSP-ответы и не содержит команд изменения или управления."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        """Открывает USB VCP-порт INAV с коротким тайм-аутом."""
        try:
            import serial
        except ModuleNotFoundError as error:
            raise RuntimeError("Не установлен pyserial. Выполните: .venv/bin/pip install pyserial") from error
        try:
            self._serial = serial.Serial(port, baudrate=baudrate, timeout=0.02)
        except serial.SerialException as error:
            raise RuntimeError(f"Не удалось открыть порт INAV {port}: {error}") from error
        self.telemetry = InavTelemetry()
        self._buffer = bytearray()
        self._commands = [
            MSP_ATTITUDE,
            MSP_ALTITUDE,
            MSP_ANALOG,
            MSP_RAW_GPS,
            MSP_SONAR_ALTITUDE,
            MSP_STATUS,
        ]
        self._command_index = 0
        self._next_request_at = 0.0
        self._request_interval = 0.08
        self._request(MSP_API_VERSION)
        self._request(MSP_FC_VARIANT)
        self._request(MSP_FC_VERSION)

    @staticmethod
    def _request_packet(command: int) -> bytes:
        """Формирует MSP-запрос без полезной нагрузки."""
        return b"$M<" + bytes((0, command, command & 0xFF))

    def _request(self, command: int) -> None:
        """Отправляет только запрос чтения MSP-параметра."""
        self._serial.write(self._request_packet(command))

    def _parse_frames(self) -> None:
        """Извлекает из буфера полные и проверенные MSP-ответы."""
        while True:
            marker = self._buffer.find(b"$M>")
            if marker < 0:
                self._buffer.clear()
                return
            if marker:
                del self._buffer[:marker]
            if len(self._buffer) < 6:
                return
            payload_size = self._buffer[3]
            frame_size = payload_size + 6
            if len(self._buffer) < frame_size:
                return
            frame = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            payload = frame[5:-1]
            checksum = frame[3] ^ frame[4]
            for byte in payload:
                checksum ^= byte
            if checksum == frame[-1]:
                self._decode(frame[4], payload)

    def _decode(self, command: int, payload: bytes) -> None:
        """Разбирает только телеметрические ответы INAV."""
        self.telemetry.updated_at = time.monotonic()
        if command == MSP_API_VERSION and len(payload) >= 3:
            self.telemetry.api_version = f"{payload[1]}.{payload[2]}"
        elif command == MSP_FC_VARIANT:
            self.telemetry.variant = payload.rstrip(b"\x00").decode("ascii", "replace")
        elif command == MSP_FC_VERSION and len(payload) >= 3:
            self.telemetry.version = ".".join(str(byte) for byte in payload[:3])
        elif command == MSP_ATTITUDE and len(payload) >= 6:
            roll, pitch, yaw = struct.unpack_from("<hhh", payload)
            self.telemetry.roll = roll / 10.0
            self.telemetry.pitch = pitch / 10.0
            self.telemetry.yaw = float(yaw)
        elif command == MSP_ALTITUDE and len(payload) >= 4:
            self.telemetry.altitude = struct.unpack_from("<i", payload)[0] / 100.0
        elif command == MSP_SONAR_ALTITUDE and len(payload) >= 4:
            self.telemetry.surface_distance = struct.unpack_from("<I", payload)[0] / 100.0
        elif command == MSP_ANALOG and len(payload) >= 7:
            self.telemetry.voltage = payload[0] / 10.0
            self.telemetry.battery_mah_drawn = struct.unpack_from("<H", payload, 1)[0]
            self.telemetry.rssi = struct.unpack_from("<H", payload, 3)[0]
            self.telemetry.battery_current = struct.unpack_from("<h", payload, 5)[0] / 100.0
        elif command == MSP_RAW_GPS and len(payload) >= 18:
            fix, satellites, latitude, longitude, _, speed, course, _ = struct.unpack_from(
                "<BBiiHHHH", payload
            )
            self.telemetry.gps_fix = fix
            self.telemetry.gps_satellites = satellites
            self.telemetry.latitude = latitude / 10_000_000.0
            self.telemetry.longitude = longitude / 10_000_000.0
            self.telemetry.ground_speed = speed / 100.0
            self.telemetry.ground_course = course / 10.0
        elif command == MSP_STATUS and len(payload) >= 10:
            self.telemetry.mode_flags = struct.unpack_from("<I", payload, 6)[0]

    def update(self) -> InavTelemetry:
        """Читает ответы и периодически запрашивает следующий параметр."""
        if self._serial.in_waiting:
            self._buffer.extend(self._serial.read(self._serial.in_waiting))
            self._parse_frames()
        now = time.monotonic()
        if now >= self._next_request_at:
            self._request(self._commands[self._command_index])
            self._command_index = (self._command_index + 1) % len(self._commands)
            self._next_request_at = now + self._request_interval
        return self.telemetry

    def close(self) -> None:
        """Закрывает USB-порт без изменения настроек контроллера."""
        self._serial.close()
