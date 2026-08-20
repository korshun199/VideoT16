"""Безопасное чтение телеметрии INAV по протоколу MSP."""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, replace

MSP_API_VERSION = 1
MSP_FC_VARIANT = 2
MSP_FC_VERSION = 3
MSP_MODE_RANGES = 34
MSP_STATUS = 101
MSP_RC = 105
MSP_RAW_GPS = 106
MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_ANALOG = 110
MSP_BOXIDS = 119
MSP_SONAR_ALTITUDE = 58

# Постоянный идентификатор режима ARM в INAV.
ARM_PERMANENT_ID = 0
# MSP_RC всегда возвращает первые каналы в порядке AERT, газ имеет индекс 3.
THROTTLE_CHANNEL_INDEX = 3
# AUX1 начинается сразу после четырёх основных каналов AERT.
NON_AUX_CHANNEL_COUNT = 4
# Рабочий диапазон RC нужен только для наглядного процента газа в OSD.
RC_MIN_US = 1000
RC_MAX_US = 2000
# Границы диапазонов режимов INAV кодируются шагами по 25 мкс от 900 мкс.
MODE_RANGE_MIN_US = 900
MODE_RANGE_STEP_US = 25


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
    armed: bool | None = None
    arm_switch: bool | None = None
    arm_aux_channel: int | None = None
    arm_switch_value: int | None = None
    rc_channels: tuple[int, ...] = ()
    throttle: int | None = None
    throttle_percent: int | None = None
    updated_at: float = 0.0


class InavMspReader:
    """Читает MSP в фоне и не содержит команд изменения или управления."""

    # Быстрые углы опрашиваются отдельно от медленной батареи и состояния.
    REQUEST_INTERVALS = {
        MSP_ATTITUDE: 0.04,
        MSP_ALTITUDE: 0.10,
        MSP_SONAR_ALTITUDE: 0.10,
        MSP_RC: 0.10,
        MSP_MODE_RANGES: 2.00,
        MSP_RAW_GPS: 0.20,
        MSP_ANALOG: 0.50,
        MSP_STATUS: 0.10,
        MSP_BOXIDS: 2.00,
    }

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        """Открывает USB VCP-порт INAV с коротким тайм-аутом."""
        try:
            import serial
        except ModuleNotFoundError as error:
            raise RuntimeError("Не установлен pyserial. Выполните: .venv/bin/pip install pyserial") from error
        self._serial_module = serial
        try:
            self._serial = serial.Serial(port, baudrate=baudrate, timeout=0.02)
        except serial.SerialException as error:
            raise RuntimeError(f"Не удалось открыть порт INAV {port}: {error}") from error
        self.telemetry = InavTelemetry()
        self._box_ids: tuple[int, ...] = ()
        self._arm_ranges: tuple[tuple[int, int, int], ...] = ()
        self._buffer = bytearray()
        self._telemetry_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._worker_error: RuntimeError | None = None
        now = time.monotonic()
        self._next_request_at = {
            command: now for command in self.REQUEST_INTERVALS
        }
        self._request(MSP_API_VERSION)
        self._request(MSP_FC_VARIANT)
        self._request(MSP_FC_VERSION)
        self._thread = threading.Thread(
            target=self._run,
            name="inav-msp-reader",
            daemon=True,
        )
        self._thread.start()

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

    def _read_available(self) -> None:
        """Забирает накопившиеся байты и обновляет телеметрию целиком."""
        if not self._serial.in_waiting:
            return
        self._buffer.extend(self._serial.read(self._serial.in_waiting))
        with self._telemetry_lock:
            self._parse_frames()

    def _request_due_commands(self, now: float) -> None:
        """Запрашивает параметры с независимыми частотами обновления."""
        for command, interval in self.REQUEST_INTERVALS.items():
            if now < self._next_request_at[command]:
                continue
            self._request(command)
            # Не догоняем пропущенные запросы пачкой после временной задержки.
            self._next_request_at[command] = now + interval

    def _run(self) -> None:
        """Независимо от YOLO читает MSP и поддерживает частоту горизонта."""
        try:
            while not self._stop_requested.is_set():
                self._read_available()
                self._request_due_commands(time.monotonic())
                self._stop_requested.wait(0.002)
        except (OSError, self._serial_module.SerialException) as error:
            self._worker_error = RuntimeError(f"Ошибка чтения INAV по MSP: {error}")
            self._stop_requested.set()

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
        elif command == MSP_RC and len(payload) >= 2:
            channel_count = len(payload) // 2
            channels = struct.unpack_from(f"<{channel_count}H", payload)
            self.telemetry.rc_channels = channels
            if len(channels) > THROTTLE_CHANNEL_INDEX:
                # INAV возвращает газ четвёртым каналом независимо от channel map.
                throttle = channels[THROTTLE_CHANNEL_INDEX]
                self.telemetry.throttle = throttle
                span = RC_MAX_US - RC_MIN_US
                percent = round((throttle - RC_MIN_US) * 100 / span)
                self.telemetry.throttle_percent = max(0, min(100, percent))
            self._update_arm_switch()
        elif command == MSP_MODE_RANGES:
            # Каждый слот содержит permanent ID, номер AUX и две границы диапазона.
            arm_ranges = []
            for offset in range(0, len(payload) - 3, 4):
                permanent_id, aux_index, start_step, end_step = payload[offset:offset + 4]
                if permanent_id != ARM_PERMANENT_ID or start_step >= end_step:
                    continue
                channel_index = NON_AUX_CHANNEL_COUNT + aux_index
                start_us = MODE_RANGE_MIN_US + start_step * MODE_RANGE_STEP_US
                end_us = MODE_RANGE_MIN_US + end_step * MODE_RANGE_STEP_US
                arm_ranges.append((channel_index, start_us, end_us))
            self._arm_ranges = tuple(arm_ranges)
            self.telemetry.arm_aux_channel = (
                arm_ranges[0][0] - NON_AUX_CHANNEL_COUNT + 1 if arm_ranges else None
            )
            self._update_arm_switch()
        elif command == MSP_ANALOG and len(payload) >= 7:
            self.telemetry.voltage = payload[0] / 10.0
            self.telemetry.battery_mah_drawn = struct.unpack_from("<H", payload, 1)[0]
            self.telemetry.rssi = struct.unpack_from("<H", payload, 3)[0]
            self.telemetry.battery_current = struct.unpack_from("<h", payload, 5)[0] / 100.0
        elif command == MSP_RAW_GPS and len(payload) >= 2:
            # Старые версии INAV возвращают 16 байт, новые могут добавлять HDOP.
            fix, satellites = struct.unpack_from("<BB", payload)
            self.telemetry.gps_fix = fix
            self.telemetry.gps_satellites = satellites
            if len(payload) >= 10:
                latitude, longitude = struct.unpack_from("<ii", payload, 2)
                self.telemetry.latitude = latitude / 10_000_000.0
                self.telemetry.longitude = longitude / 10_000_000.0
            # В старом варианте MSP курс может отсутствовать, но скорость есть.
            if len(payload) >= 14:
                speed = struct.unpack_from("<H", payload, 12)[0]
                self.telemetry.ground_speed = speed / 100.0
            if len(payload) >= 16:
                course = struct.unpack_from("<H", payload, 14)[0]
                self.telemetry.ground_course = course / 10.0
        elif command == MSP_STATUS and len(payload) >= 10:
            self.telemetry.mode_flags = struct.unpack_from("<I", payload, 6)[0]
            self._update_armed()
        elif command == MSP_BOXIDS:
            # Позиции ID совпадают с позициями битов активных режимов MSP_STATUS.
            self._box_ids = tuple(payload)
            self._update_armed()

    def _update_armed(self) -> None:
        """Определяет ARM по постоянному ID режима и активной битовой маске."""
        if self.telemetry.mode_flags is None or not self._box_ids:
            return
        try:
            arm_bit = self._box_ids.index(ARM_PERMANENT_ID)
        except ValueError:
            self.telemetry.armed = None
            return
        self.telemetry.armed = bool(self.telemetry.mode_flags & (1 << arm_bit))

    def _update_arm_switch(self) -> None:
        """Определяет положение ARM-тумблера по RC и настроенным диапазонам."""
        if not self._arm_ranges or not self.telemetry.rc_channels:
            return
        active_ranges = []
        for channel_index, start_us, end_us in self._arm_ranges:
            if channel_index >= len(self.telemetry.rc_channels):
                continue
            channel_value = self.telemetry.rc_channels[channel_index]
            active_ranges.append(start_us <= channel_value < end_us)
        if not active_ranges:
            return
        first_channel = self._arm_ranges[0][0]
        self.telemetry.arm_switch_value = self.telemetry.rc_channels[first_channel]
        self.telemetry.arm_switch = any(active_ranges)

    def update(self) -> InavTelemetry:
        """Возвращает согласованный снимок последних фоновых данных MSP."""
        if self._worker_error is not None:
            raise self._worker_error
        with self._telemetry_lock:
            return replace(self.telemetry)

    def close(self) -> None:
        """Останавливает фоновое чтение и закрывает порт без команд записи."""
        self._stop_requested.set()
        self._thread.join(timeout=1.0)
        self._serial.close()
