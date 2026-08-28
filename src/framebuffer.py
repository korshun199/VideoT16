"""Вывод обработанных кадров в Linux framebuffer без X11."""

from __future__ import annotations

import ctypes
import fcntl
import glob
import mmap
import os
import struct
from pathlib import Path


FBIOGET_VSCREENINFO = 0x4600
FBIOGET_FSCREENINFO = 0x4602


class _FixScreenInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_char * 16),
        ("smem_start", ctypes.c_ulong),
        ("smem_len", ctypes.c_uint),
        ("type", ctypes.c_uint),
        ("visual", ctypes.c_uint),
        ("xpanstep", ctypes.c_ushort),
        ("ypanstep", ctypes.c_ushort),
        ("ywrapstep", ctypes.c_ushort),
        ("pad", ctypes.c_ushort),
        ("line_length", ctypes.c_uint),
        ("mmio_start", ctypes.c_ulong),
        ("mmio_len", ctypes.c_uint),
        ("accel", ctypes.c_uint),
        ("capabilities", ctypes.c_ushort),
        ("reserved", ctypes.c_ushort * 2),
    ]


def find_framebuffer() -> str:
    """Находит framebuffer, связанный с композитным DRM-выходом."""
    for status_file in sorted(glob.glob("/sys/class/graphics/fb*/device/drm/*-Composite-1/status")):
        framebuffer = next((part for part in Path(status_file).parts if part.startswith("fb")), "")
        candidate = f"/dev/{framebuffer}" if framebuffer else ""
        if os.path.exists(candidate):
            return candidate
    candidates = sorted(glob.glob("/dev/fb*"))
    if candidates:
        return candidates[0]
    raise RuntimeError("Framebuffer не найден: проверьте /dev/fb* и Composite-1")


class FramebufferOutput:
    """Пишет BGR-кадры OpenCV в устройство framebuffer."""

    def __init__(self, device: str = "auto") -> None:
        import cv2

        self._cv2 = cv2
        self.device = find_framebuffer() if device in ("", "auto") else device
        self._fd = os.open(self.device, os.O_RDWR)
        var = bytearray(160)
        fix = _FixScreenInfo()
        try:
            fcntl.ioctl(self._fd, FBIOGET_VSCREENINFO, var, True)
            fcntl.ioctl(self._fd, FBIOGET_FSCREENINFO, fix)
        except Exception:
            os.close(self._fd)
            raise
        values = struct.unpack_from("<8I", var)
        self.width, self.height = values[0], values[1]
        self.bits_per_pixel = values[6]
        self.line_length = fix.line_length
        self.red_offset, self.red_length = struct.unpack_from("<2I", var, 32)
        self.green_offset, self.green_length = struct.unpack_from("<2I", var, 44)
        self.blue_offset, self.blue_length = struct.unpack_from("<2I", var, 56)
        self._map = mmap.mmap(self._fd, fix.smem_len, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)
        if self.bits_per_pixel not in (16, 24, 32):
            self.close()
            raise RuntimeError(f"Неподдерживаемая глубина framebuffer: {self.bits_per_pixel} бит")

    def write(self, frame) -> None:
        """Масштабирует кадр с чёрными полями и выводит его на экран."""
        cv2 = self._cv2
        height, width = frame.shape[:2]
        scale = min(self.width / width, self.height / height)
        target_width = max(1, round(width * scale))
        target_height = max(1, round(height * scale))
        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(
            resized,
            (self.height - target_height) // 2,
            self.height - target_height - (self.height - target_height) // 2,
            (self.width - target_width) // 2,
            self.width - target_width - (self.width - target_width) // 2,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        if self.bits_per_pixel == 16:
            packed = (
                ((canvas[:, :, 2].astype("uint16") >> 3) << self.red_offset)
                | ((canvas[:, :, 1].astype("uint16") >> 2) << self.green_offset)
                | ((canvas[:, :, 0].astype("uint16") >> 3) << self.blue_offset)
            ).astype("uint16")
        elif self.bits_per_pixel == 24:
            packed = canvas[:, :, ::-1]
        else:
            packed = canvas[:, :, ::-1]
            alpha = 255 * (packed[:, :, :1] * 0 + 1)
            packed = self._cv2.merge((packed[:, :, 0], packed[:, :, 1], packed[:, :, 2], alpha[:, :, 0]))
        row_bytes = packed.tobytes()
        bytes_per_pixel = self.bits_per_pixel // 8
        active_bytes = self.width * bytes_per_pixel
        for row in range(self.height):
            start = row * self.line_length
            self._map[start : start + active_bytes] = row_bytes[row * active_bytes : (row + 1) * active_bytes]

    def close(self) -> None:
        """Освобождает framebuffer."""
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FramebufferOutput":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
