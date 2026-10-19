"""Прямой вывод кадров OpenCV в DRM/KMS без X11 и framebuffer-консоли."""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import re
import struct


DRM_MODE_CONNECTED = 1
DRM_IOCTL_BASE = ord("d")


def _ioc(direction: int, number: int, size: int) -> int:
    """Формирует номер ioctl Linux DRM."""
    return (direction << 30) | (size << 16) | (DRM_IOCTL_BASE << 8) | number


DRM_IOWR = 3
DRM_IOCTL_MODE_CREATE_DUMB = _ioc(DRM_IOWR, 0xB2, 32)
DRM_IOCTL_MODE_MAP_DUMB = _ioc(DRM_IOWR, 0xB3, 16)
DRM_IOCTL_MODE_DESTROY_DUMB = _ioc(DRM_IOWR, 0xB4, 8)
DRM_FORMAT_XRGB8888 = 0x34325258
DRM_CLIENT_CAP_UNIVERSAL_PLANES = 2


class _CreateDumb(ctypes.Structure):
    _fields_ = [("height", ctypes.c_uint32), ("width", ctypes.c_uint32), ("bpp", ctypes.c_uint32),
                ("flags", ctypes.c_uint32), ("handle", ctypes.c_uint32), ("pitch", ctypes.c_uint32),
                ("size", ctypes.c_uint64)]


class _MapDumb(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32), ("offset", ctypes.c_uint64)]


class _DestroyDumb(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint32), ("pad", ctypes.c_uint32)]


class _Mode(ctypes.Structure):
    _fields_ = [
        ("clock", ctypes.c_uint32),
        ("hdisplay", ctypes.c_uint16), ("hsync_start", ctypes.c_uint16),
        ("hsync_end", ctypes.c_uint16), ("htotal", ctypes.c_uint16),
        ("hskew", ctypes.c_uint16), ("vdisplay", ctypes.c_uint16),
        ("vsync_start", ctypes.c_uint16), ("vsync_end", ctypes.c_uint16),
        ("vtotal", ctypes.c_uint16), ("vscan", ctypes.c_uint16),
        ("vrefresh", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("type", ctypes.c_uint32), ("name", ctypes.c_char * 32),
    ]


class _Resources(ctypes.Structure):
    _fields_ = [
        ("count_fbs", ctypes.c_int), ("fbs", ctypes.POINTER(ctypes.c_uint32)),
        ("count_crtcs", ctypes.c_int), ("crtcs", ctypes.POINTER(ctypes.c_uint32)),
        ("count_connectors", ctypes.c_int), ("connectors", ctypes.POINTER(ctypes.c_uint32)),
        ("count_encoders", ctypes.c_int), ("encoders", ctypes.POINTER(ctypes.c_uint32)),
        ("min_width", ctypes.c_int), ("max_width", ctypes.c_int),
        ("min_height", ctypes.c_int), ("max_height", ctypes.c_int),
    ]


class _Connector(ctypes.Structure):
    _fields_ = [
        ("connector_id", ctypes.c_uint32), ("encoder_id", ctypes.c_uint32),
        ("connector_type", ctypes.c_uint32), ("connector_type_id", ctypes.c_uint32),
        ("connection", ctypes.c_uint32), ("mm_width", ctypes.c_uint32),
        ("mm_height", ctypes.c_uint32), ("subpixel", ctypes.c_uint32),
        ("count_modes", ctypes.c_int), ("modes", ctypes.POINTER(_Mode)),
        ("count_props", ctypes.c_int), ("props", ctypes.POINTER(ctypes.c_uint32)),
        ("prop_values", ctypes.POINTER(ctypes.c_uint64)),
        ("count_encoders", ctypes.c_int), ("encoders", ctypes.POINTER(ctypes.c_uint32)),
    ]


class _PlaneResources(ctypes.Structure):
    _fields_ = [("count_planes", ctypes.c_uint32), ("planes", ctypes.POINTER(ctypes.c_uint32))]


class _Plane(ctypes.Structure):
    _fields_ = [
        ("count_formats", ctypes.c_uint32), ("formats", ctypes.POINTER(ctypes.c_uint32)),
        ("plane_id", ctypes.c_uint32), ("crtc_id", ctypes.c_uint32),
        ("fb_id", ctypes.c_uint32), ("crtc_x", ctypes.c_int32),
        ("crtc_y", ctypes.c_int32), ("x", ctypes.c_int32),
        ("y", ctypes.c_int32), ("possible_crtcs", ctypes.c_uint32),
        ("gamma_size", ctypes.c_uint32),
    ]


class DrmOutput:
    """Выводит BGR-кадры в первый подключённый DRM-коннектор."""

    def __init__(self, device: str = "/dev/dri/card1") -> None:
        import cv2

        self._cv2 = cv2
        self._fd = os.open(device, os.O_RDWR | os.O_CLOEXEC)
        self._lib = ctypes.CDLL("libdrm.so.2")
        self._lib.drmSetMaster.argtypes = [ctypes.c_int]
        self._lib.drmSetMaster.restype = ctypes.c_int
        if self._lib.drmSetMaster(self._fd) != 0:
            os.close(self._fd)
            raise RuntimeError("Не удалось получить DRM master для видеовыхода")
        self._lib.drmSetClientCap.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64]
        self._lib.drmSetClientCap(self._fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)
        self._configure_drm()

    def _configure_drm(self) -> None:
        lib = self._lib
        lib.drmModeGetResources.restype = ctypes.POINTER(_Resources)
        lib.drmModeGetConnector.restype = ctypes.POINTER(_Connector)
        lib.drmModeFreeConnector.argtypes = [ctypes.POINTER(_Connector)]
        lib.drmModeFreeResources.argtypes = [ctypes.POINTER(_Resources)]
        lib.drmModeAddFB.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint8,
                                     ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint32,
                                     ctypes.POINTER(ctypes.c_uint32)]
        lib.drmModeSetCrtc.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
                                       ctypes.POINTER(_Mode)]
        resources = lib.drmModeGetResources(self._fd)
        if not resources:
            raise RuntimeError("DRM не вернул список ресурсов")
        connector = None
        try:
            for index in range(resources.contents.count_connectors):
                current = lib.drmModeGetConnector(self._fd, resources.contents.connectors[index])
                if current and current.contents.connection == DRM_MODE_CONNECTED and current.contents.count_modes:
                    connector = current
                    break
                if current:
                    lib.drmModeFreeConnector(current)
            if not connector:
                raise RuntimeError("Подключённый DRM-коннектор не найден")
            self.connector_id = connector.contents.connector_id
            self.crtc_id = self._find_crtc(resources.contents, connector.contents.encoder_id)
            self.mode = connector.contents.modes[0]
            self.width = self.mode.hdisplay
            self.height = self.mode.vdisplay
            self._create_buffer()
            self._set_plane()
        finally:
            if connector:
                lib.drmModeFreeConnector(connector)
            lib.drmModeFreeResources(resources)

    def _find_crtc(self, resources: _Resources, encoder_id: int) -> int:
        """Находит CRTC, совместимый с выбранным encoder."""
        encoder_type = ctypes.Structure
        class Encoder(ctypes.Structure):
            _fields_ = [("encoder_id", ctypes.c_uint32), ("encoder_type", ctypes.c_uint32),
                        ("crtc_id", ctypes.c_uint32), ("possible_crtcs", ctypes.c_uint32),
                        ("possible_clones", ctypes.c_uint32)]
        self._lib.drmModeGetEncoder.restype = ctypes.POINTER(Encoder)
        self._lib.drmModeFreeEncoder.argtypes = [ctypes.POINTER(Encoder)]
        encoder = self._lib.drmModeGetEncoder(self._fd, encoder_id)
        if encoder and encoder.contents.crtc_id:
            crtc_id = encoder.contents.crtc_id
            self._lib.drmModeFreeEncoder(encoder)
            return crtc_id
        if encoder:
            possible = encoder.contents.possible_crtcs
            self._lib.drmModeFreeEncoder(encoder)
            for index in range(resources.count_crtcs):
                if possible & (1 << index):
                    return resources.crtcs[index]
        raise RuntimeError("Совместимый CRTC для Composite не найден")

    def _create_buffer(self) -> None:
        create = bytearray(32)
        struct.pack_into("<4I", create, 0, self.height, self.width, 32, 0)
        fcntl.ioctl(self._fd, DRM_IOCTL_MODE_CREATE_DUMB, create, True)
        self.handle, self.pitch = struct.unpack_from("<2I", create, 16)
        self.size = struct.unpack_from("<Q", create, 24)[0]
        self.fb_id = ctypes.c_uint32()
        if self._lib.drmModeAddFB(
            self._fd, self.width, self.height, 24, 32, self.pitch,
            self.handle, ctypes.byref(self.fb_id),
        ) != 0:
            raise OSError("drmModeAddFB завершился ошибкой")
        mapped = bytearray(16)
        struct.pack_into("<I", mapped, 0, self.handle)
        fcntl.ioctl(self._fd, DRM_IOCTL_MODE_MAP_DUMB, mapped, True)
        offset = struct.unpack_from("<Q", mapped, 8)[0]
        self._map = mmap.mmap(self._fd, self.size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ, offset=offset)

    def _set_plane(self) -> None:
        """Подменяет изображение активного композитного DRM-plane."""
        lib = self._lib
        lib.drmModeGetPlaneResources.restype = ctypes.POINTER(_PlaneResources)
        lib.drmModeGetPlane.restype = ctypes.POINTER(_Plane)
        lib.drmModeFreePlaneResources.argtypes = [ctypes.POINTER(_PlaneResources)]
        lib.drmModeFreePlane.argtypes = [ctypes.POINTER(_Plane)]
        lib.drmModeSetPlane.argtypes = [
            ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ]
        resources = lib.drmModeGetPlaneResources(self._fd)
        if not resources:
            raise RuntimeError("DRM не вернул список plane")
        try:
            plane_id = None
            for index in range(resources.contents.count_planes):
                current_id = resources.contents.planes[index]
                plane = lib.drmModeGetPlane(self._fd, current_id)
                if plane and plane.contents.possible_crtcs & 1:
                    plane_id = current_id
                if plane:
                    lib.drmModeFreePlane(plane)
                if plane_id is not None:
                    break
            if plane_id is None:
                raise RuntimeError("Совместимый DRM-plane не найден")
            result = lib.drmModeSetPlane(
                self._fd, plane_id, self.crtc_id, self.fb_id.value, 0,
                0, 0, self.width, self.height,
                0, 0, self.width << 16, self.height << 16,
            )
            if result != 0:
                raise OSError(result, "drmModeSetPlane завершился ошибкой")
            self.plane_id = plane_id
        finally:
            lib.drmModeFreePlaneResources(resources)

    def write(self, frame) -> None:
        """Масштабирует BGR-кадр и обновляет DRM-буфер."""
        cv2 = self._cv2
        height, width = frame.shape[:2]
        scale = min(self.width / width, self.height / height)
        target = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), cv2.INTER_AREA)
        canvas = cv2.copyMakeBorder(
            target, (self.height - target.shape[0]) // 2, self.height - target.shape[0] - (self.height - target.shape[0]) // 2,
            (self.width - target.shape[1]) // 2, self.width - target.shape[1] - (self.width - target.shape[1]) // 2,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        bgra = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        row_bytes = bgra.tobytes()
        for row in range(self.height):
            start = row * self.pitch
            self._map[start : start + self.width * 4] = row_bytes[row * self.width * 4 : (row + 1) * self.width * 4]

    def close(self) -> None:
        """Освобождает DRM-буфер и устройство."""
        if getattr(self, "_map", None) is not None:
            self._map.close()
            self._map = None
        if getattr(self, "_fd", None) is not None:
            self._lib.drmDropMaster(self._fd)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "DrmOutput":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
