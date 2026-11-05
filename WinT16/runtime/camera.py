"""Открытие и проверка USB-видеоустройств на Windows и Linux."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from typing import Any, Callable, Iterable

import cv2


class CameraOpenError(RuntimeError):
    """Сообщает, что выбранное видеоустройство не открылось ни одним backend."""


@dataclass(frozen=True)
class CameraInfo:
    """Хранит свойства камеры и способ её открытия через OpenCV."""

    index: int
    backend: str
    width: int
    height: int
    fps: float
    name: str = ""
    backend_id: int = cv2.CAP_ANY
    vid: int | None = None
    pid: int | None = None


def camera_backends(system_name: str | None = None) -> tuple[tuple[str, int], ...]:
    """Возвращает подходящие backend OpenCV в порядке предпочтения ОС."""
    system = (system_name or platform.system()).lower()
    if system == "windows":
        return (("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF),
                ("Автовыбор OpenCV", cv2.CAP_ANY))
    if system == "linux":
        return (("V4L2", cv2.CAP_V4L2), ("Автовыбор OpenCV", cv2.CAP_ANY))
    return (("Автовыбор OpenCV", cv2.CAP_ANY),)


def _camera_source(source: int | str) -> int | str:
    """Преобразует строковый номер камеры в индекс, сохраняя другие пути как есть."""
    if isinstance(source, bool):
        raise ValueError("Номер камеры должен быть целым числом или строкой-индексом")
    if isinstance(source, int):
        if source < 0:
            raise ValueError("Номер камеры не может быть отрицательным")
        return source
    value = source.strip()
    if not value:
        raise ValueError("Не выбран источник видео")
    if value.isdecimal():
        return int(value)
    return value


def open_camera(
    source: int | str,
    *,
    system_name: str | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
) -> tuple[Any, str]:
    """Открывает камеру через системные backend и возвращает устройство с именем backend."""
    camera = _camera_source(source)
    failures: list[str] = []
    for backend_name, backend_id in camera_backends(system_name):
        device = None
        try:
            device = capture_factory(camera, backend_id)
            if device.isOpened():
                return device, backend_name
            failures.append(f"{backend_name}: устройство не открылось")
        except Exception as error:
            failures.append(f"{backend_name}: {error}")
        if device is not None:
            try:
                device.release()
            except Exception:
                pass
    details = "; ".join(failures) or "нет доступных backend"
    raise CameraOpenError(f"Не удалось открыть камеру {source!r}. {details}")


def probe_cameras(
    max_index: int = 8,
    *,
    system_name: str | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
) -> list[CameraInfo]:
    """Проверяет индексы камер; каждый успешно открытый тестовый поток освобождается."""
    if max_index < 1:
        raise ValueError("max_index должен быть не меньше 1")
    cameras: list[CameraInfo] = []
    for index in range(max_index):
        try:
            device, backend_name = open_camera(
                index, system_name=system_name, capture_factory=capture_factory
            )
        except CameraOpenError:
            continue
        try:
            try:
                opened, image = device.read()
            except Exception:
                continue
            if not opened or image is None:
                continue
            cameras.append(CameraInfo(
                index=index,
                backend=backend_name,
                width=int(device.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(device.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(device.get(cv2.CAP_PROP_FPS)),
            ))
        finally:
            device.release()
    return cameras


def enumerate_windows_cameras() -> list[CameraInfo]:
    """Перечисляет камеры Windows с именами и идентификаторами USB-устройств."""
    if platform.system().lower() != "windows":
        raise RuntimeError("Каталог камер по именам доступен только в Windows")
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as error:
        raise RuntimeError(
            "Не установлен cv2-enumerate-cameras; переустановите зависимости WinT16"
        ) from error

    cameras: list[CameraInfo] = []
    for backend_name, backend_id in camera_backends("windows")[:2]:
        try:
            discovered = enumerate_cameras(apiPreference=backend_id)
        except TypeError:
            discovered = enumerate_cameras(backend_id)
        except Exception:
            continue
        for item in discovered:
            index = int(getattr(item, "index"))
            name = str(getattr(item, "name", "") or "")
            vid = _optional_int(getattr(item, "vid", None))
            pid = _optional_int(getattr(item, "pid", None))
            actual_backend = _optional_int(getattr(item, "backend", backend_id))
            if actual_backend is None:
                actual_backend = backend_id
            candidate = CameraInfo(
                index=index,
                backend=backend_name,
                width=0,
                height=0,
                fps=0.0,
                name=name,
                backend_id=actual_backend,
                vid=vid,
                pid=pid,
            )
            if not any(existing.index == index and existing.backend_id == backend_id
                       for existing in cameras):
                cameras.append(candidate)
    return cameras


def _optional_int(value: Any) -> int | None:
    """Преобразует VID/PID в число, сохраняя пустое или некорректное значение."""
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def find_hdmi_capture(
    cameras: Iterable[CameraInfo],
    name_keywords: Iterable[str],
    *,
    required_vid: int | None = None,
    required_pid: int | None = None,
) -> CameraInfo | None:
    """Возвращает только единственный совпавший оцифровщик; не угадывает индекс."""
    keywords = tuple(word.casefold().strip() for word in name_keywords if word.strip())
    matches = [
        camera for camera in cameras
        if (not keywords or any(word in camera.name.casefold() for word in keywords))
        and (required_vid is None or camera.vid == required_vid)
        and (required_pid is None or camera.pid == required_pid)
    ]
    return matches[0] if len(matches) == 1 else None


def open_hdmi_capture(
    *,
    name_keywords: Iterable[str],
    required_vid: int | None = None,
    required_pid: int | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    enumerator: Callable[[], list[CameraInfo]] = enumerate_windows_cameras,
) -> tuple[Any, CameraInfo]:
    """Открывает единственный найденный HDMI-оцифровщик по его имени и backend."""
    cameras = enumerator()
    keywords = tuple(word.casefold().strip() for word in name_keywords if word.strip())
    failures: list[str] = []
    for _, backend_id in camera_backends("windows")[:2]:
        backend_cameras = [camera for camera in cameras if camera.backend_id == backend_id]
        selected = find_hdmi_capture(
            backend_cameras,
            keywords,
            required_vid=required_vid,
            required_pid=required_pid,
        )
        if selected is not None:
            device = None
            try:
                device = capture_factory(selected.index, selected.backend_id)
                if device.isOpened():
                    return device, selected
                failures.append(f"{selected.backend}: устройство не открылось")
            except Exception as error:
                failures.append(f"{selected.backend}: {error}")
            if device is not None:
                device.release()
            continue
        matching_count = sum(
            1 for camera in backend_cameras
            if (not keywords or any(word in camera.name.casefold() for word in keywords))
            and (required_vid is None or camera.vid == required_vid)
            and (required_pid is None or camera.pid == required_pid)
        )
        if matching_count > 1:
            raise CameraOpenError("Найдено несколько подходящих оцифровщиков; уточните настройки.")
    if failures:
        raise CameraOpenError("Не удалось открыть оцифровщик: " + "; ".join(failures))
    names = ", ".join(
            f"{camera.name or 'без имени'} (индекс {camera.index})" for camera in cameras
        ) or "камеры не обнаружены"
    raise CameraOpenError(f"Ожидаю единственный HDMI-оцифровщик; найдено: {names}")
