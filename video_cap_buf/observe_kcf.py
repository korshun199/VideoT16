"""Необязательный экспериментальный Rust/OpenCV KCF для окна наблюдения."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np

from detector import Detection


class KcfOverlay:
    """Сопровождает одну подтверждённую рамку ограниченное время."""

    def __init__(self, library: str, max_age_ms: int) -> None:
        """Проверяет локальную библиотеку и настраивает её C-интерфейс."""
        path = Path(library)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'Не найдена библиотека Rust KCF: {path}')
        if not 0 < max_age_ms <= 2000:
            raise ValueError('Возраст KCF должен быть от 1 до 2000 мс')
        self.lib = ctypes.CDLL(str(path))
        ptr, integer, size = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        self.lib.v16_observe_new.restype = ptr
        self.lib.v16_observe_free.argtypes = [ptr]
        self.lib.v16_observe_init.argtypes = [ptr, ptr, size, integer, integer, size,
                                              integer, integer, integer, integer]
        self.lib.v16_observe_update.argtypes = [ptr, ptr, size, integer, integer, size,
                                                ctypes.POINTER(integer)]
        self.max_age_ns = max_age_ms * 1_000_000
        self.handle: int | None = None
        self.anchor: Any = None
        self.box: Detection | None = None
        self.last_frame = 0

    def reset(self) -> None:
        """Освобождает нативный объект и забывает неподтверждённую рамку."""
        if self.handle:
            self.lib.v16_observe_free(self.handle)
        self.handle = None
        self.anchor = None
        self.box = None
        self.last_frame = 0

    @staticmethod
    def _image(image: Any) -> np.ndarray:
        """Проверяет BGR-кадр и при необходимости создаёт непрерывную копию."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('KCF ожидает BGR-кадр uint8 с тремя каналами')
        return np.ascontiguousarray(image)

    def seed(self, frame: Any, result: Any) -> bool:
        """Начинает сопровождение только на кадре с точным результатом модели."""
        self.reset()
        if result.number != frame.number or not result.detections:
            return False
        image = self._image(frame.image)
        detection = max(result.detections, key=lambda item: item.confidence)
        self.handle = self.lib.v16_observe_new()
        if not self.handle:
            raise RuntimeError('Не удалось создать Rust KCF')
        code = self.lib.v16_observe_init(self.handle, image.ctypes.data, image.nbytes,
                    image.shape[1], image.shape[0], image.strides[0], detection.left, detection.top,
                    detection.right - detection.left, detection.bottom - detection.top)
        if code != 1:
            self.reset()
            return False
        self.anchor, self.box, self.last_frame = result, detection, frame.number
        return True

    def update(self, frame: Any) -> tuple[Detection, ...]:
        """Возвращает рамку только по новому кадру и до срока подтверждения."""
        if not self.handle or self.anchor is None or self.box is None:
            return ()
        if frame.number != self.last_frame + 1 or not 0 < frame.captured_ns - self.anchor.captured_ns <= self.max_age_ns:
            self.reset()
            return ()
        image = self._image(frame.image)
        output = (ctypes.c_int * 4)()
        code = self.lib.v16_observe_update(self.handle, image.ctypes.data, image.nbytes,
                    image.shape[1], image.shape[0], image.strides[0], output)
        if code != 1:
            self.reset()
            return ()
        x, y, width, height = output
        self.last_frame = frame.number
        return (Detection(self.box.class_id, f'~{self.box.label}', self.box.confidence,
                          x, y, x + width, y + height),)
