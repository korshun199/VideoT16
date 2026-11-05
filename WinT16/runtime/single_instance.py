"""Не допускает одновременный запуск двух копий одной установки Windows."""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import os
from pathlib import Path
from typing import Iterator


class AlreadyRunningError(RuntimeError):
    """Сообщает, что эта установка уже запущена в текущем сеансе Windows."""


@contextmanager
def single_instance(root: Path) -> Iterator[None]:
    """Занимает системный мьютекс на время запуска и освобождает его при выходе."""
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    kernel.CreateMutexW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_int
    identifier = hashlib.sha256(os.path.normcase(str(root.resolve())).encode("utf-8")).hexdigest()
    ctypes.set_last_error(0)
    handle = kernel.CreateMutexW(None, False, "Local\\VideoT16-" + identifier)
    if not handle:
        raise OSError(ctypes.get_last_error(), "Не удалось проверить повторный запуск")
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(handle)
        raise AlreadyRunningError("VideoT16 уже запущена. Используйте открытое окно.")
    try:
        yield
    finally:
        kernel.CloseHandle(handle)
