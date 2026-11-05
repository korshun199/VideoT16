"""Формат зашифрованного файла модели VideoT16 без записи открытой модели на диск."""

from __future__ import annotations

from pathlib import Path
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MAGIC = b"V16MOD\x00\x01"
NONCE_SIZE = 12
KEY_SIZE = 32  # Совместимость с тестами и форматом прежнего AES-256 контейнера.
KEY_SIZES = (16, KEY_SIZE)


class ModelContainerError(ValueError):
    """Сообщает о неверном ключе, повреждении или неизвестном формате модели."""


def encrypt_model_bytes(model_bytes: bytes, key: bytes, *, nonce: bytes | None = None) -> bytes:
    """Шифрует модель AES-256-GCM и защищает заголовок как дополнительные данные."""
    if not model_bytes:
        raise ValueError("Нельзя зашифровать пустую модель")
    if len(key) not in KEY_SIZES:
        raise ValueError("Ключ модели должен содержать 16 или 32 байта")
    chosen_nonce = secrets.token_bytes(NONCE_SIZE) if nonce is None else nonce
    if len(chosen_nonce) != NONCE_SIZE:
        raise ValueError("Nonce AES-GCM должен содержать 12 байт")
    header = MAGIC + chosen_nonce
    return header + AESGCM(key).encrypt(chosen_nonce, model_bytes, header)


def decrypt_model_bytes(container: bytes, key: bytes) -> bytes:
    """Проверяет заголовок и целостность, возвращая расшифрованные байты в RAM."""
    if len(key) not in KEY_SIZES:
        raise ValueError("Ключ модели должен содержать 16 или 32 байта")
    header_size = len(MAGIC) + NONCE_SIZE
    if len(container) <= header_size + 16 or not container.startswith(MAGIC):
        raise ModelContainerError("Файл не является поддерживаемым контейнером модели V16MOD")
    header = container[:header_size]
    nonce = header[len(MAGIC):]
    try:
        return AESGCM(key).decrypt(nonce, container[header_size:], header)
    except InvalidTag as error:
        raise ModelContainerError(
            "Не удалось проверить модель: неверный ключ или файл повреждён"
        ) from error


def decrypt_model_file(path: Path, key: bytes) -> bytes:
    """Считывает зашифрованную модель и расшифровывает её без временного ONNX-файла."""
    try:
        container = path.read_bytes()
    except OSError as error:
        raise OSError(f"Не удалось прочитать файл модели {path}: {error}") from error
    return decrypt_model_bytes(container, key)


def encrypt_model_file(source: Path, destination: Path, key: bytes) -> None:
    """Создаёт зашифрованную копию модели и отказывается перезаписывать файл."""
    if destination.exists():
        raise FileExistsError(f"Файл уже существует, перезапись запрещена: {destination}")
    try:
        model_bytes = source.read_bytes()
    except OSError as error:
        raise OSError(f"Не удалось прочитать исходную модель {source}: {error}") from error
    container = encrypt_model_bytes(model_bytes, key)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        output.write(container)
        output.flush()
