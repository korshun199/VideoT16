#!/usr/bin/env python3
"""Однократно создаёт закрытые ключи владельца вне каталога и Git-проекта."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import secrets
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


DEFAULT_KEYS_DIR = Path("/home/work/.local/share/VideoT16/owner_keys")


def _write_private(path: Path, contents: bytes) -> None:
    """Создаёт новый секретный файл с правами владельца и запрещает перезапись."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())


def main(argv: list[str] | None = None) -> int:
    """Создаёт общий AES-ключ модели и закрытый ключ подписи активаций один раз."""
    import argparse

    parser = argparse.ArgumentParser(description="Подготовка закрытых ключей владельца VideoT16")
    parser.add_argument("--keys-dir", type=Path, default=DEFAULT_KEYS_DIR)
    args = parser.parse_args(argv)
    folder = args.keys_dir.expanduser()
    files = (folder / "model_aes.key", folder / "activation_signing_private.pem",
             folder / "activation_signing_public.pem")
    try:
        if folder.exists() and any(path.exists() for path in files):
            raise FileExistsError("Найдены существующие ключи; ничего не перезаписано.")
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(folder, 0o700)
        signing_key = Ed25519PrivateKey.generate()
        private_pem = signing_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_pem = signing_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        public_der = signing_key.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        _write_private(files[0], secrets.token_bytes(32))
        _write_private(files[1], private_pem)
        _write_private(files[2], public_pem)
        print(f"Созданы закрытые ключи владельца: {folder}")
        print("Права каталога и файлов ограничены владельцем (0700/0600).")
        print("Открытый ключ для встраивания в WinT16:")
        print(base64.b64encode(public_der).decode("ascii"))
        print("Сохраните резервную копию каталога в защищённом месте; не добавляйте его в Git.")
        return 0
    except (OSError, FileExistsError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
