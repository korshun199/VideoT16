"""Создаёт и устанавливает версии приложения, сохраняя общие данные пользователя."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
if __package__:
    from .runtime.activation import SIGNING_PUBLIC_KEY_B64
else:
    from runtime.activation import SIGNING_PUBLIC_KEY_B64

SIGNATURE_NAME = "signature.ed25519"
SIGNATURE_DOMAIN = b"VideoT16-update-v1\0"


def _version(value: str) -> str:
    """Проверяет безопасное имя версии, используемое как имя каталога."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Некорректное имя версии")
    return value


def _digest(path: Path) -> str:
    """Вычисляет контрольную сумму файла, читая его частями."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def pack(engine: Path, archive: Path, version: str) -> None:
    """Создаёт архив с одной версией и контрольными суммами файлов."""
    version = _version(version)
    if not (engine / "Engine.exe").is_file():
        raise ValueError("В каталоге сборки отсутствует Engine.exe")
    files = sorted(path for path in engine.rglob("*") if path.is_file())
    manifest = {"format": "VideoT16-update-v1", "version": version,
                "files": {p.relative_to(engine).as_posix(): _digest(p) for p in files}}
    with archive.open("xb") as output, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True))
        for path in files:
            bundle.write(path, path.relative_to(engine).as_posix())


def sign_archive(source: Path, destination: Path, private_key_path: Path) -> None:
    """Подписывает описание архива закрытым ключом владельца, не меняя исходник."""
    private = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise ValueError("Нужен закрытый ключ Ed25519 владельца")
    with zipfile.ZipFile(source) as incoming:
        entries = incoming.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or names.count("manifest.json") != 1 or SIGNATURE_NAME in names:
            raise ValueError("Некорректный исходный архив обновления")
        if incoming.getinfo("manifest.json").file_size > 4 * 1024 * 1024:
            raise ValueError("Описание обновления слишком велико")
        manifest_bytes = incoming.read("manifest.json")
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("format") != "VideoT16-update-v1":
            raise ValueError("Неподдерживаемый формат обновления")
        files = manifest.get("files")
        if not isinstance(files, dict) or "Engine.exe" not in files or set(names) != set(files) | {"manifest.json"}:
            raise ValueError("Состав обновления не соответствует описанию")
        for name, expected in files.items():
            checksum = hashlib.sha256()
            with incoming.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != expected:
                raise ValueError(f"Повреждён файл обновления: {name}")
        signature = private.sign(SIGNATURE_DOMAIN + manifest_bytes)
        with destination.open("xb") as output, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as outgoing:
            for item in entries:
                with incoming.open(item) as source_stream, outgoing.open(item.filename, "w") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
            outgoing.writestr(SIGNATURE_NAME, signature)


def install(archive: Path, root: Path, public_key_b64: str = SIGNING_PUBLIC_KEY_B64) -> str:
    """Добавляет проверенную версию и переключает запуск, сохраняя прежнюю версию."""
    root = root.resolve()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or names.count("manifest.json") != 1 or names.count(SIGNATURE_NAME) != 1:
            raise ValueError("Архив содержит повторяющиеся файлы или не содержит описание")
        if sum(entry.file_size for entry in entries) > 2 * 1024 ** 3:
            raise ValueError("Архив обновления слишком велик")
        manifest_info = bundle.getinfo("manifest.json")
        if manifest_info.file_size > 4 * 1024 * 1024:
            raise ValueError("Описание обновления слишком велико")
        if bundle.getinfo(SIGNATURE_NAME).file_size != 64:
            raise ValueError("Обновление не содержит корректную подпись владельца")
        manifest_bytes = bundle.read(manifest_info)
        try:
            key = serialization.load_der_public_key(base64.b64decode(public_key_b64, validate=True))
            if not isinstance(key, Ed25519PublicKey):
                raise ValueError("Неверный открытый ключ обновлений")
            key.verify(bundle.read(SIGNATURE_NAME), SIGNATURE_DOMAIN + manifest_bytes)
        except (InvalidSignature, ValueError, TypeError) as error:
            raise ValueError("Подпись обновления не подтверждена владельцем") from error
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("format") != "VideoT16-update-v1":
            raise ValueError("Неподдерживаемый формат обновления")
        version = _version(manifest.get("version", ""))
        files = manifest.get("files")
        if not isinstance(files, dict) or "Engine.exe" not in files or set(names) != set(files) | {"manifest.json", SIGNATURE_NAME}:
            raise ValueError("Состав обновления не соответствует описанию")
        for name, expected in files.items():
            path = PurePosixPath(name)
            if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                    or path.is_absolute() or any(part in (".", "..") for part in path.parts)
                    or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
                raise ValueError("В архиве обнаружен недопустимый путь или контрольная сумма")
        destination = root / "versions" / version
        destination.mkdir(parents=True, exist_ok=False)
        for name, expected in files.items():
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum = hashlib.sha256()
            with bundle.open(name) as source, target.open("xb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    checksum.update(chunk)
                    output.write(chunk)
            if checksum.hexdigest() != expected:
                raise ValueError(f"Повреждён файл обновления: {name}")
    pointer = root / "current.txt"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temporary = root / f"current.{stamp}.new"
    temporary.write_text(version + "\n", encoding="ascii")
    if pointer.exists():
        shutil.copy2(pointer, root / f"current.{stamp}.previous")
    os.replace(temporary, pointer)
    return version


def main() -> int:
    """Запускает сборку архива либо оконный выбор и установку обновления."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", nargs=3, metavar=("ENGINE", "ZIP", "VERSION"))
    parser.add_argument("--apply", type=Path, metavar="ZIP",
                        help="установить подписанное обновление в каталог программы")
    args = parser.parse_args()
    if args.pack:
        pack(Path(args.pack[0]), Path(args.pack[1]), args.pack[2])
        return 0
    if args.apply:
        root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
        install(args.apply, root)
        return 0
    import tkinter as tk
    from tkinter import filedialog, messagebox
    window = tk.Tk()
    window.withdraw()
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    try:
        archive = filedialog.askopenfilename(parent=window, title="Выберите обновление VideoT16",
                                             filetypes=[("Обновление VideoT16", "*.zip")])
        if not archive:
            return 0
        version = install(Path(archive), root)
        messagebox.showinfo("VideoT16", f"Обновление {version} установлено.\nЗапустите программу.", parent=window)
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        messagebox.showerror("VideoT16", f"Обновление не установлено:\n{error}", parent=window)
        return 1
    finally:
        window.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
