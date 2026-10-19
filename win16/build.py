#!/usr/bin/env python3
"""Собирает общий payload Video16 и личный activation.dat для Windows-ПК.

Payload можно разместить в GitHub Release: он одинаков для всех пользователей.
Ключ payload хранится только у владельца в игнорируемом каталоге. Для каждого
``init.dat`` эта программа формирует отдельный ``activation.dat``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import io
import json
import os
import zipfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


PAYLOAD_MAGIC = b"V16P001\0"
DOMAIN = b"Video16/windows-payload/hardware-v3/"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def owner_key_path() -> Path:
    """Возвращает личное хранилище ключа владельца вне Git-репозитория."""
    configured = os.environ.get("VIDEO16_OWNER_KEY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".local" / "share" / "VideoT16" / "owner_windows_payload_key.bin"


def device_key(document: dict[str, object]) -> str:
    """Формирует канонический аппаратный ключ из присланного init.dat."""
    identity = document["identity"]
    computer = identity["computer"]
    bios = identity["bios"]
    tpm = identity["tpm"]
    values = (
        str(computer["UUID"]).strip().upper(),
        str(bios["SerialNumber"]).strip().upper(),
        str(tpm["TpmPresent"]).strip().lower(),
    )
    if not values[0] or not values[1]:
        raise ValueError("в init.dat отсутствует UUID или серийный номер BIOS")
    return "|".join(values)


def hardware_key(identity: str) -> bytes:
    """Выводит ключ шифрования activation из аппаратного паспорта."""
    return hmac.new(DOMAIN, identity.encode("utf-8"), hashlib.sha256).digest()


def sha256_file(path: Path) -> str:
    """Вычисляет SHA-256 файла потоково."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_key() -> bytes:
    """Создаёт постоянный закрытый ключ payload или читает ранее созданный."""
    key_path = owner_key_path()
    if key_path.is_file():
        key = key_path.read_bytes()
        if len(key) != 32:
            raise ValueError(f"неверный размер закрытого ключа: {key_path}")
        return key
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key = os.urandom(32)
    key_path.write_bytes(key)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    print(f"[OK] создан закрытый ключ владельца: {key_path}")
    return key


def make_archive(output: io.BytesIO) -> None:
    """Укладывает runtime, его исходники, настройки и модель в общий payload."""
    files = [
        "win16/runtime.py",
        "src/__init__.py", "src/local_object_detection.py", "src/realtime.py",
        "src/tracker.py", "src/onnx_detector.py", "src/osd.py",
        "src/mcm_font.py", "src/preview.py",
        "config/__init__.py", "config/osd_config.py",
        "config/runtime_settings.py", "config/runtime_settings.json",
        "models/fpv_drone_best.onnx",
    ]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            source = PROJECT_ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            name = "runtime.py" if relative == "win16/runtime.py" else relative
            archive.write(source, name)


def make_archive_with_runtime_patch(base_payload: Path, base_release: str, output: io.BytesIO) -> None:
    """Берёт проверенный payload и заменяет только Windows-оболочку runtime.py."""
    payload = base_payload.read_bytes()
    if not payload.startswith(PAYLOAD_MAGIC):
        raise ValueError(f"неверный формат исходного payload: {base_payload}")
    encrypted = payload[len(PAYLOAD_MAGIC):]
    plaintext = AESGCM(content_key()).decrypt(
        encrypted[:12], encrypted[12:], base_release.encode("utf-8")
    )
    with zipfile.ZipFile(io.BytesIO(plaintext)) as source, zipfile.ZipFile(
        output, "w", zipfile.ZIP_DEFLATED
    ) as archive:
        if "runtime.py" not in source.namelist():
            raise ValueError("в исходном payload отсутствует runtime.py")
        for info in source.infolist():
            if info.filename != "runtime.py":
                archive.writestr(info, source.read(info.filename))
        archive.write(PROJECT_ROOT / "win16" / "runtime.py", "runtime.py")


def build_release(
    release: str,
    payload_url: str,
    base_payload: Path | None = None,
    base_release: str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Создаёт неизменяемый выпуск payload и публичный release.json."""
    release_dir = PROJECT_ROOT / "win16" / "releases" / release
    payload_path = release_dir / "Video16.payload"
    manifest_path = release_dir / "release.json"
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError(f"выпуск {release!r} уже существует; используйте новое имя выпуска")
    archive = io.BytesIO()
    if base_payload is None:
        make_archive(archive)
    else:
        if not base_release:
            raise ValueError("для исправления существующего payload укажите --from-release")
        make_archive_with_runtime_patch(base_payload, base_release, archive)
    nonce = os.urandom(12)
    encrypted = AESGCM(content_key()).encrypt(nonce, archive.getvalue(), release.encode("utf-8"))
    release_dir.mkdir(parents=True, exist_ok=False)
    payload_path.write_bytes(PAYLOAD_MAGIC + nonce + encrypted)
    manifest: dict[str, object] = {
        "format": "Video16 release v1",
        "release_id": release,
        "key_id": "owner-key-v1",
        "payload_url": payload_url,
        "payload_sha256": sha256_file(payload_path),
        "payload_size": payload_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return release_dir, manifest


def write_activation(output: Path, identity: str, key: bytes) -> Path:
    """Шифрует общий ключ для одного ПК и сохраняет его в activation.dat."""
    nonce = os.urandom(12)
    wrapped = AESGCM(hardware_key(identity)).encrypt(nonce, key, identity.encode("utf-8"))
    activation = {
        "format": "Video16 activation v4",
        "key_id": "owner-key-v1",
        "device_key_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "wrapped_content_key": base64.b64encode(nonce + wrapped).decode("ascii"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(activation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    """Создаёт новый общий выпуск либо персональный activation.dat."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("init_dat", type=Path, nargs="?", help="init.dat целевого Windows-компьютера")
    parser.add_argument("--release", help="новое имя выпуска, например 2026.10.19")
    parser.add_argument("--payload-url", help="публичный URL Video16.payload")
    parser.add_argument("--from-payload", type=Path, help="проверенный payload, в котором заменить только runtime.py")
    parser.add_argument("--from-release", help="release_id исходного payload для --from-payload")
    parser.add_argument("--activation", type=Path, help="куда записать личный activation.dat")
    args = parser.parse_args()

    if args.release or args.payload_url:
        if not args.release or not args.payload_url:
            raise ValueError("для нового выпуска нужны одновременно --release и --payload-url")
        release_dir, manifest = build_release(
            args.release, args.payload_url, args.from_payload, args.from_release
        )
        print(f"[OK] выпуск payload: {release_dir / 'Video16.payload'}")
        print(f"[OK] release.json: {release_dir / 'release.json'}")
        print(f"[OK] SHA-256: {manifest['payload_sha256']}")
    if args.init_dat or args.activation:
        if not args.init_dat or not args.activation:
            raise ValueError("для activation нужны одновременно init_dat и --activation")
        document = json.loads(args.init_dat.read_text(encoding="utf-8"))
        identity = device_key(document)
        activation = write_activation(args.activation, identity, content_key())
        print(f"[OK] персональный activation: {activation}")
        print("[OK] activation можно передать пользователю; init.dat в Git не добавляется")
    if not (args.release or args.payload_url or args.init_dat or args.activation):
        raise ValueError("укажите выпуск через --release/--payload-url или init_dat с --activation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
