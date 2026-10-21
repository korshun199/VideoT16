#!/usr/bin/env python3
"""Проверяет активацию и запускает Video16 на подтверждённом Windows-ПК.

Этот файл устанавливает ``install.py``. Он не содержит payload и персональный
ключ: общий зашифрованный выпуск берётся из открытого каталога ``win16`` в
GitHub, а ``activation.dat`` владелец выпускает по одноразовому ``init.dat``.
Сам ``init.dat`` для запуска больше не требуется.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


PAYLOAD_MAGIC = b"V16P001\0"
DOMAIN = b"Video16/windows-payload/hardware-v3/"
GITHUB_RAW_BASE = "https://github.com/korshun199/VideoT16/raw/refs/heads/win16/win16"
DEFAULT_MANIFEST_URL = f"{GITHUB_RAW_BASE}/release.json"
REQUIREMENTS = (
    "cryptography>=42,<49",
    "numpy>=2.0,<3",
    "opencv-python>=4.10,<5",
    "onnxruntime>=1.20,<2",
)
REQUIRED_MODULES = ("cryptography", "numpy", "cv2", "onnxruntime")
LOCAL_PYTHON = Path(".venv") / "Scripts" / "python.exe"


class InstallError(RuntimeError):
    """Понятная ошибка установки, которую можно безопасно показать человеку."""


def configure_console() -> None:
    """Не даёт локальной кодировке консоли Windows оборвать запуск программы."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


class Journal:
    """Выводит сообщения одновременно в консоль и install.log."""

    def __init__(self, path: Path) -> None:
        """Открывает журнал в UTF-8 и создаёт его каталог при необходимости."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        """Закрывает файл журнала."""
        self._file.close()

    def line(self, message: str, state: str = "INFO") -> None:
        """Печатает одно датированное сообщение в консоль и файл."""
        stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{stamp}] [{state}] {message}"
        print(text, flush=True)
        self._file.write(text + "\n")

    def command(self, args: list[str], env: dict[str, str] | None = None) -> int:
        """Запускает команду и сохраняет весь её вывод в журнал."""
        self.line("Команда: " + " ".join(args))
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for text in process.stdout:
            text = text.rstrip()
            if text:
                self.line(text, "PIP")
        return process.wait()


def install_root() -> Path:
    """Возвращает пользовательскую папку, в которой находится start.py."""
    return Path(__file__).resolve().parent


def journal_path(root: Path, install_only: bool) -> Path:
    """Выбирает отдельный журнал установки либо обычного запуска программы."""
    return root / ("install.log" if install_only else "run.log")


def sha256_file(path: Path) -> str:
    """Возвращает SHA-256 файла без загрузки его целиком в память."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_identity() -> dict[str, object]:
    """Считывает UUID, серийный номер BIOS и признак TPM средствами Windows."""
    script = r"""
$computer = Get-CimInstance Win32_ComputerSystemProduct | Select-Object -First 1 UUID,Name
$bios = Get-CimInstance Win32_BIOS | Select-Object -First 1 SerialNumber
$tpm = Get-Tpm | Select-Object -First 1 TpmPresent,ManagedAuthLevel
[pscustomobject]@{ computer = $computer; bios = $bios; tpm = $tpm } |
    ConvertTo-Json -Compress -Depth 4
"""
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(result.stdout)


def device_key(identity: dict[str, object]) -> str:
    """Формирует стабильное каноническое представление оборудования Windows."""
    computer = identity.get("computer") or {}
    bios = identity.get("bios") or {}
    tpm = identity.get("tpm") or {}
    values = (
        str(computer.get("UUID", "")).strip().upper(),
        str(bios.get("SerialNumber", "")).strip().upper(),
        str(tpm.get("TpmPresent", "")).strip().lower(),
    )
    if not values[0] or not values[1]:
        raise InstallError("Windows не вернула UUID или серийный номер BIOS")
    return "|".join(values)


def parse_manifest(data: bytes) -> dict[str, Any]:
    """Проверяет структуру публичного манифеста одного выпуска."""
    try:
        manifest: dict[str, Any] = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstallError(f"некорректный release.json: {error}") from error
    required = ("format", "release_id", "key_id", "payload_url", "payload_sha256", "payload_size")
    if any(not manifest.get(key) for key in required):
        raise InstallError("release.json не содержит обязательные поля")
    if manifest["format"] != "Video16 release v1":
        raise InstallError("неподдерживаемый формат release.json")
    digest = str(manifest["payload_sha256"])
    if len(digest) != 64 or any(symbol not in "0123456789abcdef" for symbol in digest.lower()):
        raise InstallError("в release.json неверный SHA-256 payload")
    try:
        size = int(manifest["payload_size"])
    except (TypeError, ValueError) as error:
        raise InstallError("в release.json неверный размер payload") from error
    if size < len(PAYLOAD_MAGIC) + 12 + 16:
        raise InstallError("в release.json указан слишком маленький payload")
    return manifest


def download(url: str, destination: Path, journal: Journal) -> None:
    """Скачивает файл через временное имя и заменяет цель только после успеха."""
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    journal.line(f"Скачивание: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as target:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                target.write(block)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise InstallError(f"не удалось скачать {url}: {error}") from error
    temporary.replace(destination)


def fetch_manifest(url: str, destination: Path, journal: Journal) -> dict[str, Any]:
    """Обновляет release.json условным запросом или оставляет подтверждённую копию."""
    headers: dict[str, str] = {}
    metadata_path = destination.with_suffix(".http.json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("etag"):
                headers["If-None-Match"] = str(metadata["etag"])
            if metadata.get("last_modified"):
                headers["If-Modified-Since"] = str(metadata["last_modified"])
        except (OSError, json.JSONDecodeError):
            pass
    request = urllib.request.Request(url, headers=headers)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    journal.line(f"Проверяю обновление: {url}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
            metadata_path.write_text(
                json.dumps(
                    {
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
    except urllib.error.HTTPError as error:
        if error.code == 304 and destination.is_file():
            journal.line("release.json не изменился, использую локальную копию", "SKIP")
            return parse_manifest(destination.read_bytes())
        temporary.unlink(missing_ok=True)
        raise InstallError(f"не удалось получить release.json: HTTP {error.code}") from error
    except OSError as error:
        temporary.unlink(missing_ok=True)
        if destination.is_file():
            journal.line("Сеть недоступна, использую ранее сохранённый release.json", "WARN")
            return parse_manifest(destination.read_bytes())
        raise InstallError(f"не удалось получить release.json: {error}") from error
    temporary.replace(destination)
    return parse_manifest(destination.read_bytes())


def valid_payload(path: Path, manifest: dict[str, Any]) -> bool:
    """Проверяет размер и SHA-256 локального payload по публичному манифесту."""
    return (
        path.is_file()
        and path.stat().st_size == int(manifest["payload_size"])
        and hmac.compare_digest(sha256_file(path), str(manifest["payload_sha256"]))
    )


def fetch_payload(root: Path, manifest_url: str, journal: Journal) -> tuple[Path, dict[str, Any]]:
    """Получает выпуск в каталог программы, пропуская проверенную копию."""
    manifest_path = root / "release.json"
    manifest = fetch_manifest(manifest_url, manifest_path, journal)
    payload_path = root / "Video16.payload"
    if valid_payload(payload_path, manifest):
        journal.line(f"Payload {manifest['release_id']} уже проверен, скачивание не требуется", "SKIP")
        return payload_path, manifest
    if payload_path.exists():
        journal.line("Локальный payload не совпал с манифестом, загружаю заново", "WARN")
    download(str(manifest["payload_url"]), payload_path, journal)
    if not valid_payload(payload_path, manifest):
        payload_path.unlink(missing_ok=True)
        raise InstallError("проверка SHA-256 payload не пройдена")
    journal.line(f"Payload {manifest['release_id']} скачан и проверен", "OK")
    return payload_path, manifest


def dependencies_ready(root: Path) -> bool:
    """Проверяет локальное виртуальное окружение и его обязательные модули."""
    marker = root / "dependencies.json"
    python = root / LOCAL_PYTHON
    if not marker.is_file() or not python.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if data != {"python": sys.version, "requirements": list(REQUIREMENTS)}:
        return False
    imports = ";".join(f"import {name}" for name in REQUIRED_MODULES)
    result = subprocess.run(
        [str(python), "-c", imports],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.returncode == 0


def local_python(root: Path) -> Path:
    """Возвращает Python виртуального окружения внутри папки программы."""
    return root / LOCAL_PYTHON


def local_command_environment(root: Path) -> dict[str, str]:
    """Направляет временные файлы и кэш pip в папку программы."""
    temporary = root / ".tmp"
    cache = root / ".pip-cache"
    temporary.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "TMP": str(temporary),
        "TEMP": str(temporary),
        "TMPDIR": str(temporary),
        "PIP_CACHE_DIR": str(cache),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    })
    return environment


def ensure_dependencies(root: Path, journal: Journal) -> None:
    """Создаёт локальное окружение и ставит зависимости только внутрь папки программы."""
    if dependencies_ready(root):
        journal.line("Зависимости уже проверены, повторная установка не нужна", "SKIP")
        return
    environment = local_command_environment(root)
    python = local_python(root)
    if not python.is_file():
        journal.line("Создаю локальное окружение Python в папке программы")
        if journal.command([sys.executable, "-m", "venv", str(root / ".venv")], environment) != 0:
            raise InstallError("не удалось создать локальное окружение Python")
    journal.line("Устанавливаю зависимости в локальное окружение; это может занять несколько минут")
    command = [str(python), "-m", "pip", "install", "--upgrade", *REQUIREMENTS]
    if journal.command(command, environment) != 0:
        raise InstallError("pip не смог установить зависимости; подробности в install.log")
    imports = ";".join(f"import {name}" for name in REQUIRED_MODULES)
    result = subprocess.run([str(python), "-c", imports], capture_output=True)
    if result.returncode != 0:
        raise InstallError("после pip не найдены обязательные модули")
    marker = {"python": sys.version, "requirements": list(REQUIREMENTS)}
    (root / "dependencies.json").write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    journal.line("Зависимости установлены и проверены", "OK")


def runtime_temp_root(root: Path) -> Path:
    """Создаёт каталог расшифрованного runtime внутри папки программы."""
    return Path(tempfile.mkdtemp(prefix=".runtime-", dir=root))


def safe_extract(archive_data: bytes, destination: Path) -> None:
    """Распаковывает ZIP только по путям внутри временного каталога."""
    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise InstallError("payload содержит небезопасный путь")
        archive.extractall(destination)


def decrypt_payload(root: Path, payload_path: Path, manifest: dict[str, Any], identity: str) -> bytes:
    """Проверяет персональную активацию и расшифровывает общий payload в память."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as error:
        raise InstallError("не найдена cryptography после установки зависимостей") from error
    activation_path = root / "activation.dat"
    if not activation_path.is_file():
        raise InstallError(
            f"ожидаю персональный activation.dat в {root}; получите его у владельца программы"
        )
    try:
        activation = json.loads(activation_path.read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(fingerprint, str(activation["device_key_sha256"])):
            raise InstallError("activation.dat выдан для другого компьютера")
        wrapped = base64.b64decode(activation["wrapped_content_key"], validate=True)
        if len(wrapped) < 28:
            raise InstallError("activation.dat повреждён")
        payload = payload_path.read_bytes()
        if not payload.startswith(PAYLOAD_MAGIC):
            raise InstallError("payload имеет неверный формат")
        if str(activation["key_id"]) != str(manifest["key_id"]):
            raise InstallError("activation.dat относится к другому ключу выпуска")
        hardware = hmac.new(DOMAIN, identity.encode("utf-8"), hashlib.sha256).digest()
        content_key = AESGCM(hardware).decrypt(wrapped[:12], wrapped[12:], identity.encode("utf-8"))
        encrypted = payload[len(PAYLOAD_MAGIC):]
        return AESGCM(content_key).decrypt(encrypted[:12], encrypted[12:], str(manifest["release_id"]).encode("utf-8"))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, InstallError):
            raise
        raise InstallError(f"не удалось открыть activation.dat: {error}") from error


def run_application(root: Path, payload_path: Path, manifest: dict[str, Any], journal: Journal) -> int:
    """Подтверждает ПК, временно распаковывает runtime и запускает Video16."""
    identity = device_key(read_identity())
    plaintext = decrypt_payload(root, payload_path, manifest, identity)
    temporary_root = runtime_temp_root(root)
    try:
        safe_extract(plaintext, temporary_root)
        runtime = temporary_root / "runtime.py"
        if not runtime.is_file():
            raise InstallError("в payload отсутствует runtime.py")
        journal.line("Устройство подтверждено, запускаю Video16", "OK")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8:backslashreplace"
        environment["VIDEO16_INSTALL_ROOT"] = str(root)
        environment.update(local_command_environment(root))
        process = subprocess.Popen(
            [str(local_python(root)), str(runtime)],
            cwd=temporary_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                journal.line(text, "VIDEO16")
        result = process.wait()
        journal.line(f"Video16 завершён с кодом {result}")
        return result
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
        journal.line("Временный расшифрованный каталог удалён", "OK")


def main() -> int:
    """Обновляет выпуск при необходимости и запускает либо готовит Video16."""
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="совместимость со старым ярлыком")
    parser.add_argument("--install-only", action="store_true", help="только подготовить файлы, не запускать Video16")
    parser.add_argument("--no-pause", action="store_true", help="не ждать Enter после установки")
    parser.add_argument("--manifest", default=os.environ.get("VIDEO16_MANIFEST_URL", DEFAULT_MANIFEST_URL))
    args = parser.parse_args()
    if platform.system() != "Windows":
        print("[ERROR] start.py предназначен для Windows", flush=True)
        return 2
    root = install_root()
    journal = Journal(journal_path(root, args.install_only))
    try:
        journal.line("=== Video16: начало проверки/установки ===")
        journal.line(f"Рабочий каталог: {root}")
        journal.line(f"Python: {sys.version.split()[0]}")
        root.mkdir(parents=True, exist_ok=True)
        ensure_dependencies(root, journal)
        python = local_python(root)
        if Path(sys.executable).resolve() != python.resolve():
            journal.line("Передаю запуск локальному Python из папки программы", "OK")
            command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
            return journal.command(command, local_command_environment(root))
        payload_path, manifest = fetch_payload(root, args.manifest, journal)
        if args.install_only:
            journal.line(f"Установка готова. Поместите activation.dat в {root} и запустите start.py", "OK")
            return 0
        return run_application(root, payload_path, manifest, journal)
    except InstallError as error:
        journal.line(str(error), "ERROR")
        return 1
    except Exception as error:  # noqa: BLE001 -- полный след нужен владельцу для диагностики.
        journal.line(f"непредвиденная ошибка: {error}", "ERROR")
        journal.line(traceback.format_exc(), "TRACE")
        return 1
    finally:
        journal.line("=== Video16: завершение ===")
        journal.close()
        if args.install_only and not args.no_pause:
            print(f"\nЖурнал: {root / 'install.log'}")
            input("Нажмите Enter, чтобы закрыть окно...")


if __name__ == "__main__":
    raise SystemExit(main())
