#!/usr/bin/env python3
"""Собирает обновление на Windows и подписывает его ключом владельца на Linux."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from runtime.activation import SIGNING_PUBLIC_KEY_B64
from update import sign_archive
from tools.prepare_build_key import build_key_module_source

ROOT = Path(__file__).resolve().parent


def _quote_powershell(value: str) -> str:
    """Экранирует один текстовый аргумент PowerShell без исполнения содержимого."""
    return "'" + value.replace("'", "''") + "'"


def _run(arguments: list[str]) -> None:
    """Выполняет команду и прекращает сборку при ненулевом результате."""
    result = subprocess.run(arguments, check=False)
    if result.returncode:
        raise RuntimeError(f"Команда {Path(arguments[0]).name} завершилась с кодом {result.returncode}")


def _ssh_command(ssh: list[str], host: str, script: str) -> None:
    """Передаёт PowerShell-сценарий Windows в UTF-16 без shell-подстановок."""
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    _run(ssh + [host, "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded])


def _verify_owner_key(path: Path) -> None:
    """Проверяет, что закрытый ключ соответствует публичному ключу программы."""
    private = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise ValueError("Нужен закрытый ключ Ed25519 владельца")
    public = private.public_key().public_bytes(serialization.Encoding.DER,
                                               serialization.PublicFormat.SubjectPublicKeyInfo)
    if base64.b64encode(public).decode("ascii") != SIGNING_PUBLIC_KEY_B64:
        raise ValueError("Закрытый ключ не соответствует ключу этой версии программы")


def main() -> int:
    """Собирает только движок на Windows, затем локально подписывает ZIP обновления."""
    parser = argparse.ArgumentParser(description="Сборка подписанного обновления VideoT16")
    parser.add_argument("--config", type=Path, default=ROOT / "build.local.json")
    args = parser.parse_args()
    source_package: tempfile.TemporaryDirectory | None = None
    remote: str | None = None
    ssh: list[str] | None = None
    host: str | None = None
    remote_created = False
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        host = config["host"]
        remote_parent = config["remote_parent"].rstrip("/")
        identity = Path(config["identity_file"]).expanduser()
        private = Path(config["keys_dir"]).expanduser() / "activation_signing_private.pem"
        model_key_path = Path(config["keys_dir"]).expanduser() / "model_aes.key"
        if (not isinstance(host, str) or host.startswith("-") or any(ch.isspace() for ch in host)
                or not re.fullmatch(r"[A-Za-z]:/[A-Za-z0-9_./-]+", remote_parent)
                or not identity.is_file()):
            raise ValueError("Неверные параметры SSH или путей сборочного компьютера")
        _verify_owner_key(private)
        embedded_source = build_key_module_source(model_key_path)
        version = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        local_work = ROOT.parent / "tmp" / "WinT16-builds" / version
        local_work.mkdir(parents=True, exist_ok=False)
        temporary_root = ROOT.parent / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        source_package = tempfile.TemporaryDirectory(
            prefix="WinT16-source-", dir=temporary_root
        )
        archive = Path(source_package.name) / "source.tgz"
        files = [ROOT / "Run.py", ROOT / "update.py", ROOT / "build_update.ps1",
                 ROOT / "requirements.txt"]
        files += sorted((ROOT / "app").glob("*.py"))
        files += [
            path for path in sorted((ROOT / "runtime").glob("*.py"))
            if path.name != "_embedded_model_key.py"
        ]
        if any(not path.is_file() or path.is_symlink() for path in files):
            raise ValueError("В исходниках найдена ссылка или отсутствует файл сборки")
        with tarfile.open(archive, "x:gz") as package:
            for path in files:
                package.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
            embedded_info = tarfile.TarInfo("runtime/_embedded_model_key.py")
            embedded_info.size = len(embedded_source)
            embedded_info.mode = 0o600
            package.addfile(embedded_info, io.BytesIO(embedded_source))
        ssh = ["ssh", "-i", str(identity), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
        scp = ["scp", "-i", str(identity), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes"]
        remote = f"{remote_parent}/{version}"
        print(f"Создаю отдельную сборку {version} на Windows...", flush=True)
        _ssh_command(ssh, host, "$ErrorActionPreference='Stop'; if(-not (Test-Path -LiteralPath " +
                     _quote_powershell(remote_parent) + ")) { New-Item -ItemType Directory -Path " +
                     _quote_powershell(remote_parent) + " -ErrorAction Stop | Out-Null }; "
                     "New-Item -ItemType Directory -Path " + _quote_powershell(remote) +
                     " -ErrorAction Stop | Out-Null")
        remote_created = True
        _run(scp + [str(archive), f"{host}:{remote}/source.tgz"])
        script = ("$ErrorActionPreference='Stop'; Set-Location -LiteralPath " + _quote_powershell(remote) +
                  "; tar -xzf source.tgz; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}; "
                  "& .\\build_update.ps1 -Version " + _quote_powershell(version) +
                  "; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}")
        _ssh_command(ssh, host, script)
        unsigned = local_work / "unsigned.zip"
        print("Забираю сборку на ноутбук владельца...", flush=True)
        _run(scp + [f"{host}:{remote}/dist/update-unsigned-{version}.zip", str(unsigned)])
        output = ROOT / "dist" / f"update-{version}.zip"
        output.parent.mkdir(parents=True, exist_ok=True)
        print("Подписываю обновление локально; закрытый ключ остаётся на этом ноутбуке...", flush=True)
        sign_archive(unsigned, output, private)
        print(f"Готово: {output}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, RuntimeError) as error:
        print(f"Сборка не завершена: {error}", file=sys.stderr)
        return 1
    finally:
        if source_package is not None:
            source_package.cleanup()
        if remote_created and remote and ssh and host:
            try:
                print("Удаляю временную сборочную папку с тестового Windows-компьютера...", flush=True)
                _ssh_command(
                    ssh,
                    host,
                    "$ErrorActionPreference='Stop'; if(Test-Path -LiteralPath "
                    + _quote_powershell(remote)
                    + "){Remove-Item -LiteralPath " + _quote_powershell(remote)
                    + " -Recurse -Force}",
                )
            except (OSError, RuntimeError) as error:
                print(
                    "ВНИМАНИЕ: не удалось удалить временные материалы сборки на Windows; "
                    f"папка: {remote}; причина: {error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
