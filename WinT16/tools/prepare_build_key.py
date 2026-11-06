#!/usr/bin/env python3
"""Подготавливает локальный модуль ключа для первоначальной Windows-сборки."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.activation import ActivationError, _short_model_key
from tools.owner_keys import DEFAULT_KEYS_DIR


def build_key_module_source(key_path: Path) -> bytes:
    """Создаёт текст исходника с производным AES-128-ключом без его вывода."""
    try:
        model_key = _short_model_key(key_path.read_bytes())
    except OSError as error:
        raise ActivationError(f"Не удалось прочитать закрытый ключ модели: {error}") from error
    source = (
        '"""Секретный модуль сборки; не добавлять в Git и не передавать клиенту как файл."""\n'
        f'MODEL_AES_KEY = bytes.fromhex("{model_key.hex()}")\n'
    )
    return source.encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Создаёт runtime/_embedded_model_key.py один раз с правами владельца."""
    parser = argparse.ArgumentParser(
        description="Подготовка закрытого ключевого модуля для Windows-сборки"
    )
    parser.add_argument("--keys-dir", type=Path, default=DEFAULT_KEYS_DIR)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "runtime" / "_embedded_model_key.py")
    arguments = parser.parse_args(argv)
    output = arguments.output.expanduser()
    key_path = arguments.keys_dir.expanduser() / "model_aes.key"
    try:
        source = build_key_module_source(key_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(source)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError:
        print(f"Отказ: файл уже существует, ничего не изменено: {output}", file=sys.stderr)
        return 1
    except (OSError, ActivationError) as error:
        print(f"Не удалось подготовить ключ сборки: {error}", file=sys.stderr)
        return 1
    print(f"Готов закрытый ключевой модуль: {output}")
    print("Файл игнорируется Git; используйте его только на компьютере сборки.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
