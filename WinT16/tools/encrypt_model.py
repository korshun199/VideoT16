#!/usr/bin/env python3
"""Создаёт внешний зашифрованный контейнер модели для WinT16."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.activation import _short_model_key
from runtime.model_container import encrypt_model_file
from owner_keys import DEFAULT_KEYS_DIR


def read_key_file(path: Path, *, short_code: bool = False) -> bytes:
    """Читает ключ владельца; для коротких кодов выводит производный ключ AES-128."""
    raw = path.read_bytes()
    if len(raw) == 32:
        owner_key = raw
    else:
        try:
            owner_key = bytes.fromhex(raw.decode("ascii").strip())
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Ключ владельца должен быть 32 байта или 64 hex-символа") from error
        if len(owner_key) != 32:
            raise ValueError("Ключ владельца должен быть ровно 32 байта")
    if short_code:
        return _short_model_key(owner_key)
    return owner_key


def main(argv: list[str] | None = None) -> int:
    """Проверяет входы и шифрует ONNX-модель, не перезаписывая выходной файл."""
    parser = argparse.ArgumentParser(description="Шифрование ONNX-модели в контейнер V16MOD")
    parser.add_argument("model", type=Path, help="исходная модель .onnx")
    parser.add_argument("output", type=Path, help="выходной файл .v16m")
    parser.add_argument("--key-file", type=Path,
                        default=DEFAULT_KEYS_DIR / "model_aes.key",
                        help="закрытый файл AES-ключа владельца (по умолчанию — локальный owner_keys)")
    parser.add_argument("--short-code", action="store_true",
                        help="использовать общий AES-128 ключ для короткой активации v3")
    args = parser.parse_args(argv)
    try:
        if args.model.suffix.lower() != ".onnx":
            raise ValueError("Исходный файл должен иметь расширение .onnx")
        if args.output.suffix.lower() != ".v16m":
            raise ValueError("Файл результата должен иметь расширение .v16m")
        key = read_key_file(args.key_file, short_code=args.short_code)
        encrypt_model_file(args.model, args.output, key)
    except (OSError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1
    print(f"Зашифрованная модель создана: {args.output}")
    print("Ключ не копировался в файл настроек и не выводился в терминал.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
