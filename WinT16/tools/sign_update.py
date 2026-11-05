#!/usr/bin/env python3
"""Подписывает обновление владельцем на Linux, не передавая закрытый ключ Windows."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from owner_keys import DEFAULT_KEYS_DIR
from update import sign_archive


def main() -> int:
    """Проверяет исходный архив и создаёт отдельный подписанный файл."""
    parser = argparse.ArgumentParser(description="Подпись обновления VideoT16")
    parser.add_argument("source", type=Path, help="архив, собранный на Windows")
    parser.add_argument("destination", type=Path, help="новый подписанный архив для испытателей")
    parser.add_argument("--keys-dir", type=Path, default=DEFAULT_KEYS_DIR)
    args = parser.parse_args()
    try:
        sign_archive(args.source, args.destination, args.keys_dir / "activation_signing_private.pem")
    except (OSError, ValueError) as error:
        print(f"Не удалось подписать обновление: {error}", file=sys.stderr)
        return 1
    print(f"Подписанное обновление: {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
