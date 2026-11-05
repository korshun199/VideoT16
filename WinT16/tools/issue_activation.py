#!/usr/bin/env python3
"""Создаёт копируемый код активации для запроса конкретного компьютера."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.activation import ActivationError, issue_short_activation_code
from owner_keys import DEFAULT_KEYS_DIR


def main(argv: list[str] | None = None) -> int:
    """Принимает вставленный короткий запрос и печатает ответную строку."""
    parser = argparse.ArgumentParser(description="Код активации VideoT16 для отправки в мессенджере")
    parser.add_argument(
        "request", nargs="?",
        help="32-символьный код запроса пользователя; без аргумента спросит интерактивно",
    )
    parser.add_argument("--keys-dir", type=Path, default=DEFAULT_KEYS_DIR,
                        help="закрытый каталог ключей владельца на Linux")
    args = parser.parse_args(argv)
    folder = args.keys_dir.expanduser()
    try:
        request = args.request
        if request is None:
            request = input("Вставьте 32-символьный запрос пользователя: ").strip()
        if not request:
            raise ActivationError("Пустой запрос активации.")
        response = issue_short_activation_code(request, folder / "model_aes.key")
    except (OSError, ValueError, ActivationError) as error:
        print(f"Ошибка выпуска активации: {error}", file=sys.stderr)
        return 1
    print("Код активации для отправки пользователю:")
    print(response)
    print("Ответ содержит 32 заглавных символа; пользователь вставляет его в окно активации.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
