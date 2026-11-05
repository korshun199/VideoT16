"""Прямой запуск VideoT16 из переносимого каталога без окна настроек."""

from __future__ import annotations

import sys
import ctypes

from runtime.paths import application_root
from runtime.single_instance import AlreadyRunningError, single_instance


def main(arguments: list[str] | None = None) -> int:
    """Добавляет каталоги приложения в путь импорта и сразу запускает распознавание."""
    args = list(sys.argv[1:] if arguments is None else arguments)
    root = application_root()
    app_dir = root / "app"
    for search_path in (root, app_dir):
        if str(search_path) not in sys.path:
            sys.path.insert(0, str(search_path))

    try:
        with single_instance(root):
            from capture import main as capture_main
            return capture_main(args)
    except (AlreadyRunningError, OSError) as error:
        if sys.platform == "win32":
            ctypes.windll.user32.MessageBoxW(None, str(error), "VideoT16", 0x30)
        else:
            print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
