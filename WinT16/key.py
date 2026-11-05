#!/usr/bin/env python3
"""Короткая команда для выпуска ответа на запрос активации VideoT16."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from issue_activation import main


if __name__ == "__main__":
    raise SystemExit(main())
