#!/usr/bin/env python3
"""Конвертирует ONNX-модель в RKNN для NPU Orange Pi 5 (RK3588S)."""

import argparse
from pathlib import Path

from rknn.api import RKNN


def build_parser() -> argparse.ArgumentParser:
    """Создаёт параметры конвертации."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="Путь к ONNX-модели")
    parser.add_argument("rknn", type=Path, help="Путь к итоговой RKNN-модели")
    return parser


def main() -> int:
    """Создаёт не-квантизированную RKNN-модель для RK3588."""
    args = build_parser().parse_args()
    rknn = RKNN(verbose=False)
    try:
        print("Настройка RKNN для RK3588...")
        if rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform="rk3588",
        ) != 0:
            raise RuntimeError("Не удалось настроить RKNN")
        if rknn.load_onnx(model=str(args.onnx)) != 0:
            raise RuntimeError("Не удалось загрузить ONNX-модель")
        if rknn.build(do_quantization=False) != 0:
            raise RuntimeError("Не удалось собрать RKNN-модель")
        args.rknn.parent.mkdir(parents=True, exist_ok=True)
        if rknn.export_rknn(str(args.rknn)) != 0:
            raise RuntimeError("Не удалось сохранить RKNN-модель")
    finally:
        rknn.release()
    print(f"RKNN-модель создана: {args.rknn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
