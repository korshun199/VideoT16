#!/usr/bin/env python3

"""Создаёт предварительный INT8-вариант ONNX для оценки размера."""

import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> int:
    """Квантует веса ONNX без подмены исходной модели."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(args.source),
        model_output=str(args.target),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"INT8-вариант создан: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

