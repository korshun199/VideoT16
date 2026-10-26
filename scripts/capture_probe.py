#!/usr/bin/env python3
"""Сравнивает режимы видеозахвата без сохранения кадров и изменения профиля."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from src.capture_diagnostics import RollingCaptureMetrics, coarse_frame_signature  # noqa: E402


FOURCC_NAMES = ("YUYV", "MJPG")


@dataclass(frozen=True)
class CaptureMode:
    """Фактический или запрошенный режим одного V4L2-входа."""

    width: int
    height: int
    fps: float
    fourcc: str


def _fourcc_code(name: str) -> int:
    """Переводит четырёхсимвольное имя пиксельного формата в код OpenCV."""
    if name not in FOURCC_NAMES:
        raise ValueError(f"Неизвестный формат: {name}")
    return cv2.VideoWriter_fourcc(*name)


def _fourcc_name(value: float) -> str:
    """Возвращает читаемое имя FOURCC, не падая на неизвестном значении драйвера."""
    code = int(round(value))
    text = "".join(chr((code >> (8 * index)) & 0xFF) for index in range(4))
    return text if text.isprintable() else f"0x{code:08X}"


def _open_capture(source: int | str) -> Any:
    """Открывает V4L2-вход без GUI и без записи кадров."""
    return cv2.VideoCapture(source, cv2.CAP_V4L2)


def _read_mode(capture: Any) -> CaptureMode:
    """Читает режим, который драйвер фактически сообщает OpenCV."""
    return CaptureMode(
        width=int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        height=int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        fourcc=_fourcc_name(capture.get(cv2.CAP_PROP_FOURCC)),
    )


def _set_mode(capture: Any, mode: CaptureMode) -> CaptureMode:
    """Запрашивает режим у драйвера и возвращает его фактически принятые значения."""
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, _fourcc_code(mode.fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, mode.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, mode.height)
    capture.set(cv2.CAP_PROP_FPS, mode.fps)
    return _read_mode(capture)


def _source_argument(value: str) -> int | str:
    """Сохраняет путь устройства или переводит числовой индекс камеры в int."""
    return int(value) if value.isdigit() else value


def _read_initial_mode(source: int | str, factory: Callable[[int | str], Any]) -> CaptureMode:
    """Снимает исходный режим до теста, чтобы восстановить его в конце."""
    capture = factory(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть источник: {source}")
    try:
        return _read_mode(capture)
    finally:
        capture.release()


def _restore_mode(source: int | str, mode: CaptureMode, factory: Callable[[int | str], Any]) -> CaptureMode:
    """Возвращает устройству режим, действовавший до диагностики."""
    capture = factory(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось повторно открыть источник для восстановления: {source}")
    try:
        return _set_mode(capture, mode)
    finally:
        capture.release()


def probe_mode(
    source: int | str,
    requested: CaptureMode,
    warmup_frames: int,
    measured_frames: int,
    factory: Callable[[int | str], Any] = _open_capture,
) -> dict[str, Any]:
    """Измеряет один формат, не сохраняя изображения и не открывая окно."""
    if warmup_frames < 0 or measured_frames < 2:
        raise ValueError("Нужно не меньше двух измеряемых кадров; прогрев неотрицательный")
    capture = factory(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть источник: {source}")
    try:
        accepted = _set_mode(capture, requested)
        for _ in range(warmup_frames):
            ok, _frame = capture.read()
            if not ok:
                raise RuntimeError("Источник не вернул кадр во время прогрева")
        metrics = RollingCaptureMetrics(window_size=measured_frames)
        failures = 0
        for _ in range(measured_frames):
            started_ns = time.monotonic_ns()
            ok, frame = capture.read()
            finished_ns = time.monotonic_ns()
            if not ok or frame is None:
                failures += 1
                continue
            metrics.add(started_ns, finished_ns, coarse_frame_signature(frame))
        measured = metrics.snapshot()
        return {
            "requested": asdict(requested),
            "accepted": asdict(accepted),
            "successful_frames": measured.samples,
            "failed_frames": failures,
            "telemetry": asdict(measured),
        }
    finally:
        capture.release()


def default_report_path() -> Path:
    """Создаёт отдельный путь числового отчёта, не перезаписывая старые замеры."""
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "test_samples" / "video_trials" / "logs" / f"capture_probe_{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    """Создаёт аргументы короткой проверки YUYV и MJPEG."""
    parser = argparse.ArgumentParser(
        description="VideoT16: сравнение реальной скорости YUYV/MJPG без записи видео."
    )
    parser.add_argument("--source", default="/dev/video0", help="V4L2-узел или индекс OpenCV")
    parser.add_argument("--width", type=int, default=1280, help="запрашиваемая ширина")
    parser.add_argument("--height", type=int, default=960, help="запрашиваемая высота")
    parser.add_argument("--fps", type=float, default=60.0, help="запрашиваемая частота")
    parser.add_argument("--warmup", type=int, default=30, help="кадры прогрева формата")
    parser.add_argument("--frames", type=int, default=180, help="измеряемые кадры на формат")
    parser.add_argument("--log", type=Path, help="путь JSON-отчёта; по умолчанию в logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Запускает оба замера и в любом случае пытается вернуть исходный режим."""
    args = build_parser().parse_args(argv)
    source = _source_argument(args.source)
    requested_base = {"width": args.width, "height": args.height, "fps": args.fps}
    if args.width < 32 or args.height < 32 or args.fps <= 0:
        print("ОШИБКА: ширина, высота и FPS должны быть положительными", file=sys.stderr)
        return 2
    report: dict[str, Any] = {
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(source),
        "requested": requested_base,
        "formats": [],
        "initial_mode": None,
        "restored_mode": None,
    }
    exit_code = 0
    initial: CaptureMode | None = None
    try:
        initial = _read_initial_mode(source, _open_capture)
        report["initial_mode"] = asdict(initial)
        print(f"Источник: {source}; исходный режим: {initial.width}x{initial.height}@{initial.fps:.1f} {initial.fourcc}")
        for fourcc in FOURCC_NAMES:
            requested = CaptureMode(fourcc=fourcc, **requested_base)
            result = probe_mode(source, requested, args.warmup, args.frames)
            report["formats"].append(result)
            telemetry = result["telemetry"]
            accepted = result["accepted"]
            print(
                f"{fourcc}: драйвер={accepted['width']}x{accepted['height']}@{accepted['fps']:.1f} {accepted['fourcc']} | "
                f"успешно={result['successful_frames']}/{args.frames} | "
                f"FPS={telemetry['observed_fps']:.1f} | read p50/p90={telemetry['read_p50_ms']:.1f}/{telemetry['read_p90_ms']:.1f}ms | "
                f"gap p50/p90={telemetry['interval_p50_ms']:.1f}/{telemetry['interval_p90_ms']:.1f}ms | "
                f"грубые повторы={telemetry['coarse_same_percent']:.0f}%"
            )
    except Exception as error:  # noqa: BLE001 — результат нужен пользователю в отчёте.
        report["error"] = f"{type(error).__name__}: {error}"
        print(f"ОШИБКА: {report['error']}", file=sys.stderr)
        exit_code = 2
    finally:
        if initial is not None:
            try:
                restored = _restore_mode(source, initial, _open_capture)
                report["restored_mode"] = asdict(restored)
                print(f"Восстановлен режим: {restored.width}x{restored.height}@{restored.fps:.1f} {restored.fourcc}")
            except Exception as error:  # noqa: BLE001 — восстановление важнее обычной ошибки замера.
                report["restore_error"] = f"{type(error).__name__}: {error}"
                print(f"ОШИБКА восстановления: {report['restore_error']}", file=sys.stderr)
                exit_code = 3
        report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        path = (args.log or default_report_path()).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Отчёт: {path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
