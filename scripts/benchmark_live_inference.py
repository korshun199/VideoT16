#!/usr/bin/env python3
"""Сравнивает задержку YOLO на нескольких кадрах живого потока.

Кадры берутся с камеры один раз, хранятся только в оперативной памяти и затем
подаются всем вариантам потоков и разбиения. Видеофайл и изображения на диск
не записываются; сохраняется только текстовый отчёт с измерениями.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402

from scripts.compare_video_trials import (  # noqa: E402
    generate_tiles,
    infer_full_frame,
    infer_tiles,
    model_input_size,
)
from scripts.live_pipeline_trial import (  # noqa: E402
    _camera_api,
    _camera_source,
    configure_capture,
    load_profile,
    validate_profile,
)
from src.onnx_detector import OnnxDetector  # noqa: E402


def default_thread_counts(cpu_count: int | None = None) -> list[int]:
    """Выбирает небольшую сетку потоков с верхним значением по числу CPU."""
    available = max(1, cpu_count if cpu_count is not None else (os.cpu_count() or 1))
    candidates = [1, 2, 4, 8, 12, min(available, 12)]
    return sorted({value for value in candidates if value <= available})


def parse_thread_counts(value: str) -> list[int]:
    """Проверяет переданный список потоков, не позволяя задать ноль или дубли."""
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Потоки укажи числами через запятую, например 1,2,4,8,12") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("Количество потоков должно быть положительным")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("В списке потоков не должно быть повторов")
    return values


def build_modes(frame: Any, baseline_tile_size: int, overlap: float) -> list[dict[str, Any]]:
    """Готовит текущую плиточную схему, более крупные плитки и весь кадр."""
    height, width = frame.shape[:2]
    alternate_tile_size = max(baseline_tile_size + 1, min(max(width, height), 960))
    modes: list[dict[str, Any]] = []
    for label, tile_size in (
        ("текущие плитки", baseline_tile_size),
        ("крупные плитки", alternate_tile_size),
    ):
        tile_count = len(generate_tiles(frame, tile_size, overlap))
        if not any(mode["tile_size"] == tile_size for mode in modes):
            modes.append({"name": f"{label} ({tile_count})", "kind": "tiles", "tile_size": tile_size})
    modes.append({"name": "весь кадр (1)", "kind": "full", "tile_size": None})
    return modes


def infer_mode(detector: OnnxDetector, frame: Any, mode: dict[str, Any], overlap: float):
    """Обрабатывает один сохранённый в RAM кадр выбранной схемой."""
    if mode["kind"] == "full":
        detections, elapsed_ms, raw_confidence = infer_full_frame(detector, frame)
    else:
        detections, elapsed_ms, raw_confidence = infer_tiles(
            detector,
            frame,
            int(mode["tile_size"]),
            overlap,
        )
    return detections, elapsed_ms, raw_confidence


def summarize(records: list[dict[str, Any]], confidence_threshold: float) -> list[dict[str, Any]]:
    """Строит сравнительные итоги по медиане времени и видимым попаданиям."""
    summaries: list[dict[str, Any]] = []
    keys = sorted({(item["threads"], item["mode"]) for item in records})
    for threads, mode in keys:
        group = [item for item in records if item["threads"] == threads and item["mode"] == mode]
        times = [float(item["inference_ms"]) for item in group]
        visible = sum(int(item["visible_detections"]) > 0 for item in group)
        summaries.append(
            {
                "threads": threads,
                "mode": mode,
                "samples": len(group),
                "median_ms": round(statistics.median(times), 1),
                "min_ms": round(min(times), 1),
                "max_ms": round(max(times), 1),
                "equivalent_fps": round(1000.0 / statistics.median(times), 2),
                "samples_with_box_above_threshold": visible,
                "median_cpu_core_equivalent": round(
                    statistics.median(float(item.get("cpu_core_equivalent", 0.0)) for item in group),
                    2,
                ),
                "confidence_threshold": confidence_threshold,
                "median_raw_confidence": round(statistics.median(float(item["raw_confidence"]) for item in group), 4),
            }
        )
    return sorted(summaries, key=lambda item: (item["threads"], item["median_ms"]))


def make_logger(log_path: Path) -> logging.Logger:
    """Пишет понятный ход испытания одновременно в терминал и отдельный лог."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("videot16-live-benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def capture_samples(profile: Any, sample_count: int, logger: logging.Logger) -> tuple[list[Any], dict[str, Any]]:
    """Считывает несколько свежих кадров в RAM и сразу закрывает устройство."""
    source = _camera_source(profile.source)
    capture = cv2.VideoCapture(source, _camera_api())
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть камеру {profile.source}; проверь, не занята ли она другим окном")
    try:
        actual = configure_capture(capture, profile.capture, logger)
        frames = []
        for number in range(1, sample_count + 1):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Камера не вернула кадр для образца {number}/{sample_count}")
            frames.append(frame.copy())
            logger.info("Кадр %d/%d взят в RAM (%dx%d)", number, sample_count, frame.shape[1], frame.shape[0])
        return frames, actual
    finally:
        capture.release()


def benchmark(profile: Any, thread_counts: list[int], sample_count: int, logger: logging.Logger) -> dict[str, Any]:
    """Сравнивает режимы на одинаковых кадрах и освобождает камеру до инференса."""
    if not profile.model.is_file():
        raise FileNotFoundError(f"ONNX-модель не найдена: {profile.model}")
    model_size = model_input_size(profile.model)
    frames, capture_actual = capture_samples(profile, sample_count, logger)
    modes = build_modes(frames[0], profile.tile_size, profile.tile_overlap)
    logger.info("Источник закрыт; дальше камера не занята. Кадры остались только в оперативной памяти.")
    logger.info("Режимы: %s", "; ".join(mode["name"] for mode in modes))
    logger.info("Потоки ONNX Runtime: %s; модельный вход: %d", ",".join(map(str, thread_counts)), model_size)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for threads in thread_counts:
        logger.info("Создаю ONNX-сессию с threads=%d", threads)
        try:
            detector = OnnxDetector(
                profile.model,
                profile.min_confidence,
                False,
                model_size,
                threads=threads,
            )
        except Exception as error:  # noqa: BLE001 — не теряем остальные варианты, если один недоступен.
            logger.exception("Не удалось создать сессию threads=%d: %s", threads, error)
            failures.append({"threads": threads, "mode": "session", "error": str(error)})
            continue
        try:
            working_modes = []
            for mode in modes:
                try:
                    logger.info("Прогрев: threads=%d, %s", threads, mode["name"])
                    infer_mode(detector, frames[0], mode, profile.tile_overlap)
                    working_modes.append(mode)
                except Exception as error:  # noqa: BLE001 — остальные схемы ещё могут сработать.
                    logger.exception("Прогрев не прошёл: threads=%d, %s: %s", threads, mode["name"], error)
                    failures.append({"threads": threads, "mode": mode["name"], "error": str(error)})
            for sample_number, frame in enumerate(frames, start=1):
                for mode in working_modes:
                    cpu_started = time.process_time()
                    wall_started = time.perf_counter()
                    try:
                        detections, elapsed_ms, raw_confidence = infer_mode(
                            detector,
                            frame,
                            mode,
                            profile.tile_overlap,
                        )
                    except Exception as error:  # noqa: BLE001 — фиксируем отказ отдельного режима.
                        logger.exception("Режим завершился ошибкой: threads=%d, %s: %s", threads, mode["name"], error)
                        failures.append({"threads": threads, "mode": mode["name"], "sample": sample_number, "error": str(error)})
                        continue
                    wall_ms = (time.perf_counter() - wall_started) * 1000.0
                    cpu_ms = (time.process_time() - cpu_started) * 1000.0
                    cpu_cores = cpu_ms / max(wall_ms, 0.001)
                    visible = sum(item.confidence >= profile.confidence for item in detections)
                    record = {
                        "threads": threads,
                        "mode": mode["name"],
                        "sample": sample_number,
                        "inference_ms": round(elapsed_ms, 2),
                        "raw_confidence": round(float(raw_confidence), 5),
                        "detections": len(detections),
                        "visible_detections": visible,
                        "cpu_core_equivalent": round(cpu_cores, 2),
                    }
                    records.append(record)
                    logger.info(
                        "threads=%d | %s | кадр=%d | %.0f мс | CPU≈%.1f ядра | raw=%.0f%% | рамок=%d | выше порога=%d",
                        threads,
                        mode["name"],
                        sample_number,
                        elapsed_ms,
                        cpu_cores,
                        raw_confidence * 100,
                        len(detections),
                        visible,
                    )
        finally:
            del detector

    return {
        "profile": profile.name,
        "model": str(profile.model),
        "model_input_size": model_size,
        "capture": capture_actual,
        "samples_in_ram": sample_count,
        "tested_threads": thread_counts,
        "failures": failures,
        "results": summarize(records, profile.confidence),
        "measurements": records,
    }


def build_parser() -> argparse.ArgumentParser:
    """Описывает короткие параметры испытания без скрытых изменений системы."""
    parser = argparse.ArgumentParser(description="Замер скорости YOLO на живом USB-потоке без сохранения кадров.")
    parser.add_argument("--profile", default="sky", help="профиль камеры и модели")
    parser.add_argument("--samples", type=int, default=2, help="сколько кадров держать в RAM для сравнения (по умолчанию 2)")
    parser.add_argument("--threads", type=parse_thread_counts, help="потоки ONNX через запятую, например 1,2,4,8,12")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Проверяет параметры, запускает замер и сохраняет только числовой отчёт."""
    args = build_parser().parse_args(argv)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    log_path = PROJECT_ROOT / "logs" / f"inference_benchmark_{stamp}.log"
    report_path = PROJECT_ROOT / "logs" / f"inference_benchmark_{stamp}.json"
    logger = make_logger(log_path)
    try:
        if not 1 <= args.samples <= 10:
            raise ValueError("--samples должен быть от 1 до 10, чтобы короткий замер не перегружал компьютер")
        profile = load_profile(args.profile)
        validate_profile(profile)
        thread_counts = args.threads or default_thread_counts()
        logger.info("VideoT16: короткий замер YOLO; файлы кадров не создаются")
        logger.info("Профиль=%s | камера=%s | модель=%s | кадры=%d", profile.name, profile.source, profile.model, args.samples)
        report = benchmark(profile, thread_counts, args.samples, logger)
        report["created_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("ИТОГ: числа сохранены в %s", report_path)
        logger.info("LOG: %s", log_path)
        logger.info("Сводка: потоки | режим | медиана мс | эквивалент FPS | CPU ядра | кадров с рамкой")
        for row in report["results"]:
            logger.info(
                "%2d | %s | %7.1f | %5.2f | %4.1f | %d/%d",
                row["threads"],
                row["mode"],
                row["median_ms"],
                row["equivalent_fps"],
                row["median_cpu_core_equivalent"],
                row["samples_with_box_above_threshold"],
                row["samples"],
            )
        return 0
    except Exception as error:  # noqa: BLE001 — сохраняем понятную ошибку в журнале.
        logger.exception("ОШИБКА замера: %s", error)
        return 2
    finally:
        logging.getLogger("videot16-live-benchmark").handlers.clear()


if __name__ == "__main__":
    raise SystemExit(main())
