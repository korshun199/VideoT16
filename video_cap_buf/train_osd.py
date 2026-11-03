#!/usr/bin/env python3
"""Экспериментальное обучение небольшой сегментации OSD на синтетике."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Результаты и временные материалы не смешиваются с рабочими моделями.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "osd_training"
DEFAULT_BACKGROUNDS = OUTPUT / "backgrounds_20261102"
WIDTH = 320
HEIGHT = 192
VAL_SEED = 2_000_000_000
TRAIN_FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN)
VAL_FONTS = (cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX)
WORDS = ("AIR", "ARM", "DISARM", "GPS", "RSSI", "VOLT", "BAT", "ALT", "LQ", "MODE", "HOME")
TRAIN_BACKGROUNDS: list[np.ndarray] = []
VAL_BACKGROUNDS: list[np.ndarray] = []


class Segmenter(nn.Module):
    """Небольшая отдельная сеть: по одному кадру оценивает маску графики."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = self.block(3, 12)
        self.encoder2 = self.block(12, 24)
        self.encoder3 = self.block(24, 40)
        self.decoder2 = self.block(64, 24)
        self.decoder1 = self.block(36, 12)
        self.output = nn.Conv2d(12, 1, 1)

    @staticmethod
    def block(source: int, target: int) -> nn.Sequential:
        """Строит два свёрточных слоя с нелинейностью."""
        return nn.Sequential(nn.Conv2d(source, target, 3, padding=1), nn.ReLU(inplace=True),
                             nn.Conv2d(target, target, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Возвращает необработанные оценки OSD для каждого пикселя кадра."""
        a = self.encoder1(image)
        b = self.encoder2(F.avg_pool2d(a, 2))
        c = self.encoder3(F.avg_pool2d(b, 2))
        x = F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder2(torch.cat((x, b), dim=1))
        x = F.interpolate(x, size=a.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.decoder1(torch.cat((x, a), dim=1)))


def background(rng: np.random.Generator, validation: bool = False) -> np.ndarray:
    """Создаёт искусственный фон или берёт фрагмент подготовленного пейзажа."""
    backgrounds = VAL_BACKGROUNDS if validation else TRAIN_BACKGROUNDS
    if backgrounds and rng.random() < 0.75:
        source = backgrounds[int(rng.integers(0, len(backgrounds)))]
        height, width = source.shape[:2]
        crop_width = int(rng.integers(max(80, width // 2), width + 1))
        crop_height = min(height, max(48, round(crop_width * HEIGHT / WIDTH)))
        x = int(rng.integers(0, width - crop_width + 1))
        y = int(rng.integers(0, height - crop_height + 1))
        image = cv2.resize(source[y:y + crop_height, x:x + crop_width], (WIDTH, HEIGHT))
        if rng.random() < 0.5:
            image = cv2.flip(image, 1)
        return np.uint8(np.clip(image.astype(np.float32) * rng.uniform(0.7, 1.25)
                                + rng.uniform(-18, 18), 0, 255))
    sky = np.array((rng.integers(95, 220), rng.integers(100, 220), rng.integers(110, 235)), np.float32)
    ground = np.array((rng.integers(20, 125), rng.integers(45, 150), rng.integers(20, 125)), np.float32)
    small = rng.normal(0, rng.uniform(5, 30), size=(12, 20)).astype(np.float32)
    noise = cv2.resize(small, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    horizon = int(rng.integers(70, 165))
    image = np.empty((HEIGHT, WIDTH, 3), np.float32)
    image[:horizon] = sky
    image[horizon:] = ground
    for _ in range(int(rng.integers(2, 7))):
        points = np.array([(0, HEIGHT), (0, int(rng.integers(max(0, horizon - 20), HEIGHT))),
                           (int(rng.integers(0, WIDTH)), int(rng.integers(max(0, horizon - 30), HEIGHT))),
                           (WIDTH - 1, int(rng.integers(max(0, horizon - 20), HEIGHT))),
                           (WIDTH - 1, HEIGHT)], np.int32)
        shade = tuple(float(np.clip(c + rng.uniform(-35, 35), 0, 255)) for c in ground)
        cv2.fillPoly(image, [points], shade)
    image += noise[..., None]
    image += rng.normal(0, rng.uniform(1, 9), size=image.shape).astype(np.float32)
    image = np.uint8(np.clip(image, 0, 255))
    # Тёмные детали фона и мелкие силуэты не размечаются как OSD.
    for _ in range(int(rng.integers(4, 25))):
        x, y = int(rng.integers(0, WIDTH)), int(rng.integers(horizon, HEIGHT))
        cv2.line(image, (x, y), (x + int(rng.integers(-4, 5)), y - int(rng.integers(2, 20))),
                 tuple(int(v) for v in rng.integers(15, 110, 3)), 1, cv2.LINE_AA)
    return image


def example(seed: int, validation: bool = False, force_empty: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Рисует случайные OSD-элементы и одновременно их точную пиксельную маску."""
    rng = np.random.default_rng(seed)
    image = background(rng, validation)
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    if force_empty or rng.random() < 0.18:
        return image, mask
    fonts = VAL_FONTS if validation else TRAIN_FONTS
    colors = ((255, 255, 255), (40, 255, 40), (40, 210, 255), (35, 55, 245), (240, 240, 190))
    for _ in range(int(rng.integers(5, 19))):
        kind = int(rng.integers(0, 5))
        color = colors[int(rng.integers(0, len(colors)))]
        x, y = int(rng.integers(3, WIDTH - 48)), int(rng.integers(12, HEIGHT - 4))
        thickness = int(rng.integers(1, 3))
        if kind <= 2:
            text = str(rng.choice(WORDS)) if kind == 0 else f"{rng.integers(0, 99)}.{rng.integers(0, 10)}"
            font = fonts[int(rng.integers(0, len(fonts)))]
            scale = float(rng.uniform(0.32, 0.62))
            if rng.random() < 0.75:
                cv2.putText(image, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(mask, text, (x, y), font, scale, 255, thickness + 2, cv2.LINE_AA)
            cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            cv2.putText(mask, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
        elif kind == 3:
            end = (min(x + int(rng.integers(3, 15)), WIDTH - 1), y)
            cv2.line(image, (x, y), end, color, thickness, cv2.LINE_AA)
            cv2.line(mask, (x, y), end, 255, thickness, cv2.LINE_AA)
        else:
            end = (min(x + 9, WIDTH - 1), min(y + 6, HEIGHT - 1))
            cv2.rectangle(image, (x, y), end, color, thickness, cv2.LINE_AA)
            cv2.rectangle(mask, (x, y), end, 255, thickness, cv2.LINE_AA)
    if rng.random() < 0.45:
        image = cv2.GaussianBlur(image, (3, 3), float(rng.uniform(0.2, 0.8)))
    if rng.random() < 0.45:
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 95))])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                image = decoded
    return image, (mask > 16).astype(np.uint8)


def batch(seeds: list[int], validation: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Собирает пакет в памяти; исходные кадры и файлы на диске не создаются."""
    images, masks = zip(*(example(seed, validation) for seed in seeds))
    x = torch.from_numpy(np.stack(images).transpose(0, 3, 1, 2).copy()).float().div_(255.0)
    y = torch.from_numpy(np.stack(masks)[:, None, :, :].copy()).float()
    return x, y


def evaluate(model: Segmenter, count: int, batch_size: int) -> tuple[float, float, float]:
    """Считает precision, recall и F1 на отдельной синтетической выборке."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            seeds = [VAL_SEED + i for i in range(start, min(start + batch_size, count))]
            x, y = batch(seeds, True)
            prediction = torch.sigmoid(model(x)) >= 0.5
            truth = y >= 0.5
            tp += int((prediction & truth).sum())
            fp += int((prediction & ~truth).sum())
            fn += int((~prediction & truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    model.train()
    return precision, recall, f1


def save_checkpoint(path: Path, model: Segmenter, optimizer: torch.optim.Optimizer,
                    steps: int, best: float) -> None:
    """Сохраняет веса и параметры обучения внутри отдельного каталога опыта."""
    torch.save({"format": "videot16-osd-synthetic-v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "steps": steps, "best_f1": best,
                "width": WIDTH, "height": HEIGHT}, path)


def parse_args() -> argparse.Namespace:
    """Читает ограничение времени и параметры CPU без изменения конфигурации проекта."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=9.0, help="предел времени в часах (по умолчанию 9)")
    parser.add_argument("--seconds", type=float, default=None, help="короткий тест вместо --hours")
    parser.add_argument("--threads", type=int, default=6, help="потоки CPU (по умолчанию 6)")
    parser.add_argument("--batch", type=int, default=8, help="размер пакета (по умолчанию 8)")
    parser.add_argument("--steps", type=int, default=32, help="шагов между проверками (по умолчанию 32)")
    parser.add_argument("--val", type=int, default=64, help="число синтетических проверочных кадров")
    parser.add_argument("--backgrounds", type=Path, default=DEFAULT_BACKGROUNDS,
                        help="каталог подготовленных фонов без OSD")
    args = parser.parse_args()
    if args.hours <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error("время должно быть положительным")
    if not 1 <= args.threads <= 64 or not 1 <= args.batch <= 64 or args.steps < 1 or args.val < 1:
        parser.error("некорректное число потоков, пакет, шагов или проверочных кадров")
    return args


def main() -> int:
    """Запускает ограниченный по времени опыт и сохраняет обученные веса и отчёт."""
    args = parse_args()
    seconds = args.seconds if args.seconds is not None else args.hours * 3600
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(OUTPUT).free < 3 * 1024**3:
        raise RuntimeError("На диске осталось менее 3 ГиБ — обучение не запущено")
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=OUTPUT))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    # Два последних изображения отделены для проверки; они из того же видео.
    backgrounds = []
    if args.backgrounds.is_dir():
        for path in sorted(args.backgrounds.glob("*.jpg")):
            image = cv2.imread(str(path))
            if image is not None and image.shape[0] >= 48 and image.shape[1] >= 80:
                backgrounds.append(image)
    if len(backgrounds) >= 3:
        TRAIN_BACKGROUNDS.extend(backgrounds[:-2])
        VAL_BACKGROUNDS.extend(backgrounds[-2:])
    torch.manual_seed(20261102)
    model = Segmenter()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0]))
    log = (run_dir / "train.jsonl").open("a", encoding="utf-8", buffering=1)
    stop = False

    def request_stop(signum: int, frame: object) -> None:
        """При Ctrl+C просит сохранить текущий результат и завершиться."""
        nonlocal stop
        stop = True
        print("\nОстановка: сохраняю результат после текущего шага...", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    steps = 0
    best = -1.0
    reason = "время вышло"
    print(f"Эксперимент: {run_dir}", flush=True)
    print(f"CPU: {args.threads} потоков; предел: {seconds / 3600:.2f} ч; Ctrl+C — сохранить и завершить.", flush=True)
    print(f"Фотофоны: обучение={len(TRAIN_BACKGROUNDS)}, проверка={len(VAL_BACKGROUNDS)}; "
          "также используются искусственные фоны.", flush=True)
    print("Модель и проверка синтетические; исходная модель дронов не меняется.", flush=True)
    try:
        while not stop and time.monotonic() - started < seconds:
            losses = []
            for _ in range(args.steps):
                if stop or time.monotonic() - started >= seconds:
                    break
                if shutil.disk_usage(run_dir).free < 3 * 1024**3:
                    reason = "на диске осталось менее 3 ГиБ"
                    stop = True
                    break
                x, y = batch([steps * args.batch + i for i in range(args.batch)], False)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Получено некорректное значение ошибки обучения")
                loss.backward()
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
            if not losses:
                break
            precision, recall, f1 = evaluate(model, args.val, args.batch)
            if f1 > best:
                best = f1
                save_checkpoint(run_dir / "best.pt", model, optimizer, steps, best)
            save_checkpoint(run_dir / "last.pt", model, optimizer, steps, best)
            elapsed = time.monotonic() - started
            record = {"steps": steps, "seconds": round(elapsed, 1), "loss": round(float(np.mean(losses)), 5),
                      "synthetic_precision": round(precision, 4), "synthetic_recall": round(recall, 4),
                      "synthetic_f1": round(f1, 4), "best_f1": round(best, 4)}
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            remaining = max(0.0, seconds - elapsed) / 3600
            print(f"Шагов: {steps} | ошибка: {record['loss']:.4f} | синт. F1: {f1:.1%} | осталось: {remaining:.2f} ч", flush=True)
        if stop and reason == "время вышло":
            reason = "остановлено пользователем"
    finally:
        log.close()
    if steps == 0:
        raise RuntimeError("Ни одного шага обучения не выполнено")
    summary = {"reason": reason, "steps": steps, "seconds": round(time.monotonic() - started, 1),
               "best_synthetic_f1": round(best, 4), "threads": args.threads,
               "train_backgrounds": len(TRAIN_BACKGROUNDS), "val_backgrounds": len(VAL_BACKGROUNDS),
               "note": "Реальные OSD и дроны не размечены; качество в поле не проверено."}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ГОТОВО: {reason}; лучший результат: {run_dir / 'best.pt'}", flush=True)
    print("Качество на реальном видео необходимо проверить отдельно.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", flush=True)
        raise SystemExit(1)
