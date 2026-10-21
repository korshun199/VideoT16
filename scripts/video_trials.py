#!/usr/bin/env python3
"""Вручную собирает небольшую локальную подборку кадров для испытаний модели."""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "test_samples" / "video_trials"


def create_session_dir(
    video_path: Path,
    output_root: Path,
    now: datetime | None = None,
) -> Path:
    """Создаёт новую папку испытательной сессии, не затрагивая прошлые сессии."""
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._-") or "video"
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    session_name = f"{safe_stem}_{timestamp}"
    session_dir = output_root / session_name
    suffix = 1
    while session_dir.exists():
        session_dir = output_root / f"{session_name}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir()
    return session_dir


class FrameSampleStore:
    """Сохраняет выбранные оригинальные кадры и журнал испытательной сессии."""

    def __init__(self, output_dir: Path, source_video: str) -> None:
        """Открывает новый CSV-журнал, не перезаписывая существующий."""
        self.output_dir = output_dir
        self.source_video = Path(source_video).name
        self.manifest_path = output_dir / "manifest.csv"
        self._manifest_file = self.manifest_path.open(
            "x", encoding="utf-8", newline=""
        )
        self._manifest = csv.DictWriter(
            self._manifest_file,
            fieldnames=(
                "sample_number",
                "file",
                "source_video",
                "frame_number",
                "timestamp_seconds",
                "width",
                "height",
            ),
        )
        self._manifest.writeheader()
        self._manifest_file.flush()
        self._saved_frame_numbers: set[int] = set()
        self.saved_count = 0

    def save(
        self,
        frame,
        frame_number: int,
        timestamp_seconds: float,
    ) -> Path | None:
        """Сохраняет исходный размер кадра; повторный выбор того же кадра пропускает."""
        if frame_number in self._saved_frame_numbers:
            return None
        if frame is None or len(frame.shape) < 2:
            raise ValueError("Получен пустой или некорректный видеокадр")

        next_number = self.saved_count + 1
        time_tag = f"{max(0.0, timestamp_seconds):010.3f}".replace(".", "_")
        filename = f"sample_{next_number:04d}_frame_{frame_number:08d}_t{time_tag}.png"
        image_path = self.output_dir / filename
        try:
            written = cv2.imwrite(str(image_path), frame)
        except cv2.error as error:
            raise OSError(f"Не удалось сохранить кадр {image_path}: {error}") from error
        if not written:
            raise OSError(f"Не удалось сохранить кадр: {image_path}")

        height, width = frame.shape[:2]
        self._manifest.writerow(
            {
                "sample_number": next_number,
                "file": filename,
                "source_video": self.source_video,
                "frame_number": frame_number,
                "timestamp_seconds": f"{max(0.0, timestamp_seconds):.3f}",
                "width": width,
                "height": height,
            }
        )
        self._manifest_file.flush()
        self._saved_frame_numbers.add(frame_number)
        self.saved_count = next_number
        return image_path

    def close(self) -> None:
        """Закрывает журнал, чтобы все отметки гарантированно записались на диск."""
        if not self._manifest_file.closed:
            self._manifest_file.close()


def fit_preview(frame, max_width: int = 1280, max_height: int = 720):
    """Уменьшает только изображение в окне, сохраняя оригинал для снимка."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return frame
    preview_size = (round(width * scale), round(height * scale))
    return cv2.resize(frame, preview_size, interpolation=cv2.INTER_AREA)


def frame_timestamp(capture, frame_number: int, fps: float) -> float:
    """Возвращает позицию текущего кадра из контейнера или безопасную оценку по FPS."""
    milliseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
    if milliseconds >= 0 and (milliseconds > 0 or frame_number == 1):
        return milliseconds / 1000.0
    return max(0, frame_number - 1) / fps


def show_video(video_path: Path, output_root: Path) -> int:
    """Показывает видео и сохраняет кадры только по нажатию пробела."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    reported_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = reported_fps if 1.0 <= reported_fps <= 120.0 else 25.0
    delay_ms = max(1, round(1000.0 / fps))
    frame_number = 0
    current_frame = None
    current_timestamp = 0.0
    playing = True
    step_forward = False
    ended = False
    store: FrameSampleStore | None = None
    session_dir: Path | None = None
    window_name = "VideoT16 — отбор кадров для испытаний"

    print(f"Видео: {video_path}", flush=True)
    print(f"Частота просмотра: {fps:.1f} кадр/с; весь ролик не сохраняется.", flush=True)
    print("Пробел — сохранить кадр | P — пауза/пуск | N — следующий кадр на паузе | Q/Esc — выход", flush=True)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 800)

        while True:
            if playing or step_forward or current_frame is None:
                ok, next_frame = capture.read()
                step_forward = False
                if not ok:
                    if current_frame is None:
                        raise RuntimeError(f"В видео нет читаемых кадров: {video_path}")
                    playing = False
                    ended = True
                else:
                    current_frame = next_frame
                    frame_number += 1
                    current_timestamp = frame_timestamp(capture, frame_number, fps)

            display_frame = fit_preview(current_frame).copy()
            mode = "КОНЕЦ" if ended else ("ВОСПРОИЗВЕДЕНИЕ" if playing else "ПАУЗА")
            saved_count = store.saved_count if store is not None else 0
            cv2.putText(
                display_frame,
                f"{mode} | FRAME {frame_number} | {current_timestamp:.2f}s | SAVED {saved_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "SPACE: SAVE   P: PAUSE   N: NEXT   Q: QUIT",
                (12, max(48, display_frame.shape[0] - 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(delay_ms if playing else 30) & 0xFF

            if key == ord(" "):
                if store is None:
                    session_dir = create_session_dir(video_path, output_root)
                    store = FrameSampleStore(session_dir, video_path.name)
                saved_path = store.save(current_frame, frame_number, current_timestamp)
                if saved_path is None:
                    print(f"[УЖЕ ЕСТЬ] кадр {frame_number} пропущен как повторный.", flush=True)
                else:
                    print(
                        f"[СОХРАНЕНО] кадр {frame_number}, {current_timestamp:.3f} с: {saved_path}",
                        flush=True,
                    )
            elif key in (ord("p"), ord("P")) and not ended:
                playing = not playing
            elif key in (ord("n"), ord("N")) and not playing and not ended:
                step_forward = True
            elif key in (ord("q"), ord("Q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        if store is not None:
            store.close()

    saved_count = store.saved_count if store is not None else 0
    print(f"Готово. Для испытаний сохранено кадров: {saved_count}.", flush=True)
    if session_dir is not None:
        print(f"Папка с кадрами и manifest.csv: {session_dir}", flush=True)
    else:
        print("Кадры не выбирались; папка с данными не создавалась.", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Создаёт командные параметры ручного отбора испытательных кадров."""
    parser = argparse.ArgumentParser(
        description="Показывает ролик и сохраняет выбранные пробелом кадры для испытаний VideoT16."
    )
    parser.add_argument("video", type=Path, help="Путь к видеофайлу; ролик не копируется и не изменяется")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Каталог испытательных сессий (по умолчанию: {DEFAULT_OUTPUT_ROOT})",
    )
    return parser


def main() -> int:
    """Проверяет путь и запускает ручной просмотр без пакетного извлечения кадров."""
    args = build_parser().parse_args()
    if not args.video.is_file():
        raise SystemExit(f"Видео не найдено: {args.video}")
    try:
        return show_video(args.video, args.output_dir)
    except (RuntimeError, OSError, cv2.error) as error:
        raise SystemExit(f"Ошибка испытательного просмотра: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
