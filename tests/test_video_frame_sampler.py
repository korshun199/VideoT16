"""Проверки локального ручного отбора испытательных кадров из видео."""

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.video_trials import FrameSampleStore, create_session_dir, show_video


class VideoFrameSampleTests(unittest.TestCase):
    """Проверяет сохранение кадров для последующей проверки распознавания."""

    def test_creates_separate_session_folder(self):
        """Каждый запуск получает отдельную папку, исходный ролик не копируется."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "trials"
            source = Path(temporary) / "race.mp4"
            stamp = datetime(2026, 10, 21, 12, 30, 5)

            first = create_session_dir(source, root, now=stamp)
            second = create_session_dir(source, root, now=stamp)

            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())
            self.assertNotEqual(first, second)
            self.assertFalse(source.exists())

    def test_saves_original_frame_and_manifest_row(self):
        """Кадр сохраняется без уменьшения, CSV хранит его время и номер."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = np.full((48, 64, 3), 90, dtype=np.uint8)
            store = FrameSampleStore(directory, "race.mp4")
            try:
                saved = store.save(frame, frame_number=42, timestamp_seconds=1.68)
            finally:
                store.close()

            self.assertIsNotNone(saved)
            loaded = cv2.imread(str(saved))
            self.assertEqual(loaded.shape[:2], (48, 64))
            with (directory / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_video"], "race.mp4")
            self.assertEqual(rows[0]["frame_number"], "42")
            self.assertEqual(float(rows[0]["timestamp_seconds"]), 1.68)

    def test_does_not_save_same_video_frame_twice(self):
        """Повторное нажатие пробела на паузе не дублирует кадр."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            store = FrameSampleStore(directory, "race.mp4")
            try:
                first = store.save(frame, frame_number=10, timestamp_seconds=0.4)
                second = store.save(frame, frame_number=10, timestamp_seconds=0.4)
            finally:
                store.close()

            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(len(list(directory.glob("sample_*.png"))), 1)

    def test_reports_failed_image_write(self):
        """Ошибка записи кадра явно сообщается и не попадает в журнал."""
        with tempfile.TemporaryDirectory() as temporary:
            store = FrameSampleStore(Path(temporary), "race.mp4")
            try:
                with patch("scripts.video_trials.cv2.imwrite", return_value=False):
                    with self.assertRaisesRegex(OSError, "Не удалось сохранить кадр"):
                        store.save(np.zeros((16, 16, 3), dtype=np.uint8), 1, 0.0)
            finally:
                store.close()

    def test_space_key_saves_the_frame_shown_in_the_video_window(self):
        """Пробел в просмотрщике сохраняет выбранный кадр, а не весь ролик."""
        class FakeCapture:
            """Даёт циклу просмотра один синтетический кадр без чтения файла."""

            def __init__(self):
                self.read_count = 0

            def isOpened(self):
                """Сообщает, что тестовый источник видео готов."""
                return True

            def get(self, property_id):
                """Возвращает частоту и временную позицию тестового кадра."""
                if property_id == cv2.CAP_PROP_FPS:
                    return 25.0
                if property_id == cv2.CAP_PROP_POS_MSEC:
                    return 40.0
                return 0.0

            def read(self):
                """Возвращает единственный кадр, затем сообщает конец ролика."""
                self.read_count += 1
                if self.read_count == 1:
                    return True, np.full((32, 48, 3), 120, dtype=np.uint8)
                return False, None

            def release(self):
                """Поддерживает тот же интерфейс закрытия, что и OpenCV."""
                pass

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "test_samples"
            fake_capture = FakeCapture()
            with (
                patch("scripts.video_trials.cv2.VideoCapture", return_value=fake_capture),
                patch("scripts.video_trials.cv2.namedWindow"),
                patch("scripts.video_trials.cv2.resizeWindow"),
                patch("scripts.video_trials.cv2.imshow"),
                patch("scripts.video_trials.cv2.waitKey", side_effect=[ord(" "), ord("q")]),
                patch("scripts.video_trials.cv2.destroyAllWindows"),
            ):
                self.assertEqual(show_video(Path("race.mp4"), output_root), 0)

            sessions = list(output_root.iterdir())
            self.assertEqual(len(sessions), 1)
            self.assertEqual(len(list(sessions[0].glob("sample_*.png"))), 1)
            with (sessions[0] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["frame_number"], "1")
            self.assertEqual(rows[0]["timestamp_seconds"], "0.040")


if __name__ == "__main__":
    unittest.main()
