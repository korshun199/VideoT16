"""Проверяет независимый показ кадров без физической камеры и ONNX-модели."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video_cap_buf"))

import cv2
import numpy as np

from capture import DelayedFrameBuffer, DisplayResult, Frame, LatestFrameBuffer, PipelineSession, validate_settings
from detector import Detection
from display import DisplayWorker, _select_overlay, run_display
from editor import FIELD_SPECS


class MemoryLog:
    """Собирает события в памяти, не создавая тестовые журналы на диске."""

    def __init__(self) -> None:
        """Инициализирует список диагностических записей."""
        self.events: list[tuple[str, dict[str, object]]] = []

    def write(self, event: str, **fields: object) -> None:
        """Добавляет событие так же, как потокобезопасный JSONL-журнал."""
        self.events.append((event, fields))


class SyntheticCapture:
    """Возвращает серию искусственных кадров с частотой около 30 кадров/с."""

    def __init__(self, *_arguments: object) -> None:
        """Запоминает номер кадра; реальное USB-устройство не открывается."""
        self.number = 0
        self.opened = True

    def isOpened(self) -> bool:
        """Сообщает, что синтетический источник готов."""
        return self.opened

    def set(self, _property: int, _value: float) -> bool:
        """Принимает запрошенный режим видео для интеграционного теста."""
        return True

    def get(self, property_id: int) -> float:
        """Возвращает фактические параметры синтетического источника."""
        properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 160.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 120.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
        }
        return properties.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray]:
        """Создаёт следующий кадр и выдерживает небольшой интервал между ними."""
        time.sleep(0.01)
        self.number += 1
        image = np.full((120, 160, 3), self.number % 255, dtype=np.uint8)
        return True, image

    def release(self) -> None:
        """Закрывает синтетический источник."""
        self.opened = False


class SlowSyntheticDetector:
    """Имитирует медленное распознавание движущегося объекта."""

    def __init__(self) -> None:
        """Готовит список ошибок положения рамки относительно тестового объекта."""
        self.position_errors: list[int] = []

    def detect_with_presence(self, image: np.ndarray):
        """Задерживает инференс, чтобы показать разницу частот видео и модели."""
        position = int(image[0, 0, 0])
        time.sleep(0.07)
        detection = Detection(0, "drone", 0.9, 20 + position, 20, 60 + position, 60)
        return (detection,), (detection,), 70.0

    def annotate(
        self,
        image: np.ndarray,
        detections: tuple[Detection, ...],
        _weak_candidates: tuple[Detection, ...] = (),
    ) -> np.ndarray:
        """Проверяет рамку относительно положения в изображении и возвращает копию."""
        expected_left = 20 + int(image[0, 0, 0])
        self.position_errors.extend(abs(item.left - expected_left) for item in detections)
        return image.copy()


class LatestFrameBufferTests(unittest.TestCase):
    """Проверяет замену ожидающего кадра в однокадровом mailbox."""

    def test_mailbox_returns_newest_frame_and_skipped_count(self) -> None:
        """Медленный потребитель получает свежий кадр без копии очереди."""
        mailbox = LatestFrameBuffer()
        stop = threading.Event()
        first = Frame(1, np.zeros((2, 2, 3), np.uint8), "t1", 1, 2)
        newest = Frame(4, np.ones((2, 2, 3), np.uint8), "t4", 3, 4)
        mailbox.publish(first)
        mailbox.publish(newest)

        frame, skipped = mailbox.wait_after(0, stop)

        self.assertIs(frame, newest)
        self.assertEqual(skipped, 3)
        self.assertIs(mailbox.latest(), newest)

    def test_delayed_buffer_is_bounded_and_catches_up(self) -> None:
        """Ограниченный буфер не накапливает запоздавшие изображения."""
        image = np.zeros((2, 2, 3), np.uint8)
        buffer = DelayedFrameBuffer(image.nbytes * 2, 200_000_000)
        now = time.monotonic_ns()
        for number in (1, 2, 3):
            buffer.publish(Frame(number, image, "t", now - 1_000_000_000, now - 1_000_000_000))
        self.assertEqual(buffer.stats(), (2, image.nbytes * 2))

        frame, skipped = buffer.wait_after(0, threading.Event())

        self.assertEqual(frame.number, 3)
        self.assertEqual(skipped, 2)
        buffer.clear()
        self.assertEqual(buffer.stats(), (0, 0))

    def test_delayed_buffer_reports_insufficient_memory(self) -> None:
        """Слишком маленький буфер сообщает ошибку, а не зависает без кадров."""
        image = np.zeros((2, 2, 3), np.uint8)
        buffer = DelayedFrameBuffer(image.nbytes, 10_000_000_000)
        now = time.monotonic_ns()
        buffer.publish(Frame(1, image, "t", now, now))
        with self.assertRaisesRegex(RuntimeError, "Буфер показа мал"):
            buffer.publish(Frame(2, image, "t", now, now))
        buffer.clear()

    def test_delayed_buffer_observes_delay_and_stop(self) -> None:
        """Кадр выходит после заданной задержки, а остановка не ждёт новый кадр."""
        image = np.zeros((2, 2, 3), np.uint8)
        buffer = DelayedFrameBuffer(1024, 40_000_000)
        now = time.monotonic_ns()
        buffer.publish(Frame(1, image, "t", now, now))
        frame, _ = buffer.wait_after(0, threading.Event())
        self.assertEqual(frame.number, 1)
        self.assertGreaterEqual(time.monotonic_ns() - now, 40_000_000)
        stopped = threading.Event()
        stopped.set()
        self.assertEqual(buffer.wait_after(1, stopped), (None, 0))
        buffer.clear()

    def test_overlay_matches_frame_and_interpolates_between_observations(self) -> None:
        """Рамка на промежуточном кадре опирается на два соседних наблюдения."""
        first_box = Detection(0, "drone", 0.9, 10, 10, 30, 30)
        next_box = Detection(0, "drone", 0.8, 20, 10, 40, 30)
        first = DisplayResult(1, 1_000_000_000, 1_100_000_000, 100.0, (first_box,), (first_box,))
        last = DisplayResult(5, 1_400_000_000, 1_500_000_000, 100.0, (next_box,), (next_box,))
        image = np.zeros((2, 2, 3), np.uint8)
        middle = Frame(3, image, "t", 0, 1_200_000_000)

        detections, weak, mode, anchors = _select_overlay(middle, first, last, 0.2, 250_000_000, 0.1)

        self.assertEqual(mode, "interpolated")
        self.assertEqual(anchors, (1, 5))
        self.assertEqual(weak, ())
        self.assertEqual((detections[0].left, detections[0].right), (15, 35))
        self.assertEqual(_select_overlay(middle, first, last, 0.2, 250_000_000, 0.5)[2], "none")
        exact = Frame(1, image, "t", 0, 1_000_000_000)
        exact_boxes, _, exact_mode, exact_anchors = _select_overlay(exact, first, last, 0.2, 250_000_000, 0.1)
        self.assertEqual((exact_mode, exact_anchors, exact_boxes), ("exact", (1,), (first_box,)))

    def test_overlay_does_not_carry_detection_through_a_miss(self) -> None:
        """Пустое наблюдение или чрезмерный промежуток не создают ложную рамку."""
        box = Detection(0, "drone", 0.9, 10, 10, 30, 30)
        first = DisplayResult(1, 1_000_000_000, 1_100_000_000, 100.0, (box,), (box,))
        miss = DisplayResult(5, 1_400_000_000, 1_500_000_000, 100.0, (), ())
        frame = Frame(3, np.zeros((2, 2, 3), np.uint8), "t", 0, 1_200_000_000)

        self.assertEqual(_select_overlay(frame, first, miss, 0.2, 250_000_000, 0.1)[2], "none")
        self.assertEqual(_select_overlay(frame, first, None, 0.2, 250_000_000, 0.1)[2], "none")
        far = DisplayResult(5, 2_000_000_000, 2_100_000_000, 100.0, (box,), (box,))
        self.assertEqual(_select_overlay(frame, first, far, 0.2, 250_000_000, 0.1)[2], "none")

    def test_settings_reject_an_unbounded_box_hold(self) -> None:
        """Время удержания рамки остаётся ограниченным конфигурацией."""
        settings = {
            "source": "/dev/video-test",
            "width": 160,
            "height": 120,
            "fps": 30,
            "pixel_format": "YUYV",
            "buffer_max_mb": 16,
            "model_path": "model.onnx",
            "inference_size": 320,
            "inference_backend": "onnxruntime",
            "recognition_workers": 1,
            "inference_threads_per_worker": 1,
            "recognition_frame_policy": "latest",
            "confidence_percent": 20,
            "presence_confidence_percent": 10,
            "nms_iou_threshold": 0.45,
            "class_names": ["drone"],
            "result_buffer_max_items": 100,
            "box_color_bgr": [0, 255, 0],
            "text_color_bgr": [0, 255, 0],
            "weak_candidate_color_bgr": [255, 0, 255],
            "box_thickness": 1,
            "font_scale": 0.5,
            "display_box_hold_ms": 250,
            "display_scale_mode": "fit",
        }
        validate_settings(settings)
        settings["display_box_hold_ms"] = 5000
        with self.assertRaisesRegex(ValueError, "0 до 2000 мс"):
            validate_settings(settings)

    def test_every_runtime_setting_has_a_menu_field(self) -> None:
        """Все переменные нового профиля доступны через его окно настроек."""
        settings_path = Path(__file__).resolve().parents[1] / "video_cap_buf" / "settings.json"
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        menu_keys = {spec.key for spec in FIELD_SPECS}

        self.assertEqual(set(settings), menu_keys)

    def test_bgr_to_qimage_preserves_color_channels(self) -> None:
        """Преобразование для Qt не меняет цвета и не требует cvtColor."""
        image = np.array([[[10, 80, 210]]], dtype=np.uint8)

        qimage = DisplayWorker._to_qimage(image)

        color = qimage.pixelColor(0, 0)
        self.assertEqual((color.red(), color.green(), color.blue()), (210, 80, 10))


class SmoothDisplayPipelineTests(unittest.TestCase):
    """Проверяет, что показ обновляется чаще медленного распознавания."""

    def test_video_frames_continue_while_detector_is_busy(self) -> None:
        """Синтетическое окно показывает новые кадры между результатами модели."""
        settings = {
            "source": "/dev/video-test",
            "width": 160,
            "height": 120,
            "fps": 30,
            "pixel_format": "YUYV",
            "buffer_max_mb": 16,
            "inference_size": 320,
            "recognition_workers": 1,
            "inference_threads_per_worker": 1,
            "result_buffer_max_items": 100,
            "recognition_frame_policy": "latest",
            "confidence_percent": 20,
            "presence_confidence_percent": 10,
            "box_thickness": 1,
            "font_scale": 0.5,
            "display_box_hold_ms": 500,
            "display_delay_ms": 200,
            "display_scale_mode": "fit",
        }
        capture_log = MemoryLog()
        display_log = MemoryLog()
        detector = SlowSyntheticDetector()
        session = PipelineSession(
            settings,
            capture_log,
            MemoryLog(),
            detector,
            capture_factory=SyntheticCapture,
            duration_seconds=0.25,
        )
        try:
            result = run_display(session, display_log)
            shown = [fields for name, fields in display_log.events if name == "frame_displayed"]

            self.assertEqual(result, 0)
            self.assertGreater(session.frame_count, 8)
            self.assertGreater(session.displayed_count, session.recognized_count)
            self.assertTrue(any(item.get("recognition_frame") is not None for item in shown))
            exact = [item for item in shown if item.get("overlay_mode") == "exact"]
            interpolated = [item for item in shown if item.get("overlay_mode") == "interpolated"]
            self.assertTrue(exact, "Результаты должны попадать на свои кадры")
            self.assertTrue(interpolated, "Между находками должны показываться промежуточные кадры")
            self.assertTrue(all(item["overlay_anchors"] == (item["frame"],) for item in exact))
            self.assertTrue(all(item["overlay_anchors"][0] < item["frame"] < item["overlay_anchors"][1]
                                for item in interpolated))
            self.assertTrue(detector.position_errors)
            self.assertLessEqual(max(detector.position_errors), 2)
            self.assertTrue(all("capture_to_present_ms" in item for item in shown))
            self.assertEqual(session.frame_buffer.stats(), (0, 0))
        finally:
            session.close()
        self.assertEqual(session.preview_buffer.stats(), (0, 0))
        self.assertIsNone(session.preview_buffer.latest())


if __name__ == "__main__":
    unittest.main()
