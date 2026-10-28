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

from capture import Frame, LatestFrameBuffer, PipelineSession, validate_settings
from detector import Detection
from display import DisplayWorker, run_display
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
    """Имитирует более медленное распознавание и возвращает одну рамку."""

    def __init__(self) -> None:
        """Готовит стабильную детекцию внутри тестового кадра."""
        self.detection = Detection(0, "drone", 0.9, 20, 20, 60, 60)

    def detect_with_presence(self, _image: np.ndarray):
        """Задерживает инференс, чтобы показать разницу частот видео и модели."""
        time.sleep(0.07)
        return (self.detection,), (self.detection,), 70.0

    def annotate(
        self,
        image: np.ndarray,
        _detections: tuple[Detection, ...],
        _weak_candidates: tuple[Detection, ...] = (),
    ) -> np.ndarray:
        """Возвращает отдельное изображение и не изменяет кадр камеры."""
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
            "display_scale_mode": "fit",
        }
        capture_log = MemoryLog()
        display_log = MemoryLog()
        session = PipelineSession(
            settings,
            capture_log,
            MemoryLog(),
            SlowSyntheticDetector(),
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
            self.assertTrue(all("capture_to_present_ms" in item for item in shown))
            self.assertEqual(session.frame_buffer.stats(), (0, 0))
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
