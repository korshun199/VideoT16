"""Проверяет утилиту сравнения V4L2-форматов без настоящего оцифровщика."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.capture_probe import CaptureMode, _restore_mode, probe_mode


class FakeCapture:
    """Минимальный V4L2-подобный вход для проверки настройки формата."""

    def __init__(self) -> None:
        """Создаёт фиктивное устройство с исходным режимом 640×480@30."""
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FOURCC: float(cv2.VideoWriter_fourcc(*"YUYV")),
        }
        self.released = False
        self.read_count = 0

    def isOpened(self) -> bool:
        """Сообщает, что фиктивный V4L2-вход доступен."""
        return True

    def get(self, property_id: int) -> float:
        """Возвращает последнее установленное значение свойства."""
        return self.properties.get(property_id, 0.0)

    def set(self, property_id: int, value: float) -> bool:
        """Запоминает запрошенное свойство так же, как принявший его драйвер."""
        self.properties[property_id] = float(value)
        return True

    def read(self):
        """Отдаёт небольшой меняющийся кадр без обращения к реальному USB."""
        self.read_count += 1
        frame = np.full((8, 8, 3), self.read_count, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        """Отмечает освобождение ресурса V4L2."""
        self.released = True


class CaptureProbeTests(unittest.TestCase):
    """Проверяет протокол измерения и обратную установку режима камеры."""

    def test_probe_reports_accepted_mode_and_never_writes_frames(self) -> None:
        """Утилита считает кадры в памяти и возвращает числовую телеметрию."""
        capture = FakeCapture()
        result = probe_mode(
            "fake",
            CaptureMode(width=1280, height=960, fps=60.0, fourcc="MJPG"),
            warmup_frames=1,
            measured_frames=3,
            factory=lambda _source: capture,
        )

        self.assertEqual(result["accepted"], {
            "width": 1280,
            "height": 960,
            "fps": 60.0,
            "fourcc": "MJPG",
        })
        self.assertEqual(result["successful_frames"], 3)
        self.assertEqual(result["failed_frames"], 0)
        self.assertEqual(result["telemetry"]["samples"], 3)
        self.assertTrue(capture.released)

    def test_restore_reapplies_captured_initial_mode(self) -> None:
        """После диагностики можно вернуть прежние размер, FPS и пиксельный формат."""
        capture = FakeCapture()
        restored = _restore_mode(
            "fake",
            CaptureMode(width=640, height=480, fps=30.0, fourcc="YUYV"),
            factory=lambda _source: capture,
        )

        self.assertEqual(restored, CaptureMode(640, 480, 30.0, "YUYV"))
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
