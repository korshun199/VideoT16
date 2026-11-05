"""Проверки платформенного слоя без доступа к реальной камере."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from WinT16.runtime.camera import (
    CameraOpenError,
    CameraInfo,
    _camera_source,
    camera_backends,
    find_hdmi_capture,
    open_camera,
    open_hdmi_capture,
    probe_cameras,
)
from WinT16.runtime.metrics import read_system_metrics
from WinT16.runtime.paths import application_root


class FakeCapture:
    """Имитирует минимальный интерфейс OpenCV VideoCapture для тестов."""

    def __init__(self, source: int | str, backend: int, *, opened: bool = True) -> None:
        """Запоминает источник, backend и заданное состояние открытия."""
        self.source = source
        self.backend = backend
        self.opened = opened
        self.released = False

    def isOpened(self) -> bool:
        """Возвращает результат имитации открытия устройства."""
        return self.opened

    def read(self) -> tuple[bool, object | None]:
        """Возвращает фиктивный кадр только при успешном открытии."""
        return self.opened, object() if self.opened else None

    def get(self, property_id: int) -> float:
        """Возвращает устойчивые размеры и частоту тестового кадра."""
        del property_id
        return 640.0

    def release(self) -> None:
        """Помечает тестовое устройство как освобождённое."""
        self.released = True


class PlatformCameraTests(unittest.TestCase):
    """Проверяет разбор источника, выбор backend и очистку пробного захвата."""

    def test_windows_prefers_directshow_then_media_foundation(self) -> None:
        """Проверяет приоритет Windows backend без привязки к системе тестирования."""
        self.assertEqual(
            [name for name, _ in camera_backends("Windows")],
            ["DirectShow", "Media Foundation", "Автовыбор OpenCV"],
        )

    def test_camera_index_is_not_treated_as_filesystem_path(self) -> None:
        """Проверяет, что индекс камеры сохраняется числом и не разрешается как путь."""
        self.assertEqual(_camera_source("2"), 2)
        self.assertEqual(_camera_source(1), 1)
        self.assertEqual(_camera_source("camera://usb"), "camera://usb")

    def test_open_camera_falls_back_after_failed_backend(self) -> None:
        """Проверяет переход к следующему backend и освобождение неудачной попытки."""
        devices: list[FakeCapture] = []

        def factory(source: int | str, backend: int) -> FakeCapture:
            """Создаёт сначала неоткрывшееся, затем открытое тестовое устройство."""
            device = FakeCapture(source, backend, opened=len(devices) > 0)
            devices.append(device)
            return device

        opened, backend_name = open_camera(
            0, system_name="Windows", capture_factory=factory
        )
        self.assertIs(opened, devices[1])
        self.assertEqual(backend_name, "Media Foundation")
        self.assertTrue(devices[0].released)

    def test_open_camera_reports_all_backend_failures(self) -> None:
        """Проверяет понятную ошибку, если ни один backend не открыл камеру."""
        created: list[FakeCapture] = []

        def factory(source: int | str, backend: int) -> FakeCapture:
            """Создаёт закрытые устройства для каждого backend."""
            device = FakeCapture(source, backend, opened=False)
            created.append(device)
            return device

        with self.assertRaisesRegex(CameraOpenError, "DirectShow"):
            open_camera(3, system_name="Windows", capture_factory=factory)
        self.assertTrue(all(device.released for device in created))

    def test_probe_releases_camera_after_reading(self) -> None:
        """Проверяет, что пробный поток не удерживает камеру перед запуском приложения."""
        devices: list[FakeCapture] = []

        def factory(source: int | str, backend: int) -> FakeCapture:
            """Открывает только индекс 0, остальные имитирует отсутствующими."""
            device = FakeCapture(source, backend, opened=(source == 0))
            devices.append(device)
            return device

        found = probe_cameras(2, system_name="Windows", capture_factory=factory)
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].index, found[0].width, found[0].height), (0, 640, 640))
        self.assertTrue(all(device.released for device in devices))

    def test_hdmi_search_ignores_internal_camera_by_name(self) -> None:
        """Находит оцифровщик по имени, не полагаясь на порядковый индекс камеры."""
        cameras = [
            CameraInfo(0, "DirectShow", 0, 0, 0, "Integrated Camera", 700, 0x1234, 0x0001),
            CameraInfo(2, "DirectShow", 0, 0, 0, "USB Video Capture", 700, 0x1BCF, 0x2C99),
        ]
        selected = find_hdmi_capture(cameras, ["capture", "usb video"])
        self.assertIsNotNone(selected)
        self.assertEqual(selected.name, "USB Video Capture")

    def test_hdmi_search_refuses_ambiguous_matches(self) -> None:
        """Не выбирает случайно одну из двух камер-кандидатов."""
        cameras = [
            CameraInfo(1, "DirectShow", 0, 0, 0, "HDMI Capture A", 700),
            CameraInfo(2, "DirectShow", 0, 0, 0, "HDMI Capture B", 700),
        ]
        self.assertIsNone(find_hdmi_capture(cameras, ["capture"]))

    def test_open_hdmi_capture_uses_the_enumerated_backend(self) -> None:
        """Открывает найденный оцифровщик именно тем backend, которым его нашли."""
        camera = CameraInfo(
            7, "DirectShow", 0, 0, 0, "USB Video Capture", 700, 0x1BCF, 0x2C99
        )
        created: list[FakeCapture] = []

        def factory(source: int | str, backend: int) -> FakeCapture:
            """Запоминает открытый индекс и backend в поддельном устройстве."""
            device = FakeCapture(source, backend)
            created.append(device)
            return device

        opened, selected = open_hdmi_capture(
            name_keywords=["capture"],
            capture_factory=factory,
            enumerator=lambda: [camera],
        )
        self.assertIs(opened, created[0])
        self.assertEqual((opened.source, opened.backend), (7, 700))
        self.assertEqual(selected.name, "USB Video Capture")

    def test_hdmi_tries_second_backend_when_first_cannot_open(self) -> None:
        """Проверяет переход к Media Foundation после неудачи DirectShow."""
        cameras = [
            CameraInfo(2, "DirectShow", 0, 0, 0, "USB Video Capture", 700),
            CameraInfo(5, "Media Foundation", 0, 0, 0, "USB Video Capture", 1400),
        ]
        devices: list[FakeCapture] = []

        def factory(source: int | str, backend: int) -> FakeCapture:
            """Первое устройство закрыто, второе доступно."""
            device = FakeCapture(source, backend, opened=backend == 1400)
            devices.append(device)
            return device

        opened, selected = open_hdmi_capture(name_keywords=["capture"],
                                              capture_factory=factory,
                                              enumerator=lambda: cameras)
        self.assertTrue(devices[0].released)
        self.assertIs(opened, devices[1])
        self.assertEqual(selected.backend, "Media Foundation")


class PlatformMetricsTests(unittest.TestCase):
    """Проверяет, что недоступный датчик температуры не мешает работе приложения."""

    def test_temperature_sensor_failure_is_non_fatal(self) -> None:
        """Возвращает загрузку CPU и пустую температуру при недоступном датчике."""
        with patch("WinT16.runtime.metrics.psutil.cpu_percent", return_value=37.5), \
                patch("WinT16.runtime.metrics.psutil.sensors_temperatures",
                      side_effect=NotImplementedError):
            metrics = read_system_metrics()
        self.assertEqual(metrics.cpu_percent, 37.5)
        self.assertIsNone(metrics.temperature_c)


class ApplicationRootTests(unittest.TestCase):
    """Проверяет, что все версии запуска используют одну постоянную папку установки."""

    def test_explicit_install_root_is_used_even_in_source_process(self) -> None:
        """Подхватывает путь, который Run.exe передаёт движку через окружение."""
        with TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"VIDEOT16_APP_ROOT": temporary}):
                self.assertEqual(application_root(), Path(temporary).resolve())

    def test_frozen_engine_finds_shared_root_from_version_directory(self) -> None:
        """При прямом запуске Engine.exe находит settings.json и current.txt выше себя."""
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            version_dir = root / "versions" / "test-version"
            version_dir.mkdir(parents=True)
            (root / "current.txt").write_text("test-version", encoding="ascii")
            (root / "settings.json").write_text("{}", encoding="utf-8")
            executable = version_dir / "Engine.exe"
            with patch.dict("os.environ", {"VIDEOT16_APP_ROOT": ""}), \
                    patch("WinT16.runtime.paths.sys.frozen", True, create=True), \
                    patch("WinT16.runtime.paths.sys.executable", str(executable)):
                self.assertEqual(application_root(), root.resolve())


if __name__ == "__main__":
    unittest.main()
