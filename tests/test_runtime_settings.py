"""Проверки настроек, изменяемых оператором через веб-панель."""

import json
import tempfile
import unittest
from pathlib import Path

from config.osd_config import ARM_BANNER_STYLE, AXIS_STYLE, FLIGHT_STATUS_STYLE, HORIZON_STYLE, OBJECT_STYLE, OSD_TEXT
from config.runtime_settings import apply_osd_settings, load_settings, save_settings


class RuntimeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.styles = {"object": OBJECT_STYLE.copy(), "text": OSD_TEXT.copy(), "horizon": HORIZON_STYLE.copy(), "axis": AXIS_STYLE.copy(), "flight_status": FLIGHT_STATUS_STYLE.copy(), "arm_banner": ARM_BANNER_STYLE.copy()}

    def test_save_load_merges_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            saved = save_settings({"detection": {"confidence_percent": 73}}, path)
            loaded = load_settings(path)
        self.assertEqual(saved["detection"]["confidence_percent"], 73)
        self.assertEqual(loaded["detection"]["inference_size"], 320)

    def test_desktop_settings_are_saved_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            saved = save_settings({"desktop": {"camera_device": "/dev/video9", "inference_size": 320, "capture_width": 1280}}, path)
            self.assertEqual(saved["desktop"]["camera_device"], "/dev/video9")
            self.assertEqual(saved["desktop"]["capture_width"], 1280)
            with self.assertRaises(ValueError):
                save_settings({"desktop": {"confidence_percent": 0}}, path)

    def test_saved_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_settings({}, path)
            with path.open(encoding="utf-8") as stream:
                self.assertIsInstance(json.load(stream), dict)


if __name__ == "__main__":
    unittest.main()
