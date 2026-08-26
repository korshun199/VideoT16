"""Проверки настроек, изменяемых оператором через веб-панель."""

import json
import tempfile
import unittest
from pathlib import Path

from config.osd_config import (
    ARM_BANNER_STYLE,
    AXIS_STYLE,
    FLIGHT_STATUS_STYLE,
    HORIZON_STYLE,
    OBJECT_STYLE,
    OSD_TEXT,
)
from config.runtime_settings import apply_osd_settings, load_settings, save_settings


class RuntimeSettingsTests(unittest.TestCase):
    def setUp(self):
        self.styles = {
            "object": OBJECT_STYLE.copy(),
            "text": OSD_TEXT.copy(),
            "horizon": HORIZON_STYLE.copy(),
            "axis": AXIS_STYLE.copy(),
            "flight_status": FLIGHT_STATUS_STYLE.copy(),
            "arm_banner": ARM_BANNER_STYLE.copy(),
        }

    def test_save_load_merges_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            saved = save_settings({"detection": {"confidence_percent": 73}}, path)
            loaded = load_settings(path)

        self.assertEqual(saved["detection"]["confidence_percent"], 73)
        self.assertEqual(loaded["detection"]["inference_size"], 256)
        self.assertIn("arm_banner", loaded)

    def test_invalid_confidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_settings({"detection": {"confidence_percent": 101}}, Path(directory) / "x.json")

    def test_colors_and_offsets_are_applied(self):
        settings = load_settings()
        settings["osd_text"]["color_rgb"] = [10, 20, 30]
        settings["osd_text"]["shadow_offset_x"] = 4
        settings["osd_text"]["shadow_offset_y"] = 5
        settings["arm_banner"]["armed_color_rgb"] = [1, 2, 3]
        apply_osd_settings(settings, self.styles)

        self.assertEqual(self.styles["text"]["color"], (30, 20, 10))
        self.assertEqual(self.styles["text"]["shadow_offset"], (4, 5))
        self.assertEqual(self.styles["arm_banner"]["armed_color"], (3, 2, 1))

    def test_saved_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_settings({}, path)
            with path.open(encoding="utf-8") as stream:
                self.assertIsInstance(json.load(stream), dict)


if __name__ == "__main__":
    unittest.main()
