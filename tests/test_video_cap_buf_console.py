"""Проверяет краткий итог в консоли и сохранение подробностей отдельно."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "video_cap_buf"))

from capture import _print_console_summary


class ConsoleSummaryTests(unittest.TestCase):
    """Проверяет, что терминал не печатает времена и память буферов."""

    def test_summary_contains_counts_only(self) -> None:
        """Оставляет короткие счётчики и скрывает подробную диагностику."""
        summary = {
            "frames_captured": 53,
            "frames_recognized": 12,
            "frames_with_presence_candidate": 5,
            "frames_with_box": 4,
            "frames_skipped": 41,
            "frames_displayed": 53,
            "stale_results": 0,
            "capture_ms": 2000.0,
            "post_capture_drain_ms": 2411.0,
            "capture_to_results_ready_ms": 4412.0,
            "source_buffer_frames": 0,
            "source_buffer_bytes": 0,
            "result_buffer_items": 0,
            "result_buffer_estimated_bytes": 0,
            "preview_frame": 53,
            "preview_frame_bytes": 6220800,
        }
        output = StringIO()

        with redirect_stdout(output):
            _print_console_summary(summary)

        text = output.getvalue()
        self.assertIn("Захвачено: 53", text)
        self.assertIn("проверено моделью: 12", text)
        self.assertNotIn("Время захвата", text)
        self.assertNotIn("готовности результатов", text)
        self.assertNotIn("Перед очисткой", text)
        self.assertNotIn("свежий кадр окна", text)


if __name__ == "__main__":
    unittest.main()
