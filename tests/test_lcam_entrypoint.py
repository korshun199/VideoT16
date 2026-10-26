"""Проверяет два штатных способа запуска VideoT16 из корневой команды lcam."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LcamEntrypointTests(unittest.TestCase):
    """Изолированно проверяет единственный запуск без камеры и Qt."""

    def setUp(self) -> None:
        """Создаёт временную копию запуска и безопасный фиктивный Python."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        fake_python = self.root / ".venv" / "bin" / "python3"
        fake_python.parent.mkdir(parents=True)
        fake_python.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n", encoding="utf-8")
        fake_python.chmod(0o755)
        self.launcher = self.root / "lcam"
        shutil.copy2(PROJECT_ROOT / "lcam", self.launcher)

    def tearDown(self) -> None:
        """Удаляет только временную копию тестового проекта."""
        self.temp_dir.cleanup()

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Запускает копию lcam через Bash и возвращает текст результата."""
        return subprocess.run(
            ["bash", str(self.launcher), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_no_arguments_runs_saved_settings_without_menu(self) -> None:
        """Корневая команда запускает основной конвейер без меню и аргументов."""
        result = self._run()

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            result.stdout.splitlines(),
            [str(self.root / "scripts/live_pipeline_trial.py")],
        )

    def test_settings_menu_flag_is_rejected(self) -> None:
        """Удалённый редактор не запускается через устаревший флаг --m."""
        result = self._run("--m")

        self.assertEqual(result.returncode, 2)
        self.assertIn("./lcam", result.stderr)
        self.assertNotIn("config/editor.py", result.stderr)

    def test_other_arguments_are_rejected(self) -> None:
        """Принимается только запуск без аргументов."""
        result = self._run("--v2")

        self.assertEqual(result.returncode, 2)
        self.assertIn("Использование: ./lcam", result.stderr)
        self.assertNotIn("--v2", result.stderr)


if __name__ == "__main__":
    unittest.main()
