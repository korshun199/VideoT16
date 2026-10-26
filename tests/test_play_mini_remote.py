"""Проверяет безопасный разбор имени видео для SSH-плеера."""

import base64
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from scripts.play_mini_remote import decode_video_name


class PlayMiniRemoteTests(unittest.TestCase):
    """Проверяет корректные имена и отсекает попытки выйти из каталога видео."""

    def test_name_with_spaces_and_quotes_is_decoded_as_plain_filename(self) -> None:
        """Base64-имя с пробелами не превращается в команду удалённой оболочки."""
        encoded_name = base64.b64encode("Полёт 01 'test'.mp4".encode()).decode()
        with patch.dict(os.environ, {"VIDEO_NAME_B64": encoded_name}):
            self.assertEqual(decode_video_name(), "Полёт 01 'test'.mp4")

    def test_directory_traversal_is_rejected(self) -> None:
        """Путь с переходом к родительскому каталогу завершается ошибкой."""
        encoded_name = base64.b64encode("../private.mp4".encode()).decode()
        error_output = io.StringIO()
        with patch.dict(os.environ, {"VIDEO_NAME_B64": encoded_name}):
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit) as raised:
                    decode_video_name()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("без пути", error_output.getvalue())

    def test_unsupported_extension_is_rejected(self) -> None:
        """Скрипт не передаёт на минибук произвольные документы как видео."""
        encoded_name = base64.b64encode("notes.txt".encode()).decode()
        error_output = io.StringIO()
        with patch.dict(os.environ, {"VIDEO_NAME_B64": encoded_name}):
            with contextlib.redirect_stderr(error_output):
                with self.assertRaises(SystemExit):
                    decode_video_name()
        self.assertIn("поддерживаемого видеофайла", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
