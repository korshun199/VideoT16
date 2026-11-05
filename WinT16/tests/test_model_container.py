"""Проверяет шифрованный формат модели и его поведение при повреждении."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from WinT16.runtime.model_container import (
    KEY_SIZE,
    MAGIC,
    ModelContainerError,
    decrypt_model_bytes,
    decrypt_model_file,
    encrypt_model_bytes,
    encrypt_model_file,
)


class ModelContainerTests(unittest.TestCase):
    """Проверяет шифрование, расшифрование и аутентификацию данных модели."""

    def test_round_trip_returns_original_model_bytes(self) -> None:
        """Проверяет, что корректный ключ возвращает исходный payload без изменений."""
        model = b"fake-onnx-model-payload"
        key = bytes(range(KEY_SIZE))
        encrypted = encrypt_model_bytes(model, key, nonce=b"n" * 12)
        self.assertTrue(encrypted.startswith(MAGIC))
        self.assertNotIn(model, encrypted)
        self.assertEqual(decrypt_model_bytes(encrypted, key), model)

    def test_wrong_key_is_rejected(self) -> None:
        """Проверяет отказ расшифрования и проверки целостности при другом ключе."""
        encrypted = encrypt_model_bytes(b"onnx", b"a" * KEY_SIZE, nonce=b"b" * 12)
        with self.assertRaises(ModelContainerError):
            decrypt_model_bytes(encrypted, b"c" * KEY_SIZE)

    def test_tampered_ciphertext_is_rejected(self) -> None:
        """Проверяет, что даже изменение одного байта контейнера обнаруживается."""
        encrypted = bytearray(
            encrypt_model_bytes(b"onnx", b"a" * KEY_SIZE, nonce=b"b" * 12)
        )
        encrypted[-1] ^= 1
        with self.assertRaises(ModelContainerError):
            decrypt_model_bytes(bytes(encrypted), b"a" * KEY_SIZE)

    def test_invalid_key_length_and_plain_onnx_are_rejected(self) -> None:
        """Проверяет минимальные структурные ограничения контейнера."""
        with self.assertRaises(ValueError):
            encrypt_model_bytes(b"onnx", b"short")
        with self.assertRaises(ModelContainerError):
            decrypt_model_bytes(b"plain-onnx", b"a" * KEY_SIZE)

    def test_file_flow_preserves_source_and_refuses_to_overwrite(self) -> None:
        """Проверяет создание .v16m, отсутствие изменений исходника и запрет перезаписи."""
        temp_root = Path(__file__).resolve().parents[2] / "tmp" / "WinT16"
        temp_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=temp_root) as temp_dir:
            folder = Path(temp_dir)
            source = folder / "model.onnx"
            encrypted = folder / "model.v16m"
            original = b"synthetic-onnx-payload"
            key = b"k" * KEY_SIZE
            source.write_bytes(original)
            encrypt_model_file(source, encrypted, key)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(decrypt_model_file(encrypted, key), original)
            with self.assertRaises(FileExistsError):
                encrypt_model_file(source, encrypted, key)


if __name__ == "__main__":
    unittest.main()
