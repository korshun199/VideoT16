"""Проверяет состав офлайн-установщика и защиту существующих данных."""

from pathlib import Path
import tempfile
import unittest
import json

from WinT16.install import install_payload, validate_payload


class InstallerTests(unittest.TestCase):
    """Проверяет установку в новый каталог и отказ на небезопасных данных."""

    def _payload(self, root: Path) -> Path:
        """Создаёт минимальный имитированный состав выпуска."""
        payload = root / "payload"
        (payload / "versions" / "v1").mkdir(parents=True)
        (payload / "models").mkdir()
        for name in ("Run.exe", "Update.exe", "logo.png"):
            (payload / name).write_bytes(b"test")
        (payload / "settings.json").write_text(json.dumps({"model_path": "models/model.v16m"}))
        (payload / "current.txt").write_text("v1\n", encoding="ascii")
        (payload / "versions" / "v1" / "Engine.exe").write_bytes(b"program")
        (payload / "models" / "model.v16m").write_bytes(b"encrypted")
        return payload

    def test_install_preserves_complete_package(self) -> None:
        """Устанавливает комплект в новый каталог со всеми нужными файлами."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            destination = root / "new install"
            install_payload(payload, destination)
            self.assertEqual((destination / "versions" / "v1" / "Engine.exe").read_bytes(), b"program")
            self.assertEqual((destination / "models" / "model.v16m").read_bytes(), b"encrypted")

    def test_existing_folder_is_never_overwritten(self) -> None:
        """Не меняет папку, где могут находиться активация и настройки."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            payload = self._payload(root)
            destination = root / "existing"
            destination.mkdir()
            (destination / "auth.dat").write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                install_payload(payload, destination)
            self.assertEqual((destination / "auth.dat").read_bytes(), b"keep")

    def test_device_keys_and_raw_models_rejected(self) -> None:
        """Запрещает упаковать персональный ключ или открытую модель."""
        for name in ("device.key", "auth.dat", "secret.onnx", "activation_signing_private.pem"):
            with self.subTest(name=name), tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
                payload = self._payload(Path(temporary))
                (payload / "versions" / "v1" / name).write_bytes(b"private")
                with self.assertRaisesRegex(ValueError, "Запрещённый"):
                    validate_payload(payload)


if __name__ == "__main__":
    unittest.main()
