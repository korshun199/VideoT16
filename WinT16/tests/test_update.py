"""Проверяет обновление без удаления активации и пользовательских данных."""

from pathlib import Path
import tempfile
import unittest
import zipfile
import json
import base64

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from WinT16.update import SIGNATURE_DOMAIN, install, pack, sign_archive


class UpdateTests(unittest.TestCase):
    """Проверяет упаковку, сохранение данных и отказ при повреждении архива."""

    def _key(self, root: Path) -> tuple[Ed25519PrivateKey, Path, str]:
        """Создаёт временную тестовую пару ключей для проверки подписи."""
        private = Ed25519PrivateKey.generate()
        path = root / "signer.pem"
        path.write_bytes(private.private_bytes(serialization.Encoding.PEM,
                                               serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        public = private.public_key().public_bytes(serialization.Encoding.DER,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)
        return private, path, base64.b64encode(public).decode("ascii")

    def _archive(self, root: Path, engine: Path) -> tuple[Path, str]:
        """Создаёт и подписывает архив тестовым ключом."""
        _, key, public = self._key(root)
        unsigned, signed = root / "unsigned.zip", root / "update.zip"
        pack(engine, unsigned, "v2")
        sign_archive(unsigned, signed, key)
        return signed, public

    def test_update_preserves_state_and_previous_version(self) -> None:
        """Установка новой версии не затрагивает настройки, ключ и предыдущий запуск."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            engine = root / "build"
            engine.mkdir()
            (engine / "Engine.exe").write_bytes(b"new executable")
            (engine / "library.dll").write_bytes(b"library")
            archive, public = self._archive(root, engine)
            installed = root / "installed"
            (installed / "versions" / "v1").mkdir(parents=True)
            (installed / "versions" / "v1" / "Engine.exe").write_bytes(b"old executable")
            (installed / "current.txt").write_text("v1\n")
            (installed / "auth.dat").write_bytes(b"private activation")
            (installed / "settings.json").write_bytes(b"user settings")
            self.assertEqual(install(archive, installed, public), "v2")
            self.assertEqual((installed / "current.txt").read_text().strip(), "v2")
            self.assertEqual((installed / "auth.dat").read_bytes(), b"private activation")
            self.assertEqual((installed / "settings.json").read_bytes(), b"user settings")
            self.assertEqual((installed / "versions" / "v1" / "Engine.exe").read_bytes(), b"old executable")
            self.assertEqual((installed / "versions" / "v2" / "Engine.exe").read_bytes(), b"new executable")

    def test_rejects_unlisted_archive_entry_before_switching(self) -> None:
        """Недоверенный лишний файл не меняет действующую версию."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            engine = root / "build"
            engine.mkdir()
            (engine / "Engine.exe").write_bytes(b"executable")
            archive, public = self._archive(root, engine)
            with zipfile.ZipFile(archive, "a") as bundle:
                bundle.writestr("../auth.dat", b"overwrite")
            installed = root / "installed"
            installed.mkdir()
            (installed / "current.txt").write_text("v1\n")
            (installed / "auth.dat").write_bytes(b"private activation")
            with self.assertRaises(ValueError):
                install(archive, installed, public)
            self.assertEqual((installed / "current.txt").read_text(), "v1\n")
            self.assertEqual((installed / "auth.dat").read_bytes(), b"private activation")

    def test_rejects_wrong_digest_and_keeps_previous_version_active(self) -> None:
        """Повреждённый файл не переключает запуск и не изменяет существующую версию."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            private, _, public = self._key(root)
            manifest = {"format": "VideoT16-update-v1", "version": "v2",
                        "files": {"Engine.exe": "0" * 64}}
            manifest_bytes = json.dumps(manifest).encode()
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("manifest.json", manifest_bytes)
                bundle.writestr("signature.ed25519", private.sign(SIGNATURE_DOMAIN + manifest_bytes))
                bundle.writestr("Engine.exe", b"wrong content")
            installed = root / "installed"
            (installed / "versions" / "v1").mkdir(parents=True)
            (installed / "versions" / "v1" / "Engine.exe").write_bytes(b"working")
            (installed / "current.txt").write_text("v1\n")
            with self.assertRaisesRegex(ValueError, "Повреждён"):
                install(archive, installed, public)
            self.assertEqual((installed / "current.txt").read_text(), "v1\n")
            self.assertEqual((installed / "versions" / "v1" / "Engine.exe").read_bytes(), b"working")

    def test_unsigned_or_wrong_signer_rejected(self) -> None:
        """Не принимает неподписанный архив и архив с чужой подписью."""
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / "tmp") as temporary:
            root = Path(temporary)
            engine = root / "build"
            engine.mkdir()
            (engine / "Engine.exe").write_bytes(b"executable")
            signed, public = self._archive(root, engine)
            installed = root / "installed"
            installed.mkdir()
            with self.assertRaises(ValueError):
                install(root / "unsigned.zip", installed, public)
            unrelated = Ed25519PrivateKey.generate().public_key().public_bytes(
                serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
            with self.assertRaisesRegex(ValueError, "Подпись"):
                install(signed, installed, base64.b64encode(unrelated).decode("ascii"))


if __name__ == "__main__":
    unittest.main()
