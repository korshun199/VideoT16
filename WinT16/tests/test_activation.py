"""Проверяет запросы устройства, подпись ответа и аппаратную привязку активации."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from WinT16.runtime.activation import (
    ActivationError,
    DeviceIdentity,
    _load_or_create_device_key,
    accept_activation_code,
    accept_short_activation_code,
    activation_key,
    decode_request_code,
    encode_request_code,
    ensure_request,
    identity_from_values,
    issue_auth,
    issue_activation_code,
    issue_short_activation_code,
    verify_auth,
)
from WinT16.runtime.model_container import decrypt_model_bytes, encrypt_model_bytes


class TestProtector:
    """Имитирует DPAPI для тестов, сохраняя формат проверяемого защищённого blob."""

    def protect(self, data: bytes) -> bytes:
        """Возвращает тестовую оболочку, отличимую от открытого PEM."""
        return b"test-protected:" + data

    def unprotect(self, data: bytes) -> bytes:
        """Снимает оболочку и отвергает неподготовленные тестовые данные."""
        prefix = b"test-protected:"
        if not data.startswith(prefix):
            raise ValueError("test protection blob is invalid")
        return data[len(prefix):]


def signing_files(folder: Path) -> tuple[Path, Path, str]:
    """Создаёт тестовые файлы подписи и DER-ключ, который закрепит клиент."""
    private = Ed25519PrivateKey.generate()
    private_path = folder / "signer-private.pem"
    public_path = folder / "signer-public.der"
    private_path.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_der = private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    public_path.write_bytes(public_der)
    return private_path, public_path, base64.b64encode(public_der).decode("ascii")


class HardwareIdentityTests(unittest.TestCase):
    """Проверяет нормализацию аппаратных полей и fail-closed при нехватке данных."""

    def test_identity_ignores_unavailable_or_placeholder_values(self) -> None:
        """Одинаковый набор реальных hardware IDs даёт одинаковый хеш."""
        first = identity_from_values({
            "system_uuid": "ABCD-EF01-2345", "baseboard_serial": "Board-98765",
            "processor_id": "To Be Filled By O.E.M.",
        })
        second = identity_from_values({
            "system_uuid": "abcd ef01 2345", "baseboard_serial": "board 98765",
        })
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.available_fields, ("baseboard_serial", "system_uuid"))

    def test_identity_requires_two_real_hardware_fields(self) -> None:
        """Отпечаток не создаётся из имени пользователя или единственного слабого ID."""
        with self.assertRaises(ActivationError):
            identity_from_values({"system_uuid": "00000000", "baseboard_serial": "OEM"})


class ActivationFlowTests(unittest.TestCase):
    """Проверяет полный тестовый цикл init.dat → персональный auth.dat → AES-ключ."""

    def setUp(self) -> None:
        """Создаёт временный каталог, identity, тестовую DPAPI и подписывающий ключ."""
        temp_root = Path(__file__).resolve().parents[2] / "tmp" / "WinT16"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / "client"
        self.app.mkdir()
        self.owner = self.root / "owner"
        self.owner.mkdir()
        self.identity = DeviceIdentity("a" * 64, ("baseboard_serial", "system_uuid"))
        self.protector = TestProtector()
        self.signer_private, self.signer_public, self.public_b64 = signing_files(self.owner)
        self.model_key = bytes(range(32))
        self.model_key_path = self.owner / "model_aes.key"
        self.model_key_path.write_bytes(self.model_key)

    def test_request_is_created_once_and_contains_only_public_device_key(self) -> None:
        """Создаёт воспроизводимый request и убеждается, что его секрет хранится DPAPI-blob."""
        request = ensure_request(self.app, self.identity, self.protector)
        request_path = self.app / "init.dat"
        device_key_path = self.app / "device.key"
        self.assertEqual(request["hardware_id"], self.identity.fingerprint)
        self.assertEqual(ensure_request(self.app, self.identity, self.protector), request)
        self.assertNotIn(b"BEGIN PRIVATE KEY", device_key_path.read_bytes())
        self.assertNotIn(b"model_aes", request_path.read_bytes())

    def test_owner_authenticates_request_and_device_unwraps_shared_model_key(self) -> None:
        """Корректный подписанный ответ открывает ровно общий ключ, заданный владельцем."""
        ensure_request(self.app, self.identity, self.protector)
        auth_path = self.app / "auth.dat"
        issue_auth(self.app / "init.dat", auth_path, self.signer_private, self.model_key_path)
        device_key = _load_or_create_device_key(self.app, self.protector, self.identity)
        self.assertEqual(
            verify_auth(auth_path, self.identity, device_key, self.public_b64), self.model_key
        )

    def test_request_code_round_trip_and_owner_issues_copyable_response(self) -> None:
        """Проверяет путь обмена двумя текстовыми строками без передачи файлов.»"""
        request = ensure_request(self.app, self.identity, self.protector)
        request_code = encode_request_code(request)
        self.assertEqual(decode_request_code(request_code), request)
        response_code = issue_activation_code(
            request_code, self.signer_private, self.model_key_path
        )
        accepted_key = accept_activation_code(
            self.app, self.identity, response_code, self.protector, self.public_b64
        )
        self.assertEqual(accepted_key, self.model_key)
        self.assertTrue((self.app / "auth.dat").is_file())

    def test_tampered_request_or_response_code_is_rejected(self) -> None:
        """Не принимает обрезанный запрос и изменённый ответный код.»"""
        request = ensure_request(self.app, self.identity, self.protector)
        request_code = encode_request_code(request)
        with self.assertRaises(ActivationError):
            decode_request_code(request_code[:-2])
        response_code = issue_activation_code(
            request_code, self.signer_private, self.model_key_path
        )
        changed = response_code[:-1] + ("A" if response_code[-1] != "A" else "B")
        with self.assertRaises(ActivationError):
            accept_activation_code(
                self.app, self.identity, changed, self.protector, self.public_b64
            )
        self.assertFalse((self.app / "auth.dat").exists())

    def test_modified_signature_is_rejected(self) -> None:
        """Изменение подписанных параметров не позволяет запустить распознавание."""
        ensure_request(self.app, self.identity, self.protector)
        auth_path = self.app / "auth.dat"
        issue_auth(self.app / "init.dat", auth_path, self.signer_private, self.model_key_path)
        envelope = json.loads(auth_path.read_text())
        envelope["payload"]["hardware_id"] = "b" * 64
        auth_path.write_text(json.dumps(envelope))
        device_key = _load_or_create_device_key(self.app, self.protector, self.identity)
        with self.assertRaisesRegex(ActivationError, "Подпись"):
            verify_auth(auth_path, self.identity, device_key, self.public_b64)

    def test_valid_auth_for_another_hardware_is_rejected(self) -> None:
        """Перенос подписанного auth.dat на другой hardware fingerprint завершается отказом."""
        ensure_request(self.app, self.identity, self.protector)
        auth_path = self.app / "auth.dat"
        issue_auth(self.app / "init.dat", auth_path, self.signer_private, self.model_key_path)
        device_key = _load_or_create_device_key(self.app, self.protector, self.identity)
        other_pc = DeviceIdentity("b" * 64, self.identity.available_fields)
        with self.assertRaisesRegex(ActivationError, "другого оборудования"):
            verify_auth(auth_path, other_pc, device_key, self.public_b64)

    def test_unusable_device_key_is_preserved_and_new_request_can_be_made(self) -> None:
        """Перенос папки на новый ПК сохраняет старый blob и готовит новый запрос."""
        path = self.app / "device.key"
        path.write_text('{"format":"unknown"}')
        request = ensure_request(self.app, self.identity, self.protector)
        self.assertEqual(request["hardware_id"], self.identity.fingerprint)
        self.assertTrue(list(self.app.glob("device.key.unusable.*")))
        self.assertTrue(path.is_file())

    def test_authorized_key_decrypts_shared_encrypted_model(self) -> None:
        """Один общий контейнер открывается ключом только из персонального auth.dat."""
        ensure_request(self.app, self.identity, self.protector)
        auth_path = self.app / "auth.dat"
        issue_auth(self.app / "init.dat", auth_path, self.signer_private, self.model_key_path)
        device_key = _load_or_create_device_key(self.app, self.protector, self.identity)
        authorized_key = verify_auth(auth_path, self.identity, device_key, self.public_b64)
        encrypted_model = encrypt_model_bytes(b"shared-model-onnx", self.model_key)
        self.assertEqual(decrypt_model_bytes(encrypted_model, authorized_key), b"shared-model-onnx")

    def test_short_codes_are_exactly_32_characters_and_unlock_shared_model(self) -> None:
        """Короткий запрос и ответ укладываются в 32 символа и открывают общий AES-128 файл."""
        from WinT16.runtime.activation import ensure_short_request

        request_code = ensure_short_request(self.app, self.identity)
        self.assertEqual(len(request_code), 32)
        self.assertTrue(request_code.isupper())
        self.assertEqual(ensure_short_request(self.app, self.identity), request_code)
        activation_code = issue_short_activation_code(request_code, self.model_key_path)
        self.assertEqual(len(activation_code), 32)
        self.assertTrue(activation_code.isupper())
        accepted_key = accept_short_activation_code(self.app, self.identity, activation_code)
        self.assertEqual(len(accepted_key), 16)
        self.assertTrue((self.app / "auth.dat").is_file())
        self.assertEqual(activation_key(self.app, self.identity), accepted_key)
        encrypted_model = encrypt_model_bytes(b"shared-model-onnx", accepted_key)
        self.assertEqual(decrypt_model_bytes(encrypted_model, accepted_key), b"shared-model-onnx")

    def test_short_activation_code_is_bound_to_hardware(self) -> None:
        """Ответ, выпущенный по запросу одного ПК, не подходит другому отпечатку."""
        from WinT16.runtime.activation import ensure_short_request

        request_code = ensure_short_request(self.app, self.identity)
        activation_code = issue_short_activation_code(request_code, self.model_key_path)
        other_pc = DeviceIdentity("b" * 64, self.identity.available_fields)
        with self.assertRaisesRegex(ActivationError, "другого компьютера"):
            accept_short_activation_code(self.app, other_pc, activation_code)

    def test_short_code_rejects_non_32_character_input(self) -> None:
        """Проверяет понятное ограничение длины ручного ввода."""
        with self.assertRaisesRegex(ActivationError, "ровно 32"):
            accept_short_activation_code(self.app, self.identity, "A" * 31)


if __name__ == "__main__":
    unittest.main()
