"""Офлайн-активация WinT16 с привязкой к оборудованию и защите DPAPI."""

from __future__ import annotations

import base64
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# Открытая часть ключа выпуска WinT16. Закрытая часть хранится только у владельца.
SIGNING_PUBLIC_KEY_B64 = "MCowBQYDK2VwAyEAV9kD735imeGGxbw1F/WzeF8plsdcZYtUuyMKuFcCI6s="
REQUEST_FORMAT = "VideoT16-init-v1"
REQUEST_CODE_PREFIX = "V16R1:"
AUTH_FORMAT = "VideoT16-auth-v1"
DEVICE_KEY_FILE = "device.key"
REQUEST_FILE = "init.dat"
AUTH_FILE = "auth.dat"
MODEL_KEY_SIZE = 32
SHORT_MODEL_KEY_SIZE = 16
RSA_KEY_SIZE = 3072
RSA_LABEL = b"VideoT16 model key v1"
SHORT_REQUEST_FORMAT = "VideoT16-request-v2"
SHORT_AUTH_FORMAT = "VideoT16-auth-v2"
SHORT_CODE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
PHONE_REQUEST_FORMAT = "VideoT16-request-v3"
PHONE_AUTH_FORMAT = "VideoT16-auth-v3"
PHONE_CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
PHONE_CODE_LENGTH = 12


class ActivationError(RuntimeError):
    """Сообщает о некорректном запросе, активации или локальном ключе."""


@dataclass(frozen=True)
class DeviceIdentity:
    """Хранит стабильный отпечаток и только названия доступных аппаратных полей."""

    fingerprint: str
    available_fields: tuple[str, ...]


def _canonical_json(value: object) -> bytes:
    """Кодирует данные подписи однозначным JSON-представлением UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _valid_hardware_value(value: object) -> str | None:
    """Нормализует идентификатор и отбрасывает типичные заглушки изготовителей."""
    if not isinstance(value, str):
        return None
    cleaned = "".join(character for character in value.strip().upper() if character.isalnum())
    if len(cleaned) < 5 or cleaned in {
        "NONE", "UNKNOWN", "DEFAULTSTRING", "TOBEFILLEDBYOEM", "SYSTEMSERIALNUMBER",
        "FFFFFFFF", "00000000", "123456789",
    }:
        return None
    return cleaned


def identity_from_values(values: dict[str, object]) -> DeviceIdentity:
    """Строит отпечаток из UUID системы, серийника платы и ID процессора."""
    accepted = {
        name: value for name in ("system_uuid", "baseboard_serial", "processor_id")
        if (value := _valid_hardware_value(values.get(name))) is not None
    }
    if len(accepted) < 2:
        raise ActivationError(
            "Windows не предоставила минимум два аппаратных идентификатора. "
            "Проверьте UUID компьютера, серийный номер платы и ID процессора."
        )
    digest = hashlib.sha256(_canonical_json(accepted)).hexdigest()
    return DeviceIdentity(digest, tuple(sorted(accepted)))


def read_windows_identity() -> DeviceIdentity:
    """Читает аппаратные UUID/серийные номера через штатный CIM-интерфейс Windows."""
    if platform.system().lower() != "windows":
        raise ActivationError("Аппаратную активацию можно запросить только в Windows.")
    script = (
        "$ErrorActionPreference='Stop';"
        "$p=Get-CimInstance Win32_ComputerSystemProduct;"
        "$b=Get-CimInstance Win32_BaseBoard;"
        "$c=Get-CimInstance Win32_Processor | Select-Object -First 1;"
        "[pscustomobject]@{system_uuid=$p.UUID;baseboard_serial=$b.SerialNumber;"
        "processor_id=$c.ProcessorId}|ConvertTo-Json -Compress"
    )
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True, capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise ActivationError(f"Не удалось прочитать аппаратный ID Windows: {error}") from error
    if not isinstance(raw, dict):
        raise ActivationError("Windows вернула данные оборудования в неизвестном формате.")
    return identity_from_values(raw)


class DpapiProtector:
    """Шифрует локальный закрытый ключ средствами DPAPI текущей учётной записи."""

    @staticmethod
    def _transform(data: bytes, *, protect: bool) -> bytes:
        """Вызывает CryptProtectData/CryptUnprotectData и освобождает память Windows."""
        if os.name != "nt":
            raise ActivationError("DPAPI доступен только в Windows.")
        class DataBlob(ctypes.Structure):
            """Представляет DATA_BLOB из WinCrypt.h."""

            _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]

        source_buffer = ctypes.create_string_buffer(data)
        source = DataBlob(len(data), ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_byte)))
        output = DataBlob()
        crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
        description = wintypes.LPWSTR()
        if protect:
            function = crypt32.CryptProtectData
            function.argtypes = [
                ctypes.POINTER(DataBlob), wintypes.LPCWSTR, ctypes.POINTER(DataBlob),
                ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(DataBlob),
            ]
            succeeded = function(ctypes.byref(source), "VideoT16 device key", None,
                                 None, None, 0x1, ctypes.byref(output))
        else:
            function = crypt32.CryptUnprotectData
            function.argtypes = [
                ctypes.POINTER(DataBlob), ctypes.POINTER(wintypes.LPWSTR),
                ctypes.POINTER(DataBlob), ctypes.c_void_p, ctypes.c_void_p,
                wintypes.DWORD, ctypes.POINTER(DataBlob),
            ]
            succeeded = function(ctypes.byref(source), ctypes.byref(description), None,
                                 None, None, 0x1, ctypes.byref(output))
        if not succeeded:
            raise ActivationError(f"Windows DPAPI завершился ошибкой {ctypes.get_last_error()}.")
        try:
            return ctypes.string_at(output.pbData, output.cbData)
        finally:
            kernel32.LocalFree(output.pbData)
            if description:
                kernel32.LocalFree(description)

    def protect(self, data: bytes) -> bytes:
        """Защищает байты ключа DPAPI без привязки к локальному паролю приложения."""
        return self._transform(data, protect=True)

    def unprotect(self, data: bytes) -> bytes:
        """Возвращает DPAPI-данные только в исходной учётной записи Windows."""
        return self._transform(data, protect=False)


def _read_json(path: Path, max_bytes: int = 65536) -> dict[str, object]:
    """Читает небольшой JSON-файл и проверяет, что его верхний уровень — объект."""
    try:
        raw = path.read_bytes()
        if len(raw) > max_bytes:
            raise ActivationError(f"Файл слишком большой: {path.name}")
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivationError(f"Не удалось прочитать {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ActivationError(f"Неверный формат файла {path.name}.")
    return value


def _write_new_json(path: Path, value: dict[str, object]) -> None:
    """Создаёт JSON исключительно новым файлом и не перезаписывает имеющиеся данные."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=True, indent=2).encode("ascii") + b"\n"
    try:
        with path.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        raise ActivationError(f"Файл уже существует, перезапись запрещена: {path.name}")


def _load_or_create_device_key(
    app_root: Path, protector: DpapiProtector, identity: DeviceIdentity,
) -> rsa.RSAPrivateKey:
    """Загружает локальный DPAPI-ключ или создаёт его один раз для этого каталога."""
    key_path = app_root / DEVICE_KEY_FILE
    if key_path.exists():
        try:
            record = _read_json(key_path)
            if record.get("format") != "VideoT16-device-key-v1":
                raise ActivationError("Локальный device.key имеет неподдерживаемый формат.")
            protected = base64.b64decode(str(record["protected_private_key"]), validate=True)
            private_bytes = protector.unprotect(protected)
            key = serialization.load_pem_private_key(private_bytes, password=None)
            if not isinstance(key, rsa.RSAPrivateKey):
                raise ActivationError("Локальный ключ устройства имеет неверный тип.")
            return key
        except (ActivationError, KeyError, ValueError, TypeError):
            # Копия каталога на другой ПК не раскрывает DPAPI-ключ; сохраняем
            # старый blob и выпускаем локальный ключ для нового запроса.
            backup = key_path.with_name(
                f"device.key.unusable.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )
            key_path.rename(backup)

    key = rsa.generate_private_key(public_exponent=65537, key_size=RSA_KEY_SIZE)
    private_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    protected = protector.protect(private_bytes)
    _write_new_json(key_path, {
        "format": "VideoT16-device-key-v1",
        "protected_private_key": base64.b64encode(protected).decode("ascii"),
    })
    return key


def ensure_request(
    app_root: Path,
    identity: DeviceIdentity,
    protector: DpapiProtector | None = None,
) -> dict[str, object]:
    """Создаёт запрос init.dat с аппаратным отпечатком и открытым ключом устройства."""
    active_protector = protector or DpapiProtector()
    device_key = _load_or_create_device_key(app_root, active_protector, identity)
    public_der = device_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    request = {
        "format": REQUEST_FORMAT,
        "hardware_id": identity.fingerprint,
        "hardware_fields": list(identity.available_fields),
        "device_public_key": base64.b64encode(public_der).decode("ascii"),
    }
    request_path = app_root / REQUEST_FILE
    if request_path.exists():
        try:
            existing = _read_json(request_path)
        except ActivationError:
            existing = None
        if existing == request:
            return request
        backup = request_path.with_name(
            f"init.previous.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.dat"
        )
        request_path.rename(backup)
    _write_new_json(request_path, request)
    return request


def encode_request_code(request: dict[str, object]) -> str:
    """Упаковывает запрос устройства в одну копируемую строку для мессенджера."""
    if request.get("format") != REQUEST_FORMAT:
        raise ActivationError("Нельзя показать запрос неизвестного формата.")
    encoded = base64.urlsafe_b64encode(_canonical_json(request)).decode("ascii").rstrip("=")
    return f"{REQUEST_CODE_PREFIX}{encoded}"


def decode_request_code(code: str) -> dict[str, object]:
    """Проверяет копируемую строку запроса и возвращает безопасный JSON-объект."""
    normalized = code.strip()
    if not normalized.startswith(REQUEST_CODE_PREFIX) or len(normalized) > 16384:
        raise ActivationError("Запрос не распознан; нужна вся строка, начинающаяся с V16R1:.")
    encoded = normalized[len(REQUEST_CODE_PREFIX):]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        value = json.loads(base64.b64decode(padded, altchars=b"-_", validate=True).decode("ascii"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivationError("Строка запроса повреждена или скопирована не целиком.") from error
    if not isinstance(value, dict) or value.get("format") != REQUEST_FORMAT:
        raise ActivationError("В запросе неподдерживаемый формат.")
    if not isinstance(value.get("hardware_id"), str) or len(value["hardware_id"]) != 64:
        raise ActivationError("В запросе отсутствует аппаратный отпечаток.")
    return value


def issue_auth(request_path: Path, output_path: Path, signing_private_path: Path,
               model_key_path: Path) -> None:
    """Создаёт JSON-файл ответа для инженерных тестов, не перезаписывая существующее."""
    envelope = _build_auth_envelope(request_path, signing_private_path, model_key_path)
    _write_new_json(output_path, envelope)


def _build_auth_envelope(
    request_source: str | Path, signing_private_path: Path, model_key_path: Path,
) -> dict[str, object]:
    """Подписывает ответ и оборачивает общий AES-ключ открытым RSA-ключом ПК."""
    if isinstance(request_source, Path):
        request = _read_json(request_source)
    elif request_source.startswith(REQUEST_CODE_PREFIX):
        request = decode_request_code(str(request_source))
    else:
        request = _read_json(Path(request_source))
    if request.get("format") != REQUEST_FORMAT:
        raise ActivationError("init.dat имеет неизвестную версию.")
    hardware_id = request.get("hardware_id")
    if not isinstance(hardware_id, str) or len(hardware_id) != 64:
        raise ActivationError("В init.dat отсутствует корректный аппаратный отпечаток.")
    try:
        public_der = base64.b64decode(str(request["device_public_key"]), validate=True)
        device_public = serialization.load_der_public_key(public_der)
        signing_key = serialization.load_pem_private_key(signing_private_path.read_bytes(), password=None)
        model_key = model_key_path.read_bytes()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ActivationError(f"Не удалось прочитать ключи или запрос: {error}") from error
    if (not isinstance(device_public, rsa.RSAPublicKey)
            or device_public.key_size != RSA_KEY_SIZE
            or not isinstance(signing_key, Ed25519PrivateKey)):
        raise ActivationError("Неподходящий тип ключа устройства или подписи выпуска.")
    if len(model_key) != MODEL_KEY_SIZE:
        raise ActivationError("Ключ модели должен содержать ровно 32 байта.")
    wrapped_key = device_public.encrypt(
        model_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(), label=RSA_LABEL),
    )
    payload = {
        "version": 1,
        "hardware_id": hardware_id,
        "device_key_sha256": hashlib.sha256(public_der).hexdigest(),
        "wrapped_model_key": base64.b64encode(wrapped_key).decode("ascii"),
    }
    return {
        "format": AUTH_FORMAT,
        "payload": payload,
        "signature": base64.b64encode(signing_key.sign(_canonical_json(payload))).decode("ascii"),
    }


def issue_activation_code(request_source: str | Path, signing_private_path: Path,
                          model_key_path: Path) -> str:
    """Возвращает короткий текстовый код ответа, пригодный для отправки в мессенджере."""
    envelope = _build_auth_envelope(request_source, signing_private_path, model_key_path)
    encoded = base64.urlsafe_b64encode(_canonical_json(envelope)).decode("ascii").rstrip("=")
    return f"V16A1:{encoded}"


def _verify_auth_envelope(envelope: dict[str, object], identity: DeviceIdentity,
                          device_private: rsa.RSAPrivateKey,
                          public_key_b64: str) -> bytes:
    """Проверяет подписанный конверт и раскрывает AES-ключ только локальным RSA-ключом."""
    if envelope.get("format") != AUTH_FORMAT:
        raise ActivationError("auth.dat имеет неизвестный формат.")
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ActivationError("В auth.dat отсутствуют подписанные данные.")
    try:
        verifier = serialization.load_der_public_key(base64.b64decode(public_key_b64, validate=True))
        if not isinstance(verifier, Ed25519PublicKey):
            raise ValueError("Закреплённый ключ выпуска не Ed25519")
        signature = base64.b64decode(str(envelope["signature"]), validate=True)
        verifier.verify(signature, _canonical_json(payload))
    except (KeyError, ValueError, TypeError, InvalidSignature) as error:
        raise ActivationError("Подпись auth.dat неверна или файл повреждён.") from error
    if payload.get("version") != 1 or payload.get("hardware_id") != identity.fingerprint:
        raise ActivationError("auth.dat выпущен для другого оборудования.")
    public_der = device_private.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    if payload.get("device_key_sha256") != hashlib.sha256(public_der).hexdigest():
        raise ActivationError("auth.dat привязан к другому ключу устройства.")
    try:
        wrapped = base64.b64decode(str(payload["wrapped_model_key"]), validate=True)
        model_key = device_private.decrypt(
            wrapped,
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(), label=RSA_LABEL),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise ActivationError("Ключ модели в auth.dat не подходит этому устройству.") from error
    if len(model_key) != MODEL_KEY_SIZE:
        raise ActivationError("В auth.dat получен ключ модели неверной длины.")
    return model_key


def verify_auth(auth_path: Path, identity: DeviceIdentity,
                device_private: rsa.RSAPrivateKey,
                public_key_b64: str = SIGNING_PUBLIC_KEY_B64) -> bytes:
    """Проверяет auth.dat с диска и возвращает AES-ключ модели только в память."""
    return _verify_auth_envelope(_read_json(auth_path), identity, device_private, public_key_b64)


def accept_activation_code(
    app_root: Path,
    identity: DeviceIdentity,
    code: str,
    protector: DpapiProtector | None = None,
    public_key_b64: str = SIGNING_PUBLIC_KEY_B64,
) -> bytes:
    """Проверяет вставленный код и сохраняет auth.dat только после полной криптопроверки."""
    normalized = code.strip()
    prefix = "V16A1:"
    if not normalized.startswith(prefix) or len(normalized) > 16384:
        raise ActivationError("Код активации не распознан; вставьте всю строку V16A1:... .")
    encoded = normalized[len(prefix):]
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
        envelope = json.loads(raw.decode("ascii"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ActivationError("Код активации повреждён или обрезан.") from error
    if not isinstance(envelope, dict):
        raise ActivationError("Код активации имеет неизвестный формат.")
    private_key = load_device_private_key(app_root, identity, protector)
    model_key = _verify_auth_envelope(envelope, identity, private_key, public_key_b64)

    auth_path = app_root / AUTH_FILE
    if auth_path.exists():
        existing = None
        try:
            existing = _read_json(auth_path)
        except ActivationError:
            pass
        if existing == envelope:
            return model_key
        backup = auth_path.with_name(
            f"auth.previous.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.dat"
        )
        temporary = app_root / f"auth.incoming.{os.getpid()}.tmp"
        _write_new_json(temporary, envelope)
        auth_path.rename(backup)
        temporary.replace(auth_path)
    else:
        _write_new_json(auth_path, envelope)
    return model_key


def load_device_private_key(
    app_root: Path,
    identity: DeviceIdentity,
    protector: DpapiProtector | None = None,
) -> rsa.RSAPrivateKey:
    """Открывает локальный ключ для этапа проверки серверного ответа."""
    return _load_or_create_device_key(app_root, protector or DpapiProtector(), identity)


def activation_key(
    app_root: Path,
    identity: DeviceIdentity,
    protector: DpapiProtector | None = None,
) -> bytes:
    """Проверяет auth.dat v2 или прежний v1 и отдаёт ключ модели в память."""
    auth_path = app_root / AUTH_FILE
    if auth_path.is_file():
        envelope = _read_json(auth_path)
        if envelope.get("format") == SHORT_AUTH_FORMAT:
            return _verify_short_activation_code(
                str(envelope.get("code", "")), identity
            )
        if envelope.get("format") == PHONE_AUTH_FORMAT:
            code = str(envelope.get("code", ""))
            return _verify_phone_activation_code(
                code, identity, _embedded_model_key()
            )
        if envelope.get("format") == AUTH_FORMAT:
            raise ActivationError(
                "Сохранена активация старого формата. Нужен новый короткий код для этого компьютера."
            )
    device_private = load_device_private_key(app_root, identity, protector)
    if not auth_path.is_file():
        raise ActivationError("Файл активации auth.dat не найден.")
    return verify_auth(auth_path, identity, device_private)


def _short_hardware_code(identity: DeviceIdentity) -> str:
    """Кодирует 20 байт отпечатка компьютера в 32 символа Base32."""
    raw = bytes.fromhex(identity.fingerprint)[:20]
    return base64.b32encode(raw).decode("ascii").rstrip("=")


def _phone_code_from_digest(digest: bytes) -> str:
    """Преобразует 60 случайных бит в 12 символов без путаемых букв."""
    if len(digest) < 8:
        raise ActivationError("Недостаточно данных для кода активации.")
    number = int.from_bytes(digest[:8], "big") >> 4
    characters: list[str] = []
    for _ in range(PHONE_CODE_LENGTH):
        number, remainder = divmod(number, len(PHONE_CODE_ALPHABET))
        characters.append(PHONE_CODE_ALPHABET[remainder])
    return "".join(reversed(characters))


def _normalize_phone_code(code: str) -> str:
    """Нормализует диктуемый код: убирает пробелы/дефисы и явные омонимы."""
    normalized = "".join(code.split()).replace("-", "").upper()
    # При диктовке буква O часто путается с нулём, а I/L — с единицей.
    normalized = normalized.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    if (len(normalized) != PHONE_CODE_LENGTH
            or any(character not in PHONE_CODE_ALPHABET for character in normalized)):
        raise ActivationError(
            "Код должен содержать ровно 12 заглавных букв или цифр. "
            "Для диктовки используются буквы без I, L, O."
        )
    return normalized


def _phone_hardware_code(identity: DeviceIdentity) -> str:
    """Стабильно кодирует 60 бит отпечатка оборудования в короткий запрос."""
    digest = hashlib.sha256(
        b"VideoT16 hardware request v3\0" + bytes.fromhex(identity.fingerprint)
    ).digest()
    return _phone_code_from_digest(digest)


def _embedded_model_key() -> bytes:
    """Загружает 128-битный ключ модели, добавляемый владельцем только при сборке."""
    try:
        from runtime._embedded_model_key import MODEL_AES_KEY
    except (ImportError, AttributeError) as error:
        raise ActivationError(
            "В этой сборке не встроен ключ модели. Нужна сборка владельца VideoT16."
        ) from error
    if not isinstance(MODEL_AES_KEY, bytes) or len(MODEL_AES_KEY) != SHORT_MODEL_KEY_SIZE:
        raise ActivationError("Встроенный ключ модели имеет неверный формат.")
    return MODEL_AES_KEY


def ensure_short_request(app_root: Path, identity: DeviceIdentity) -> str:
    """Создаёт или переиспользует 12-значный запрос этого компьютера."""
    code = _phone_hardware_code(identity)
    path = app_root / REQUEST_FILE
    record = {"format": PHONE_REQUEST_FORMAT, "code": code}
    if path.is_file():
        try:
            if _read_json(path) == record:
                return code
        except ActivationError:
            pass
        backup = path.with_name(
            f"init.previous.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.dat"
        )
        path.rename(backup)
    _write_new_json(path, record)
    return code


def _decode_short_code(code: str) -> bytes:
    """Проверяет 32-символьный код и возвращает ровно 20 декодированных байт."""
    normalized = "".join(code.split()).upper()
    if len(normalized) != 32 or any(character not in SHORT_CODE_ALPHABET for character in normalized):
        raise ActivationError("Код должен содержать ровно 32 заглавных символа A–Z или 2–7.")
    try:
        raw = base64.b32decode(normalized + "=" * (-len(normalized) % 8))
    except (ValueError, TypeError) as error:
        raise ActivationError("Код повреждён.") from error
    if len(raw) != 20:
        raise ActivationError("Код имеет неверную длину данных.")
    return raw


def _short_model_key(owner_key: bytes) -> bytes:
    """Детерминированно получает совместимый 128-битный ключ модели из ключа владельца."""
    if len(owner_key) != MODEL_KEY_SIZE:
        raise ActivationError("Ключ владельца модели должен содержать 32 байта.")
    return hashlib.sha256(b"VideoT16 short model key v2\0" + owner_key).digest()[:SHORT_MODEL_KEY_SIZE]


def issue_short_activation_code(request_code: str, model_key_path: Path) -> str:
    """Выпускает 12-символьный ответ; 32-символьные запросы остаются совместимы."""
    compact_request = "".join(request_code.split()).replace("-", "").upper()
    if len(compact_request) == PHONE_CODE_LENGTH:
        request = _normalize_phone_code(compact_request)
        try:
            owner_key = model_key_path.read_bytes()
        except OSError as error:
            raise ActivationError(f"Не удалось прочитать ключ модели владельца: {error}") from error
        model_key = _short_model_key(owner_key)
        response = hmac.new(
            model_key,
            b"VideoT16 phone activation v3\0" + request.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return _phone_code_from_digest(response)

    # Старый выпуск оставлен, чтобы можно было повторно активировать v2-клиенты.
    hardware = _decode_short_code(request_code)
    try:
        owner_key = model_key_path.read_bytes()
    except OSError as error:
        raise ActivationError(f"Не удалось прочитать ключ модели владельца: {error}") from error
    model_key = _short_model_key(owner_key)
    mask = hashlib.sha256(b"VideoT16 activation mask v2\0" + hardware).digest()[:SHORT_MODEL_KEY_SIZE]
    masked_key = bytes(left ^ right for left, right in zip(model_key, mask))
    check = hashlib.sha256(b"VideoT16 activation check v2\0" + hardware + model_key).digest()[:4]
    return base64.b32encode(masked_key + check).decode("ascii").rstrip("=")


def _verify_short_activation_code(code: str, identity: DeviceIdentity) -> bytes:
    """Проверяет короткий ответ для текущего ПК и извлекает AES-128 ключ модели."""
    payload = _decode_short_code(code)
    hardware = bytes.fromhex(identity.fingerprint)[:20]
    masked_key, provided_check = payload[:SHORT_MODEL_KEY_SIZE], payload[SHORT_MODEL_KEY_SIZE:]
    mask = hashlib.sha256(b"VideoT16 activation mask v2\0" + hardware).digest()[:SHORT_MODEL_KEY_SIZE]
    model_key = bytes(left ^ right for left, right in zip(masked_key, mask))
    expected = hashlib.sha256(b"VideoT16 activation check v2\0" + hardware + model_key).digest()[:4]
    if not hmac.compare_digest(provided_check, expected):
        raise ActivationError("Код активации выпущен для другого компьютера или введён с ошибкой.")
    return model_key


def _verify_phone_activation_code(
    code: str, identity: DeviceIdentity, model_key: bytes,
) -> bytes:
    """Сверяет 12-символьный ответ с компьютером и возвращает его ключ модели."""
    if len(model_key) != SHORT_MODEL_KEY_SIZE:
        raise ActivationError("Ключ модели для телефонной активации имеет неверную длину.")
    normalized = _normalize_phone_code(code)
    request = _phone_hardware_code(identity)
    expected = _phone_code_from_digest(hmac.new(
        model_key,
        b"VideoT16 phone activation v3\0" + request.encode("ascii"),
        hashlib.sha256,
    ).digest())
    if not hmac.compare_digest(normalized, expected):
        raise ActivationError(
            "Ответный код не подходит этому компьютеру или в нём допущена ошибка."
        )
    return model_key


def accept_short_activation_code(
    app_root: Path,
    identity: DeviceIdentity,
    code: str,
    model_key: bytes | None = None,
) -> bytes:
    """Принимает 12-символьный ответ либо старый 32-символьный и сохраняет auth.dat."""
    normalized = "".join(code.split()).upper()
    if len(normalized) != 32:
        normalized = _normalize_phone_code(normalized)
        active_key = model_key if model_key is not None else _embedded_model_key()
        key = _verify_phone_activation_code(normalized, identity, active_key)
        record = {"format": PHONE_AUTH_FORMAT, "code": _normalize_phone_code(normalized)}
        return _store_short_auth_record(app_root, record, key)

    # Старые v2-коды и сохранённые активации продолжают работать без изменений.
    key = _verify_short_activation_code(code, identity)
    normalized = "".join(code.split()).upper()
    record = {"format": SHORT_AUTH_FORMAT, "code": normalized}
    return _store_short_auth_record(app_root, record, key)


def _store_short_auth_record(
    app_root: Path, record: dict[str, object], key: bytes,
) -> bytes:
    """Атомарно сохраняет проверенный auth.dat, не теряя предыдущую активацию."""
    path = app_root / AUTH_FILE
    if path.is_file():
        try:
            if _read_json(path) == record:
                return key
        except ActivationError:
            pass
        backup = path.with_name(
            f"auth.previous.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.dat"
        )
        temporary = app_root / f"auth.incoming.{os.getpid()}.tmp"
        _write_new_json(temporary, record)
        path.rename(backup)
        temporary.replace(path)
    else:
        _write_new_json(path, record)
    return key
