"""Проверки Windows init/install/start без Windows и сети."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
START_PATH = ROOT / "win16" / "start.py"
SPEC = importlib.util.spec_from_file_location("windows_start", START_PATH)
assert SPEC is not None and SPEC.loader is not None
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)

BUILDER_PATH = ROOT / "win16" / "build.py"
BUILDER_SPEC = importlib.util.spec_from_file_location("windows_builder", BUILDER_PATH)
assert BUILDER_SPEC is not None and BUILDER_SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)

INIT_PATH = ROOT / "win16" / "init.py"
INIT_SPEC = importlib.util.spec_from_file_location("windows_init", INIT_PATH)
assert INIT_SPEC is not None and INIT_SPEC.loader is not None
INIT = importlib.util.module_from_spec(INIT_SPEC)
INIT_SPEC.loader.exec_module(INIT)

RUNTIME_PATH = ROOT / "win16" / "runtime.py"
RUNTIME_SPEC = importlib.util.spec_from_file_location("windows_runtime", RUNTIME_PATH)
assert RUNTIME_SPEC is not None and RUNTIME_SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(RUNTIME_SPEC)
RUNTIME_SPEC.loader.exec_module(RUNTIME)

INSTALL_PATH = ROOT / "win16" / "install.py"
INSTALL_SPEC = importlib.util.spec_from_file_location("windows_install", INSTALL_PATH)
assert INSTALL_SPEC is not None and INSTALL_SPEC.loader is not None
INSTALLER = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(INSTALLER)


class WindowsBootstrapTests(unittest.TestCase):
    """Проверяет паспорт, выпуск и защиту от повторной загрузки payload."""

    def manifest(self, payload: bytes) -> dict[str, object]:
        """Создаёт независимый корректный манифест для тестового payload."""
        return {
            "format": "Video16 release v1",
            "release_id": "test-1",
            "key_id": "owner-key-v1",
            "payload_url": "https://example.invalid/Video16.payload",
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "payload_size": len(payload),
        }

    def test_manifest_is_accepted(self):
        """Корректный публичный манифест принимается."""
        data = json.dumps(self.manifest(b"V16P001\0" + b"x" * 28)).encode("utf-8")
        self.assertEqual(BOOTSTRAP.parse_manifest(data)["release_id"], "test-1")

    def test_manifest_rejects_wrong_hash(self):
        """Манифест с не-SHA256 хешем нельзя использовать для установки."""
        manifest = self.manifest(b"V16P001\0" + b"x" * 28)
        manifest["payload_sha256"] = "bad"
        with self.assertRaises(BOOTSTRAP.InstallError):
            BOOTSTRAP.parse_manifest(json.dumps(manifest).encode("utf-8"))

    def test_verified_payload_is_reused(self):
        """Payload с совпадающим размером и SHA-256 не требует повторной загрузки."""
        payload = b"V16P001\0" + b"x" * 28
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.payload"
            path.write_bytes(payload)
            self.assertTrue(BOOTSTRAP.valid_payload(path, self.manifest(payload)))

    def test_changed_payload_is_not_reused(self):
        """Повреждённый payload не проходит проверку и должен быть скачан заново."""
        payload = b"V16P001\0" + b"x" * 28
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.payload"
            path.write_bytes(payload + b"broken")
            self.assertFalse(BOOTSTRAP.valid_payload(path, self.manifest(payload)))

    def test_tracked_release_manifest_matches_tracked_payload(self):
        """Публичные файлы Git-выпуска согласованы по размеру и контрольной сумме."""
        manifest = BOOTSTRAP.parse_manifest((ROOT / "win16" / "release.json").read_bytes())
        self.assertTrue(BOOTSTRAP.valid_payload(ROOT / "win16" / "Video16.payload", manifest))

    def test_install_and_run_use_separate_logs(self):
        """Обычный запуск сохраняет ошибки в run.log, а установка — в install.log."""
        root = Path("D:/Мои программы/VideoT16")
        self.assertEqual(BOOTSTRAP.journal_path(root, True), root / "install.log")
        self.assertEqual(BOOTSTRAP.journal_path(root, False), root / "run.log")

    def test_windows_owner_key_uses_dedicated_local_appdata_path(self):
        """Новый Windows-ключ хранится отдельно от прежнего Linux-пути."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "profile" / ".local" / "share" / "VideoT16"
            legacy.mkdir(parents=True)
            (legacy / "owner_windows_payload_key.bin").write_bytes(b"legacy")
            with (
                mock.patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local-appdata")}, clear=True),
                mock.patch.object(BUILDER.platform, "system", return_value="Windows"),
                mock.patch.object(BUILDER.Path, "home", return_value=root / "profile"),
            ):
                self.assertEqual(
                    BUILDER.owner_key_path(),
                    root / "local-appdata" / "VideoT16" / "owner_windows_payload_key.bin",
                )

    def test_scripts_use_their_own_user_folder(self):
        """init/install/start не должны переносить файлы в заранее заданный диск."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(INSTALLER, "__file__", str(root / "install.py")):
                self.assertEqual(INSTALLER.install_root(), root)
            with mock.patch.object(BOOTSTRAP, "__file__", str(root / "start.py")):
                self.assertEqual(BOOTSTRAP.install_root(), root)

    def test_runtime_dependencies_and_temporary_files_stay_in_user_folder(self):
        """Окружение Python, кэш и расшифрованный runtime создаются рядом с программой."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                BOOTSTRAP.local_python(root),
                root / ".venv" / "Scripts" / "python.exe",
            )
            environment = BOOTSTRAP.local_command_environment(root)
            for key in ("TEMP", "TMP", "TMPDIR", "PIP_CACHE_DIR"):
                self.assertEqual(Path(environment[key]).parent, root)
            temporary = BOOTSTRAP.runtime_temp_root(root)
            self.assertEqual(temporary.parent, root)
            temporary.rmdir()

    def test_dependencies_install_into_local_venv(self):
        """pip запускается из локального .venv, не из системного Python."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = BOOTSTRAP.local_python(root)
            python.parent.mkdir(parents=True)
            python.touch()
            journal = mock.Mock()
            journal.command.return_value = 0
            with (
                mock.patch.object(BOOTSTRAP, "dependencies_ready", return_value=False),
                mock.patch.object(BOOTSTRAP.subprocess, "run", return_value=mock.Mock(returncode=0)),
            ):
                BOOTSTRAP.ensure_dependencies(root, journal)
            args, environment = journal.command.call_args.args
            self.assertEqual(args[:3], [str(python), "-m", "pip"])
            self.assertEqual(Path(environment["PIP_CACHE_DIR"]).parent, root)
            self.assertTrue((root / "dependencies.json").is_file())

    def test_runtime_output_is_configured_for_run_log_capture(self):
        """Запуск Video16 направляет stdout и stderr в журнал родительского процесса."""
        source = START_PATH.read_text(encoding="utf-8")
        self.assertIn("stdout=subprocess.PIPE", source)
        self.assertIn("stderr=subprocess.STDOUT", source)

    def test_windows_camera_index_uses_directshow(self):
        """Выбранная цифровая камера Windows повторно открывается тем же DirectShow."""
        calls: list[tuple[object, object]] = []

        class FakeCv2:
            CAP_DSHOW = "directshow"

            @staticmethod
            def VideoCapture(source, backend):
                calls.append((source, backend))
                return "capture"

        def original(source, settings=None):
            return (source, settings)

        opened = RUNTIME.make_windows_open_capture(original, FakeCv2)
        self.assertEqual(opened(1), "capture")
        self.assertEqual(calls, [(1, "directshow")])
        self.assertEqual(opened("rtsp://camera"), ("rtsp://camera", None))

    def test_new_payload_contains_core_only(self):
        """Новый payload содержит исходники ядра, но не модели и файлы настроек."""
        archive_data = io.BytesIO()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for relative in (
                "win16/runtime.py",
                "win16/launcher_ui.py",
                "src/__init__.py",
                "src/local_object_detection.py",
                "src/realtime.py",
                "src/tracker.py",
                "src/onnx_detector.py",
                "src/osd.py",
                "src/mcm_font.py",
                "src/preview.py",
                "config/__init__.py",
                "config/osd_config.py",
                "config/runtime_settings.py",
                "config/runtime_settings.json",
                "models/user-model.onnx",
            ):
                target = project / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"test")
            with mock.patch.object(BUILDER, "PROJECT_ROOT", project):
                BUILDER.make_archive(archive_data)

        with zipfile.ZipFile(archive_data) as archive:
            names = set(archive.namelist())
        self.assertIn("runtime.py", names)
        self.assertIn("src/local_object_detection.py", names)
        self.assertIn("launcher_ui.py", names)
        self.assertIn("config/runtime_settings.py", names)
        self.assertNotIn("models/user-model.onnx", names)
        self.assertNotIn("config/runtime_settings.json", names)
        self.assertFalse(any(name.endswith(".json") for name in names))

    def test_launcher_discovers_models_in_user_program_folder(self):
        """Меню перечисляет модели из постоянного каталога пользователя."""
        ui = self.load_launcher_ui()
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory) / "models"
            models.mkdir()
            (models / "custom.onnx").write_bytes(b"model")
            self.assertEqual(ui.available_models(models), ["custom.onnx"])

    def test_launcher_explains_empty_user_models_folder(self):
        """При пустом каталоге моделей меню подсказывает пользователю, что сделать."""
        ui = self.load_launcher_ui()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ui.LauncherError, r"models.*\.onnx"):
                ui.choose_options(Path(directory), [], [0])

    def load_launcher_ui(self):
        """Загружает модуль меню без запуска графического интерфейса."""
        ui_path = ROOT / "win16" / "launcher_ui.py"
        ui_spec = importlib.util.spec_from_file_location("windows_launcher_ui", ui_path)
        assert ui_spec is not None and ui_spec.loader is not None
        ui = importlib.util.module_from_spec(ui_spec)
        ui_spec.loader.exec_module(ui)
        return ui

    def test_selection_requires_known_model_camera_and_confidence(self):
        """Окно не должно передавать в runtime произвольную модель или камеру."""
        ui = self.load_launcher_ui()
        self.assertEqual(
            ui.validate_selection("model.onnx", 2, 55, ["model.onnx"], [2]),
            {
                "format": "Video16 Windows settings v1",
                "model": "model.onnx",
                "camera_index": 2,
                "confidence_percent": 55,
            },
        )
        with self.assertRaises(ui.LauncherError):
            ui.validate_selection("other.onnx", 2, 55, ["model.onnx"], [2])

    def test_personal_activation_decrypts_common_payload(self):
        """Один payload открывается только с активацией для совпавшего паспорта."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        identity = "UUID-TEST|BIOS-TEST|true"
        release = "test-1"
        key = os.urandom(32)
        plain = b"PK\\x03\\x04 test archive"
        nonce = os.urandom(12)
        payload = BOOTSTRAP.PAYLOAD_MAGIC + nonce + AESGCM(key).encrypt(
            nonce, plain, release.encode("utf-8")
        )
        manifest = self.manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "test.payload"
            payload_path.write_bytes(payload)
            BUILDER.write_activation(root / "activation.dat", identity, key)
            self.assertEqual(
                BOOTSTRAP.decrypt_payload(root, payload_path, manifest, identity), plain
            )

    def test_activation_for_another_hardware_is_rejected(self):
        """Чужой activation.dat не открывает один и тот же зашифрованный выпуск."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        issued_identity = "UUID-FIRST|BIOS-FIRST|true"
        current_identity = "UUID-SECOND|BIOS-SECOND|true"
        key = os.urandom(32)
        nonce = os.urandom(12)
        payload = BOOTSTRAP.PAYLOAD_MAGIC + nonce + AESGCM(key).encrypt(
            nonce, b"PK\\x03\\x04 test archive", b"test-1"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload_path = root / "Video16.payload"
            payload_path.write_bytes(payload)
            BUILDER.write_activation(root / "activation.dat", issued_identity, key)
            with self.assertRaises(BOOTSTRAP.InstallError):
                BOOTSTRAP.decrypt_payload(root, payload_path, self.manifest(payload), current_identity)

    def test_init_creates_passport_without_hardware_file_dependency(self):
        """init.py сохраняет паспорт, но запуск затем использует отпечаток, не путь файла."""
        identity = {
            "computer": {"UUID": "uuid-test"},
            "bios": {"SerialNumber": "bios-test"},
            "tpm": {"TpmPresent": True},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(INIT, "read_identity", return_value=identity):
            path = INIT.create_init(Path(directory) / "init.dat")
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(INIT.device_key(document["identity"]), "UUID-TEST|BIOS-TEST|true")


if __name__ == "__main__":
    unittest.main()
