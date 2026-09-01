"""Локальная веб-панель оператора VideoT16 без внешних зависимостей."""

from __future__ import annotations

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config.runtime_settings import SETTINGS_PATH, load_settings, save_settings

ROOT = Path(__file__).parent
PROJECT_DIR = ROOT.parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_RESTART_SERVICE = "videot16.service"
FONTS = [
    "FONT_HERSHEY_SIMPLEX",
    "FONT_HERSHEY_PLAIN",
    "FONT_HERSHEY_DUPLEX",
    "FONT_HERSHEY_COMPLEX",
    "FONT_HERSHEY_TRIPLEX",
    "FONT_HERSHEY_COMPLEX_SMALL",
    "FONT_HERSHEY_SCRIPT_SIMPLEX",
    "FONT_HERSHEY_SCRIPT_COMPLEX",
]


def temperature() -> str:
    """Возвращает температуру платы, если доступна команда Raspberry Pi."""
    try:
        result = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, check=False)
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def available_models() -> list[str]:
    """Возвращает разрешённые ONNX-модели из каталога проекта."""
    if not MODEL_DIR.is_dir():
        return []
    return sorted(path.name for path in MODEL_DIR.glob("*.onnx") if path.is_file())


def selected_model(settings: dict) -> str:
    """Возвращает имя выбранной модели без выхода за каталог models."""
    value = settings.get("detection", {}).get("model_path", "")
    if isinstance(value, str) and value.startswith("models/") and "/" not in value[7:]:
        name = value[7:]
        if name in available_models():
            return name
    models = available_models()
    return models[0] if models else ""


def restart_detection() -> tuple[bool, str]:
    """Перезапускает только службу распознавания через ограниченный sudo-helper."""
    result = subprocess.run(
        ["sudo", "-n", "/usr/local/sbin/videot16-restart"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or "служба не перезапустилась"
    return True, "распознавание перезапущено"


class ConfigHandler(BaseHTTPRequestHandler):
    """Обрабатывает страницу и JSON API настроек."""

    settings_path = SETTINGS_PATH

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 — имя метода задаёт BaseHTTPRequestHandler.
        route = urlparse(self.path).path
        if route == "/":
            self._send(200, "text/html; charset=utf-8", (ROOT / "index.html").read_bytes())
        elif route == "/api/settings":
            settings = load_settings(self.settings_path)
            body = json.dumps({"settings": settings, "fonts": FONTS, "models": available_models(), "selected_model": selected_model(settings)}, ensure_ascii=False).encode()
            self._send(200, "application/json; charset=utf-8", body)
        elif route == "/api/status":
            body = json.dumps({"temperature": temperature(), "settings": str(self.settings_path)}).encode()
            self._send(200, "application/json; charset=utf-8", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:  # noqa: N802 — имя метода задаёт BaseHTTPRequestHandler.
        route = urlparse(self.path).path
        if route == "/api/configuration/finish":
            marker = self.settings_path.parent / "wifi_finish.request"
            marker.write_text("finish\n", encoding="utf-8")
            body = json.dumps({"ok": True, "message": "Wi-Fi отключается"}, ensure_ascii=False).encode()
            self._send(200, "application/json; charset=utf-8", body)
            return
        if route == "/api/model/restart":
            ok, message = restart_detection()
            status = 200 if ok else 503
            self._send(status, "application/json; charset=utf-8", json.dumps({"ok": ok, "message": message}, ensure_ascii=False).encode())
            return
        if route not in {"/api/settings", "/api/model"}:
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            if route == "/api/model":
                model_name = payload.get("model") if isinstance(payload, dict) else None
                if not isinstance(model_name, str) or model_name not in available_models():
                    raise ValueError("модель не входит в список разрешённых ONNX-файлов")
                settings = load_settings(self.settings_path)
                settings.setdefault("detection", {})["model_path"] = f"models/{model_name}"
                saved = save_settings(settings, self.settings_path)
                self._send(200, "application/json; charset=utf-8", json.dumps({"settings": saved, "selected_model": model_name}, ensure_ascii=False).encode())
                return
            saved = save_settings(payload, self.settings_path)
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as error:
            self._send(400, "application/json; charset=utf-8", json.dumps({"error": str(error)}).encode())
            return
        self._send(200, "application/json; charset=utf-8", json.dumps({"settings": saved}, ensure_ascii=False).encode())

    def log_message(self, format: str, *args: object) -> None:
        print(f"[WEB] {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Адрес прослушивания; 0.0.0.0 — для локальной сети")
    parser.add_argument("--port", type=int, default=8080, help="Порт панели")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH, help="JSON-файл настроек")
    args = parser.parse_args()
    ConfigHandler.settings_path = args.settings
    server = ThreadingHTTPServer((args.host, args.port), ConfigHandler)
    print(f"Панель настроек: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
