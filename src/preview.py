"""Временный MJPEG-просмотр обработанного кадра через localhost."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.mcm_font import McmFont


class PreviewServer:
    """Отдаёт последний кадр распознавания в браузер без хранения видеофайлов."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8081) -> None:
        """Запускает HTTP-сервер в фоновом потоке."""
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._last_update = 0.0
        self._update_interval = 0.2  # Веб: 5 кадров/с, чтобы не перегружать Raspberry.
        self._mcm_font = McmFont(Path(__file__).resolve().parents[1] / "assets/osd/vision.mcm")
        self._closed = False
        state = self

        class Handler(BaseHTTPRequestHandler):
            """Обрабатывает страницу и MJPEG-поток."""

            def log_message(self, format: str, *args: object) -> None:
                """Не записывает запросы браузера в журнал распознавания."""

            def do_GET(self) -> None:  # noqa: N802
                """Отдаёт веб-страницу или поток кадров."""
                if self.path == "/":
                    body = (
                        b"<!doctype html><meta charset='utf-8'><title>VideoT16</title>"
                        b"<h3>VideoT16 preview</h3><img src='/stream.mjpg' "
                        b"style='max-width:100%;height:auto'>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store")
                self.end_headers()
                try:
                    while True:
                        with state._condition:
                            state._condition.wait_for(
                                lambda: state._jpeg is not None or state._closed,
                                timeout=2.0,
                            )
                            if state._closed:
                                return
                            jpeg = state._jpeg
                        if jpeg is None:
                            continue
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode("ascii")
                            + b"\r\n\r\n" + jpeg + b"\r\n"
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Веб-просмотр: http://{host}:{port}/", flush=True)

    def update(self, frame, canvas_mirror=None) -> None:
        """Кодирует кадр с зеркалом Canvas OSD в JPEG для веб-просмотра."""
        now = time.monotonic()
        if now - self._last_update < self._update_interval:
            return
        self._last_update = now

        if canvas_mirror is not None:
            columns, rows, lines = canvas_mirror.snapshot()
            self._mcm_font.draw(frame, lines, columns, rows)

        # Умеренное качество снижает задержку MJPEG на Raspberry Pi.
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
        if ok:
            with self._condition:
                self._jpeg = encoded.tobytes()
                self._condition.notify_all()

    def close(self) -> None:
        """Останавливает сервер и освобождает порт."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)
