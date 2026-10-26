"""Удалённый координатор запуска GTK/GStreamer-плеера через SSH."""

from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path


# Каталог пользовательских видео на минибуке.
VIDEO_DIRECTORY = Path("/home/work/video").resolve()
# Каталог только для файлов и журналов этого помощника.
CACHE_DIRECTORY = Path.home() / ".cache" / "VideoT16"
# Файл программы, состояние запуска и журнал процесса плеера.
PLAYER_FILE = CACHE_DIRECTORY / "step_video_player.py"
PID_FILE = CACHE_DIRECTORY / "step-player.json"
LOG_FILE = CACHE_DIRECTORY / "step-player.log"
LEGACY_PID_FILE = CACHE_DIRECTORY / "vlc-mini.pid"
# Разрешённые расширения, которые передаёт установленный GStreamer.
VIDEO_EXTENSIONS = {
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts",
    ".ts", ".webm", ".wmv",
}
# Маркеры позволяют отличать наше окно от посторонних процессов пользователя.
PLAYER_TOKEN_PREFIX = "VIDEOT16_PLAY_MINI_TOKEN="
LEGACY_MARKER = b"VIDEOT16_PLAY_MINI=1"


def fail(message: str) -> None:
    """Печатает понятную ошибку и завершает удалённую команду с отказом."""
    print(f"Ошибка: {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def decode_video_name() -> str:
    """Декодирует base64-имя и запрещает передавать путь вместо имени файла."""
    try:
        video_name = base64.b64decode(
            os.environ.get("VIDEO_NAME_B64", ""), validate=True
        ).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        fail("имя файла передано в неверном формате")
    if (
        not video_name
        or Path(video_name).name != video_name
        or video_name in {".", ".."}
        or Path(video_name).suffix.lower() not in VIDEO_EXTENSIONS
    ):
        fail("укажи только имя поддерживаемого видеофайла, без пути")
    return video_name


def process_arguments(process_directory: Path) -> list[str] | None:
    """Безопасно читает аргументы процесса того же пользователя."""
    try:
        if process_directory.stat().st_uid != os.getuid():
            return None
        raw_arguments = (process_directory / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    return [item.decode("utf-8", "replace") for item in raw_arguments if item]


def process_environment(process_directory: Path) -> set[bytes] | None:
    """Безопасно читает переменные процесса того же пользователя."""
    try:
        if process_directory.stat().st_uid != os.getuid():
            return None
        return set((process_directory / "environ").read_bytes().split(b"\0"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def is_managed_player(process_directory: Path, token: str | None = None) -> bool:
    """Проверяет маркер, путь скрипта и видео перед управлением процессом."""
    arguments = process_arguments(process_directory)
    environment = process_environment(process_directory)
    if not arguments or len(arguments) < 3 or environment is None:
        return False
    try:
        process_name = (process_directory / "comm").read_text(encoding="utf-8").strip()
        state_fields = (process_directory / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        if not state_fields or state_fields[0] == "Z":
            return False
        player_script = Path(arguments[1]).resolve()
        video_file = Path(arguments[2]).resolve()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, IndexError):
        return False

    marker = next(
        (item for item in environment if item.startswith(PLAYER_TOKEN_PREFIX.encode())),
        None,
    )
    if marker is None:
        return False
    if token is not None and marker != f"{PLAYER_TOKEN_PREFIX}{token}".encode():
        return False
    return (
        process_name.startswith("python")
        and Path(arguments[0]).name.startswith("python")
        and player_script == PLAYER_FILE
        and video_file.parent == VIDEO_DIRECTORY
        and video_file.suffix.lower() in VIDEO_EXTENSIONS
    )


def managed_player_pids(token: str | None = None) -> list[int]:
    """Находит только процессы плеера, подтверждённые маркером и путями."""
    process_ids: list[int] = []
    for process_directory in Path("/proc").iterdir():
        if process_directory.name.isdigit() and is_managed_player(process_directory, token):
            process_ids.append(int(process_directory.name))
    return process_ids


def is_legacy_vlc(process_directory: Path) -> bool:
    """Распознаёт VLC, который был запущен прежней версией play-mini."""
    try:
        if process_directory.stat().st_uid != os.getuid():
            return False
        process_name = (process_directory / "comm").read_text(encoding="utf-8").strip()
        environment = (process_directory / "environ").read_bytes().split(b"\0")
        arguments = [
            item.decode("utf-8", "replace")
            for item in (process_directory / "cmdline").read_bytes().split(b"\0")
            if item
        ]
        if LEGACY_MARKER not in environment or process_name != "vlc":
            return False
        if not arguments or Path(arguments[0]).name != "vlc":
            return False
        if not {"--no-one-instance", "--fullscreen", "--play-and-exit"}.issubset(arguments):
            return False
        video_file = Path(arguments[-1]).resolve()
        return video_file.is_file() and video_file.parent == VIDEO_DIRECTORY
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, RuntimeError):
        return False


def managed_legacy_vlc_pids() -> list[int]:
    """Находит только старые управляемые VLC-процессы из этого скрипта."""
    process_ids: list[int] = []
    for process_directory in Path("/proc").iterdir():
        if process_directory.name.isdigit() and is_legacy_vlc(process_directory):
            process_ids.append(int(process_directory.name))
    return process_ids


def stop_processes(process_ids: list[int], description: str) -> None:
    """Штатно завершает только заранее проверенные процессы и ждёт их выхода."""
    for process_id in process_ids:
        process_directory = Path(f"/proc/{process_id}")
        if not (is_managed_player(process_directory) or is_legacy_vlc(process_directory)):
            continue
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            fail(f"нет прав остановить {description} PID={process_id}")

    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        still_running = [
            process_id
            for process_id in process_ids
            if Path(f"/proc/{process_id}").exists()
            and (is_managed_player(Path(f"/proc/{process_id}")) or is_legacy_vlc(Path(f"/proc/{process_id}")))
        ]
        if not still_running:
            break
        time.sleep(0.1)

    still_running = [
        process_id
        for process_id in process_ids
        if Path(f"/proc/{process_id}").exists()
        and (is_managed_player(Path(f"/proc/{process_id}")) or is_legacy_vlc(Path(f"/proc/{process_id}")))
    ]
    if still_running:
        fail(f"{description} не закрылся штатно для PID: " + ", ".join(map(str, still_running)))


def stop_managed_players() -> None:
    """Закрывает наш новый плеер и при необходимости старое окно VLC."""
    player_pids = managed_player_pids()
    legacy_pids = managed_legacy_vlc_pids()
    if player_pids:
        stop_processes(player_pids, "плеер VideoT16")
    if legacy_pids:
        stop_processes(legacy_pids, "старый VLC VideoT16")
    if not player_pids and not legacy_pids:
        print("Активный плеер VideoT16 не найден.")
    else:
        print("Плеер VideoT16 закрыт; чужие окна не затронуты.")
    for state_file in (PID_FILE, LEGACY_PID_FILE):
        try:
            state_file.unlink(missing_ok=True)
        except OSError as error:
            print(f"Предупреждение: не удалось очистить {state_file}: {error}", file=sys.stderr)


def desktop_environment() -> tuple[dict[str, str], str]:
    """Находит переменные активного графического сеанса пользователя."""
    gui_keys = {
        "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE", "XDG_SESSION_ID", "XDG_CURRENT_DESKTOP",
        "DBUS_SESSION_BUS_ADDRESS", "GDK_BACKEND",
    }

    def has_display(environment: dict[str, str]) -> bool:
        """Проверяет, что в окружении есть адрес X11 или Wayland."""
        return bool(environment.get("DISPLAY") or environment.get("WAYLAND_DISPLAY"))

    try:
        manager_result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        manager_result = None

    if manager_result is not None and manager_result.returncode == 0:
        manager_environment = dict(
            line.split("=", 1) for line in manager_result.stdout.splitlines() if "=" in line
        )
        if has_display(manager_environment):
            return {key: value for key, value in manager_environment.items() if key in gui_keys}, "systemd --user"

    priorities = {
        "gnome-shell": 100, "sway": 100, "weston": 95, "lxqt-session": 90,
        "lxsession": 90, "xfce4-session": 90, "mate-session": 90,
        "cinnamon": 90, "plasmashell": 90, "kwin_wayland": 85,
        "kwin_x11": 85, "openbox": 70, "xfwm4": 70,
    }
    candidates: list[tuple[int, str, dict[str, str]]] = []
    for process_directory in Path("/proc").iterdir():
        if not process_directory.name.isdigit():
            continue
        try:
            if process_directory.stat().st_uid != os.getuid():
                continue
            process_name = (process_directory / "comm").read_text(encoding="utf-8").strip()
            raw_environment = (process_directory / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        environment: dict[str, str] = {}
        for item in raw_environment:
            if b"=" in item:
                key, value = item.split(b"=", 1)
                environment[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
        if not has_display(environment):
            continue
        priority = priorities.get(process_name, 0)
        if environment.get("XDG_SESSION_ID"):
            priority += 5
        candidates.append((priority, process_name, environment))

    if not candidates:
        fail("не нашла активный графический сеанс; войди в рабочий стол минибука и повтори")
    _, process_name, selected = max(candidates, key=lambda item: item[0])
    return {key: value for key, value in selected.items() if key in gui_keys}, process_name


def check_player_dependencies() -> None:
    """Проверяет GTK4, GStreamer и GTK-видеосинк без установки пакетов."""
    try:
        import gi

        gi.require_version("Gtk", "4.0")
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as error:
        fail(f"на минибуке не доступны GTK4/GStreamer для Python: {error}")
    Gst.init(None)
    for element_name in ("playbin", "gtk4paintablesink"):
        if Gst.ElementFactory.find(element_name) is None:
            fail(f"в GStreamer минибука отсутствует элемент {element_name}")


def install_player_source() -> None:
    """Передаёт проверенный локальный Python-файл во временный каталог пользователя."""
    try:
        source = base64.b64decode(os.environ.get("PLAYER_SOURCE_B64", ""), validate=True)
        if not source or len(source) > 250_000:
            fail("исходник плеера отсутствует или слишком велик")
        compile(source.decode("utf-8"), str(PLAYER_FILE), "exec")
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        temporary_file = CACHE_DIRECTORY / f".{PLAYER_FILE.name}.{os.getpid()}.tmp"
        temporary_file.write_bytes(source)
        os.replace(temporary_file, PLAYER_FILE)
    except (ValueError, UnicodeDecodeError, OSError, SyntaxError) as error:
        fail(f"не удалось передать исходник плеера: {error}")


def start_player() -> None:
    """Проверяет видео, передаёт плеер и запускает его в найденном рабочем столе."""
    active_players = managed_player_pids()
    if active_players:
        fail("плеер уже открыт; сначала выполни ./play-mini stop")
    if managed_legacy_vlc_pids():
        fail("ещё открыто старое видео в VLC; выполни ./play-mini stop")
    if not VIDEO_DIRECTORY.is_dir():
        fail(f"каталог видео не найден: {VIDEO_DIRECTORY}")

    video_name = decode_video_name()
    video_file = VIDEO_DIRECTORY / video_name
    try:
        video_file = video_file.resolve(strict=True)
    except FileNotFoundError:
        fail(f"видео не найдено: {video_name}")
    if video_file.parent != VIDEO_DIRECTORY or not video_file.is_file():
        fail("разрешены только обычные видеофайлы непосредственно в /home/work/video")

    check_player_dependencies()
    source_payload = os.environ.get("PLAYER_SOURCE_B64", "")
    if not source_payload:
        fail("не передан исходник плеера с этого ноутбука")
    install_player_source()
    desktop_vars, session_source = desktop_environment()

    # Удаляем устаревшие GUI-переменные SSH и подставляем окружение рабочего стола.
    launch_environment = os.environ.copy()
    gui_keys = (
        "DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
        "XDG_SESSION_TYPE", "XDG_SESSION_ID", "XDG_CURRENT_DESKTOP",
        "DBUS_SESSION_BUS_ADDRESS", "GDK_BACKEND",
    )
    for gui_key in gui_keys:
        launch_environment.pop(gui_key, None)
    launch_environment.update(desktop_vars)
    launch_environment.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    launch_environment["VIDEOT16_PLAY_MINI_TOKEN"] = uuid.uuid4().hex

    try:
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("w", encoding="utf-8") as log_file:
            log_file.write(f"Видео: {video_file}\nСеанс: {session_source}\n")
            log_file.flush()
            process = subprocess.Popen(
                [sys.executable, str(PLAYER_FILE), str(video_file)],
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=launch_environment,
                start_new_session=True,
            )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(0.1)
        if process.poll() is not None:
            log_text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
            fail("плеер завершился при запуске. Журнал:\n" + log_text[-3000:])
        PID_FILE.write_text(
            json.dumps({"pid": process.pid, "token": launch_environment["VIDEOT16_PLAY_MINI_TOKEN"]}),
            encoding="utf-8",
        )
    except (OSError, subprocess.SubprocessError) as error:
        if "process" in locals() and process.poll() is None:
            process.terminate()
        fail(f"не удалось запустить окно плеера: {error}")

    print(f"Плеер VideoT16 запущен в сеансе {session_source}; PID={process.pid}")
    print(f"Видео: {video_file}")
    print("Кнопка «Стоп» ставит на паузу; крестик окна закрывает плеер.")
    print(f"Журнал удалённого плеера: {LOG_FILE}")


def main() -> int:
    """Выполняет на минибуке команду start или stop, переданную по SSH."""
    action = os.environ.get("REMOTE_ACTION", "")
    if action == "stop":
        stop_managed_players()
        return 0
    if action == "start":
        start_player()
        return 0
    fail("неизвестная команда; используй start или stop")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
