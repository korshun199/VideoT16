"""Локальное распознавание объектов с USB-камеры или RTSP-потока."""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from config.osd_config import (
    ARM_BANNER_STYLE,
    AXIS_STYLE,
    FLIGHT_STATUS_STYLE,
    HORIZON_STYLE,
    OBJECT_STYLE,
    OSD_TEXT,
    SYSTEM_STATUS_STYLE,
)
from config.runtime_settings import apply_osd_settings, load_settings
from src.realtime import Detection, LatestFrameCapture, LatestInferenceWorker, SystemStatusMonitor
from src.tracker import DetectionTracker
from src.osd import OSD_MAX_X, OSD_MAX_Y

RUSSIAN_LABELS = [
    "человек", "велосипед", "автомобиль", "мотоцикл", "самолёт", "автобус", "поезд", "грузовик", "лодка",
    "светофор", "пожарный гидрант", "знак стоп", "паркомат", "скамейка", "птица", "кошка", "собака",
    "лошадь", "овца", "корова", "слон", "медведь", "зебра", "жираф", "рюкзак", "зонт", "сумочка", "галстук",
    "чемодан", "фрисби", "лыжи", "сноуборд", "спортивный мяч", "воздушный змей", "бейсбольная бита",
    "бейсбольная перчатка", "скейтборд", "доска для сёрфинга", "теннисная ракетка", "бутылка", "бокал", "чашка",
    "вилка", "нож", "ложка", "миска", "банан", "яблоко", "сэндвич", "апельсин", "брокколи", "морковь",
    "хот-дог", "пицца", "пончик", "торт", "стул", "диван", "растение в горшке", "кровать", "обеденный стол",
    "унитаз", "телевизор", "ноутбук", "мышь", "пульт", "клавиатура", "мобильный телефон", "микроволновка",
    "духовка", "тостер", "раковина", "холодильник", "книга", "часы", "ваза", "ножницы", "плюшевый мишка",
    "фен", "зубная щётка",
]

RUSSIAN_DRONE_LABELS = {
    "quadcopter": "квадрокоптер",
    "fixed-wing": "самолётный дрон",
}

RUSSIAN_MILITARY_LABELS = {
    "armored_car": "бронемашина",
    "car": "автомобиль",
    "person": "человек",
    "plane": "самолёт",
    "rszo": "РСЗО",
    "sau": "САУ",
    "tank": "танк",
    "trench": "окоп",
    "truck": "грузовик",
    "vehicle": "техника",
}

WINDOW_NAME = "Локальное распознавание объектов"

def parse_source(value: str) -> int | str:
    """Преобразует номер камеры в число, а URL оставляет строкой."""
    return int(value) if value.isdigit() else value


def parse_resolution(value: str) -> tuple[int, int]:
    """Преобразует разрешение вида 1920x1080 в ширину и высоту."""
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("Разрешение должно быть в формате ШИРИНАxВЫСОТА")
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("Ширина и высота должны быть положительными")
    return width, height


def parse_monitor_geometry(output: str, monitor_name: str) -> tuple[int, int, int, int]:
    """Извлекает ширину, высоту и координаты активного монитора из xrandr."""
    pattern = re.compile(
        rf"^{re.escape(monitor_name)}\s+connected(?:\s+primary)?\s+"
        r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match:
            width, height, x, y = (int(value) for value in match.groups())
            return width, height, x, y
    raise RuntimeError(f"Монитор {monitor_name} не подключён или не имеет активного режима")


def resolve_monitor_geometry(monitor_name: str) -> tuple[int, int, int, int]:
    """Получает текущую геометрию выбранного X11-монитора через xrandr."""
    if not shutil.which("xrandr"):
        raise RuntimeError("Для параметра --monitor требуется команда xrandr")
    result = subprocess.run(
        ["xrandr", "--query", "--current"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or "неизвестная ошибка xrandr"
        raise RuntimeError(f"Не удалось прочитать мониторы: {message}")
    return parse_monitor_geometry(result.stdout, monitor_name)


def build_parser() -> argparse.ArgumentParser:
    """Создаёт интерфейс командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="logitech", help="logitech, номер камеры, /dev/videoN или RTSP URL")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Путь к локальным весам YOLO")
    parser.add_argument("--settings", type=Path, default=Path("config/runtime_settings.json"), help="Файл настроек оператора")
    parser.add_argument("--labels", choices=("ru", "en"), default="ru", help="Язык подписей объектов")
    parser.add_argument("--generic-label", action="store_true", help="Показывать для всех объектов подпись OBJECT")
    parser.add_argument("--confidence", type=float, default=0.35, help="Минимальная уверенность 0..1")
    parser.add_argument("-p", "--percent", "--confidence-percent", dest="confidence_percent", type=float, help="Минимальная уверенность в процентах 0..100")
    parser.add_argument("--device", default="cpu", help="Устройство: cpu или auto для RKNN")
    parser.add_argument(
        "--inference-size",
        type=int,
        default=640,
        help="Размер стороны изображения для YOLO, например 320 или 640",
    )
    parser.add_argument(
        "--inference-interval",
        type=int,
        default=1,
        help="Запускать инференс на каждом N-м кадре",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=0,
        help="Желаемая частота камеры; 0 — оставить значение камеры",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="Число потоков CPU для Torch и ONNX; 0 — безопасное значение",
    )
    parser.add_argument("--alert-wav", type=Path, help="Локальный WAV для сигнала обнаружения")
    parser.add_argument("--alert-delay", type=float, default=3.0, help="Сколько секунд удерживать обнаружение до сигнала")
    parser.add_argument("--alert-cooldown", type=float, default=3.0, help="Пауза между сигналами в секундах")
    parser.add_argument("--output", type=Path, help="Путь записи обработанного видео")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("snapshots"))
    parser.add_argument("--headless", action="store_true", help="Работать без окна предпросмотра")
    parser.add_argument(
        "--framebuffer",
        nargs="?",
        const="auto",
        help="Выводить обработанное видео в Linux framebuffer; auto — найти Composite-1",
    )
    parser.add_argument(
        "--drm",
        nargs="?",
        const="/dev/dri/card1",
        help="Выводить обработанное видео напрямую через DRM/KMS",
    )
    parser.add_argument("--window-width", type=int, default=1280, help="Ширина окна предпросмотра")
    parser.add_argument("--window-height", type=int, default=720, help="Высота окна предпросмотра")
    parser.add_argument("--resolution", type=parse_resolution, help="Размер окна, например 1920x1080")
    parser.add_argument("--monitor", help="Имя монитора xrandr, например HDMI-1")
    parser.add_argument("--fullscreen", action="store_true", help="Развернуть окно на весь экран")
    parser.add_argument("--window-x", type=int, default=None, help="Положение окна по горизонтали")
    parser.add_argument("--window-y", type=int, default=None, help="Положение окна по вертикали")
    parser.add_argument("--max-frames", type=int, default=0, help="Остановиться после N кадров; 0 — без лимита")
    parser.add_argument("--list-cameras", action="store_true", help="Проверить камеры 0..4")
    parser.add_argument("--inav-port", help="USB-порт INAV, например /dev/ttyACM0")
    parser.add_argument("--inav-baudrate", type=int, default=115200, help="Скорость MSP-порта INAV")
    parser.add_argument(
        "--battery-capacity-mah",
        type=int,
        default=0,
        help="Полная ёмкость аккумулятора в мАч для расчёта остатка",
    )
    return parser


def list_cameras() -> None:
    """Показывает доступные первые десять устройств V4L2."""
    import cv2

    for index in range(10):
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if capture.isOpened():
            print(f"Камера {index}: доступна")
            capture.release()


def create_writer(path: Path, capture: cv2.VideoCapture) -> cv2.VideoWriter:
    """Создаёт MP4-запись с параметрами входного потока."""
    import cv2

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if 1 <= fps <= 120 else 25.0
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось открыть запись: {path}")
    return writer


def open_capture(source: int | str):
    """Открывает USB-камеру через V4L2 или поток через авто-бэкенд."""
    import cv2

    if isinstance(source, int) or (isinstance(source, str) and source.startswith("/dev/video")):
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    return cv2.VideoCapture(source)


def configure_low_latency_capture(capture, camera_fps: float = 0) -> None:
    """Настраивает V4L2 на минимальную очередь и MJPEG без принуждения разрешения."""
    import cv2

    # Не копим старые кадры: приоритет имеет актуальность, а не полнота очереди.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if camera_fps > 0:
        capture.set(cv2.CAP_PROP_FPS, camera_fps)


def resolve_source(source: int | str) -> int | str:
    """Находит внешнюю Logitech по имени V4L2 или стабильному симлинку."""
    if source != "logitech":
        return source
    candidates = sorted(glob.glob("/dev/v4l/by-id/*Logitech*video-index0"))
    if candidates:
        return candidates[0]
    for name_file in sorted(glob.glob("/sys/class/video4linux/video*/name")):
        try:
            camera_name = Path(name_file).read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if "logitech" in camera_name or "brio" in camera_name:
            device = Path(name_file).parent.name
            return f"/dev/{device}"
    raise RuntimeError("Logitech Brio не найдена. Проверьте: v4l2-ctl --list-devices")


def configure_local_caches() -> None:
    """Переносит кэши библиотек в каталог проекта."""
    cache_dir = Path(".cache")
    (cache_dir / "ultralytics").mkdir(parents=True, exist_ok=True)
    (cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str((cache_dir / "ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str((cache_dir / "matplotlib").resolve()))


def play_alert(path: Path) -> subprocess.Popen[bytes] | None:
    """Запускает WAV через доступный системный проигрыватель."""
    players = (
        ("paplay", ["paplay", str(path)]),
        ("aplay", ["aplay", "-q", str(path)]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)]),
    )
    for name, command in players:
        if shutil.which(name):
            return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


def extract_detections(result, generic_label: bool = False) -> tuple[Detection, ...]:
    """Отделяет координаты рамок от объекта результата Ultralytics."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        name = "OBJECT" if generic_label else result.names[class_id]
        detections.append(Detection(x1, y1, x2, y2, str(name), confidence))
    return tuple(detections)


def draw_detections(frame, detections: tuple[Detection, ...]):
    """Рисует на свежем кадре последние рамки и уверенность."""
    import cv2

    # Копия сохраняет исходный кадр неизменным для фонового распознавания.
    annotated = frame.copy()
    for detection in detections:
        label = f"{detection.name} {detection.confidence * 100:.0f}%"
        object_font = getattr(cv2, OBJECT_STYLE["font"])
        cv2.rectangle(
            annotated,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            OBJECT_STYLE["color"],
            OBJECT_STYLE["box_thickness"],
        )
        text_y = max(
            detection.y1 - OBJECT_STYLE["text_offset_y"],
            OBJECT_STYLE["text_min_y"],
        )
        cv2.putText(
            annotated,
            label,
            (detection.x1, text_y),
            object_font,
            OBJECT_STYLE["font_scale"],
            OBJECT_STYLE["color"],
            OBJECT_STYLE["text_thickness"],
            cv2.LINE_AA,
        )
    return annotated


def put_osd_text(frame, text: str, position: tuple[int, int]):
    """Рисует маленький жёлтый текст с минимальной тенью вправо и вниз."""
    import cv2

    x, y = position
    font = getattr(cv2, OSD_TEXT["font"])
    shadow_x, shadow_y = OSD_TEXT["shadow_offset"]
    # Тень смещена вправо и вниз на величину из настроечного файла.
    cv2.putText(
        frame,
        text,
        (x + shadow_x, y + shadow_y),
        font,
        OSD_TEXT["font_scale"],
        OSD_TEXT["shadow_color"],
        OSD_TEXT["shadow_thickness"],
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        OSD_TEXT["font_scale"],
        OSD_TEXT["color"],
        OSD_TEXT["thickness"],
        cv2.LINE_AA,
    )
    return frame


def draw_system_status(frame, text: str):
    """Рисует синюю строку температуры и загрузки CPU внизу слева."""
    import cv2

    frame_height = frame.shape[0]
    x = SYSTEM_STATUS_STYLE["left"]
    y = frame_height - SYSTEM_STATUS_STYLE["bottom"]
    font = getattr(cv2, SYSTEM_STATUS_STYLE["font"])
    offset_x, offset_y = SYSTEM_STATUS_STYLE["shadow_offset"]
    cv2.putText(
        frame,
        text,
        (x + offset_x, y + offset_y),
        font,
        SYSTEM_STATUS_STYLE["font_scale"],
        SYSTEM_STATUS_STYLE["shadow_color"],
        SYSTEM_STATUS_STYLE["shadow_thickness"],
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        SYSTEM_STATUS_STYLE["font_scale"],
        SYSTEM_STATUS_STYLE["color"],
        SYSTEM_STATUS_STYLE["thickness"],
        cv2.LINE_AA,
    )
    return frame


def inference_status_text(inference, now: float) -> str:
    """Формирует ожидание, чистое вычисление и свежесть результата YOLO."""
    if inference.sequence == 0 or inference.completed_at <= 0:
        return "AI W--- R--- A---"
    submitted_at = getattr(inference, "submitted_at", None)
    started_at = getattr(inference, "started_at", None)
    age_ms = max(0.0, (now - inference.completed_at) * 1000)
    if submitted_at is None or started_at is None:
        return f"AI W--- R--- A{age_ms:.0f}ms"
    wait_ms = max(0.0, (started_at - submitted_at) * 1000)
    run_ms = max(0.0, (inference.completed_at - started_at) * 1000)
    return f"AI W{wait_ms:.0f} R{run_ms:.0f} A{age_ms:.0f}ms"


def course_to_cardinal(course: float) -> str:
    """Преобразует курс в ближайшее направление по сторонам света."""
    directions = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return directions[int((course + 22.5) // 45) % len(directions)]


def draw_telemetry(frame, telemetry, battery_capacity_mah: int = 0):
    """Рисует на кадре последние данные INAV без управления полётником."""
    import cv2

    """ lines = [f"FIPIK {telemetry.variant} {telemetry.version} 2026".strip()] """
    lines = [f"FIPIK BPS  2026".strip()]
    if telemetry.roll is not None:
        lines.append(f"R {telemetry.roll:.1f}  P {telemetry.pitch:.1f}  Y {telemetry.yaw:.0f}")
    if telemetry.altitude is not None:
        lines.append(f"ALT {telemetry.altitude:.2f} m")
    if telemetry.surface_distance is not None:
        # AGL — высота над поверхностью по дальномеру полётного контроллера.
        lines.append(f"AGL {telemetry.surface_distance:.2f} m")
    if telemetry.voltage is not None:
        rssi_percent = min(100, round((telemetry.rssi or 0) * 100 / 1023))
        battery_line = f"BAT {telemetry.voltage:.1f} V"
        if telemetry.battery_current is not None:
            battery_line += f"  I {telemetry.battery_current:.2f} A"
        if telemetry.battery_mah_drawn is not None:
            battery_line += f"  USED {telemetry.battery_mah_drawn} mAh"
            if battery_capacity_mah > 0:
                # Процент рассчитывается по расходу относительно полной ёмкости.
                remaining_mah = max(0, battery_capacity_mah - telemetry.battery_mah_drawn)
                remaining_percent = round(remaining_mah * 100 / battery_capacity_mah)
                battery_line += f"  LEFT {remaining_mah} mAh {remaining_percent}%"
        lines.append(f"{battery_line}  RSSI {rssi_percent}%")
    if telemetry.gps_fix is not None:
        lines.append(f"GPS fix {telemetry.gps_fix}  SAT {telemetry.gps_satellites or 0}")
        # После первого GPS-пакета строка GS остаётся на экране даже без фикса.
        ground_speed = telemetry.ground_speed if telemetry.ground_speed is not None else 0.0
        # Скорость INAV приходит в метрах в секунду; дополнительно показываем км/ч.
        speed_kmh = ground_speed * 3.6
        course_text = ""
        if telemetry.ground_course is not None:
            course_text = f"  {course_to_cardinal(telemetry.ground_course)} {telemetry.ground_course:.0f} DEG"
        lines.append(f"GS {ground_speed:.1f} m/s {speed_kmh:.1f} km/h{course_text}")
    if telemetry.updated_at <= 0:
        lines = ["INAV: WAITING MSP"]
    for index, line in enumerate(lines):
        position = (OSD_TEXT["left"], OSD_TEXT["top"] + index * OSD_TEXT["line_spacing"])

        put_osd_text(frame, line, position)
    return frame


def draw_flight_status(frame, telemetry):
    """Рисует справа состояние ARM и вертикальный индикатор газа."""
    import cv2

    if telemetry.updated_at <= 0:
        return frame

    frame_width = frame.shape[1]
    right = FLIGHT_STATUS_STYLE["right"]
    top = FLIGHT_STATUS_STYLE["top"]
    line_spacing = FLIGHT_STATUS_STYLE["line_spacing"]
    font = getattr(cv2, OSD_TEXT["font"])

    arm_value = "--" if telemetry.armed is None else ("ON" if telemetry.armed else "OFF")
    arm_switch_value = "--" if telemetry.arm_switch is None else ("ON" if telemetry.arm_switch else "OFF")
    arm_switch_details = ""
    if telemetry.arm_aux_channel is not None and telemetry.arm_switch_value is not None:
        arm_switch_details = f" A{telemetry.arm_aux_channel} {telemetry.arm_switch_value}"
    throttle_value = "--" if telemetry.throttle_percent is None else f"{telemetry.throttle_percent}%"
    status_lines = (
        #f"ARM {arm_value}",
        #f"ARM SW {arm_switch_value}{arm_switch_details}",
        f"THR {throttle_value}",
    )
    for index, text in enumerate(status_lines):
        text_width = cv2.getTextSize(
            text,
            font,
            OSD_TEXT["font_scale"],
            OSD_TEXT["thickness"],
        )[0][0]
        position = (frame_width - right - text_width, top + index * line_spacing)
        put_osd_text(frame, text, position)

    if telemetry.throttle_percent is None:
        return frame

    bar_width = FLIGHT_STATUS_STYLE["bar_width"]
    bar_height = FLIGHT_STATUS_STYLE["bar_height"]
    bar_x = frame_width - right - bar_width
    bar_y = top + FLIGHT_STATUS_STYLE["bar_top_offset"]
    bar_end = (bar_x + bar_width, bar_y + bar_height)

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        bar_end,
        FLIGHT_STATUS_STYLE["shadow_color"],
        FLIGHT_STATUS_STYLE["shadow_thickness"],
    )
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        bar_end,
        FLIGHT_STATUS_STYLE["bar_color"],
        FLIGHT_STATUS_STYLE["bar_thickness"],
    )

    # Заполнение и маркер движутся снизу вверх от 0 до 100 процентов.
    inner_height = max(1, bar_height - 2)
    filled_height = round(inner_height * telemetry.throttle_percent / 100)
    if filled_height > 0:
        cv2.rectangle(
            frame,
            (bar_x + 2, bar_y + bar_height - filled_height),
            (bar_x + bar_width - 2, bar_y + bar_height - 1),
            FLIGHT_STATUS_STYLE["fill_color"],
            -1,
        )
    marker_y = bar_y + bar_height - round(bar_height * telemetry.throttle_percent / 100)
    marker_overhang = FLIGHT_STATUS_STYLE["marker_overhang"]
    marker_start = (bar_x - marker_overhang, marker_y)
    marker_end = (bar_x + bar_width + marker_overhang, marker_y)
    cv2.line(
        frame,
        marker_start,
        marker_end,
        FLIGHT_STATUS_STYLE["marker_shadow_color"],
        FLIGHT_STATUS_STYLE["marker_shadow_thickness"],
        cv2.LINE_AA,
    )
    cv2.line(
        frame,
        marker_start,
        marker_end,
        FLIGHT_STATUS_STYLE["marker_color"],
        FLIGHT_STATUS_STYLE["marker_thickness"],
        cv2.LINE_AA,
    )
    return frame


def draw_arm_banner(frame, telemetry, now: float | None = None):
    """Рисует снизу мигающий ARM OFF или постоянный режим полёта."""
    import cv2

    if telemetry.updated_at <= 0 or telemetry.armed is None:
        return frame

    # Для пилотской надписи используем положение назначенного ARM-тумблера.
    # Фактическое состояние контроллера отдельно сохраняется в telemetry.armed.
    arm_enabled = telemetry.arm_switch if telemetry.arm_switch is not None else telemetry.armed
    if arm_enabled:
        text = telemetry.flight_mode or "ACRO"
        color = ARM_BANNER_STYLE["armed_color"]
    else:
        blink_period = max(0.1, ARM_BANNER_STYLE["blink_period"])
        current_time = time.monotonic() if now is None else now
        if current_time % blink_period >= blink_period / 2:
            return frame
        text = "ARM OFF"
        color = ARM_BANNER_STYLE["disarmed_color"]

    font = getattr(cv2, ARM_BANNER_STYLE["font"])
    font_scale = ARM_BANNER_STYLE["font_scale"]
    thickness = ARM_BANNER_STYLE["thickness"]
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    position = (
        max(0, (frame.shape[1] - text_size[0]) // 2),
        max(text_size[1], frame.shape[0] - ARM_BANNER_STYLE["bottom_margin"]),
    )
    shadow_offset = ARM_BANNER_STYLE["shadow_offset"]
    shadow_position = (position[0] + shadow_offset[0], position[1] + shadow_offset[1])
    cv2.putText(
        frame,
        text,
        shadow_position,
        font,
        font_scale,
        ARM_BANNER_STYLE["shadow_color"],
        ARM_BANNER_STYLE["shadow_thickness"],
        cv2.LINE_AA,
    )
    cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_artificial_horizon(frame, telemetry):
    """Рисует в центре кадра горизонт и неподвижную ось самолёта."""
    import math

    import cv2

    if telemetry.roll is None or telemetry.pitch is None:
        return frame

    frame_height, frame_width = frame.shape[:2]
    center_x = frame_width // 2
    center_y = frame_height // 2
    pitch_limit = HORIZON_STYLE["pitch_limit"]
    pitch = max(-pitch_limit, min(pitch_limit, telemetry.pitch))
    horizon_y = int(center_y + pitch * HORIZON_STYLE["pitch_scale"])
    horizon_length = max(
        HORIZON_STYLE["min_length"],
        int(min(frame_width, frame_height) * HORIZON_STYLE["length_ratio"]),
    )
    angle = math.radians(-telemetry.roll)
    half_length = horizon_length / 2
    delta_x = int(math.cos(angle) * half_length)
    delta_y = int(math.sin(angle) * half_length)
    start = (center_x - delta_x, horizon_y - delta_y)
    end = (center_x + delta_x, horizon_y + delta_y)

    # Жёлтая линия горизонта с чёрной окантовкой остаётся видимой на любом фоне.
    cv2.line(
        frame,
        start,
        end,
        HORIZON_STYLE["shadow_color"],
        HORIZON_STYLE["shadow_thickness"],
        cv2.LINE_AA,
    )
    cv2.line(frame, start, end, HORIZON_STYLE["color"], HORIZON_STYLE["thickness"], cv2.LINE_AA)

    # Белая неподвижная ось и короткие крылья показывают положение центра кадра.
    vertical_length = AXIS_STYLE["vertical_half_length"]
    horizontal_length = AXIS_STYLE["horizontal_half_length"]
    axis_top = (center_x, center_y - vertical_length)
    axis_bottom = (center_x, center_y + vertical_length)
    left_wing = (center_x - horizontal_length, center_y)
    right_wing = (center_x + horizontal_length, center_y)
    cv2.line(frame, axis_top, axis_bottom, AXIS_STYLE["shadow_color"], AXIS_STYLE["shadow_thickness"], cv2.LINE_AA)
    cv2.line(frame, axis_top, axis_bottom, AXIS_STYLE["color"], AXIS_STYLE["thickness"], cv2.LINE_AA)
    cv2.line(frame, left_wing, right_wing, AXIS_STYLE["shadow_color"], AXIS_STYLE["shadow_thickness"], cv2.LINE_AA)
    cv2.line(frame, left_wing, right_wing, AXIS_STYLE["color"], AXIS_STYLE["thickness"], cv2.LINE_AA)
    return frame


def black_background(frame, target_width: int, target_height: int):
    """Помещает кадр по центру чёрного холста без искажения пропорций."""
    import numpy as np

    frame_height, frame_width = frame.shape[:2]
    scale = min(target_width / frame_width, target_height / frame_height)
    image_width = max(1, int(frame_width * scale))
    image_height = max(1, int(frame_height * scale))
    import cv2

    resized = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    left = (target_width - image_width) // 2
    top = (target_height - image_height) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def run(args: argparse.Namespace) -> int:
    """Запускает захват, локальный инференс и отображение результата."""
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Локальная модель не найдена: {model_path}. Положите веса в этот путь.")
    runtime_settings = load_settings(args.settings)
    apply_osd_settings(
        runtime_settings,
        {
            "object": OBJECT_STYLE,
            "text": OSD_TEXT,
            "system_status": SYSTEM_STATUS_STYLE,
            "horizon": HORIZON_STYLE,
            "axis": AXIS_STYLE,
            "flight_status": FLIGHT_STATUS_STYLE,
            "arm_banner": ARM_BANNER_STYLE,
        },
    )
    detection_settings = runtime_settings["detection"]
    args.confidence_percent = float(detection_settings["confidence_percent"])
    args.inference_size = int(detection_settings["inference_size"])
    args.inference_interval = int(detection_settings["inference_interval"])
    args.generic_label = bool(detection_settings["generic_label"])
    if args.confidence_percent is not None:
        if not 0 < args.confidence_percent <= 100:
            raise ValueError("--confidence-percent должен быть больше 0 и не больше 100")
        args.confidence = args.confidence_percent / 100
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence должен быть больше 0 и не больше 1")
    if args.inference_size < 32:
        raise ValueError("--inference-size должен быть не меньше 32")
    if args.inference_interval < 1:
        raise ValueError("--inference-interval должен быть не меньше 1")
    if args.cpu_threads < 0:
        raise ValueError("--cpu-threads не может быть отрицательным")
    if args.alert_cooldown < 0:
        raise ValueError("--alert-cooldown не может быть отрицательным")
    if args.battery_capacity_mah < 0:
        raise ValueError("--battery-capacity-mah не может быть отрицательной")
    if args.alert_wav and not args.alert_wav.is_file():
        raise FileNotFoundError(f"WAV-файл не найден: {args.alert_wav}")
    if args.monitor:
        monitor_width, monitor_height, monitor_x, monitor_y = resolve_monitor_geometry(args.monitor)
        args.window_width = monitor_width
        args.window_height = monitor_height
        args.window_x = monitor_x
        args.window_y = monitor_y
        print(
            f"Монитор {args.monitor}: {monitor_width}x{monitor_height}+{monitor_x}+{monitor_y}",
            flush=True,
        )

    try:
        configure_local_caches()
        import cv2
        print("Загрузка локальной модели YOLO на CPU...", flush=True)
        if model_path.suffix.lower() == ".onnx":
            from src.onnx_detector import OnnxDetector
        else:
            from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Зависимости не установлены. Выполните: ./scripts/setup_ubuntu.sh") from error

    if args.cpu_threads and model_path.suffix.lower() != ".onnx":
        import torch

        torch.set_num_threads(args.cpu_threads)
        print(f"Потоки Torch на CPU: {args.cpu_threads}", flush=True)
    model = None
    onnx_detector = None
    if model_path.suffix.lower() == ".onnx":
        onnx_detector = OnnxDetector(
            model_path,
            args.confidence,
            args.generic_label,
            args.inference_size,
        )
    else:
        model = YOLO(str(model_path))
    if model is not None and args.labels == "ru":
        names = dict(model.model.names)
        class_count = len(names)
        if class_count == 1:
            model.model.names = {0: "Фипик"}
        elif class_count == len(RUSSIAN_LABELS):
            model.model.names = dict(enumerate(RUSSIAN_LABELS))
        else:
            model.model.names = {
                index: RUSSIAN_MILITARY_LABELS.get(
                    str(name), RUSSIAN_DRONE_LABELS.get(str(name), str(name))
                )
                for index, name in names.items()
            }
    print("Модель загружена. Открываю камеру...", flush=True)
    capture = open_capture(resolve_source(parse_source(args.source)))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть источник камеры: {args.source}")
    configure_low_latency_capture(capture, args.camera_fps)
    print(
        f"Камера: {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS",
        flush=True,
    )
    inav_reader = None
    if args.inav_port:
        from src.inav_msp import InavMspReader

        inav_reader = InavMspReader(args.inav_port, args.inav_baudrate)
        print(f"INAV подключён: {args.inav_port} (только чтение MSP)", flush=True)
    framebuffer_output = None
    drm_output = None
    if args.drm:
        from src.drm_output import DrmOutput

        drm_output = DrmOutput(args.drm)
        print(
            f"DRM подключён: {args.drm} Composite {drm_output.width}x{drm_output.height}",
            flush=True,
        )
    elif args.framebuffer:
        from src.framebuffer_output import FramebufferOutput

        framebuffer_output = FramebufferOutput(args.framebuffer)
        print(
            f"Framebuffer подключён: {framebuffer_output.device} "
            f"{framebuffer_output.width}x{framebuffer_output.height} "
            f"{framebuffer_output.bits_per_pixel} бит",
            flush=True,
        )
    fullscreen_pending = False
    if not args.headless:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.resolution and not args.monitor:
            args.window_width, args.window_height = args.resolution
        cv2.resizeWindow(WINDOW_NAME, args.window_width, args.window_height)
        if args.window_x is not None or args.window_y is not None:
            cv2.moveWindow(
                WINDOW_NAME,
                args.window_x or 0,
                args.window_y or 0,
            )
        # Полный экран включается после первого кадра, когда X11 уже знает монитор окна.
        fullscreen_pending = args.fullscreen

    writer: cv2.VideoWriter | None = create_writer(args.output, capture) if args.output else None
    frame_capture = LatestFrameCapture(capture)
    system_status = SystemStatusMonitor()
    predict_args = {"conf": args.confidence, "verbose": False}
    predict_args["imgsz"] = args.inference_size
    if args.device != "auto":
        predict_args["device"] = args.device
    def predict_frame(frame) -> tuple[Detection, ...]:
        """Выполняет один инференс и сохраняет только лёгкие данные рамок."""
        if onnx_detector is not None:
            return onnx_detector(frame)
        result = model.predict(frame, **predict_args)[0]
        return extract_detections(result, args.generic_label)

    inference_worker = LatestInferenceWorker(predict_frame)
    tracker = DetectionTracker()
    # Между результатами YOLO сопровождаем объект, не повторяя старую рамку.
    last_tracker_inference_sequence = -1
    # Храним последнюю позицию между редкими результатами YOLO.
    tracked_detections = ()
    last_osd_position: tuple[int, int] | None = None
    osd_pointer_visible = False
    last_target_confidence = 0.0
    last_status_log = 0.0
    last_osd_update = 0.0
    runtime_mtime = args.settings.stat().st_mtime_ns if args.settings.exists() else None
    last_settings_check = 0.0

    def reload_runtime_settings() -> None:
        """Подхватывает изменения панели без перезапуска видеопотока."""
        nonlocal runtime_mtime
        settings = load_settings(args.settings)
        detection = settings["detection"]
        args.confidence = max(0.01, min(1.0, float(detection["confidence_percent"]) / 100))
        args.inference_size = max(32, int(detection["inference_size"]))
        args.inference_interval = max(1, int(detection["inference_interval"]))
        args.generic_label = bool(detection["generic_label"])
        predict_args["conf"] = args.confidence
        predict_args["imgsz"] = args.inference_size
        apply_osd_settings(
            settings,
            {
                "object": OBJECT_STYLE,
                "text": OSD_TEXT,
                "system_status": SYSTEM_STATUS_STYLE,
                "horizon": HORIZON_STYLE,
                "axis": AXIS_STYLE,
                "flight_status": FLIGHT_STATUS_STYLE,
                "arm_banner": ARM_BANNER_STYLE,
            },
        )
        if onnx_detector is not None:
            onnx_detector.confidence = args.confidence
            onnx_detector.size = args.inference_size
            onnx_detector.generic_label = args.generic_label
        runtime_mtime = args.settings.stat().st_mtime_ns if args.settings.exists() else None

    frame_number = 0
    last_camera_sequence = 0
    last_inference_sequence = 0
    last_alert_at = 0.0
    detection_started_at: float | None = None
    alert_sent_for_detection = False
    alert_process: subprocess.Popen[bytes] | None = None
    try:
        while args.max_frames <= 0 or frame_number < args.max_frames:
            now_monotonic = time.monotonic()
            if now_monotonic - last_settings_check >= 1.0:
                last_settings_check = now_monotonic
                current_mtime = args.settings.stat().st_mtime_ns if args.settings.exists() else None
                if current_mtime != runtime_mtime:
                    reload_runtime_settings()
            # Raspberry-версия захвата хранит только последний кадр и предоставляет latest().
            frame = frame_capture.latest()
            camera_sequence = frame_number
            new_camera_frame = True
            if new_camera_frame:
                last_camera_sequence = camera_sequence
            if new_camera_frame and camera_sequence % args.inference_interval == 0:
                inference_worker.submit(frame)
            inference = inference_worker.latest()
            if inference.sequence != last_tracker_inference_sequence:
                # Обновляем трекер только один раз на новый результат YOLO.
                # Иначе быстрый главный цикл преждевременно исчерпывал пропуски.
                tracked_detections = tracker.update(inference.detections)
                last_tracker_inference_sequence = inference.sequence
                if inav_reader and not tracked_detections and osd_pointer_visible:
                    # Штатный Betaflight OSD хранит последний символ.
                    # Явно скрываем его, когда объект исчез из кадра.
                    if last_osd_position is not None:
                        inav_reader.set_crosshairs_position(
                            *last_osd_position,
                            visible=False,
                        )
                    osd_pointer_visible = False
            # На J7 передаём чистый кадр: рамку рисует штатный OSD полётника.
            annotated = frame.copy()
            if inav_reader and tracked_detections:
                target = tracked_detections[0]
                last_target_confidence = target.confidence
                center_x = (target.x1 + target.x2) / 2
                center_y = (target.y1 + target.y2) / 2
                osd_position = (
                    round(center_x * OSD_MAX_X / max(1, frame.shape[1] - 1)),
                    round(center_y * OSD_MAX_Y / max(1, frame.shape[0] - 1)),
                )
                if osd_position != last_osd_position and now_monotonic - last_osd_update >= 0.02:
                    inav_reader.set_crosshairs_position(*osd_position)
                    last_osd_position = osd_position
                    last_osd_update = now_monotonic
                    osd_pointer_visible = True
            if now_monotonic - last_status_log >= 1.0:
                last_status_log = now_monotonic
                if osd_pointer_visible and last_osd_position is not None:
                    print(
                        "\033[32m"
                        f"OSD: X={last_osd_position[0]} Y={last_osd_position[1]} "
                        f"confidence={round(last_target_confidence * 100)}%"
                        "\033[0m",
                        flush=True,
                    )
                else:
                    best_confidence = getattr(onnx_detector, "last_best_confidence", 0.0)
                    print(
                        "\033[33m"
                        f"confidence={round(best_confidence * 100)}%"
                        "\033[0m",
                        flush=True,
                    )
            if inav_reader:
                # Читаем MSP для связи, но штатный OSD уже находится в видеокадре.
                inav_reader.update()
            now = time.monotonic()
            # Задержку сигнала считаем только по новым результатам нейросети.
            if inference.sequence != last_inference_sequence:
                last_inference_sequence = inference.sequence
                if inference.detections:
                    if detection_started_at is None:
                        detection_started_at = inference.completed_at
                        alert_sent_for_detection = False
                else:
                    detection_started_at = None
                    alert_sent_for_detection = False

                if args.alert_wav and detection_started_at is not None:
                    held_for = inference.completed_at - detection_started_at
                    process_finished = alert_process is None or alert_process.poll() is not None
                    if (
                        held_for >= args.alert_delay
                        and not alert_sent_for_detection
                        and process_finished
                        and now - last_alert_at >= args.alert_cooldown
                    ):
                        alert_process = play_alert(args.alert_wav)
                        last_alert_at = now
                        alert_sent_for_detection = True
            if writer:
                writer.write(annotated)
            if framebuffer_output:
                framebuffer_output.write(annotated)
            if drm_output:
                drm_output.write(annotated)
            frame_number += 1

            if not args.headless:
                display_frame = black_background(annotated, args.window_width, args.window_height)
                cv2.imshow(WINDOW_NAME, display_frame)
                if fullscreen_pending:
                    cv2.waitKey(1)
                    if args.window_x is not None or args.window_y is not None:
                        cv2.moveWindow(WINDOW_NAME, args.window_x or 0, args.window_y or 0)
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    fullscreen_pending = False
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                    target = args.snapshot_dir / f"frame_{frame_number:06d}.jpg"
                    cv2.imwrite(str(target), annotated)
                    print(f"Снимок сохранён: {target}")
    finally:
        inference_worker.close()
        frame_capture.close()
        capture.release()
        if inav_reader:
            inav_reader.close()
        if writer:
            writer.release()
        if framebuffer_output:
            framebuffer_output.close()
        if drm_output:
            drm_output.close()
        cv2.destroyAllWindows()
    print(f"Обработано кадров: {frame_number}")
    return 0


def main() -> int:
    """Точка входа приложения."""
    args = build_parser().parse_args()
    if args.list_cameras:
        list_cameras()
        return 0
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Остановлено пользователем.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
