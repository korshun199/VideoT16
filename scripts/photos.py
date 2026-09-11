#!/usr/bin/env python3
"""Создаёт и показывает два синхронных видео из фотографий FPV."""

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.onnx_detector import OnnxDetector


# ============================================================
#                         НАСТРОЙКИ
# ============================================================

КОРЕНЬ = Path(__file__).resolve().parent
КАТАЛОГ_ФОТО = КОРЕНЬ / "../dataset/fpv/object_photos/images"
МОДЕЛЬ = КОРЕНЬ / "../runs/fpv_training/fpv_quadcopter_merged/weights/best.onnx"
КАТАЛОГ_ВИДЕО = КОРЕНЬ / "../dataset/fpv/object_photos/videos"

ВИДЕО_БЕЗ_РАСПОЗНАВАНИЯ = КАТАЛОГ_ВИДЕО / "clean.mp4"
ВИДЕО_С_РАСПОЗНАВАНИЕМ = КАТАЛОГ_ВИДЕО / "detected.mp4"

# Каждая фотография показывается одну секунду.
СЕКУНД_НА_ФОТО = 1.0
FPS_ВИДЕО = 5

# Итоговое разрешение обоих видео.
ШИРИНА_ВИДЕО = 1280
ВЫСОТА_ВИДЕО = 720

# Настройки распознавания.
ПОРОГ_УВЕРЕННОСТИ = 0.60
РАЗМЕР_МОДЕЛИ = 640
ПОДПИСЬ_ОБЪЕКТА = "FPV-DRON"

# Размер окна программы на ноутбуке.
# Размер окна программы на ноутбуке.
ШИРИНА_ОКНА = 1100
ВЫСОТА_ОКНА = 760


РАСШИРЕНИЯ_ФОТО = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def получить_фотографии():
    """Возвращает отсортированный список фотографий исходного каталога."""

    if not КАТАЛОГ_ФОТО.is_dir():
        raise RuntimeError(f"Каталог фотографий не найден: {КАТАЛОГ_ФОТО}")

    фотографии = sorted(
        путь
        for путь in КАТАЛОГ_ФОТО.iterdir()
        if путь.is_file() and путь.suffix.lower() in РАСШИРЕНИЯ_ФОТО
    )

    if not фотографии:
        raise RuntimeError(f"В каталоге нет фотографий: {КАТАЛОГ_ФОТО}")

    return фотографии


def вписать_в_кадр(изображение):
    """Вписывает фотографию в 1280×720 без искажения пропорций."""

    высота, ширина = изображение.shape[:2]
    масштаб = min(ШИРИНА_ВИДЕО / ширина, ВЫСОТА_ВИДЕО / высота)
    новая_ширина = max(1, round(ширина * масштаб))
    новая_высота = max(1, round(высота * масштаб))
    изображение = cv2.resize(
        изображение,
        (новая_ширина, новая_высота),
        interpolation=cv2.INTER_AREA,
    )
    кадр = np.zeros((ВЫСОТА_ВИДЕО, ШИРИНА_ВИДЕО, 3), dtype=np.uint8)
    x = (ШИРИНА_ВИДЕО - новая_ширина) // 2
    y = (ВЫСОТА_ВИДЕО - новая_высота) // 2
    кадр[y:y + новая_высота, x:x + новая_ширина] = изображение
    return кадр


def нарисовать_распознавание(изображение, обнаружения):
    """Рисует только реальные результаты модели и их уверенность."""

    результат = изображение.copy()

    for обнаружение in обнаружения:
        cv2.rectangle(
            результат,
            (обнаружение.x1, обнаружение.y1),
            (обнаружение.x2, обнаружение.y2),
            (0, 255, 0),
            4,
        )
        подпись = f"{обнаружение.name} {обнаружение.confidence * 100:.0f}%"
        cv2.putText(
            результат,
            подпись,
            (обнаружение.x1, max(30, обнаружение.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return результат


def открыть_writer(путь):
    """Открывает MP4-файл для записи и проверяет кодек."""

    writer = cv2.VideoWriter(
        str(путь),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS_ВИДЕО,
        (ШИРИНА_ВИДЕО, ВЫСОТА_ВИДЕО),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать видео: {путь}")

    return writer


def создать_два_видео(сообщения):
    """В фоне создаёт чистое и обработанное видео из одного набора фотографий."""

    чистая_часть = ВИДЕО_БЕЗ_РАСПОЗНАВАНИЯ.with_suffix(".part.mp4")
    обработанная_часть = ВИДЕО_С_РАСПОЗНАВАНИЕМ.with_suffix(".part.mp4")
    writer_чистый = None
    writer_обработанный = None

    try:
        фотографии = получить_фотографии()

        if not МОДЕЛЬ.is_file():
            raise RuntimeError(f"Модель не найдена: {МОДЕЛЬ}")

        КАТАЛОГ_ВИДЕО.mkdir(parents=True, exist_ok=True)
        detector = OnnxDetector(
            МОДЕЛЬ,
            ПОРОГ_УВЕРЕННОСТИ,
            False,
            РАЗМЕР_МОДЕЛИ,
            ПОДПИСЬ_ОБЪЕКТА,
        )
        writer_чистый = открыть_writer(чистая_часть)
        writer_обработанный = открыть_writer(обработанная_часть)
        повторов = max(1, round(СЕКУНД_НА_ФОТО * FPS_ВИДЕО))

        for номер, путь in enumerate(фотографии, 1):
            изображение = cv2.imread(str(путь))

            if изображение is None:
                сообщения.put(("текст", f"Пропуск повреждённого файла: {путь.name}"))
                continue

            обнаружения = detector(изображение)
            чистый_кадр = вписать_в_кадр(изображение)
            обработанный_кадр = вписать_в_кадр(
                нарисовать_распознавание(изображение, обнаружения)
            )

            for _ in range(повторов):
                writer_чистый.write(чистый_кадр)
                writer_обработанный.write(обработанный_кадр)

            сообщения.put(("прогресс", номер, len(фотографии), путь.name))

        writer_чистый.release()
        writer_чистый = None
        writer_обработанный.release()
        writer_обработанный = None
        чистая_часть.replace(ВИДЕО_БЕЗ_РАСПОЗНАВАНИЯ)
        обработанная_часть.replace(ВИДЕО_С_РАСПОЗНАВАНИЕМ)
        сообщения.put(("готово", len(фотографии)))

    except Exception as ошибка:
        сообщения.put(("ошибка", str(ошибка)))

    finally:
        if writer_чистый is not None:
            writer_чистый.release()
        if writer_обработанный is not None:
            writer_обработанный.release()


class ПросмотрДвухВидео:
    """Управляет созданием и синхронным просмотром двух вариантов видео."""

    def __init__(self, окно):
        self.окно = окно
        self.сообщения = queue.Queue()
        self.capture = None
        self.режим = "чистый"
        self.пауза = False
        self.фото = None
        self.закрывается = False
        self.последний_кадр = None

        self._создать_интерфейс()
        self._проверить_готовность()
        self._обработать_сообщения()

    def _создать_интерфейс(self):
        """Создаёт предпросмотр, прогресс и кнопки режимов."""

        self.окно.title("VideoT16 — тест распознавания на фотографиях")
        self.окно.geometry(f"{ШИРИНА_ОКНА}x{ВЫСОТА_ОКНА}+0+0")
        self.окно.configure(bg="#181818")

        self.экран = tk.Label(self.окно, bg="#000000")
        self.экран.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        self.прогресс = ttk.Progressbar(self.окно, mode="determinate")
        self.прогресс.pack(fill=tk.X, padx=8, pady=4)

        self.статус = tk.Label(
            self.окно,
            text="Подготовка...",
            bg="#181818",
            fg="#eeeeee",
            anchor="w",
            font=("DejaVu Sans", 11),
        )
        self.статус.pack(fill=tk.X, padx=10)

        панель = tk.Frame(self.окно, bg="#181818")
        панель.pack(fill=tk.X, padx=8, pady=8)

        self.кнопка_чистое = tk.Button(
            панель,
            text="БЕЗ РАСПОЗНАВАНИЯ",
            command=lambda: self.переключить_режим("чистый"),
            bg="#2878b5",
            fg="white",
            font=("DejaVu Sans", 11, "bold"),
            height=2,
        )
        self.кнопка_чистое.pack(side=tk.LEFT)

        self.кнопка_детектор = tk.Button(
            панель,
            text="С РАСПОЗНАВАНИЕМ",
            command=lambda: self.переключить_режим("детектор"),
            bg="#555555",
            fg="white",
            font=("DejaVu Sans", 11, "bold"),
            height=2,
        )
        self.кнопка_детектор.pack(side=tk.LEFT, padx=8)

        self.кнопка_пауза = tk.Button(
            панель,
            text="ПАУЗА",
            command=self.переключить_паузу,
            font=("DejaVu Sans", 11, "bold"),
            height=2,
        )
        self.кнопка_пауза.pack(side=tk.LEFT)

        self.кнопка_стоп = tk.Button(
            панель,
            text="СТОП",
            command=self.остановить,
            font=("DejaVu Sans", 11, "bold"),
            height=2,
        )
        self.кнопка_стоп.pack(side=tk.LEFT, padx=8)

        tk.Button(
            панель,
            text="ВЫХОД",
            command=self.закрыть,
            bg="#a72b2b",
            fg="white",
            font=("DejaVu Sans", 11, "bold"),
            height=2,
        ).pack(side=tk.RIGHT)

        self.окно.protocol("WM_DELETE_WINDOW", self.закрыть)
        self.окно.bind("<Escape>", lambda _event: self.закрыть())
        self.окно.bind("<space>", lambda _event: self.переключить_паузу())

    def _проверить_готовность(self):
        """Открывает готовые видео либо запускает их создание."""

        if ВИДЕО_БЕЗ_РАСПОЗНАВАНИЯ.is_file() and ВИДЕО_С_РАСПОЗНАВАНИЕМ.is_file():
            self.прогресс.configure(value=100, maximum=100)
            self.открыть_видео(0)
            return

        self._включить_кнопки(False)
        self.статус.configure(text="Создаю два видео. Это выполняется один раз...")
        threading.Thread(
            target=создать_два_видео,
            args=(self.сообщения,),
            daemon=True,
        ).start()

    def _включить_кнопки(self, включить):
        """Блокирует управление, пока видео ещё создаются."""

        состояние = tk.NORMAL if включить else tk.DISABLED
        self.кнопка_чистое.configure(state=состояние)
        self.кнопка_детектор.configure(state=состояние)
        self.кнопка_пауза.configure(state=состояние)
        self.кнопка_стоп.configure(state=состояние)

    def _обработать_сообщения(self):
        """Переносит прогресс фоновой генерации в интерфейс Tk."""

        while True:
            try:
                сообщение = self.сообщения.get_nowait()
            except queue.Empty:
                break

            if сообщение[0] == "прогресс":
                _, номер, всего, имя = сообщение
                self.прогресс.configure(maximum=всего, value=номер)
                осталось = всего - номер
                self.статус.configure(
                    text=f"Создание: {номер}/{всего} • осталось {осталось} • {имя}"
                )
            elif сообщение[0] == "текст":
                self.статус.configure(text=сообщение[1])
            elif сообщение[0] == "готово":
                self._включить_кнопки(True)
                self.статус.configure(
                    text=f"Готово: создано два видео из {сообщение[1]} фотографий"
                )
                self.открыть_видео(0)
            elif сообщение[0] == "ошибка":
                self.статус.configure(text=f"Ошибка: {сообщение[1]}", fg="#ff5555")
                messagebox.showerror("Ошибка", сообщение[1])

        if not self.закрывается:
            self.окно.after(100, self._обработать_сообщения)

    def путь_режима(self):
        """Возвращает файл текущего режима просмотра."""

        if self.режим == "детектор":
            return ВИДЕО_С_РАСПОЗНАВАНИЕМ
        return ВИДЕО_БЕЗ_РАСПОЗНАВАНИЯ

    def открыть_видео(self, номер_кадра):
        """Открывает выбранный вариант видео с указанного кадра."""

        if self.capture is not None:
            self.capture.release()

        self.capture = cv2.VideoCapture(str(self.путь_режима()))

        if not self.capture.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {self.путь_режима()}")

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, номер_кадра)
        self.пауза = False
        self.кнопка_пауза.configure(text="ПАУЗА")
        self._показать_следующий_кадр()

    def переключить_режим(self, режим):
        """Переключает чистое и распознанное видео на том же кадре."""

        if режим == self.режим or self.capture is None:
            return

        номер_кадра = max(0, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.режим = режим
        self.кнопка_чистое.configure(bg="#2878b5" if режим == "чистый" else "#555555")
        self.кнопка_детектор.configure(bg="#2a9d4b" if режим == "детектор" else "#555555")
        self.открыть_видео(номер_кадра)

    def переключить_паузу(self):
        """Ставит просмотр на паузу либо продолжает его."""

        if self.capture is None:
            return
        self.пауза = not self.пауза
        self.кнопка_пауза.configure(text="ПРОДОЛЖИТЬ" if self.пауза else "ПАУЗА")

    def остановить(self):
        """Возвращает текущее видео к первому кадру и ставит его на паузу."""

        if self.capture is None:
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.пауза = True
        self.кнопка_пауза.configure(text="ПРОДОЛЖИТЬ")
        self._прочитать_и_показать()

    def _прочитать_и_показать(self):
        """Читает один кадр текущего видео и выводит его в окно."""

        if self.capture is None:
            return False

        ok, кадр = self.capture.read()

        if not ok or кадр is None:
            return False

        self.последний_кадр = кадр
        ширина_экрана = max(self.экран.winfo_width(), 640)
        высота_экрана = max(self.экран.winfo_height(), 360)
        масштаб = min(
            ширина_экрана / кадр.shape[1],
            высота_экрана / кадр.shape[0],
        )
        размер = (
            max(1, int(кадр.shape[1] * масштаб)),
            max(1, int(кадр.shape[0] * масштаб)),
        )
        изображение = cv2.resize(кадр, размер, interpolation=cv2.INTER_AREA)
        изображение = cv2.cvtColor(изображение, cv2.COLOR_BGR2RGB)
        self.фото = ImageTk.PhotoImage(Image.fromarray(изображение))
        self.экран.configure(image=self.фото)

        текущий = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        всего = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.статус.configure(
            text=(
                f"{'С РАСПОЗНАВАНИЕМ' if self.режим == 'детектор' else 'БЕЗ РАСПОЗНАВАНИЯ'}"
                f" • кадр {текущий}/{всего}"
            ),
            fg="#eeeeee",
        )
        return True

    def _показать_следующий_кадр(self):
        """Поддерживает воспроизведение текущего видео с заданным FPS."""

        if self.закрывается or self.capture is None:
            return

        if not self.пауза and not self._прочитать_и_показать():
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._прочитать_и_показать()

        self.окно.after(max(1, round(1000 / FPS_ВИДЕО)), self._показать_следующий_кадр)

    def закрыть(self):
        """Освобождает видео и закрывает программу."""

        self.закрывается = True
        if self.capture is not None:
            self.capture.release()
        self.окно.destroy()


def main():
    """Запускает оконное создание и сравнение двух видео."""

    окно = tk.Tk()

    try:
        ПросмотрДвухВидео(окно)
    except RuntimeError as ошибка:
        окно.withdraw()
        messagebox.showerror("VideoT16", str(ошибка))
        окно.destroy()
        return 1

    окно.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
