#!/usr/bin/env python3
"""Оконная запись видео с USB-преобразователя MacroSilicon."""

import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


# ============================================================
#                         НАСТРОЙКИ
# ============================================================

# USB-видеопреобразователь.
КАМЕРА = "/dev/video0"

# Параметры входного видео и готовой записи.
ШИРИНА = 1280
ВЫСОТА = 720
КАДРОВ_В_СЕКУНДУ = 30
КОДЕК = "mp4v"

# Каталог для готовых видеозаписей.
КАТАЛОГ_ЗАПИСЕЙ = Path(__file__).resolve().parent / "recordings"

# Размер оконного предпросмотра на экране ноутбука.
ШИРИНА_ОКНА = 1000
ВЫСОТА_ПРЕДПРОСМОТРА = 560
ИНТЕРВАЛ_ОБНОВЛЕНИЯ_МС = 10

# Цвета интерфейса.
ЦВЕТ_ФОНА = "#181818"
ЦВЕТ_ТЕКСТА = "#f0f0f0"
ЦВЕТ_ЗАПИСИ = "#d62828"
ЦВЕТ_ГОТОВНОСТИ = "#2a9d4b"


def открыть_камеру():
    """Открывает видеопреобразователь с минимальной очередью кадров."""

    камера = cv2.VideoCapture(КАМЕРА, cv2.CAP_V4L2)
    камера.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    камера.set(cv2.CAP_PROP_FRAME_WIDTH, ШИРИНА)
    камера.set(cv2.CAP_PROP_FRAME_HEIGHT, ВЫСОТА)
    камера.set(cv2.CAP_PROP_FPS, КАДРОВ_В_СЕКУНДУ)
    камера.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not камера.isOpened():
        raise RuntimeError(f"Не удалось открыть видеоустройство {КАМЕРА}")

    return камера


def создать_запись(ширина, высота, fps):
    """Создаёт новый MP4-файл и возвращает объект записи и его путь."""

    КАТАЛОГ_ЗАПИСЕЙ.mkdir(parents=True, exist_ok=True)
    метка_времени = datetime.now().strftime("%Y%m%d_%H%M%S")
    путь = КАТАЛОГ_ЗАПИСЕЙ / f"video_{метка_времени}.mp4"
    кодек = cv2.VideoWriter_fourcc(*КОДЕК)
    запись = cv2.VideoWriter(str(путь), кодек, fps, (ширина, высота))

    if not запись.isOpened():
        raise RuntimeError(f"Не удалось создать видеофайл {путь}")

    return запись, путь


class ПрограммаЗаписи:
    """Показывает видео и управляет записью одной кнопкой."""

    def __init__(self, окно):
        self.окно = окно
        self.камера = открыть_камеру()
        self.запись = None
        self.путь = None
        self.начало_записи = None
        self.кадров_записано = 0
        self.фото = None
        self.закрывается = False

        self.ширина = int(self.камера.get(cv2.CAP_PROP_FRAME_WIDTH)) or ШИРИНА
        self.высота = int(self.камера.get(cv2.CAP_PROP_FRAME_HEIGHT)) or ВЫСОТА
        self.fps = self.камера.get(cv2.CAP_PROP_FPS)

        if self.fps <= 0:
            self.fps = КАДРОВ_В_СЕКУНДУ

        self._создать_интерфейс()
        self._обновить_кадр()

    def _создать_интерфейс(self):
        """Создаёт окно предпросмотра и кнопки управления."""

        self.окно.title("VideoT16 — запись видео")
        self.окно.configure(bg=ЦВЕТ_ФОНА)
        self.окно.geometry(f"{ШИРИНА_ОКНА}x{ВЫСОТА_ПРЕДПРОСМОТРА + 115}+0+0")
        self.окно.minsize(720, 520)

        self.экран = tk.Label(self.окно, bg="#000000")
        self.экран.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))

        панель = tk.Frame(self.окно, bg=ЦВЕТ_ФОНА)
        панель.pack(fill=tk.X, padx=8, pady=8)

        self.кнопка_записи = tk.Button(
            панель,
            text="●  ЗАПИСЬ",
            command=self.начать_запись,
            bg=ЦВЕТ_ГОТОВНОСТИ,
            fg="white",
            activebackground="#207a3a",
            activeforeground="white",
            font=("DejaVu Sans", 14, "bold"),
            width=14,
            height=2,
        )
        self.кнопка_записи.pack(side=tk.LEFT)

        self.кнопка_стоп = tk.Button(
            панель,
            text="■  СТОП",
            command=self.остановить_запись,
            bg=ЦВЕТ_ЗАПИСИ,
            fg="white",
            activebackground="#a71f1f",
            activeforeground="white",
            disabledforeground="#aaaaaa",
            font=("DejaVu Sans", 14, "bold"),
            width=12,
            height=2,
            state=tk.DISABLED,
        )
        self.кнопка_стоп.pack(side=tk.LEFT, padx=(8, 0))

        кнопка_выхода = tk.Button(
            панель,
            text="ВЫХОД",
            command=self.закрыть,
            bg="#444444",
            fg="white",
            activebackground="#666666",
            activeforeground="white",
            font=("DejaVu Sans", 12, "bold"),
            width=10,
            height=2,
        )
        кнопка_выхода.pack(side=tk.RIGHT)

        self.статус = tk.Label(
            панель,
            text=f"ГОТОВО  •  {self.ширина}×{self.высота}  •  {self.fps:.0f} FPS",
            bg=ЦВЕТ_ФОНА,
            fg=ЦВЕТ_ТЕКСТА,
            font=("DejaVu Sans", 12),
            anchor="w",
        )
        self.статус.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=18)

        self.окно.protocol("WM_DELETE_WINDOW", self.закрыть)
        self.окно.bind("<Escape>", lambda _event: self.закрыть())
        self.окно.bind("q", lambda _event: self.закрыть())
        self.окно.bind("Q", lambda _event: self.закрыть())
        self.окно.bind("<space>", lambda _event: self.переключить_запись())

    def переключить_запись(self):
        """Переключает запись клавишей пробела."""

        if self.запись is None:
            self.начать_запись()
            return

        self.остановить_запись()

    def начать_запись(self):
        """Начинает новую запись после нажатия зелёной кнопки."""

        if self.запись is not None:
            return

        try:
            self.запись, self.путь = создать_запись(
                self.ширина,
                self.высота,
                self.fps,
            )
        except RuntimeError as ошибка:
            messagebox.showerror("Ошибка записи", str(ошибка))
            return

        self.начало_записи = time.monotonic()
        self.кадров_записано = 0
        self.кнопка_записи.configure(state=tk.DISABLED)
        self.кнопка_стоп.configure(state=tk.NORMAL)
        print(f"Запись началась: {self.путь}")

    def остановить_запись(self):
        """Закрывает текущий видеофайл и возвращает кнопку в исходное состояние."""

        if self.запись is None:
            return

        прошло = time.monotonic() - self.начало_записи
        self.запись.release()
        self.запись = None
        self.кнопка_записи.configure(state=tk.NORMAL)
        self.кнопка_стоп.configure(state=tk.DISABLED)
        self.статус.configure(
            text=f"СОХРАНЕНО: {self.путь.name}  •  {прошло:.1f} с",
            fg=ЦВЕТ_ГОТОВНОСТИ,
        )
        print(f"Запись завершена: {self.путь}")

    def _обновить_кадр(self):
        """Читает свежий кадр, показывает его и при необходимости записывает."""

        if self.закрывается:
            return

        ok, кадр = self.камера.read()

        if ok and кадр is not None:
            if self.запись is not None:
                self.запись.write(кадр)
                self.кадров_записано += 1
                прошло = time.monotonic() - self.начало_записи
                self.статус.configure(
                    text=(
                        f"● ЗАПИСЬ  {прошло:06.1f} с  •  "
                        f"{self.кадров_записано} кадров  •  {self.путь.name}"
                    ),
                    fg=ЦВЕТ_ЗАПИСИ,
                )

            ширина_экрана = max(self.экран.winfo_width(), 640)
            высота_экрана = max(self.экран.winfo_height(), 360)
            масштаб = min(
                ширина_экрана / кадр.shape[1],
                высота_экрана / кадр.shape[0],
            )
            новая_ширина = max(1, int(кадр.shape[1] * масштаб))
            новая_высота = max(1, int(кадр.shape[0] * масштаб))
            предпросмотр = cv2.resize(кадр, (новая_ширина, новая_высота))
            предпросмотр = cv2.cvtColor(предпросмотр, cv2.COLOR_BGR2RGB)
            self.фото = ImageTk.PhotoImage(Image.fromarray(предпросмотр))
            self.экран.configure(image=self.фото)
        else:
            self.статус.configure(
                text="Нет видеосигнала от преобразователя",
                fg=ЦВЕТ_ЗАПИСИ,
            )

        self.окно.after(ИНТЕРВАЛ_ОБНОВЛЕНИЯ_МС, self._обновить_кадр)

    def закрыть(self):
        """Безопасно закрывает запись, камеру и окно."""

        if self.закрывается:
            return

        self.закрывается = True
        self.остановить_запись()
        self.камера.release()
        self.окно.destroy()


def main():
    """Запускает оконное приложение записи."""

    окно = tk.Tk()

    try:
        ПрограммаЗаписи(окно)
    except RuntimeError as ошибка:
        окно.withdraw()
        messagebox.showerror("VideoT16", str(ошибка))
        окно.destroy()
        return 1

    окно.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
