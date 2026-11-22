"""Изолированное экспериментальное сопровождение без ожидания в окне."""

from collections import deque
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Callable

from observe_kcf import KcfOverlay


class AsyncKcf:
    """Обрабатывает ограниченный поток кадров и отдаёт рамку по точному номеру."""

    def __init__(self, library: str, max_age_ms: int, log: Any,
                 tracker_factory: Callable[..., Any] = KcfOverlay,
                 queue_size: int = 32, history_size: int = 24) -> None:
        """Проверяет зависимости до запуска и ограничивает память для истории."""
        if queue_size < 1 or history_size < 2:
            raise ValueError('Нужны положительная очередь и история от двух кадров')
        self.tracker = tracker_factory(library, max_age_ms)
        self.log = log
        self.queue: Queue[tuple[str, Any, int]] = Queue(maxsize=queue_size)
        self.history: deque[Any] = deque(maxlen=history_size)
        self.outputs: dict[int, tuple[Any, tuple[Any, ...]]] = {}
        self.lock = Lock()
        self.stopping = Event()
        self.disabled = False
        self.generation = 0
        self.active_generation = -1
        self.last_result_number = 0
        self.cutoffs: dict[int, int] = {}
        self.thread = Thread(target=self._run, name='experimental-kcf', daemon=True)
        self.history_size = history_size

    def start(self) -> None:
        """Запускает единственного владельца нативного трекера."""
        self.thread.start()

    def _offer(self, kind: str, value: Any) -> None:
        """Не задерживает источник и отключает модуль при потере событий."""
        with self.lock:
            if self.disabled or self.stopping.is_set():
                return
            if kind == 'result':
                if value.number <= self.last_result_number:
                    return
                self.last_result_number = value.number
                # Новая проверка отзывает только рамки начиная с её кадра.
                self.cutoffs[self.generation] = value.number
                self.generation += 1
                for number in [number for number in self.outputs if number >= value.number]:
                    self.outputs.pop(number, None)
            try:
                self.queue.put_nowait((kind, value, self.generation))
            except Full:
                self.disabled = True
                self.outputs.clear()
                self.log.write('experimental_kcf_disabled', reason='queue_full')

    def submit_frame(self, frame: Any) -> None:
        """Передаёт ссылку на неизменяемый исходный кадр без ожидания."""
        self._offer('frame', frame)

    def submit_result(self, result: Any) -> None:
        """Передаёт результат модели, включая пустой, в порядке получения."""
        self._offer('result', result)

    def take(self, frame_number: int) -> tuple[Any, tuple[Any, ...]] | None:
        """Возвращает только результат для данного кадра, никогда не ожидая его."""
        with self.lock:
            if self.disabled or self.stopping.is_set():
                return None
            result = self.outputs.pop(frame_number, None)
            for old in [number for number in self.outputs if number < frame_number]:
                self.outputs.pop(old, None)
            return result

    def _put_output(self, frame: Any, boxes: tuple[Any, ...]) -> None:
        """Ограничивает число результатов и сохраняет исходное подтверждение."""
        if not boxes:
            return
        with self.lock:
            cutoff = self.cutoffs.get(self.active_generation)
            if (self.disabled or self.stopping.is_set() or self.active_generation < 0
                    or (cutoff is not None and frame.number >= cutoff)):
                return
            self.outputs[frame.number] = (self.tracker.anchor, boxes)
            while len(self.outputs) > self.history_size:
                self.outputs.pop(min(self.outputs))

    def _handle_result(self, result: Any) -> None:
        """Переинициализирует трекер на исходном кадре и догоняет историю."""
        self.tracker.reset()
        with self.lock:
            for number in [number for number in self.outputs if number >= result.number]:
                self.outputs.pop(number, None)
        frame = next((item for item in self.history if item.number == result.number), None)
        if frame is None or not self.tracker.seed(frame, result):
            return
        for next_frame in self.history:
            if next_frame.number > result.number:
                self._put_output(next_frame, self.tracker.update(next_frame))

    def _run(self) -> None:
        """В одном потоке обрабатывает события и изолирует ошибку эксперимента."""
        try:
            while not self.stopping.is_set():
                with self.lock:
                    if self.disabled:
                        break
                try:
                    kind, value, generation = self.queue.get(timeout=0.05)
                except Empty:
                    continue
                if kind == 'frame':
                    self.history.append(value)
                    self._put_output(value, self.tracker.update(value))
                else:
                    with self.lock:
                        if generation != self.generation:
                            continue
                        self.active_generation = generation
                        self.cutoffs = {key: value for key, value in self.cutoffs.items()
                                        if key >= generation}
                    self._handle_result(value)
        except Exception as error:
            with self.lock:
                self.disabled = True
                self.outputs.clear()
            self.log.write('experimental_kcf_disabled', reason='worker_error', message=str(error))
        finally:
            self.tracker.reset()
            self.history.clear()
            with self.lock:
                self.outputs.clear()
            while True:
                try:
                    self.queue.get_nowait()
                except Empty:
                    break

    def close(self) -> bool:
        """Завершает поток без бесконечного ожидания; ложь означает тайм-аут."""
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        return not self.thread.is_alive()
