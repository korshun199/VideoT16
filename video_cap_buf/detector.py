"""Обнаружение объектов ONNX Runtime/OpenCV DNN и нанесение рамок на кадры."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Описывает один найденный объект в координатах исходного кадра."""

    class_id: int
    label: str
    confidence: float
    left: int
    top: int
    right: int
    bottom: int


class OpenCVDnnDetector:
    """Загружает ONNX-модель и декодирует координаты для потока отображения."""

    def __init__(
        self,
        model_path: Path,
        input_size: int,
        confidence_percent: float,
        nms_iou: float,
        class_names: list[str],
        box_color_bgr: tuple[int, int, int],
        text_color_bgr: tuple[int, int, int],
        box_thickness: int,
        font_scale: float,
        threads: int,
        presence_confidence_percent: float | None = None,
        weak_candidate_color_bgr: tuple[int, int, int] = (255, 0, 255),
    ) -> None:
        """Проверяет параметры и загружает модель средствами OpenCV DNN."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if input_size <= 0:
            raise ValueError("inference_size должен быть положительным")
        if not 0 <= confidence_percent <= 100:
            raise ValueError("confidence_percent должен быть от 0 до 100")
        if presence_confidence_percent is None:
            presence_confidence_percent = confidence_percent
        if not 0 <= presence_confidence_percent <= confidence_percent:
            raise ValueError(
                "presence_confidence_percent должен быть от 0 до confidence_percent"
            )
        if not 0 <= nms_iou <= 1:
            raise ValueError("nms_iou должен быть от 0 до 1")
        if not class_names or any(not name.strip() for name in class_names):
            raise ValueError("class_names должен содержать непустые названия классов")
        if box_thickness <= 0 or font_scale <= 0 or threads <= 0:
            raise ValueError("Толщина рамки, размер шрифта и число потоков должны быть положительными")

        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_percent / 100.0
        self.presence_confidence_threshold = presence_confidence_percent / 100.0
        self.nms_iou = nms_iou
        self.class_names = tuple(class_names)
        self.box_color_bgr = box_color_bgr
        self.text_color_bgr = text_color_bgr
        self.weak_candidate_color_bgr = weak_candidate_color_bgr
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.threads = threads

        cv2.setNumThreads(threads)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, image: np.ndarray) -> tuple[tuple[Detection, ...], float]:
        """Распознаёт кадр и возвращает координаты без копирования полного изображения."""
        output, geometry, inference_ms = self._prepare_and_infer(image)
        detections = self._decode(output, *geometry)
        return detections, inference_ms

    def detect_with_presence(
        self, image: np.ndarray
    ) -> tuple[tuple[Detection, ...], tuple[Detection, ...], float]:
        """Одним инференсом оценивает присутствие и отдельно отбирает рамки для показа."""
        output, geometry, inference_ms = self._prepare_and_infer(image)
        presence_candidates = self._decode(
            output, *geometry, confidence_threshold=self.presence_confidence_threshold
        )
        # При NMS низкий порог не может удалить кандидата с большей
        # уверенностью, поэтому рамки извлекаются из уже отфильтрованного списка.
        localized_detections = tuple(
            candidate for candidate in presence_candidates
            if candidate.confidence >= self.confidence_threshold
        )
        return presence_candidates, localized_detections, inference_ms

    def _prepare_and_infer(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, float, int, int], float]:
        """Готовит один кадр и запускает модель ровно один раз."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Ожидался цветной BGR-кадр с тремя каналами")

        height, width = image.shape[:2]
        model_image, scale, pad_left, pad_top = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            model_image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        inference_started_ns = time.perf_counter_ns()
        output = self._run_inference(blob)
        inference_ms = (time.perf_counter_ns() - inference_started_ns) / 1_000_000
        geometry = (width, height, scale, pad_left, pad_top)
        return output, geometry, inference_ms

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """Выполняет прямой проход через OpenCV DNN; ORT переопределяет этот шов."""
        self.net.setInput(blob)
        return self.net.forward()

    def annotate(
        self,
        image: np.ndarray,
        detections: tuple[Detection, ...],
        weak_candidates: tuple[Detection, ...] = (),
    ) -> np.ndarray:
        """Создаёт копию кадра с зелёными рамками и слабыми пурпурными кандидатами."""
        return self._draw(image, detections, weak_candidates)

    def process(self, image: np.ndarray) -> tuple[np.ndarray, tuple[Detection, ...], float, float]:
        """Совместимо распознаёт кадр, рисует рамки и возвращает времена этапов."""
        detections, inference_ms = self.detect(image)
        drawing_started_ns = time.perf_counter_ns()
        annotated = self.annotate(image, detections)
        drawing_ms = (time.perf_counter_ns() - drawing_started_ns) / 1_000_000
        return annotated, detections, inference_ms, drawing_ms

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Масштабирует изображение без искажения пропорций и добавляет поля."""
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_left = (self.input_size - resized_width) // 2
        pad_top = (self.input_size - resized_height) // 2
        model_image = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        model_image[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
        return model_image, scale, pad_left, pad_top

    def _decode(
        self,
        output: np.ndarray,
        image_width: int,
        image_height: int,
        scale: float,
        pad_left: int,
        pad_top: int,
        confidence_threshold: float | None = None,
    ) -> tuple[Detection, ...]:
        """Декодирует выход YOLOv8 и применяет отдельное подавление дублей по классу."""
        prediction = np.asarray(output)
        if prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 2:
            raise ValueError(f"Неожиданная форма выхода модели: {prediction.shape}")

        expected_columns = 4 + len(self.class_names)
        if prediction.shape[0] == expected_columns:
            prediction = prediction.T
        elif prediction.shape[1] != expected_columns:
            raise ValueError(
                f"Выход модели {prediction.shape} не соответствует {len(self.class_names)} классам"
            )

        by_class: dict[int, list[tuple[tuple[int, int, int, int], float]]] = {}
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        scores = prediction[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        row_indexes = np.arange(prediction.shape[0])
        confidences = scores[row_indexes, class_ids]
        selected = np.flatnonzero(np.isfinite(confidences) & (confidences >= threshold))
        for row_index in selected:
            row = prediction[row_index]
            class_id = int(class_ids[row_index])
            confidence = float(confidences[row_index])

            center_x, center_y, box_width, box_height = (float(value) for value in row[:4])
            left = int(round((center_x - box_width / 2 - pad_left) / scale))
            top = int(round((center_y - box_height / 2 - pad_top) / scale))
            right = int(round((center_x + box_width / 2 - pad_left) / scale))
            bottom = int(round((center_y + box_height / 2 - pad_top) / scale))
            left = min(max(left, 0), image_width - 1)
            top = min(max(top, 0), image_height - 1)
            right = min(max(right, 0), image_width)
            bottom = min(max(bottom, 0), image_height)
            if right <= left or bottom <= top:
                continue

            by_class.setdefault(class_id, []).append(((left, top, right - left, bottom - top), confidence))

        detections: list[Detection] = []
        for class_id, candidates in by_class.items():
            boxes = [candidate[0] for candidate in candidates]
            confidences = [candidate[1] for candidate in candidates]
            kept = cv2.dnn.NMSBoxes(boxes, confidences, threshold, self.nms_iou)
            for kept_index in np.asarray(kept).reshape(-1):
                (left, top, box_width, box_height), confidence = candidates[int(kept_index)]
                detections.append(Detection(
                    class_id=class_id,
                    label=self.class_names[class_id],
                    confidence=confidence,
                    left=left,
                    top=top,
                    right=left + box_width,
                    bottom=top + box_height,
                ))

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return tuple(detections)

    def _draw(
        self,
        image: np.ndarray,
        detections: tuple[Detection, ...],
        weak_candidates: tuple[Detection, ...] = (),
    ) -> np.ndarray:
        """Рисует уверенные рамки и слабых кандидатов разными цветами."""
        annotated = image.copy()
        # Слабая группа рисуется первой; уверенная рамка остаётся визуально главной.
        drawing_groups = (
            (
                weak_candidates,
                self.weak_candidate_color_bgr,
                self.weak_candidate_color_bgr,
                "? ",
            ),
            (detections, self.box_color_bgr, self.text_color_bgr, ""),
        )
        for group, box_color, text_color, label_prefix in drawing_groups:
            for detection in group:
                cv2.rectangle(
                    annotated,
                    (detection.left, detection.top),
                    (detection.right, detection.bottom),
                    box_color,
                    self.box_thickness,
                )
                label = (
                    f"{label_prefix}{detection.label} "
                    f"{detection.confidence * 100:.0f}%"
                )
                text_y = detection.top - 8 if detection.top > 24 else detection.top + 22
                cv2.putText(
                    annotated,
                    label,
                    (detection.left, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    text_color,
                    max(1, self.box_thickness),
                    cv2.LINE_AA,
                )
        return annotated


class OnnxRuntimeDetector(OpenCVDnnDetector):
    """Выполняет тот же YOLO-инференс через отдельную сессию ONNX Runtime."""

    def __init__(
        self,
        model_path: Path,
        input_size: int,
        confidence_percent: float,
        nms_iou: float,
        class_names: list[str],
        box_color_bgr: tuple[int, int, int],
        text_color_bgr: tuple[int, int, int],
        box_thickness: int,
        font_scale: float,
        threads: int,
        presence_confidence_percent: float | None = None,
        weak_candidate_color_bgr: tuple[int, int, int] = (255, 0, 255),
    ) -> None:
        """Проверяет вход модели и создаёт CPU-сессию с заданным числом потоков."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        if input_size <= 0:
            raise ValueError("inference_size должен быть положительным")
        if not 0 <= confidence_percent <= 100:
            raise ValueError("confidence_percent должен быть от 0 до 100")
        if presence_confidence_percent is None:
            presence_confidence_percent = confidence_percent
        if not 0 <= presence_confidence_percent <= confidence_percent:
            raise ValueError(
                "presence_confidence_percent должен быть от 0 до confidence_percent"
            )
        if not 0 <= nms_iou <= 1:
            raise ValueError("nms_iou должен быть от 0 до 1")
        if not class_names or any(not name.strip() for name in class_names):
            raise ValueError("class_names должен содержать непустые названия классов")
        if box_thickness <= 0 or font_scale <= 0 or threads <= 0:
            raise ValueError("Толщина рамки, размер шрифта и число потоков должны быть положительными")

        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                "Для backend=onnxruntime установите onnxruntime в общее .venv"
            ) from error

        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_percent / 100.0
        self.presence_confidence_threshold = presence_confidence_percent / 100.0
        self.nms_iou = nms_iou
        self.class_names = tuple(class_names)
        self.box_color_bgr = box_color_bgr
        self.text_color_bgr = text_color_bgr
        self.weak_candidate_color_bgr = weak_candidate_color_bgr
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.threads = threads

        # Каждый экземпляр распознавателя имеет одну нить ORT; кадры
        # распараллеливаются между независимыми экземплярами модели.
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.providers = self.session.get_providers()

        # Не даём незаметно подать кадр не того статического размера.
        expected_shape = (1, 3, input_size, input_size)
        actual_shape = model_input.shape
        if len(actual_shape) != len(expected_shape) or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(actual_shape, expected_shape)
        ):
            raise ValueError(
                f"Размер входа модели {actual_shape} не совпадает с ожидаемым "
                f"{expected_shape}; выберите правильные model_path и inference_size"
            )
        if model_input.type != "tensor(float)":
            raise ValueError(
                f"Модель ожидает {model_input.type}, а подготовка подаёт tensor(float)"
            )

    def _run_inference(self, blob: np.ndarray) -> np.ndarray:
        """Выполняет прямой проход через уже загруженную сессию ONNX Runtime."""
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        return outputs[0]
