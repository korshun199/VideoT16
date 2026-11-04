"""Точечное подавление ложной рамки на знакомой надписи AIR в верхней части кадра."""

from __future__ import annotations

import base64
import zlib

import cv2
import numpy as np

from detector import Detection


# Контуры букв из проверенного примера OSD; это не фотография и не модель.
_AIR_EDGES = (
    "eNrtl1uuAyEMQ3P2v+nqoqE8Co5v6SdRH1LHjYkJSYi4du1nRrETEJ15jO3zO9Dws8FaEPW9Rw1fnw/7GHPFCubvpVHFITsPvcTJ6ouz6jNDGZRPkJliLP63RCWUVVmc5T+4oyjrPuIt3wkzpWxyBbkrTFxGWTPMdXVMCcmmf2gilbUow6N8iPCAjrAmJU2aA8ow0odae1JlnUPSEnFP2ZMlW2BUn7dIqvwMGabD9IXVBY++g+g98GssiF5Sz23rO/v1OcJC60xpsZsW8W2NzVvSwsMJZXRpqJNnGFu2SzQp0S2JWXaFNzNWty66YkfaA/5TY9nPSKOSyKPpUWJJ1R1hPWxYUSKlpT2fBi/26JxSdRJinv+I2I6EqGmRfh7Q0+cUUq3KHno9V6Nm7MUVQt0qnDtJjOXz2rVrvr0AUCECTg=="
)
_AIR_EDGES_1280 = "eNrtlEEShSAMQ/vuf2lnFARpAujyz+/OFtI0xEb8YDCrneEKY8oUHn0wBUQKz4842yALCYzGUPenOzXcHPLXZ2mOvuLAQoFBTPoXzuyC3VC4Wd6ALW8ICTyYYUan2TaY0+wLGNVK2CkzmgHrHSV8WVuxy+z2gHZ0vAC7CJi3xFBb+kzLH/IJVj7Tg5SFMmo6AbM/IPi/OuO3QdN66k5nUeUKqjmQquBWYZ1dLEw0syaj2tE25Rf3PzbjAEFwAUs="
_AIR_EDGES_960 = "eNrVk9sOgCAMQzn//9MGEd1KF3wzzoQItN2FrbUPDXb33TxPzjPy+l3J3Es6Iu4I0AgDccr5pZ2Jlnm580mBSprqc8QlTJqRc8xmmLxgsjBHEKF8RYWuFcmyK2Y9F+2ibqAVU7PEvcPWJ0/+7JjymLNCgsUyUzPHvqXoodD2mKGQ6Uh9OorvJ+0HdgDppQDS"


def _reference(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    """Восстанавливает небольшой эталон контуров или сообщает о повреждении."""
    data = zlib.decompress(base64.b64decode(encoded, validate=True))
    if len(data) != shape[0] * shape[1]:
        raise ValueError("Повреждён образец надписи AIR")
    return np.frombuffer(data, dtype=np.uint8).reshape(shape) * 255


def _contains_air(candidate: Detection, air_box: tuple[int, int, int, int]) -> bool:
    """Отбирает только компактную рамку, охватывающую большую часть всей надписи."""
    x1, y1, x2, y2 = air_box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return False
    overlap_w = max(0, min(candidate.right, x2) - max(candidate.left, x1))
    overlap_h = max(0, min(candidate.bottom, y2) - max(candidate.top, y1))
    box_area = max(0, candidate.right - candidate.left) * max(0, candidate.bottom - candidate.top)
    return overlap_w * overlap_h >= 0.65 * width * height and box_area <= 7 * width * height


class AirFilter:
    """Проверяет образец AIR и исключает только совпадающие рамки модели."""

    def __init__(self) -> None:
        """Загружает эталон один раз, без обращения к сети и файлам пользователя."""
        self.references = (
            (1.0, _reference(_AIR_EDGES, (41, 115))),
            (2 / 3, _reference(_AIR_EDGES_1280, (27, 77))),
            (0.5, _reference(_AIR_EDGES_960, (21, 58))),
        )

    def filter(
        self, image: np.ndarray, candidates: tuple[Detection, ...]
    ) -> tuple[tuple[Detection, ...], tuple[Detection, ...], float]:
        """Возвращает оставшиеся и исключённые рамки и оценку сходства AIR."""
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Фильтр AIR получил некорректный кадр")
        height, width = image.shape[:2]
        # Ищем только при наличии рамки в верхней половине и недалеко от центра.
        eligible = tuple(item for item in candidates if
                         item.top < height * 0.27 and item.bottom > 0
                         and item.right > width * 0.25 and item.left < width * 0.75)
        if not eligible:
            return candidates, (), 0.0
        base_scale, reference = min(self.references, key=lambda entry: abs(entry[0] - width / 1920.0))
        best_score = 0.0
        best_box = None
        for item in eligible:
            margin = max(8, int(width * 0.015))
            left, top = max(0, item.left - margin), max(0, item.top - margin)
            right, bottom = min(width, item.right + margin), min(height, item.bottom + margin)
            if right <= left or bottom <= top:
                continue
            grey = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(grey, 80, 160)
            for multiplier in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
                scale = width / 1920.0 / base_scale * multiplier
                tw = max(1, round(reference.shape[1] * scale))
                th = max(1, round(reference.shape[0] * scale))
                if tw > edges.shape[1] or th > edges.shape[0] or tw < 8 or th < 4:
                    continue
                template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_NEAREST)
                _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(edges, template, cv2.TM_CCOEFF_NORMED))
                if score > best_score:
                    best_score = float(score)
                    x, y = left + location[0], top + location[1]
                    best_box = (x, y, x + tw, y + th)
        if best_box is None or best_score < 0.75:
            return candidates, (), best_score
        removed = tuple(item for item in eligible if _contains_air(item, best_box))
        return tuple(item for item in candidates if item not in removed), removed, best_score
