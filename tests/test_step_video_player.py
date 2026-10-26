"""Проверяет шаги времени плеера без открытия окна и внешнего видеоустройства."""

import unittest

from scripts.step_video_player import (
    NANOSECONDS_PER_SECOND,
    format_time_ns,
    stepped_position_ns,
    timeline_position_ns,
)


class StepVideoPlayerTests(unittest.TestCase):
    """Проверяет односекундный переход и формат позиции ролика."""

    def test_forward_and_backward_move_exactly_one_second(self) -> None:
        """Каждое нажатие сдвигает позицию ровно на одну секунду."""
        current = 12 * NANOSECONDS_PER_SECOND + 250_000_000

        self.assertEqual(
            stepped_position_ns(current, 30 * NANOSECONDS_PER_SECOND, 1),
            13 * NANOSECONDS_PER_SECOND + 250_000_000,
        )
        self.assertEqual(
            stepped_position_ns(current, 30 * NANOSECONDS_PER_SECOND, -1),
            11 * NANOSECONDS_PER_SECOND + 250_000_000,
        )

    def test_step_stays_inside_video_bounds(self) -> None:
        """Назад не уходит до начала, а вперёд не перескакивает длительность."""
        duration = 4 * NANOSECONDS_PER_SECOND

        self.assertEqual(stepped_position_ns(0, duration, -1), 0)
        self.assertEqual(
            stepped_position_ns(3 * NANOSECONDS_PER_SECOND + 500_000_000, duration, 1),
            duration,
        )

    def test_unknown_duration_still_clamps_start(self) -> None:
        """При неизвестной длине ролика ограничивается хотя бы нижняя граница."""
        self.assertEqual(stepped_position_ns(250_000_000, None, -1), 0)
        self.assertEqual(
            stepped_position_ns(250_000_000, None, 1),
            1_250_000_000,
        )

    def test_invalid_step_direction_is_rejected(self) -> None:
        """Принимаются только кнопки шага назад и вперёд."""
        with self.assertRaisesRegex(ValueError, "-1 или 1"):
            stepped_position_ns(0, None, 0)

    def test_position_format_is_readable_for_short_and_long_videos(self) -> None:
        """Время короче часа показывается как ММ:СС, длинное — с часами."""
        self.assertEqual(format_time_ns(65 * NANOSECONDS_PER_SECOND), "01:05")
        self.assertEqual(format_time_ns(3_661 * NANOSECONDS_PER_SECOND), "01:01:01")
        self.assertEqual(format_time_ns(None), "--:--")

    def test_timeline_maps_slider_to_video_time(self) -> None:
        """Начало, середина и конец шкалы выбирают соответствующее время видео."""
        duration = 80 * NANOSECONDS_PER_SECOND

        self.assertEqual(timeline_position_ns(0.0, duration), 0)
        self.assertEqual(timeline_position_ns(0.5, duration), 40 * NANOSECONDS_PER_SECOND)
        self.assertEqual(timeline_position_ns(1.0, duration), duration)

    def test_timeline_clamps_out_of_range_and_rejects_unknown_duration(self) -> None:
        """Шкала не выходит за границы ролика и недоступна без его длительности."""
        duration = 10 * NANOSECONDS_PER_SECOND

        self.assertEqual(timeline_position_ns(-0.5, duration), 0)
        self.assertEqual(timeline_position_ns(1.5, duration), duration)
        self.assertIsNone(timeline_position_ns(0.5, None))
        self.assertIsNone(timeline_position_ns(0.5, 0))

    def test_timeline_rejects_non_finite_position(self) -> None:
        """Защищает перемотку от NaN и бесконечности в значении ползунка."""
        with self.assertRaisesRegex(ValueError, "конечным числом"):
            timeline_position_ns(float("nan"), NANOSECONDS_PER_SECOND)


if __name__ == "__main__":
    unittest.main()
