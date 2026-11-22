//! Контроль времени и корректности рамок для системы наблюдения VideoT16.
//!
//! Модуль не читает видео, не выполняет распознавание или KCF и не связан
//! с управлением оборудованием. Он принимает уже полученные наблюдения.

/// Рамка в пикселях исходного кадра; правая и нижняя границы исключены.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl BoundingBox {
    /// Проверяет, что рамка имеет положительную площадь и помещается в кадре.
    pub fn is_valid(self, width: u32, height: u32) -> bool {
        width > 0 && height > 0 && self.left < self.right && self.top < self.bottom
            && self.right <= width && self.bottom <= height
    }
}

/// Происхождение отображаемой рамки.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Результат детектора на текущем кадре.
    Detection,
    /// Неподтверждённое последующим распознаванием сопровождение.
    Tracking,
}

/// Явно маркированная рамка текущего кадра.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Overlay {
    pub frame: u64,
    pub anchor_frame: u64,
    pub age_ns: u64,
    pub bbox: BoundingBox,
    pub origin: Origin,
}

/// Ошибка порядка кадров или некорректного наблюдения.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationError {
    FrameOrder,
    TimeOrder,
    InvalidBox,
    InvalidConfidence,
    NotCurrentFrame,
}

/// Ограничивает возраст результата без удержания старых координат.
pub struct ObservationGate {
    max_age_ns: u64,
    latest_frame: Option<u64>,
    latest_time_ns: Option<u64>,
    anchor: Option<(u64, u64)>,
}

impl ObservationGate {
    /// Создаёт пустой контроллер с максимальным временем подтверждения.
    pub fn new(max_age_ns: u64) -> Self {
        Self { max_age_ns, latest_frame: None, latest_time_ns: None, anchor: None }
    }

    /// Принимает только следующий кадр и проверяет монотонность времени.
    pub fn begin_frame(&mut self, frame: u64, time_ns: u64) -> Result<(), ObservationError> {
        if self.latest_frame.map_or(false, |last| frame <= last) {
            return Err(ObservationError::FrameOrder);
        }
        if self.latest_time_ns.map_or(false, |last| time_ns <= last) {
            return Err(ObservationError::TimeOrder);
        }
        self.latest_frame = Some(frame);
        self.latest_time_ns = Some(time_ns);
        Ok(())
    }

    /// Принимает подтверждённую рамку только для текущего кадра.
    pub fn detect(&mut self, frame: u64, bbox: BoundingBox, width: u32, height: u32,
                  confidence: f32) -> Result<Overlay, ObservationError> {
        let time = self.current_time(frame)?;
        if !bbox.is_valid(width, height) { return Err(ObservationError::InvalidBox); }
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ObservationError::InvalidConfidence);
        }
        self.anchor = Some((frame, time));
        Ok(Overlay { frame, anchor_frame: frame, age_ns: 0, bbox, origin: Origin::Detection })
    }

    /// Выдаёт рамку только при наличии нового результата сопровождения.
    pub fn track(&mut self, frame: u64, bbox: BoundingBox, width: u32, height: u32)
        -> Result<Option<Overlay>, ObservationError>
    {
        let time = self.current_time(frame)?;
        if !bbox.is_valid(width, height) { return Err(ObservationError::InvalidBox); }
        let Some((anchor_frame, anchor_time)) = self.anchor else { return Ok(None) };
        let age = time.saturating_sub(anchor_time);
        if age > self.max_age_ns {
            self.anchor = None;
            return Ok(None);
        }
        Ok(Some(Overlay { frame, anchor_frame, age_ns: age, bbox, origin: Origin::Tracking }))
    }

    /// Сбрасывает подтверждение при потере объекта или завершении сеанса.
    pub fn lose(&mut self) {
        self.anchor = None;
    }

    /// Возвращает время только для текущего кадра.
    fn current_time(&self, frame: u64) -> Result<u64, ObservationError> {
        match (self.latest_frame, self.latest_time_ns) {
            (Some(current), Some(time)) if current == frame => Ok(time),
            _ => Err(ObservationError::NotCurrentFrame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox() -> BoundingBox { BoundingBox { left: 5, top: 10, right: 25, bottom: 30 } }

    #[test]
    fn fresh_observations_are_distinguished_from_tracking() {
        let mut gate = ObservationGate::new(100);
        gate.begin_frame(1, 1_000).unwrap();
        let detection = gate.detect(1, bbox(), 100, 100, 0.8).unwrap();
        assert_eq!(detection.origin, Origin::Detection);
        gate.begin_frame(2, 1_050).unwrap();
        let tracked = gate.track(2, bbox(), 100, 100).unwrap().unwrap();
        assert_eq!((tracked.frame, tracked.anchor_frame, tracked.age_ns, tracked.origin), (2, 1, 50, Origin::Tracking));
    }

    #[test]
    fn expired_or_lost_tracks_never_return_a_box() {
        let mut gate = ObservationGate::new(100);
        gate.begin_frame(1, 1_000).unwrap();
        gate.detect(1, bbox(), 100, 100, 0.8).unwrap();
        gate.begin_frame(2, 1_101).unwrap();
        assert_eq!(gate.track(2, bbox(), 100, 100).unwrap(), None);
        gate.begin_frame(3, 1_102).unwrap();
        gate.detect(3, bbox(), 100, 100, 0.8).unwrap();
        gate.lose();
        gate.begin_frame(4, 1_103).unwrap();
        assert_eq!(gate.track(4, bbox(), 100, 100).unwrap(), None);
    }

    #[test]
    fn rejects_invalid_and_out_of_order_inputs() {
        let mut gate = ObservationGate::new(100);
        gate.begin_frame(1, 1_000).unwrap();
        assert_eq!(gate.begin_frame(1, 1_001), Err(ObservationError::FrameOrder));
        assert_eq!(gate.begin_frame(2, 1_000), Err(ObservationError::TimeOrder));
        assert_eq!(gate.detect(2, bbox(), 100, 100, 0.8), Err(ObservationError::NotCurrentFrame));
        assert_eq!(gate.detect(1, bbox(), 20, 20, 0.8), Err(ObservationError::InvalidBox));
        assert_eq!(gate.detect(1, bbox(), 100, 100, f32::NAN), Err(ObservationError::InvalidConfidence));
    }
}
