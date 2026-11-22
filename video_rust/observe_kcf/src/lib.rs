//! Экспериментальный доступ к OpenCV KCF для наблюдения, без управления устройствами.
use std::ffi::c_void;
use std::ptr::NonNull;

#[link(name = "videot16_kcf")]
extern "C" {
    fn videot16_kcf_new() -> *mut c_void;
    fn videot16_kcf_free(ptr: *mut c_void);
    fn videot16_kcf_init(ptr: *mut c_void, data: *const u8, width: i32, height: i32,
        stride: usize, x: i32, y: i32, w: i32, h: i32) -> i32;
    fn videot16_kcf_update(ptr: *mut c_void, data: *const u8, width: i32, height: i32,
        stride: usize, output: *mut i32) -> i32;
}

/// Целочисленная рамка в координатах исходного кадра.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect { pub x: i32, pub y: i32, pub width: i32, pub height: i32 }

/// BGR-кадр с явно указанной геометрией и шагом строк.
pub struct Frame<'a> { pub pixels: &'a [u8], pub width: i32, pub height: i32, pub stride: usize }

impl Frame<'_> {
    /// Проверяет размеры и доступность всей памяти изображения.
    fn valid(&self) -> bool {
        if self.width <= 0 || self.height <= 0 || self.width > 32768 || self.height > 32768 { return false; }
        let row = self.width as usize * 3;
        if self.stride < row { return false; }
        self.stride.checked_mul(self.height as usize - 1)
            .and_then(|value| value.checked_add(row))
            .map_or(false, |needed| needed <= self.pixels.len())
    }
}

/// KCF не передаётся между потоками и не удерживает ссылки на входные кадры.
pub struct Kcf { handle: NonNull<c_void>, initialized: bool, geometry: Option<(i32, i32)> }

impl Kcf {
    /// Создаёт независимый объект KCF или сообщает ошибку загрузки OpenCV.
    pub fn new() -> Result<Self, &'static str> {
        let handle = NonNull::new(unsafe { videot16_kcf_new() }).ok_or("не удалось создать KCF")?;
        Ok(Self { handle, initialized: false, geometry: None })
    }

    /// Задаёт рамку на кадре; повторная инициализация требует нового экземпляра.
    pub fn init(&mut self, frame: &Frame<'_>, bbox: Rect) -> Result<(), &'static str> {
        if self.initialized { return Err("трекер уже инициализирован"); }
        if !frame.valid() { return Err("некорректный кадр"); }
        let code = unsafe { videot16_kcf_init(self.handle.as_ptr(), frame.pixels.as_ptr(),
            frame.width, frame.height, frame.stride, bbox.x, bbox.y, bbox.width, bbox.height) };
        if code != 1 { return Err("не удалось инициализировать KCF"); }
        self.initialized = true;
        self.geometry = Some((frame.width, frame.height));
        Ok(())
    }

    /// Возвращает новую рамку либо None при потере; старую рамку не сохраняет.
    pub fn update(&mut self, frame: &Frame<'_>) -> Result<Option<Rect>, &'static str> {
        if !self.initialized { return Err("трекер не инициализирован"); }
        if !frame.valid() || self.geometry != Some((frame.width, frame.height)) { return Err("некорректная геометрия кадра"); }
        let mut values = [0i32; 4];
        let code = unsafe { videot16_kcf_update(self.handle.as_ptr(), frame.pixels.as_ptr(),
            frame.width, frame.height, frame.stride, values.as_mut_ptr()) };
        match code {
            1 if values[0] >= 0 && values[1] >= 0 && values[2] > 0 && values[3] > 0
                && values[2] <= frame.width && values[3] <= frame.height
                && values[0] <= frame.width - values[2] && values[1] <= frame.height - values[3] =>
                Ok(Some(Rect { x: values[0], y: values[1], width: values[2], height: values[3] })),
            0 => { self.initialized = false; Ok(None) },
            _ => { self.initialized = false; Err("ошибка обновления KCF") },
        }
    }
}

impl Drop for Kcf {
    /// Всегда освобождает объект OpenCV.
    fn drop(&mut self) { unsafe { videot16_kcf_free(self.handle.as_ptr()); } }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(offset: usize) -> Vec<u8> {
        let mut image = vec![0; 160 * 120 * 3];
        for y in 0..32usize { for x in 0..32usize { for c in 0..3usize {
            image[((40+y)*160 + 50+offset+x)*3+c] = ((x*37+y*19+c*31+x*y*3)%251) as u8;
        } } }
        image
    }

    #[test]
    fn follows_small_synthetic_translation() {
        let mut tracker = Kcf::new().unwrap();
        let first = image(0);
        tracker.init(&Frame { pixels: &first, width: 160, height: 120, stride: 480 },
            Rect { x: 50, y: 40, width: 32, height: 32 }).unwrap();
        let mut last = None;
        for offset in [0, 2, 4, 6, 8] {
            let pixels = image(offset);
            last = tracker.update(&Frame { pixels: &pixels, width: 160, height: 120, stride: 480 }).unwrap();
            assert!(last.is_some(), "потеряна искусственная цель при смещении {offset}");
        }
        assert!((last.unwrap().x - 58).abs() <= 4);
    }

    #[test]
    fn rejects_short_frames_invalid_rectangles_and_wrong_dimensions() {
        let mut tracker = Kcf::new().unwrap();
        let short = Frame { pixels: &[0; 8], width: 160, height: 120, stride: 480 };
        assert!(tracker.init(&short, Rect { x: 0, y: 0, width: 10, height: 10 }).is_err());
        let pixels = image(0);
        let good = Frame { pixels: &pixels, width: 160, height: 120, stride: 480 };
        assert!(tracker.init(&good, Rect { x: 155, y: 40, width: 32, height: 32 }).is_err());
        tracker.init(&good, Rect { x: 50, y: 40, width: 32, height: 32 }).unwrap();
        assert!(tracker.update(&Frame { pixels: &pixels, width: 120, height: 120, stride: 480 }).is_err());
    }
}
