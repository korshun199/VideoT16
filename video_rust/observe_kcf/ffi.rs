//! Ограниченный C ABI для экспериментального вызова Rust KCF из интерфейса.
#[path = "src/lib.rs"]
mod kcf;

use kcf::{Frame, Kcf, Rect};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

/// Создаёт объект; null означает ошибку создания.
#[no_mangle]
pub extern "C" fn v16_observe_new() -> *mut Kcf {
    catch_unwind(|| Kcf::new().map(|value| Box::into_raw(Box::new(value))))
        .ok().and_then(Result::ok).unwrap_or(std::ptr::null_mut())
}

/// Освобождает объект, созданный v16_observe_new; вызов повторно запрещён.
#[no_mangle]
pub unsafe extern "C" fn v16_observe_free(handle: *mut Kcf) {
    if !handle.is_null() { drop(Box::from_raw(handle)); }
}

/// Проверяет указатели; выделение памяти и её время жизни гарантирует вызывающая сторона.
unsafe fn frame<'a>(pixels: *const u8, len: usize, width: i32, height: i32, stride: usize)
    -> Option<Frame<'a>>
{
    if pixels.is_null() || len > isize::MAX as usize { return None; }
    Some(Frame { pixels: slice::from_raw_parts(pixels, len), width, height, stride })
}

/// Принимает BGR-кадр и подтверждённую рамку; 1 — успех, -1 — ошибка.
#[no_mangle]
pub unsafe extern "C" fn v16_observe_init(handle: *mut Kcf, pixels: *const u8, len: usize,
    width: i32, height: i32, stride: usize, x: i32, y: i32, w: i32, h: i32) -> i32
{
    catch_unwind(AssertUnwindSafe(|| {
        let Some(tracker) = handle.as_mut() else { return -1 };
        let Some(frame) = frame(pixels, len, width, height, stride) else { return -1 };
        if tracker.init(&frame, Rect { x, y, width: w, height: h }).is_ok() { 1 } else { -1 }
    })).unwrap_or(-1)
}

/// Возвращает 1 с рамкой, 0 при потере, -1 при ошибке; output содержит 4 i32.
#[no_mangle]
pub unsafe extern "C" fn v16_observe_update(handle: *mut Kcf, pixels: *const u8, len: usize,
    width: i32, height: i32, stride: usize, output: *mut i32) -> i32
{
    catch_unwind(AssertUnwindSafe(|| {
        let Some(tracker) = handle.as_mut() else { return -1 };
        let Some(frame) = frame(pixels, len, width, height, stride) else { return -1 };
        if output.is_null() { return -1; }
        match tracker.update(&frame) {
            Ok(Some(bbox)) => {
                let out = slice::from_raw_parts_mut(output, 4);
                out.copy_from_slice(&[bbox.x, bbox.y, bbox.width, bbox.height]);
                1
            }
            Ok(None) => 0,
            Err(_) => -1,
        }
    })).unwrap_or(-1)
}
