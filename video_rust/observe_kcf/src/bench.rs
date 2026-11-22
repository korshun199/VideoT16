//! Предварительная проверка KCF на последовательности BGR-кадров из stdin.
#[path = "lib.rs"]
mod kcf;
use kcf::{Frame, Kcf, Rect};
use std::io::{self, Read};
use std::time::Instant;

/// Читает целый кадр; ранний конец входа считается ошибкой.
fn read_frame(input: &mut impl Read, buffer: &mut [u8]) -> io::Result<bool> {
    let mut count = 0;
    while count < buffer.len() {
        let n = input.read(&mut buffer[count..])?;
        if n == 0 {
            return if count == 0 { Ok(false) } else { Err(io::ErrorKind::UnexpectedEof.into()) };
        }
        count += n;
    }
    Ok(true)
}

/// Тестирует отдельный заранее размеченный эпизод и печатает рамки и время.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = io::stdin().lock();
    let mut pixels = vec![0u8; 1280 * 720 * 3];
    if !read_frame(&mut input, &mut pixels)? { return Err("пустой поток кадров".into()); }
    let mut tracker = Kcf::new()?;
    tracker.init(&Frame { pixels: &pixels, width: 1280, height: 720, stride: 3840 },
                 Rect { x: 415, y: 207, width: 95, height: 38 })?;
    println!("frame,ok,x,y,w,h,update_ms");
    println!("2940,1,415,207,95,38,0");
    let mut index = 2941u32;
    while read_frame(&mut input, &mut pixels)? {
        let started = Instant::now();
        let bbox = tracker.update(&Frame { pixels: &pixels, width: 1280, height: 720, stride: 3840 })?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        match bbox {
            Some(r) => println!("{index},1,{},{},{},{},{ms:.3}", r.x, r.y, r.width, r.height),
            None => { println!("{index},0,0,0,0,0,{ms:.3}"); break; }
        }
        index += 1;
    }
    Ok(())
}
