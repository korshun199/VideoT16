// Узкий интерфейс OpenCV KCF для модуля наблюдения; управление устройствами отсутствует.
#include <opencv2/tracking.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>

struct KcfHolder { cv::Ptr<cv::TrackerKCF> tracker; };

extern "C" {
// Создаёт независимый экземпляр трекера; nullptr означает ошибку.
void* videot16_kcf_new() noexcept {
    try { return new KcfHolder{cv::TrackerKCF::create()}; } catch (...) { return nullptr; }
}

// Освобождает экземпляр. Передавать только указатель, полученный из new.
void videot16_kcf_free(void* ptr) noexcept { delete static_cast<KcfHolder*>(ptr); }

// Проверяет геометрию BGR-кадра перед передачей OpenCV.
static bool valid_image(const std::uint8_t* data, int width, int height, std::size_t stride) {
    return data && width > 0 && height > 0 && width <= 32768 && height <= 32768
        && stride >= static_cast<std::size_t>(width) * 3;
}

// Инициализирует трекер вручную переданной подтверждённой рамкой.
int videot16_kcf_init(void* ptr, const std::uint8_t* data, int width, int height,
                      std::size_t stride, int x, int y, int w, int h) noexcept {
    if (!ptr || !valid_image(data, width, height, stride) || x < 0 || y < 0 ||
        w <= 0 || h <= 0 || w > width || h > height || x > width - w || y > height - h) return -1;
    try {
        cv::Mat frame(height, width, CV_8UC3, const_cast<std::uint8_t*>(data), stride);
        static_cast<KcfHolder*>(ptr)->tracker->init(frame, cv::Rect(x, y, w, h));
        return 1;
    } catch (...) { return -2; }
}

// Возвращает 1 при найденной рамке, 0 при потере и отрицательное число при ошибке.
int videot16_kcf_update(void* ptr, const std::uint8_t* data, int width, int height,
                        std::size_t stride, int* output) noexcept {
    if (!ptr || !output || !valid_image(data, width, height, stride)) return -1;
    try {
        cv::Mat frame(height, width, CV_8UC3, const_cast<std::uint8_t*>(data), stride);
        cv::Rect box;
        if (!static_cast<KcfHolder*>(ptr)->tracker->update(frame, box)) return 0;
        output[0] = box.x; output[1] = box.y; output[2] = box.width; output[3] = box.height;
        return 1;
    } catch (...) { return -2; }
}
}
