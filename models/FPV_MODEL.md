# Модель обнаружения FPV-дронов

Файл `fpv_drone_best.pt` — готовая модель Ultralytics YOLOv8n.

Классы:

- `quadcopter` — квадрокоптер;
- `fixed-wing` — самолётный дрон.

Источник: [TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1).

Ссылки для восстановления:

- [Оригинальные веса `best.pt` на Hugging Face](https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt);
- [Копия `.pt` в GitHub](https://raw.githubusercontent.com/korshun199/VideoT16/main/models/fpv_drone_best.pt);
- [Копия `.onnx` в GitHub](https://raw.githubusercontent.com/korshun199/VideoT16/main/models/fpv_drone_best.onnx).

Автоматическое восстановление с проверкой целостности:

```bash
./scripts/download_fpv_model.sh
```

Лицензия модели: CC BY 4.0. Перед коммерческим использованием проверьте условия лицензии источника.

SHA-256:

```text
bf24a20e69b288a0c7e4855c72146149fa5c25845fa5e54beda2b93cf79824
```

Для компьютера используется `fpv_drone_best.pt`. Для Orange Pi 5 модель нужно экспортировать в формат RKNN скриптами из каталога `scripts/`.
