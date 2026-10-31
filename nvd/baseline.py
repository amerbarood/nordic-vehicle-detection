"""Off-the-shelf baseline (experiments/): COCO-pretrained YOLOv9-E via Ultralytics, no fine-tuning.

Draws every detection on the image so the false positives / misses that
motivated fine-tuning are visible. Requires `pip install ultralytics`.
"""

from __future__ import annotations

from pathlib import Path

OUTPUT_DIR = Path("outputs/baseline")


def run(images: list[Path], conf: float = 0.3, car_only: bool = False) -> list[Path]:
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov9e.pt")
    car_class = [k for k, v in model.names.items() if v == "car"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path in images:
        img = cv2.imread(str(path))
        result = model(img, conf=conf, classes=car_class if car_only else None, verbose=False)[0]
        for box, cls, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{model.names[int(cls)]} {score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = OUTPUT_DIR / f"baseline_{Path(path).name}"
        cv2.imwrite(str(out), img)
        print(f"{Path(path).name}: {len(result.boxes)} detections -> {out}")
        outputs.append(out)
    return outputs
