"""Train a detector: python train.py <data.yaml> <model> <imgsz> <name> [devices, e.g. 0 or 1,2,3]"""
import sys

from ultralytics import YOLO

data, model, imgsz, name, *device = sys.argv[1:]
YOLO(model).train(
    data=data,
    imgsz=int(imgsz),
    # fixed length, last weights: the fold models never pick an epoch by their held-out plans
    epochs=50,
    patience=0,
    batch=16 * len(device[0].split(",")) if device else 16,
    device=device[0] if device else None,
    cache="ram",
    name=name,
    project="runs",
    # plans come in any orientation; colour jitter is irrelevant for line drawings
    flipud=0.5,
    degrees=0,
    hsv_h=0,
    hsv_s=0,
    mosaic=1.0,
    close_mosaic=10,
    workers=16,
    plots=True,
)
