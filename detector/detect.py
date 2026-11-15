"""Symbols and rooms on one picture, for drawings uploaded in the app.

python detect.py <image>   # JSON {"symbols": [[class, conf, x0, y0, x1, y1]], "rooms": [...]} in the image's pixels

Uses the single tiled yolo26m symbol model (82.7% exact counts against the ensemble's 85.3%, at a third of the CPU
time) and the room model, with thresholds tuned on the val plans as for the dataset plans.
"""
import json
import sys
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

import evaluate
import rooms
import tiles
from prepare import OUT

RUNS = Path.home() / "junction-data/detector-runs/detect/runs"
HERE = Path(__file__).parent
MAX_SIDE = 2000  # 90% of CubiCasa plans are under ~1800 px; larger pictures draw symbols bigger than trained on
Image.MAX_IMAGE_PIXELS = None


def model(run):
    return YOLO(RUNS / run / "weights/best.pt")


def boxes(r, x=0, y=0):
    return [[int(c), float(s), x0 + x, y0 + y, x1 + x, y1 + y]
            for c, s, (x0, y0, x1, y1) in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())]


def kept(dets, names, preds, labels):
    """Detections above the class thresholds tuned on the val plans, deduplicated, as [class, conf, box]."""
    thr = evaluate.tune(json.loads((HERE / preds).read_text()), labels, names)
    return [[names[c], s, *b] for c, s, *b in evaluate.dedupe(dets, thr, names)]


def main(path):
    im = Image.open(path).convert("RGBA")
    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")  # transparent → white paper
    k = min(1, MAX_SIDE / max(im.size))
    im = im.resize((round(im.width * k), round(im.height * k)))
    W, H = im.size
    origins = tiles.tiles(W, H)
    crops = [im.crop((x, y, min(x + tiles.TILE, W), min(y + tiles.TILE, H))) for x, y in origins]
    symbols = [d for (x, y), r in zip(origins, model("full-m-1024").predict(crops, imgsz=tiles.TILE, conf=0.05, verbose=False))
               for d in boxes(r, x, y)  # each detection counts in the tile nearest its centre, as in tiles.predict
               if tiles.owner((d[2] + d[4]) / 2, W) == x and tiles.owner((d[3] + d[5]) / 2, H) == y]
    found = boxes(model("rooms-m-1280").predict(im, imgsz=1280, conf=0.05, verbose=False)[0])
    out = {"symbols": kept(symbols, evaluate.CLASSES, "preds-full-m-1024.json", OUT / "labels"),
           "rooms": kept(found, rooms.CLASSES, "preds-rooms.json", rooms.ROOMS / "labels")}
    print(json.dumps({key: [[c, round(s, 3), *(round(v / k, 1) for v in b)] for c, s, *b in dets]
                      for key, dets in out.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
