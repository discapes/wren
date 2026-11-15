"""Whole-plan room and storey detector. Stoa's answers follow the CubiCasa5k space types exactly:
bedrooms = Bedroom*, kitchens = Kitchen*, bathrooms = Bath* not labelled WC, WC rooms = Bath* labelled WC,
balconies = Outdoor Balcony*, storeys = Floorplan groups. Saunas are sauna-bench clusters (symbol detector) plus Sauna
spaces holding no bench cluster (evaluate.saunas).

python rooms.py build                 # same plan split as detector-ds -> detector-rooms
"""
import shutil
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from prepare import CUBICASA, NUM, OUT, apply, compose, parse_transform

ROOMS = OUT.parent / "detector-rooms"
CLASSES = ["storey", "bedroom", "kitchen", "bathroom", "wc_room", "balcony", "sauna_room"]


def space_class(cls, label):
    t = cls.split()[1:]
    if t[0] == "Bedroom":
        return "bedroom"
    if t[0] == "Kitchen":
        return "kitchen"
    if t[0] == "Bath":
        return "wc_room" if label.upper().startswith("WC") else "bathroom"
    if t[:2] == ["Outdoor", "Balcony"]:
        return "balcony"
    if t[0] == "Sauna":
        return "sauna_room"


def polygon_box(el, m):
    """Bounding box of an element's own polygon children (not its labels and dimension marks)."""
    pts = []
    for p in el.findall("{*}polygon"):
        v = [float(x) for x in NUM.findall(p.get("points", ""))]
        pts += [apply(m, x, y) for x, y in zip(v[0::2], v[1::2])]
    return pts


def plan_boxes(svg):
    out = []

    def walk(el, m, floor):
        m = compose(m, parse_transform(el.get("transform")))
        cls = el.get("class") or ""
        if cls.startswith("Floorplan"):
            floor = []
        if cls.startswith("Space ") or cls.startswith("Wall"):
            pts = polygon_box(el, m)
            if floor is not None:
                floor += pts
            if cls.startswith("Space "):
                label = next((t.text.strip() for t in el.iter() if t.tag.endswith("text") and t.text and t.text.strip()), "")
                kind = space_class(cls, label)
                if kind and pts:
                    out.append((kind, pts))
        for c in el:
            walk(c, m, floor)
        if cls.startswith("Floorplan") and floor:
            out.append(("storey", floor))

    walk(ET.parse(svg).getroot(), (1, 0, 0, 1, 0, 0), None)
    return [(k, min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p)) for k, p in out]


def write(item):
    split, img = item
    d, num = img.stem.rsplit("-", 1)
    plan = CUBICASA / d / num
    w, h = Image.open(img).size
    sw, sh = Image.open(plan / "F1_scaled.png").size
    sx, sy = w / sw, h / sh
    lines = []
    for k, x0, y0, x1, y1 in plan_boxes(plan / "model.svg"):
        x0, x1, y0, y1 = max(0, x0 * sx), min(w, x1 * sx), max(0, y0 * sy), min(h, y1 * sy)
        if x1 - x0 > 2 and y1 - y0 > 2:
            lines.append(f"{CLASSES.index(k)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
    (ROOMS / "images" / split / img.name).hardlink_to(img)
    (ROOMS / "labels" / split / f"{img.stem}.txt").write_text("".join(lines))


def build():
    shutil.rmtree(ROOMS, ignore_errors=True)
    items = []
    for split in ("train", "val", "test"):
        for d in ("images", "labels"):
            (ROOMS / d / split).mkdir(parents=True)
        items += [(split, p) for p in sorted((OUT / "images" / split).glob("*.png"))]
    with Pool() as pool:
        pool.map(write, items, chunksize=16)
    (ROOMS / "data.yaml").write_text(
        f"path: {ROOMS}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))


def detect(model, images, imgsz=1280):
    """Whole-plan detections: {stem: [[class, conf, x0, y0, x1, y1]]}."""
    preds = {}
    for i in range(0, len(images), 32):  # one big list makes ultralytics hold every image in memory
        for r in model.predict([str(p) for p in images[i:i + 32]], imgsz=int(imgsz), conf=0.05, verbose=False, batch=8):
            preds[Path(r.path).stem] = [[int(c), round(float(s), 3), *[round(v, 1) for v in b]]
                                        for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist())]
    return preds


if __name__ == "__main__":
    {"build": build}[sys.argv[1]](*sys.argv[2:])
