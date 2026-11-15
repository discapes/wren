"""Build a YOLO dataset from all CubiCasa5k plans; Stoa's plans are split train/val/test by plan (see split_of).

SVG coordinates are in the F1_scaled.png pixel frame; Stoa ships F1_original.png, so images are
F1_original and boxes are rescaled by original/scaled. All floors of a plan are drawn in the same image.
"""
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from svgpathtools import parse_path

CUBICASA = Path.home() / "junction-data/cubicasa5k/cubicasa5k"
STOA = Path.home() / "junction-data/stoa-full/junction-dataset/data/cubicasa5k"
OUT = Path.home() / "junction-data/detector-ds"
Image.MAX_IMAGE_PIXELS = None  # some plans are ~90 megapixels

CLASSES = ["door", "window", "toilet", "sink", "shower", "bathtub", "sauna_bench"]
# first tokens of the SVG class attribute -> our class. Stoa's shower+bathtub answers leave ShowerCab elements out.
MAP = {
    "Door": "door",
    "Window": "window",
    "FixedFurniture Toilet": "toilet",
    "FixedFurniture Sink": "sink",
    "FixedFurniture DoubleSink": "sink",
    "FixedFurniture DoubleSinkRight": "sink",
    "FixedFurniture RoundSink": "sink",
    "FixedFurniture SideSink": "sink",
    "FixedFurniture CornerSink": "sink",
    "FixedFurniture Shower": "shower",
    "FixedFurniture Bathtub": "bathtub",
    "FixedFurniture BathtubRound": "bathtub",
    "FixedFurniture Jacuzzi": "bathtub",
    "FixedFurniture SaunaBench": "sauna_bench",
    "FixedFurniture SaunaBenchHigh": "sauna_bench",
    "FixedFurniture SaunaBenchMid": "sauna_bench",
    "FixedFurniture SaunaBenchLow": "sauna_bench",
}
NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")


def parse_transform(t):
    m = (1, 0, 0, 1, 0, 0)
    for op, args in re.findall(r"(\w+)\(([^)]*)\)", t or ""):
        v = [float(x) for x in NUM.findall(args)]
        if op == "matrix":
            n = tuple(v)
        elif op == "translate":
            n = (1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0)
        elif op == "scale":
            n = (v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        else:
            raise ValueError(op)
        m = compose(m, n)
    return m


def compose(m, n):
    a, b, c, d, e, f = m
    A, B, C, D, E, F = n
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)


def apply(m, x, y):
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def local_points(el):
    """Corner points of an element's geometry in its own coordinate frame."""
    tag = el.tag.split("}")[-1]
    if tag == "polygon" or tag == "polyline":
        v = [float(x) for x in NUM.findall(el.get("points", ""))]
        return list(zip(v[0::2], v[1::2]))
    if tag == "rect":
        x, y, w, h = (float(el.get(k, 0)) for k in ("x", "y", "width", "height"))
        return [(x, y), (x + w, y + h)]
    if tag == "path" and el.get("d"):
        x0, x1, y0, y1 = parse_path(el.get("d")).bbox()
        return [(x0, y0), (x1, y1)]
    return []


def collect(el, m, pts):
    m = compose(m, parse_transform(el.get("transform")))
    pts += [apply(m, *p) for p in local_points(el)]
    for c in el:
        collect(c, m, pts)


def element_points(el, m):
    """All points under el in the root frame. Furniture uses its BoundaryPolygon only."""
    pts = []
    boundary = [c for c in el if c.get("class") == "BoundaryPolygon"]
    if boundary:
        m = compose(m, parse_transform(el.get("transform")))
        for c in boundary:
            collect(c, m, pts)
    else:
        collect(el, m, pts)
    return pts


def walk(el, m, out):
    m2 = compose(m, parse_transform(el.get("transform")))
    cls = el.get("class") or ""
    key = next((k for k in MAP if cls == k or cls.startswith(k + " ")), None)
    if key:
        pts = element_points(el, m)
        if pts:
            xs, ys = zip(*pts)
            out.append((MAP[key], min(xs), min(ys), max(xs), max(ys)))
        return
    for c in el:
        walk(c, m2, out)


def boxes(plan):
    out = []
    walk(ET.parse(plan / "model.svg").getroot(), (1, 0, 0, 1, 0, 0), out)
    return out


def split_of(plans, stoa):
    """CubiCasa5k plans outside Stoa's set all train. Stoa's plans: 70% train, 10% val (thresholds), 20% test."""
    ours = sorted(n for n in plans if n in stoa)
    random.Random(0).shuffle(ours)
    n = len(ours)
    split = {name: "train" for name in plans}
    split.update({name: "test" for name in ours[: n // 5]})
    split.update({name: "val" for name in ours[n // 5 : n // 5 + n // 10]})
    return split


def write(item):
    name, split = item
    plan = CUBICASA / name.rsplit("-", 1)[0] / name.rsplit("-", 1)[1]
    orig = Image.open(plan / "F1_original.png")
    sw, sh = Image.open(plan / "F1_scaled.png").size
    sx, sy = orig.width / sw, orig.height / sh
    lines = []
    for cls, x0, y0, x1, y1 in boxes(plan):
        x0, x1 = max(0, x0 * sx), min(orig.width, x1 * sx)
        y0, y1 = max(0, y0 * sy), min(orig.height, y1 * sy)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cx, cy = (x0 + x1) / 2 / orig.width, (y0 + y1) / 2 / orig.height
        lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {(x1 - x0) / orig.width:.6f} {(y1 - y0) / orig.height:.6f}")
    shutil.copyfile(plan / "F1_original.png", OUT / "images" / split / f"{name}.png")
    (OUT / "labels" / split / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
    return split


def main():
    stoa = {p.stem for p in STOA.glob("*.png")}
    plans = [f"{p.parent.parent.name}-{p.parent.name}" for p in CUBICASA.glob("*/*/model.svg")]
    split = split_of(plans, stoa)
    for s in ("train", "val", "test"):
        for d in ("images", "labels"):
            (OUT / d / s).mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        done = Counter(pool.map(write, sorted(split.items()), chunksize=16))
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(dict(done), file=sys.stderr)


if __name__ == "__main__":
    main()
