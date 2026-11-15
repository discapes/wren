"""Full-resolution tiling: thin symbols (windows, door leaves) vanish when a 2700 px plan is shrunk to 1280.

python tiles.py build                      # detector-ds -> detector-ds-tiles (all splits)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from prepare import OUT

TILE, OVERLAP = 1024, 256
TILES = OUT.parent / "detector-ds-tiles"


def starts(size):
    """Tile origins along one axis; the last tile ends at the image edge."""
    if size <= TILE:
        return [0]
    s = list(range(0, size - TILE, TILE - OVERLAP))
    return s + [size - TILE]


def tiles(w, h):
    return [(x, y) for y in starts(h) for x in starts(w)]


def tile_plan(item):
    split, img_path = item
    im = Image.open(img_path)
    W, H = im.size
    boxes = []
    for line in (OUT / "labels" / split / f"{img_path.stem}.txt").read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = line.split()
            cx, cy, bw, bh = float(cx) * W, float(cy) * H, float(bw) * W, float(bh) * H
            boxes.append((int(c), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    for x, y in tiles(W, H):
        tw, th = min(TILE, W - x), min(TILE, H - y)
        lines = []
        for c, x0, y0, x1, y1 in boxes:
            cx0, cy0, cx1, cy1 = max(x0, x), max(y0, y), min(x1, x + tw), min(y1, y + th)
            # boxes cut by the tile edge stay only if at least half of them is inside
            if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < 0.5 * (x1 - x0) * (y1 - y0):
                continue
            lines.append(f"{c} {((cx0 + cx1) / 2 - x) / tw:.6f} {((cy0 + cy1) / 2 - y) / th:.6f} {(cx1 - cx0) / tw:.6f} {(cy1 - cy0) / th:.6f}\n")
        name = f"{img_path.stem}-{x}-{y}"
        im.crop((x, y, x + tw, y + th)).convert("RGB").save(TILES / "images" / split / f"{name}.png")
        (TILES / "labels" / split / f"{name}.txt").write_text("".join(lines))


def build():
    items = [(split, p) for split in ("train", "val", "test") for p in sorted((OUT / "images" / split).glob("*.png"))]
    for split in ("train", "val", "test"):
        for d in ("images", "labels"):
            (TILES / d / split).mkdir(parents=True, exist_ok=True)
    with Pool() as pool:
        pool.map(tile_plan, items, chunksize=8)
    (TILES / "data.yaml").write_text((OUT / "data.yaml").read_text().replace(str(OUT), str(TILES)))
    print(sum(1 for _ in (TILES / "images/train").glob("*.png")), "train tiles", file=sys.stderr)


def owner(c, size):
    """Origin of the tile whose centre is nearest to c, so each detection is counted in exactly one tile."""
    return min(starts(size), key=lambda s: abs(c - (s + min(TILE, size - s) / 2)))


def detect(model, images, imgsz=TILE):
    """Tiled detections per plan: {stem: [[class, conf, x0, y0, x1, y1]]} in the plan's pixels."""
    preds = {}
    for p in images:
        im = Image.open(p).convert("RGB")
        W, H = im.size
        origins = tiles(W, H)
        crops = [im.crop((x, y, min(x + TILE, W), min(y + TILE, H))) for x, y in origins]
        dets = []
        for (x, y), r in zip(origins, model.predict(crops, imgsz=int(imgsz), conf=0.05, verbose=False, batch=16)):
            for c, s, (x0, y0, x1, y1) in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy.tolist()):
                if owner(x + (x0 + x1) / 2, W) == x and owner(y + (y0 + y1) / 2, H) == y:
                    dets.append([int(c), round(float(s), 3), round(x0 + x, 1), round(y0 + y, 1), round(x1 + x, 1), round(y1 + y, 1)])
        preds[p.stem] = dets
    return preds


if __name__ == "__main__":
    {"build": build}[sys.argv[1]](*sys.argv[2:])
