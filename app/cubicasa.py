"""CubiCasa plans: the detector's symbol and room boxes → graph, on the plan PNG.

Predictions come from ../detector (folds.py: every plan is predicted by models that never trained on it). Class
thresholds are tuned and duplicates dropped exactly as evaluate.py scores them.
"""
import json
import sys
from pathlib import Path

from PIL import Image
from shapely.geometry import box

from graph import DATA, Graph, Node, View

DETECTOR = Path(__file__).resolve().parent.parent / "detector"
sys.path.insert(0, str(DETECTOR))
import evaluate  # noqa: E402
import rooms  # noqa: E402

DIR = DATA / "data/cubicasa5k"
# per model, the prediction files in order of preference: the ensemble covers the val and test plans, which none of
# its models trained on; the out-of-fold predictions cover every plan
SOURCES = {"symbols": ("preds-symbols.json", "preds-symbols-oof.json"), "rooms": ("preds-rooms.json", "preds-rooms-oof.json")}
SYMBOL_KIND = {"door": "opening", "window": "opening"}  # the other symbols are elements
ROOM_CAT = {"bedroom": "bedroom", "kitchen": "kitchen", "bathroom": "bathroom", "wc_room": "wc", "balcony": "balcony"}

Image.MAX_IMAGE_PIXELS = None
_models = {}


def models():
    """name → [(predictions, class names, tuned thresholds)] per prediction file, loaded once per process."""
    if not _models:
        for name, names, labels in (("symbols", evaluate.CLASSES, evaluate.OUT / "labels"),
                                    ("rooms", rooms.CLASSES, rooms.ROOMS / "labels")):
            for f in SOURCES[name]:
                if (DETECTOR / f).exists():
                    preds = json.loads((DETECTOR / f).read_text())
                    _models.setdefault(name, []).append((preds, names, evaluate.tune(preds, labels, names)))
    return _models


def detections(model, stem):
    """[(class name, conf, (x0, y0, x1, y1))] above threshold, deduplicated, from the first file covering the plan."""
    for preds, names, thr in models().get(model, []):
        if stem in preds:
            return [(names[c], s, tuple(b)) for c, s, *b in evaluate.dedupe(preds[stem], thr, names)]
    return []


def items():
    return sorted({s for preds, _, _ in models()["symbols"] for s in preds if (DIR / f"{s}.png").exists()})


def read(stem):
    rel = f"data/cubicasa5k/{stem}.png"
    w, h = Image.open(DATA / rel).size
    g = Graph(stem, "cubicasa5k")
    g.views["png"] = View("png", rel, (0, 0, w, h))
    return build(g, "png", detections("rooms", stem), detections("symbols", stem))


def build(g, view, room_dets, symbols):
    """Storeys, rooms, symbols and saunas from [(class, conf, box)] detections on the picture `view`."""
    def add(node, b):
        g.add(node)
        x0, y0, x1, y1 = (round(v, 1) for v in b)
        g.anchor(node.id, view, [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]])

    storeys = [d for d in room_dets if d[0] == "storey"] or [("storey", 1.0, g.views[view].box)]
    for i, (_, conf, b) in enumerate(storeys):
        add(Node(f"storey/{i}", "storey", "storey", geom=box(*b), conf=conf, via="detector"), b)

    def storey_of(b):
        c = box(*b).centroid
        return next((s.id for s in g.find(kind="storey") if s.geom.contains(c)), "storey/0")

    for i, (cls, conf, b) in enumerate(d for d in room_dets if d[0] in ROOM_CAT):
        add(Node(f"room/{i}", "space", ROOM_CAT[cls], storey_of(b), geom=box(*b), conf=conf, via="detector"), b)
    for i, (cls, conf, b) in enumerate(symbols):
        add(Node(f"{cls}/{i}", SYMBOL_KIND.get(cls, "element"), cls, storey_of(b), geom=box(*b), conf=conf,
                 via="detector"), b)
    found = [(b, conf) for cls, conf, b in symbols + room_dets if cls in ("sauna_bench", "sauna_room")]
    benches = [b for cls, _, b in symbols if cls == "sauna_bench"]
    sauna_rooms = [b for cls, _, b in room_dets if cls == "sauna_room"]
    for i, u in enumerate(evaluate.saunas(benches, sauna_rooms)):
        conf = max(c for b, c in found if evaluate.overlap(b, u) > 0.5)
        add(Node(f"sauna/{i}", "space", "sauna", storey_of(u), geom=box(*u), conf=conf, via="detector"), u)
    return g.derive()
