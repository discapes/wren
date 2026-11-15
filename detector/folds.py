"""5-fold cross-training over Stoa's cubicasa plans, so every plan is predicted by models that never saw it.

Fold 0 is the test split of prepare.split_of, which keeps its scores comparable with earlier runs; folds 1-4 split
Stoa's train and val plans. The 213 CubiCasa5k plans outside Stoa's set train in every fold. Thresholds are tuned on
the out-of-fold predictions of folds 1-4 (evaluate.tune never reads the test plans).

python folds.py build                       # fold lists of symbol tiles and room images -> detector-folds/<k>/
python folds.py predict <k> symbols|rooms <run> <imgsz>   # fold k plans -> preds-fold<k>-<run>.json
python folds.py merge <run>                 # preds-fold*-<run>.json -> preds-<run>.json
"""
import json
import random
import sys
from pathlib import Path

from prepare import OUT
from rooms import ROOMS
from tiles import TILES

FOLDS = OUT.parent / "detector-folds"
K = 5


def assignment():
    """plan -> fold (None for plans outside Stoa's set), written once so the remote machine reuses it."""
    path = FOLDS / "folds.json"
    if not path.exists():
        from prepare import STOA
        stoa = {p.stem for p in STOA.glob("*.png")}
        test = sorted(p.stem for p in (OUT / "images/test").glob("*.png"))
        rest = sorted(p.stem for s in ("train", "val") for p in (OUT / "images" / s).glob("*.png") if p.stem in stoa)
        random.Random(0).shuffle(rest)
        folds = {n: 0 for n in test} | {n: 1 + i % (K - 1) for i, n in enumerate(rest)}
        others = [p.stem for p in (OUT / "images/train").glob("*.png") if p.stem not in stoa]
        FOLDS.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(folds | {n: None for n in others}))
    return json.loads(path.read_text())


def plan_of(tile):
    return tile.stem.rsplit("-", 2)[0]


def build():
    folds = assignment()
    for root, kind, name in ((TILES, "symbols", plan_of), (ROOMS, "rooms", lambda p: p.stem)):
        images = sorted(p for s in ("train", "val", "test") for p in (root / "images" / s).glob("*.png"))
        names = (OUT / "data.yaml" if kind == "symbols" else ROOMS / "data.yaml").read_text().split("names:")[1]
        for k in range(K):
            d = FOLDS / str(k)
            d.mkdir(parents=True, exist_ok=True)
            train = [p for p in images if folds[name(p)] != k]
            held = [p for p in images if folds[name(p)] == k]
            (d / f"{kind}-train.txt").write_text("".join(f"{p}\n" for p in train))
            # validation only logs progress; the last epoch's weights are used, so held-out plans never pick a model
            (d / f"{kind}-val.txt").write_text("".join(f"{p}\n" for p in held[::10]))
            (d / f"{kind}.yaml").write_text(f"train: {d / f'{kind}-train.txt'}\nval: {d / f'{kind}-val.txt'}\nnames:{names}")


def predict(k, kind, run, imgsz):
    from ultralytics import YOLO

    import rooms
    import tiles

    folds = assignment()
    plans = sorted(p for s in ("train", "val", "test") for p in (OUT / "images" / s).glob("*.png") if folds[p.stem] == int(k))
    weights = next(Path("runs").glob(f"**/{run}/weights/last.pt"))
    detect = tiles.detect if kind == "symbols" else rooms.detect
    out = Path(f"preds-fold{k}-{run.split('-', 1)[1]}.json")
    out.write_text(json.dumps(detect(YOLO(weights), plans, imgsz)))
    print(out, len(plans))


def merge(run):
    preds = {}
    for p in sorted(Path().glob(f"preds-fold*-{run}.json")):
        preds |= json.loads(p.read_text())
    Path(f"preds-{run}.json").write_text(json.dumps(preds))
    print(len(preds), "plans")


if __name__ == "__main__":
    {"build": build, "predict": predict, "merge": merge}[sys.argv[1]](*sys.argv[2:])
