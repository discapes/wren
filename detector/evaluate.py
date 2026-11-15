"""Score the symbol and room detectors on Stoa's cubicasa count questions.

python evaluate.py ensemble <out.json> <preds.json>...   # average confidences of matching boxes across models
python evaluate.py score <symbol preds.json> [<room preds.json>] [test|tuning]   # thresholds tuned on train+val plans; accuracy on Stoa's questions
"""
import json
import sys
from collections import Counter
from pathlib import Path

from prepare import CLASSES, OUT

QUESTIONS = Path.home() / "junction-data/stoa-full/junction-dataset/questions.jsonl"
# question prefix (English; Finnish duplicates are skipped) -> (model, classes whose counts are summed)
TEMPLATES = {
    "How many windows": ("symbols", ["window"]),
    "How many toilet bowls": ("symbols", ["toilet"]),
    "How many sinks": ("symbols", ["sink"]),
    "How many showers and bathtubs": ("symbols", ["shower", "bathtub"]),
    "How many doors": ("symbols", ["door"]),
    "How many saunas": ("symbols", ["sauna"]),
    "How many floors": ("rooms", ["storey"]),
    "How many bedrooms": ("rooms", ["bedroom"]),
    "How many kitchens": ("rooms", ["kitchen"]),
    "How many bathrooms": ("rooms", ["bathroom"]),
    "How many separate WC rooms": ("rooms", ["wc_room"]),
    "How many balconies": ("rooms", ["balcony"]),
}
THRESHOLDS = [i / 100 for i in range(5, 91, 5)]


def overlap(a, b):
    """Intersection over the smaller box: catches a pane box nested inside a whole-window box."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def dedupe(dets, thr, names):
    """Detections above their class threshold, minus boxes mostly covered by a more confident one of the same class.
    A shower and a bathtub on the same spot are one fixture."""
    same = lambda a, b: a == b or {names[a], names[b]} == {"shower", "bathtub"}
    kept = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if d[1] >= thr[names[d[0]]] and not any(same(k[0], d[0]) and overlap(k[2:], d[2:]) > 0.5 for k in kept):
            kept.append(d)
    return kept


def groups(boxes, gap=30):
    """Boxes lying within gap px of each other, grouped (the benches of one sauna)."""
    out = []
    for b in boxes:
        near = [g for g in out if any(b[0] < o[2] + gap and o[0] < b[2] + gap and b[1] < o[3] + gap and o[1] < b[3] + gap for o in g)]
        out = [g for g in out if g not in near] + [[b] + [o for g in near for o in g]]
    return out


def saunas(benches, rooms):
    """Sauna boxes: each cluster of benches, plus each sauna room holding no bench cluster (Stoa counts both)."""
    out = [(min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g)) for g in groups(benches)]
    return out + [r for r in rooms if not any(overlap(r, s) > 0.5 for s in out)]


def counts(dets, thr, names):
    return Counter(names[d[0]] for d in dedupe(dets, thr, names))


def tune(preds, labels, names):
    """Per class, the confidence threshold that gets the most counts exactly right (Stoa scores exact counts) on the
    train and val plans that have predictions; the test plans are never used. labels: a labels/ root with split dirs."""
    val = {p.stem: Counter(names[int(l.split()[0])] for l in p.read_text().splitlines() if l.strip())
           for split in ("train", "val") for p in (labels / split).glob("*.txt") if p.stem in preds}
    thr = {}
    for cls in names:
        err = {t: sum(counts(preds[n], {**{c: 2 for c in names}, cls: t}, names)[cls] != gt[cls] for n, gt in val.items())
               for t in THRESHOLDS}
        thr[cls] = min(err, key=err.get)
    return thr


def ensemble(out, *paths):
    """Cluster same-class boxes across models; a cluster's confidence is the mean over models (0 where a model missed it)."""
    runs = [json.loads(Path(p).read_text()) for p in paths]
    merged = {}
    for name in runs[0]:
        clusters = []  # [class, summed conf, box of the most confident member]
        for dets in (r[name] for r in runs):
            for c, s, *box in sorted(dets, key=lambda d: -d[1]):
                hit = next((k for k in clusters if k[0] == c and overlap(k[2], box) > 0.5), None)
                if hit:
                    hit[1] += s
                else:
                    clusters.append([c, s, box])
        merged[name] = [[c, round(s / len(runs), 3), *box] for c, s, box in clusters]
    Path(out).write_text(json.dumps(merged))


def score(symbols, rooms=None, plans="test"):
    """Accuracy on Stoa's questions for the test plans, or with plans=tuning for the train and val plans that have
    predictions (where choices between models and rules are made, so the test plans are scored once at the end)."""
    from rooms import CLASSES as ROOM_CLASSES, ROOMS

    models = {"symbols": (json.loads(Path(symbols).read_text()), CLASSES, OUT / "labels")}
    if rooms:
        models["rooms"] = (json.loads(Path(rooms).read_text()), ROOM_CLASSES, ROOMS / "labels")
    thr = {m: tune(preds, labels, names) for m, (preds, names, labels) in models.items()}

    splits = ("test",) if plans == "test" else ("train", "val")
    chosen = {p.stem for s in splits for p in (OUT / "labels" / s).glob("*.txt")}
    for preds, _, _ in models.values():
        chosen &= set(preds)
    rows = []
    for line in QUESTIONS.open():
        q = json.loads(line)
        name = Path(q["path"]).stem
        if "/cubicasa5k/" not in q["path"] or name not in chosen:
            continue
        model, classes = next((v for k, v in TEMPLATES.items() if q["question"].startswith(k)), (None, None))
        if model in models:
            kept = {m: dedupe(preds[name], thr[m], names) for m, (preds, names, _) in models.items()}
            c = Counter(models[m][1][d[0]] for m in kept for d in kept[m])
            boxes = lambda m, cls: [d[2:] for d in kept.get(m, []) if models[m][1][d[0]] == cls]
            c["sauna"] = len(saunas(boxes("symbols", "sauna_bench"), boxes("rooms", "sauna_room")))
            rows.append(("+".join(classes), int(q["answer"]), sum(c[k] for k in classes)))

    lines = [f"thresholds: {thr}", "", f"{plans} plans", "", "| question | n | exact | within ±1 | mean abs error |", "|---|---|---|---|---|"]
    for key in sorted({r[0] for r in rows}) + ["all"]:
        sel = [r for r in rows if key in (r[0], "all")]
        n = len(sel)
        exact = sum(a == p for _, a, p in sel) / n
        near = sum(abs(a - p) <= 1 for _, a, p in sel) / n
        mae = sum(abs(a - p) for _, a, p in sel) / n
        lines.append(f"| {key} | {n} | {exact:.1%} | {near:.1%} | {mae:.2f} |")
    report = "\n".join(lines)
    if plans == "test":
        Path(symbols).with_suffix(".md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    {"ensemble": ensemble, "score": score}[sys.argv[1]](*sys.argv[2:])
