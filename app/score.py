"""Score answers against questions.jsonl (or another question file): counts exact, areas/lengths/volumes within 2%.

uv run score.py [source | file.jsonl] [sample size] [--llm] [--out results.jsonl]
--llm: the model answers what no template covers; --out: every question's answer, steps and verdict
"""
import json
import re
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import agent
import ask
from graph import DATA


def correct(q, value):
    if not isinstance(value, (int, float)):
        return False
    if q["type"] == "count":
        return value == q["answer"]
    return abs(value - q["answer"]) <= 0.02 * abs(q["answer"])


def solve(q, llm):
    """(question, answer, correct): templates first, the model for the rest when llm is on."""
    try:
        a, _, _ = ask.answer(q["path"], q["question"], llm)
    except Exception as e:
        a = ask.Answer(None, "", [], repr(e), "llm")
    if a is None:
        return q, a, False
    if q["type"] == "text":  # "not in the model" is a valid answer to some text questions, so null values are judged too
        return q, a, agent.judge(q["question"], q["answer"], f"{a.value} {a.unit}. {a.text}")
    return q, a, a.value is not None and correct(q, a.value)


def main(source=None, n=None, llm=False, out=None):
    path = DATA / "questions.jsonl"
    if source and source.endswith(".jsonl"):
        path, source = Path(source), None
    qs = [json.loads(line) for line in open(path)]
    qs = [q for q in qs if (llm or q["type"] != "text") and (source is None or f"/{source}/" in q["path"])]
    # cubicasa models trained on part of Stoa's plans: score only the held-out test split
    test = {p.stem for p in (Path.home() / "junction-data/detector-ds/images/test").glob("*.png")}
    qs = [q for q in qs if "/cubicasa5k/" not in q["path"] or Path(q["path"]).stem in test]
    if n:
        random.seed(0)
        qs = random.sample(qs, min(n, len(qs)))
    stats = defaultdict(Counter)
    misses = defaultdict(list)
    tokens, usd = 0, 0.0
    results = open(out, "w") if out else None
    with ThreadPoolExecutor(4 if llm else 1) as pool:
        for q, a, ok in pool.map(lambda q: solve(q, llm), qs):
            if results:
                results.write(json.dumps({**q, "ok": ok, "value": a and a.value, "text": a and a.text, "via": a and a.via,
                                          "usage": a and a.usage, "steps": a and [w for w, _ in a.steps]},
                                         ensure_ascii=False, default=str) + "\n")
            key = (q["path"].split("/")[1], a.via if a else "unmatched")
            stats[key]["n"] += 1
            stats[key]["ok"] += ok
            tokens += int(m.group(1)) if a and (m := re.search(r"(\d+) tokens", a.usage)) else 0
            usd += float(m.group(1)) if a and (m := re.search(r"\$([\d.]+)", a.usage)) else 0
            if not ok:
                misses[key].append((q["path"].split("/")[-1], q["question"][:110], str(q["answer"])[:80], a and str(a.value)[:80]))
    total = Counter()
    for key, s in sorted(stats.items()):
        total.update(s)
        print(f"{key[0]:16} {key[1]:18} {s['ok']:6}/{s['n']:<6} {100 * s['ok'] / s['n']:5.1f}%")
        for m in misses[key][:3]:
            print("    ", m)
    print(f"total {total['ok']}/{total['n']} = {100 * total['ok'] / total['n']:.1f}%, model tokens {tokens}, ${usd:.2f}")


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    args = [a for a in sys.argv[1:] if a not in ("--llm", "--out", out)]
    main(args[0] if args else None, int(args[1]) if len(args) > 1 else None, "--llm" in sys.argv, out)
