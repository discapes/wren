"""Score pdfplan on the Swiss PDFs with their DXF ignored: dataset template questions per template, next to the
DXF graph in graph.db on the same questions.

uv run score_pdfplan.py [number of PDFs] [misses shown per template]
"""
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import ask
import graph
import pdfplan
import score


def run(item):
    """[(question, template, pdfplan correct, its value, DXF graph correct)] for one PDF."""
    path, qs = item
    g, d = pdfplan.read(path.split("/", 2)[2]), graph.load(path.split("/")[-1][:-4])
    out = []
    for q in qs:
        a, b = ask.match(g, q["question"]), ask.match(d, q["question"])
        ok = lambda a: a is not None and a.value is not None and score.correct(q, a.value)
        out.append((q, a and a.via, ok(a), a and a.value, ok(b)))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    shown = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    by = defaultdict(list)
    for line in open(graph.DATA / "questions.jsonl"):
        q = json.loads(line)
        if "/swiss_dwellings/" in q["path"]:
            by[q["path"]].append(q)
    random.seed(1)
    paths = random.sample(sorted(by), min(n, len(by)))
    stats, misses = defaultdict(Counter), defaultdict(list)
    with ProcessPoolExecutor(10) as pool:
        for res in pool.map(run, [(p, by[p]) for p in paths]):
            for q, via, ok, value, dxf in res:
                stats[via].update(n=1, ok=ok, dxf=dxf)
                if not ok:
                    misses[via].append((q["path"].split("/")[-1][:-4], q["question"][:70], q["answer"], value))
    total = Counter()
    for via, s in sorted(stats.items(), key=lambda x: str(x[0])):
        total.update(s)
        print(f"{str(via):16} {s['ok']:6}/{s['n']:<6} {100 * s['ok'] / s['n']:5.1f}%   DXF graph {100 * s['dxf'] / s['n']:5.1f}%")
        for m in misses[via][:shown]:
            print("    ", m)
    print(f"total {total['ok']}/{total['n']} = {100 * total['ok'] / total['n']:.1f}% over {len(paths)} PDFs "
          f"(DXF graph {100 * total['dxf'] / total['n']:.1f}%)")
