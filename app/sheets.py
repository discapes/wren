"""Vector PDF sheets (opening_design): ruled schedule tables → table and row nodes, rows outlined on the page."""
import re
from collections import defaultdict

import pymupdf
from shapely import STRtree, box, unary_union

from graph import DATA, Graph, Node, View

DIR = DATA / "data/opening_design"
TOL = 1.5  # pt: line positions closer than this are the same rule


def items():
    return sorted(str(p.relative_to(DIR)) for p in DIR.rglob("*.pdf"))


def rules(page):
    """Axis-aligned line segments in display coordinates: ([(y, x0, x1)], [(x, y0, y1)])."""
    m = page.rotation_matrix
    hs, vs = [], []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                segs = [(it[1], it[2])]
            elif it[0] == "re":
                r = it[1]
                segs = [(r.tl, r.tr), (r.bl, r.br), (r.tl, r.bl), (r.tr, r.br)]
            else:
                continue
            for a, b in segs:
                a, b = a * m, b * m
                if abs(a.y - b.y) < 0.5 and abs(a.x - b.x) > 3:
                    hs.append((a.y, min(a.x, b.x), max(a.x, b.x)))
                elif abs(a.x - b.x) < 0.5 and abs(a.y - b.y) > 3:
                    vs.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    return hs, vs


def levels(values):
    """Distinct positions, merging those within TOL."""
    out = []
    for v in sorted(values):
        if not out or v - out[-1] > TOL:
            out.append(v)
    return out


def grids(hs, vs):
    """Connected groups of rules that form a table grid: [(hs, vs)] with at least 3 rows and 2 columns."""
    geoms = [box(x0, y - 0.6, x1, y + 0.6) for y, x0, x1 in hs] + [box(x - 0.6, y0, x + 0.6, y1) for x, y0, y1 in vs]
    if not geoms:
        return []
    parts = unary_union(geoms)
    parts = list(getattr(parts, "geoms", [parts]))
    tree = STRtree(parts)
    groups = defaultdict(lambda: ([], []))
    for i, g in enumerate(geoms):
        k = int(tree.query(g, predicate="intersects")[0])
        (groups[k][0] if i < len(hs) else groups[k][1]).append(hs[i] if i < len(hs) else vs[i - len(hs)])
    # a schedule's header is often drawn as its own grid just above the body: join grids stacked at the same width
    found = sorted(groups.values(), key=lambda g: min(y for y, *_ in g[0]) if g[0] else 0)
    merged = []
    for h, v in found:
        if not h:
            continue
        top, x0, x1 = min(y for y, *_ in h), min(a for _, a, _ in h), max(b for *_, b in h)
        prev = next((m for m in merged if abs(m[2] - x0) < 3 and abs(m[3] - x1) < 3 and 0 <= top - m[4] < 25), None)
        if prev:
            prev[0].extend(h), prev[1].extend(v)
            prev[4] = max(y for y, *_ in h)
        else:
            merged.append([list(h), list(v), x0, x1, max(y for y, *_ in h)])
    return [(h, v) for h, v, *_ in merged if len(levels(y for y, *_ in h)) >= 4 and len(levels(x for x, *_ in v)) >= 3]


def cells(ys, vs, x0, x1, y0, y1):
    """Column boundaries of the band y0..y1: the vertical rules crossing its middle, plus the table edges."""
    mid = (y0 + y1) / 2
    return levels([x0, x1] + [x for x, a, b in vs if a - TOL <= mid <= b + TOL and x0 - TOL <= x <= x1 + TOL])


def table(hs, vs, words):
    """(title, columns, rows) of one grid; a row is (bbox, [cell texts])."""
    ys = levels(y for y, *_ in hs)
    x0, x1 = min(a for _, a, _ in hs), max(b for *_, b in hs)
    bands = []  # (y0, y1, column boundaries, cell texts)
    for y0, y1 in zip(ys, ys[1:]):
        xs = cells(ys, vs, x0, x1, y0, y1)
        texts = [" ".join(w[4] for w in words if a <= w[7] < b and y0 <= w[8] < y1) for a, b in zip(xs, xs[1:])]
        if any(texts):
            bands.append((y0, y1, xs, texts))
    if not bands:
        return None
    width = max(len(b[3]) for b in bands)
    first = next((i for i, b in enumerate(bands) if len(b[3]) == width and any(re.search(r"\d", t) for t in b[3])), None)
    if first is None:
        return None
    title = " ".join(t for b in bands[:first] if len(b[3]) == 1 for t in b[3])
    xs = bands[first][2]
    columns = []
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2
        parts = [t for _, _, bx, texts in bands[:first] if len(texts) > 1
                 for (c0, c1), t in zip(zip(bx, bx[1:]), texts) if c0 <= mid < c1 and t]
        parts = [p for i, p in enumerate(dict.fromkeys(parts)) if not any(q.startswith(p) for q in parts[i + 1:])]
        columns.append(" ".join(parts) or f"column {len(columns) + 1}")
    rows = []
    for y0, y1, bx, texts in bands[first:]:
        if len(texts) != width:  # a spanning line inside the body (notes, group headings) belongs to no column
            continue
        rows.append(((x0, y0, x1, y1), texts))
    return title, columns, rows


def read(rel, source="opening_design"):
    """The PDF data/<source>/<rel>: one view per page, its ruled tables as table and row nodes."""
    doc = pymupdf.open(DATA / "data" / source / rel)
    stem = re.sub(r"[^\w.-]+", "-", rel.removesuffix(".pdf")).strip("-")
    g = Graph(stem, source, rel.rsplit("/", 1)[-1].removesuffix(".pdf"))
    for pno, page in enumerate(doc):
        vid = f"p{pno + 1}"
        g.views[vid] = View(vid, f"data/{source}/{rel}", (0, 0, page.rect.width, page.rect.height), pno)
        sid = g.add(Node(f"sheet/{pno + 1}", "storey", "sheet", name=f"page {pno + 1}", via="pdf")).id
        m = page.rotation_matrix
        words = []
        for w in page.get_text("words"):
            r = pymupdf.Rect(w[:4]) * m
            words.append((*r, w[4], 0, 0, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
        for t, (hs, vs) in enumerate(grids(*rules(page))):
            got = table(hs, vs, words)
            if not got:
                continue
            title, columns, rows = got
            if len(rows) < 2 or all(c.startswith("column ") for c in columns):
                continue
            tid = f"{sid}/table/{t}"
            g.add(Node(tid, "table", "schedule", sid, title or None, attrs={"columns": columns}, via="pdf"))
            for i, (bb, texts) in enumerate(rows):
                nid = f"{tid}/row/{i}"
                g.add(Node(nid, "row", "row", tid, number=texts[0] or None,
                           attrs={c: v for c, v in zip(columns, texts) if v}, via="pdf"))
                bx0, by0, bx1, by1 = (round(v, 1) for v in bb)
                g.anchor(nid, vid, [[[bx0, by0], [bx1, by0], [bx1, by1], [bx0, by1], [bx0, by0]]])
    return g.derive()
