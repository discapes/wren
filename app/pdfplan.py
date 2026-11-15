"""Vector PDF floor plans: room tags, room and apartment outlines, door swings and legend symbols → graph nodes.

Reads the page's text and vector paths only (no model):
- room tags: stacked text lines with a name, a room number and/or a printed area ("Bad", "3.05", "15.61 m²",
  "205 SF")
- room outlines: the first drawn filled shape around the tag that holds no other tag, then refined to the smallest
  shape in the line style most room outlines on the sheet share
- apartments: the smallest shape around an apartment label ("Whg. A") that holds two rooms or more; unlabelled
  shapes in the same line style are units too
- scale from a printed "1:100" or "1/4" = 1'-0"", else from printed areas against outline areas
- doors: 60–120° swing arcs (curves or short line pieces) with a leaf line from the hinge; a double door's two
  leaves are one door
- fixtures and windows: the legend's symbols (a drawing left of a word such as "WC" or "Fenster") matched by the
  structure of their paths against every closed shape on the plan
"""
import re
from collections import Counter, defaultdict
from statistics import median

import numpy as np
import pymupdf
import shapely
from shapely import STRtree, unary_union
from shapely.affinity import scale as scale_by, translate
from shapely.geometry import MultiPoint, Point, Polygon, box

import vocab
from graph import DATA, Graph, Node, View

PT = 0.0254 / 72  # m per PDF point at 1:1
SCALE = re.compile(r"\b1\s?:\s?(\d{1,4})\b")
FEET = re.compile(r"(\d+)(?:/(\d+))?\"\s?=\s?1'\s?-?\s?0\"")  # 3/32" = 1'-0"
AREA = re.compile(r"(\d+(?:[.,]\d+)*)\s?(m²|m2|SF|sq\.? ?ft|ft²)(?!\w)", re.I)
SQFT = 0.09290304
REF = re.compile(r"[A-Z]{0,3}[-.]?\d[\w./-]*")
NUMBER = re.compile(r"[A-Z]{0,2}\d+(?:[.-]\d+)*[a-z]?")
UNIT = re.compile(r"(?:Whg|Wohnung|App|Appartement|Apt|Apartment|Unit|Asunto|As)\.?\s?([A-Z]{1,2}|\d{1,3})", re.I)
SYMBOLS = {  # legend words → category of the symbol drawn left of them
    "toilet": r"WC|Toilette?s?|WC-istuin",
    "washbasin": r"Lavabos?|Waschbecken|Washbasins?|Lavandino|Pesuallas",
    "sink": r"Sinks?|Spüle|Evier|Tiskiallas",
    "bathtub": r"Badewannen?|Bathtubs?|Baignoires?|Vasca da bagno|Kylpyamme",
    "shower": r"Duschen?|Showers?|Douches?|Doccia|Suihku",
    "window": r"Fenster|Windows?|Fenêtres?|Finestra|Ikkuna",
    "stove": r"Küchenzeile|Kochherd|Herd|Stove|Cuisinière|Liesi",
    "washing_machine": r"Waschmaschine|Washing machine|Lave-linge|Pyykinpesukone",
}
SYMBOLS = {cat: re.compile(rf"(?:{w})", re.I) for cat, w in SYMBOLS.items()}
KIND = {"window": "opening"}


BEZIER = [((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t ** 3) for t in np.linspace(0, 1, 9)]


def matrix_points(items, m):
    """Path items → rings of display coordinates; a new ring starts where an item does not continue the last."""
    a, b, c, d, e, f = m
    rings = []
    for it in items:
        if it[0] == "re":
            r = it[1]
            pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
        elif it[0] == "qu":
            q = it[1]
            pts = [(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll, q.ul)]
        elif it[0] == "c":
            ps = [(p.x, p.y) for p in it[1:5]]
            pts = [tuple(sum(w * p[i] for w, p in zip(ws, ps)) for i in (0, 1)) for ws in BEZIER]
        else:
            pts = [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        if (a, b, c, d, e, f) != (1, 0, 0, 1, 0, 0):
            pts = [(a * x + c * y + e, b * x + d * y + f) for x, y in pts]
        if it[0] in ("re", "qu") or not rings or abs(rings[-1][-1][0] - pts[0][0]) + abs(rings[-1][-1][1] - pts[0][1]) > 0.05:
            rings.append(pts)
        else:
            rings[-1].extend(pts[1:])
    return rings


def shapes(page):
    """[(signature, rings, bbox, filled, stroke style)] for every path; the signature names its items with rectangles
    and quads as 4 lines."""
    m = page.rotation_matrix
    out, clips = [], {}
    for d in page.get_drawings(extended=True):
        if d["type"] == "clip":
            clips = {lv: r for lv, r in clips.items() if lv < d["level"]} | {d["level"]: tuple(d["scissor"])}
            continue
        if d["type"] == "group":
            continue
        # a path cut by a clip (a hatch tile clipped to a wall) does not show its own outline
        x0, y0, x1, y1 = d["rect"]
        if any(lv < d["level"] and (x0 < r[0] - 1 or y0 < r[1] - 1 or x1 > r[2] + 1 or y1 > r[3] + 1)
               for lv, r in clips.items()):
            continue
        sig = "".join(it[0] for it in d["items"]).replace("re", "llll").replace("qu", "llll")
        rings = matrix_points(d["items"], m)
        style = (tuple(round(c, 2) for c in d.get("color") or ()), round(d.get("width") or 0, 2), d.get("dashes"))
        out.append((sig, rings, tuple(d["rect"] * m), d.get("fill") is not None, style))
    return out


def polygons(paths):
    """Closed rings of the paths as valid polygons, and the stroke style of each. Circles and ellipses (grid bubbles,
    tags, fixtures) are left out: rooms have straight walls somewhere."""
    out, styles = [], []
    for sig, rings, _, _, style in paths:
        if set(sig) == {"c"}:
            continue
        for r in rings:
            if len(r) >= 4 and abs(r[0][0] - r[-1][0]) + abs(r[0][1] - r[-1][1]) < 0.1:
                p = Polygon(r)
                p = p if p.is_valid else p.buffer(0)
                if p.area > 1 and p.geom_type == "Polygon":
                    out.append(p)
                    styles.append(style)
    return out, styles


def lines(page):
    """Horizontal text lines: [(text, (x0, y0, x1, y1))] in display coordinates."""
    m = page.rotation_matrix
    out = []
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", ()):
            text = " ".join(s["text"].strip() for s in ln["spans"] if s["text"].strip())
            if text:
                out.append((text, tuple(pymupdf.Rect(ln["bbox"]) * m)))
    return out


def parse(text):
    """A text line → (name, number, printed area)."""
    area = None
    if a := AREA.search(text):
        if a.group(2).lower().startswith("m"):
            area = float(a.group(1).replace(",", "."))
        else:
            area = round(float(a.group(1).replace(",", "")) * SQFT, 2)
        text = (text[:a.start()] + text[a.end():]).strip()
    words = text.split()
    number = None
    if words and NUMBER.fullmatch(words[-1]) and (len(words) == 1 or not NUMBER.fullmatch(words[0])):
        number = words.pop()
    if words and NUMBER.fullmatch(words[0]) and number is None and len(words) > 1:  # "3.05 Bad"
        number = words.pop(0)
    name = " ".join(words) or None
    # a sheet or detail reference ("A-301") is no room name, nor title block text (addresses, notes)
    if name and (REF.fullmatch(name) or re.search(r"[,|@:;]", name) or len(re.sub(r"\(.*?\)", "", name).split()) > 3):
        name = None
    return name, number, area


def tags(page_lines):
    """Room tags: text lines stacked into one label → [{name, number, area, rect}]; a room may have only a number."""
    ls = sorted(page_lines, key=lambda t: t[1][1])
    groups = []
    for text, r in ls:
        h = r[3] - r[1]
        cx = (r[0] + r[2]) / 2
        g = next((g for g in groups if 0 <= r[1] - g["rect"][3] < 0.6 * h and abs(cx - (g["rect"][0] + g["rect"][2]) / 2)
                  < max(r[2] - r[0], g["rect"][2] - g["rect"][0]) / 2), None)
        if g is None:
            g = {"parts": [], "rect": r}
            groups.append(g)
        g["parts"].append(text)
        g["rect"] = (min(g["rect"][0], r[0]), min(g["rect"][1], r[1]), max(g["rect"][2], r[2]), max(g["rect"][3], r[3]))
    out = []
    for g in groups:
        names, numbers, areas = [], [], []
        for text in g["parts"]:
            name, number, area = parse(text)
            names += [name] if name else []
            numbers += [number] if number else []
            areas += [area] if area is not None else []
        name = " ".join(names) or None
        if (name and re.search(r"[^\W\d_]", name) or not name and numbers) and len(numbers) <= 1 and len(areas) <= 1:
            out.append({"name": name, "number": numbers[0] if numbers else None,
                        "area": areas[0] if areas else None, "rect": g["rect"]})
    return out


def enclosing(polys, tree, point, at_least=0.0, ok=lambda p: True, first_drawn=False):
    """The smallest (or first drawn) polygon containing the point with more than `at_least` area that passes `ok`."""
    hits = [polys[i] for i in sorted(tree.query(point, predicate="within"))]
    hits = hits if first_drawn else sorted(hits, key=lambda p: p.area)
    return next((p for p in hits if p.area > at_least and ok(p)), None)


def chains(segments):
    """Polylines through segments that meet end to end, where no third segment joins: [[points]]."""
    key = lambda p: (round(p[0], 1), round(p[1], 1))
    ends = defaultdict(list)
    for i, (a, b) in enumerate(segments):
        ends[key(a)].append(i)
        ends[key(b)].append(i)
    seen, out = set(), []
    for i in range(len(segments)):
        if i in seen:
            continue
        seen.add(i)
        line = list(segments[i])
        for forward in (True, False):  # grow from the last point, then from the first
            while True:
                tip = key(line[-1] if forward else line[0])
                nxt = [j for j in ends[tip] if j not in seen]
                if len(ends[tip]) != 2 or not nxt:
                    break
                j = nxt[0]
                seen.add(j)
                a, b = segments[j]
                p = b if key(a) == tip else a
                line = line + [p] if forward else [p] + line
        out.append(line)
    return out


def arcs(paths, k):
    """Door swings: 60–120° arcs of a 0.4–5 m radius (12–200 pt unscaled), drawn as curves or as short line pieces,
    with a leaf line from the hinge to one end of the arc.
    → [(hinge, radius, end points, polygon of the swing)]"""
    lo, hi = (0.4 / k, 5.0 / k) if k else (12, 200)
    candidates, pieces = [], []
    for sig, rings, *_ in paths:
        if sig and set(sig) == {"c"} and len(rings) == 1:
            candidates.append(rings[0])
        elif sig and set(sig) == {"l"}:
            for r in rings:
                if len(r) > 4:
                    candidates.append(r)
                elif len(r) == 2 and np.hypot(r[1][0] - r[0][0], r[1][1] - r[0][1]) < 0.4 * hi:
                    pieces.append(r)
    candidates += [c for c in chains(pieces) if len(c) > 4]
    # door leaves: straight segments, from every path, to find the one running from an arc's hinge to its end
    segments = [(p, q) for sig, rings, *_ in paths if "l" in sig for r in rings for p, q in zip(r, r[1:])]
    ends = np.array(segments).reshape(-1, 4) if segments else np.zeros((0, 4))
    seg_tree = STRtree(shapely.box(np.minimum(ends[:, 0], ends[:, 2]), np.minimum(ends[:, 1], ends[:, 3]),
                                   np.maximum(ends[:, 0], ends[:, 2]), np.maximum(ends[:, 1], ends[:, 3])))
    near = lambda u, v, tol: abs(u[0] - v[0]) + abs(u[1] - v[1]) < tol
    out = []
    for pts in map(np.array, candidates):
        a, b, mid = pts[0], pts[-1], pts[len(pts) // 2]
        (x1, y1), (x2, y2), (x3, y3) = a, mid, b  # the circle through a, mid, b
        d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(d) < 1e-6 or np.hypot(*(a - b)) < 1e-3:
            continue
        ux = ((x1 * x1 + y1 * y1) * (y2 - y3) + (x2 * x2 + y2 * y2) * (y3 - y1) + (x3 * x3 + y3 * y3) * (y1 - y2)) / d
        uy = ((x1 * x1 + y1 * y1) * (x3 - x2) + (x2 * x2 + y2 * y2) * (x1 - x3) + (x3 * x3 + y3 * y3) * (x2 - x1)) / d
        c = np.array([ux, uy])
        r = np.hypot(*(a - c))
        if not lo <= r <= hi or np.abs(np.hypot(*(pts - c).T) - r).max() > 0.05 * r:
            continue
        sweep = np.degrees(np.arccos(np.clip(np.dot(a - c, b - c) / r / r, -1, 1)))
        leaf = any(near(p, c, 0.1 * r) and (near(q, a, 0.1 * r) or near(q, b, 0.1 * r)) for i in seg_tree.query(Point(c).buffer(0.1 * r))
                   for p, q in (segments[i], segments[i][::-1]))
        if 60 <= sweep <= 120 and leaf:
            out.append((c, r, (a, b), MultiPoint([*pts, c]).convex_hull))
    return out


def doors(swings):
    """Swing arcs grouped into doors: two leaves whose arc ends meet, hinged a leaf width apart each, are one double
    door. → [polygon]
    An arc drawn twice at the same hinge stays two doors, as the plan's source holds two door entities there."""
    parent = list(range(len(swings)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, (ci, ri, ei, _) in enumerate(swings):
        for j in range(i):
            cj, rj, ej, _ = swings[j]
            if abs(np.hypot(*(ci - cj)) - ri - rj) < 0.2 * (ri + rj) and abs(ri - rj) < 0.35 * max(ri, rj) and min(np.hypot(*(p - q)) for p in ei for q in ej) < 0.08 * ri:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, s in enumerate(swings):
        groups[find(i)].append(s[3])
    return [unary_union(g).envelope for g in groups.values()]


def legend(paths, page_lines):
    """Legend entries: {signature: category} for each symbol word with a drawing just left of it."""
    found = {}
    for text, r in page_lines:
        cat = next((c for c, rx in SYMBOLS.items() if rx.fullmatch(text.strip())), None)
        if not cat:
            continue
        h = r[3] - r[1]
        region = (r[0] - 8 * h, r[1] - h, r[0] - 0.2 * h, r[3] + h)
        inside = [p for p in paths if p[2][0] >= region[0] and p[2][2] <= region[2]
                  and p[2][1] >= region[1] and p[2][3] <= region[3]]
        if inside:
            inside.sort(key=lambda p: (p[2][2] - p[2][0]) * (p[2][3] - p[2][1]), reverse=True)
            found[signature(inside[0], inside[1:])] = cat
    return found


def signature(outer, inner):
    """Structure of a symbol: the items of its outline path, then the sorted items of the paths inside it. A single
    line running the outline's full length along its long side is "l=" (window glazing), any other line "l"."""
    ring = np.array(outer[1][0])
    edges = np.diff(ring, axis=0)
    u = edges[np.argmax(np.hypot(*edges.T))] if len(edges) else np.zeros(2)

    def tag(p):
        if p[0] != "l":
            return p[0]
        v = np.subtract(p[1][0][-1], p[1][0][0])
        nu, nv = np.hypot(*u), np.hypot(*v)
        return "l=" if nu and abs(u[0] * v[1] - u[1] * v[0]) < 0.05 * nu * nv and nv > 0.8 * nu else "l"

    return (outer[0], *sorted(tag(p) for p in inner))


def inside(b, outer, tol=0.1):
    """Whether bbox b lies within the outer bbox and is not a line drawn along one of its edges (a wall end)."""
    x0, y0, x1, y1 = outer
    if not (b[0] >= x0 - tol and b[1] >= y0 - tol and b[2] <= x1 + tol and b[3] <= y1 + tol):
        return False
    on_edge = (b[2] - b[0] < tol and min(abs(b[0] - x0), abs(b[0] - x1)) < tol
               or b[3] - b[1] < tol and min(abs(b[1] - y0), abs(b[1] - y1)) < tol)
    return not on_edge


def symbols(paths, protos, exclude):
    """Plan symbols matching a legend signature: [(category, polygon)]; `exclude` is the legend area.
    A symbol drawn twice on the same spot is two symbols (the source holds two entities there)."""
    if not protos:
        return []
    copies = Counter((p[0], tuple(round(v, 1) for v in p[2])) for p in paths)
    paths = list({(p[0], tuple(round(v, 1) for v in p[2])): p for p in paths}.values())
    boxes = shapely.box(*np.array([p[2] for p in paths]).T)
    tree = STRtree(boxes)
    outers = {s[0] for s in protos}
    out, used = [], set()
    for i, p in enumerate(paths):
        if p[0] not in outers or i in used or boxes[i].area < 1 or exclude.intersects(boxes[i]):
            continue
        inner = [j for j in tree.query(boxes[i]) if j != i and inside(paths[j][2], p[2])]
        sig = signature(p, [paths[j] for j in inner])
        if sig in protos:
            out += [(protos[sig], boxes[i])] * copies[p[0], tuple(round(v, 1) for v in p[2])]
            used.update(inner)
    return out


def scale(text, rooms):
    """m per PDF point: from a printed 1:N or 3/32" = 1'-0", else from printed areas against the outline areas (pt²)."""
    ns = Counter(int(n) for n in SCALE.findall(text) if 10 <= int(n) <= 1000)
    ns.update(round(12 * int(b or 1) / int(a)) for a, b in FEET.findall(text))
    if ns:
        n = ns.most_common(1)[0][0]
        return n * PT, f"1:{n} printed"
    ratios = [(t["area"] / p.area) ** 0.5 for t, p in rooms if t["area"] and p is not None]
    if len(ratios) >= 3:
        return median(ratios), f"from {len(ratios)} printed room areas"
    return None, "unknown"


def category(name):
    """Room category of a label; "2BD - LIVING", "ST - BEDROOM" name a unit type before the room."""
    name = name or ""
    return next((c for part in [name, *name.split(" - ")[1:]] if (c := vocab.room_cat(part)) != "unknown"), "unknown")


def rings(geom):
    return [[[round(x, 2), round(y, 2)] for x, y in p.exterior.coords] for p in getattr(geom, "geoms", [geom])]


def add(g, page, vid, sid, offset=0.0, skip=()):
    """Floor-plan content of one PDF page as nodes under the storey `sid`, outlined in view `vid`.
    Geometry is in metres when the page's scale is known (else in points), shifted right by `offset` so pages of one
    graph never overlap in derive. Rooms without a measured outline get no geometry, so derive gives them no area.
    Text inside the `skip` boxes (schedule tables already read) is not a room tag."""
    paths = shapes(page)
    polys, styles = polygons(paths)
    tree = STRtree(polys)
    filled, filled_styles = polygons([p for p in paths if p[3]])
    ftree = STRtree(filled)
    page_lines = [(t, r) for t, r in lines(page) if not any(
        a[0] <= (r[0] + r[2]) / 2 <= a[2] and a[1] <= (r[1] + r[3]) / 2 <= a[3] for a in skip)]
    units = [(m.group(1), r, t) for t, r in page_lines if (m := UNIT.fullmatch(t))]
    unit_rects = {r for _, r, _ in units}
    found = [t for t in tags([l for l in page_lines if l[1] not in unit_rects])
             if t["number"] or t["area"] is not None or vocab.room_cat(t["name"]) != "unknown"]
    page_area = page.rect.width * page.rect.height
    centres = [box(*t["rect"]).centroid for t in found]

    style_of = {id(p): st for p, st in zip(filled + polys, filled_styles + styles)}

    def alone(p, c):
        """Whether p holds no tag but c's, apart from tags in smaller shapes inside p (a lift inside a stair hall)."""
        return all(d is c or not p.contains(d) or any(
            q.area < p.area and not q.contains(c) and q.within(p.buffer(0.5))
            for q in (polys[i] for i in tree.query(d, predicate="within"))) for d in centres)

    def room_of(t, c, style=None):
        """The tag's room: the first drawn filled shape around it that holds no other tag (a room's fill lies under
        the stairs, fixtures and furniture drawn on it), else the smallest unfilled one. With `style`, the smallest
        shape drawn in the line style of this sheet's rooms."""
        args = c, 1.5 * box(*t["rect"]).area, lambda p: p.area < 0.25 * page_area and alone(p, c)
        if style:
            return enclosing(*styled, *args)
        return enclosing(filled, ftree, *args, first_drawn=True) or enclosing(polys, tree, *args)

    rooms = [(t, room_of(t, c)) for t, c in zip(found, centres)]
    common = Counter(style_of[id(p)] for _, p in rooms if p is not None).most_common(1)
    if common:
        same = [p for p, st in zip(polys, styles) if st == common[0][0]]
        styled = same, STRtree(same)
        rooms = [(t, room_of(t, c, common[0][0]) or p) for (t, p), c in zip(rooms, centres)]
    # outside a closed shape a tag needs a printed area, or a number and a known room name: other text there is
    # legend, title block, captions ("DOOR TYPE A1"), scale bars or notes
    rooms = [(t, p) for t, p in rooms if p is not None or t["name"] and (
        t["area"] is not None or t["number"] and category(t["name"]) != "unknown")]
    k, how = scale(" ".join(t for t, _ in page_lines), rooms)
    g.meta.setdefault("pdf_scale", {})[vid] = how
    if k:
        g.meta["m_per_unit"] = 1.0
    # a bare number names a room only in a room-sized shape (door, wall and detail tags sit in small frames)
    rooms = [(t, p) for t, p in rooms if t["name"] or p.area > 4 * box(*t["rect"]).area and (not k or p.area * k * k > 0.5)]
    # an outline far off the printed area is some other shape around the tag
    rooms = [(t, None if p is not None and t["area"] and k and not 0.67 < p.area * k * k / t["area"] < 1.5 else p)
             for t, p in rooms]
    outlines = [p for _, p in rooms if p is not None]
    room_styles = {style_of[id(p)] for p in outlines if id(p) in style_of}
    otree = STRtree(outlines) if outlines else None

    def node(n, outline, model=True):
        g.add(n)
        g.anchor(n.id, vid, rings(outline))
        n.geom = translate(scale_by(outline, k or 1, k or 1, origin=(0, 0)), offset) if model else None

    # apartments: the shape around each label that holds two rooms or more; shapes drawn in the same line style
    # without a label are units too (stair cores, shared areas)
    unit_polys, unit_styles, unit_ids = [], set(), []
    for letter, r, text in units:
        p = enclosing(polys, tree, box(*r).centroid, 0, lambda p: sum(p.contains(d) for d in centres) >= 2)
        if p is not None:
            unit_polys.append(p)
            unit_styles.add(styles[polys.index(p)])
        nid = f"{sid}/unit/{letter}"
        node(Node(nid, "unit", "apartment", sid, text, letter, via="pdf-plan"), p if p is not None else box(*r))
        unit_ids.append(nid)
    unit_styles -= room_styles  # a line style shared with rooms does not mark units
    for i, p in enumerate(p for p, s in zip(polys, styles) if s in unit_styles and not any(p.equals(u) for u in unit_polys)):
        node(Node(f"{sid}/unit/{i}", "unit", "unit", sid, via="pdf-plan"), p)
        unit_polys.append(p)
        unit_ids.append(f"{sid}/unit/{i}")

    for i, (t, p) in enumerate(rooms):
        if p is not None and otree is not None:  # a room drawn inside another is cut out of it
            inner = [outlines[j] for j in otree.query(p, predicate="contains_properly")]
            p = p.difference(unary_union(inner)) if inner else p
        cat = category(t["name"])
        attrs = {"outdoor": cat in vocab.OUTDOOR}
        if t["area"] is not None:
            attrs["area_printed_m2"] = t["area"]
        nid = f"{sid}/sp/{t['number']}" if t["number"] and f"{sid}/sp/{t['number']}" not in g.nodes else f"{sid}/sp/{i}"
        n = Node(nid, "space", cat, sid, t["name"], t["number"], attrs=attrs, conf=1.0 if p is not None else 0.7,
                 via="pdf-plan")
        node(n, p if p is not None else box(*t["rect"]), p is not None and k is not None)
        if n.geom is None:  # derive places rooms in units by their outline; this one has none, so by its tag
            c = box(*t["rect"]).centroid
            n.parent = min(((u.area, nid) for u, nid in zip(unit_polys, unit_ids) if u.contains(c)), default=(0, sid))[1]
    # the legend's symbols and words are not on the plan
    protos = legend(paths, page_lines)
    legend_words = [box(*r) for t, r in page_lines if any(rx.fullmatch(t.strip()) for rx in SYMBOLS.values())]
    exclude = unary_union([b.buffer(10 * (b.bounds[3] - b.bounds[1])) for b in legend_words]) if legend_words else Polygon()
    for i, d in enumerate(x for x in doors(arcs(paths, k)) if not exclude.intersects(x)):
        node(Node(f"{sid}/door/{i}", "opening", "door", sid, via="pdf-plan"), d)
    for i, (cat, b) in enumerate(symbols(paths, protos, exclude)):
        node(Node(f"{sid}/{cat}/{i}", KIND.get(cat, "element"), cat, sid, via="pdf-legend"), b)


def add_pages(g, rel, source):
    """Every page of the PDF as a storey ("sheet/<n>", view "p<n>", as sheets.read names them) with its floor-plan
    content."""
    for pno, page in enumerate(pymupdf.open(DATA / "data" / source / rel)):
        vid = f"p{pno + 1}"
        g.views.setdefault(vid, View(vid, f"data/{source}/{rel}", (0, 0, page.rect.width, page.rect.height), pno))
        sid = f"sheet/{pno + 1}"
        if sid not in g.nodes:
            g.add(Node(sid, "storey", "sheet", name=f"page {pno + 1}", via="pdf"))
        tables = [a.shape[0] for n in g.find(kind="row") for a in g.anchors[n.id] if a.view == vid and a.shape]
        add(g, page, vid, sid, pno * 1e5, [(*r[0], *r[2]) for r in tables])
    g.derive()
    # derive puts a room in any unit around it; where units overlap (a stair core drawn inside an apartment's
    # boundary), the smallest is the room's
    units = [n for n in g.find(kind="unit") if n.geom is not None]
    for s in g.find(kind="space"):
        if s.geom is not None:
            p = s.geom.representative_point()
            s.parent = min(((u.geom.area, u.id) for u in units if u.geom.contains(p)), default=(0, s.parent))[1]
    g.index()
    g.manifest = g.make_manifest()
    return g


def read(rel, source="swiss_dwellings"):
    """The PDF data/<source>/<rel> as a floor-plan graph, one storey per page."""
    stem = re.sub(r"[^\w.-]+", "-", rel.removesuffix(".pdf")).strip("-")
    g = Graph(stem, source, rel.rsplit("/", 1)[-1].removesuffix(".pdf"))
    return add_pages(g, rel, source)
