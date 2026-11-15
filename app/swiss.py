"""Swiss Dwellings reader: semantic DXF layers → graph, with the PDF and PNG renders as views."""
import re
from collections import Counter

import ezdxf
from ezdxf import bbox
import numpy as np
import pymupdf
from PIL import Image
from shapely import STRtree, unary_union
from shapely.geometry import LineString, Point, Polygon

import vocab
from graph import DATA, Graph, Node, View

DIR = DATA / "data/swiss_dwellings"
UNIT = re.compile(r"(?:Whg|App)\. (\S+)")
NUMBER = re.compile(r"[A-Z]?\d+(\.\d+)*")
AREA = re.compile(r"([\d.]+) m²")


def items():
    return sorted(p.stem for p in DIR.glob("*.dxf"))


def points(e):
    return [tuple(p[:2]) for p in e.get_points()]


def closed(e):
    p = points(e)
    return e.closed or (len(p) > 2 and p[0] == p[-1])


def shape(e):
    t = e.dxftype()
    if t == "CIRCLE":
        return Point(e.dxf.center.x, e.dxf.center.y).buffer(e.dxf.radius)
    if t == "ARC":
        return LineString([tuple(v)[:2] for v in e.flattening(0.02)])
    p = points(e)
    return LineString(p) if len(p) > 1 else Point(p[0])


def fixture_cat(sig):
    """Fixture category from the entities drawn inside its bounding rectangle (P<n> = n-point polyline, C = circle)."""
    if sig == "P2 P48":
        return "toilet"
    if sig == "C P48":
        return "washbasin"
    if "P69" in sig:
        return "bathtub"
    if sig == "C P5":
        return "shower"
    if sig.count("C") >= 4:
        return "stove"
    if sig == "P2 P2":
        return "shaft"
    if sig and set(sig.split()) == {"P2"}:
        return "stair"
    return "fixture"


def fit(msp, page):
    """Similarity transform DXF → PDF points (y flipped) from room texts that occur once in both."""
    words = {}
    for w in page.get_text("words"):
        words.setdefault(w[4], []).append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2))
    texts = list(msp.query('TEXT[layer=="ROOM_TEXT"]'))
    once = Counter(e.dxf.text for e in texts)
    A, b = [], []
    for e in texts:
        t = e.dxf.text
        if once[t] == 1 and len(words.get(t, ())) == 1:
            (x, y), (u, v) = tuple(e.dxf.insert)[:2], words[t][0]
            A += [[x, 1, 0], [-y, 0, 1]]
            b += [u, v]
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(b))
    err = np.abs(np.array(A) @ sol - b).max()
    return sol, float(err)


def rings(geom, f):
    parts = getattr(geom, "geoms", [geom])
    return [[[round(c, 2) for c in f(x, y)] for x, y in p.exterior.coords] for p in parts]


def swiss_layers(doc):
    """Whether a DXF follows this dataset's layer convention, which `read` turns into an exact graph."""
    return all(name in doc.layers for name in ("ROOM_OUTLINE", "ROOM_TEXT")) and bool(
        doc.modelspace().query('TEXT[layer=="ROOM_TEXT"]'))


def read(stem, source="swiss_dwellings", to_pdf=None):
    """The DXF data/<source>/<stem>.dxf with its PDF (and PNG, if there is one) as views. to_pdf maps DXF to PDF
    points when it is known (a PDF drawn from the DXF); else it is fitted from room texts printed in both."""
    rel = f"data/{source}/{stem}"
    msp = ezdxf.readfile(DATA / f"{rel}.dxf").modelspace()
    title = next((e.dxf.text for e in msp.query('TEXT[layer=="ANNOTATION"]')), "")
    g = Graph(stem, source, title, {"m_per_unit": 1.0})
    g.add(Node("storey", "storey", "storey", name=title, via="dxf"))
    handles, outline = {}, {}  # node id → DXF handle, overlay geometry

    def add(node, e, display=None):
        g.add(node)
        handles[node.id] = e.dxf.handle
        outline[node.id] = node.geom if display is None else display

    # units, named by their "Whg. A" / "App. A" label
    texts = list(msp.query('TEXT[layer=="ROOM_TEXT"]'))
    unit_labels = [(Point(tuple(e.dxf.insert)[:2]), e.dxf.text, m.group(1)) for e in texts if (m := UNIT.fullmatch(e.dxf.text))]
    for i, e in enumerate(x for x in msp.query('LWPOLYLINE[layer=="UNIT_OUTLINE"]') if len(x) > 2):
        poly = Polygon(points(e)).buffer(0)
        label = next((lab for lab in unit_labels if poly.contains(lab[0])), None)
        if label:
            add(Node(f"unit/{label[2]}", "unit", "apartment", "storey", label[1], label[2], poly, via="dxf"), e)
        else:
            add(Node(f"unit/{i}", "unit", "unit", "storey", geom=poly, via="dxf"), e)

    # spaces: outline polygon + the stacked texts inside it (name, number, printed area)
    rooms = [(e, Polygon(points(e)).buffer(0)) for e in msp.query('LWPOLYLINE[layer=="ROOM_OUTLINE"]') if len(e) > 2]
    tree = STRtree([p for _, p in rooms])
    # outlines have no holes: a room drawn inside another (e.g. inside a corridor) is cut out of it
    rooms = [(e, p.difference(unary_union([rooms[j][1] for j in tree.query(p, predicate="contains_properly")])))
             for e, p in rooms]
    tree = STRtree([p for _, p in rooms])
    found = [{} for _ in rooms]
    for e in texts:
        t, pt = e.dxf.text, Point(tuple(e.dxf.insert)[:2])
        if UNIT.fullmatch(t):
            continue
        hit = tree.query(pt, predicate="within")
        i = hit[0] if len(hit) else tree.query_nearest(pt, max_distance=1.0)
        if not np.size(i):
            continue
        slot = "area" if AREA.fullmatch(t) else "number" if NUMBER.fullmatch(t) else "name"
        found[int(np.atleast_1d(i)[0])].setdefault(slot, t)
    for i, ((e, poly), f) in enumerate(zip(rooms, found)):
        cat = vocab.room_cat(f.get("name"))
        attrs = {"outdoor": cat in vocab.OUTDOOR}
        if "area" in f:
            attrs["area_printed_m2"] = float(AREA.fullmatch(f["area"]).group(1))
        nid = f"sp/{f['number']}" if "number" in f and f"sp/{f['number']}" not in g.nodes else f"sp/{i}"
        add(Node(nid, "space", cat, "storey", f.get("name"), f.get("number"), poly, attrs, via="dxf"), e)

    # doors: one closed rectangle per door; leaves (arcs, lines) only widen the overlay
    leaves = [shape(e) for e in msp.query('*[layer=="DOOR"]') if not (e.dxftype() == "LWPOLYLINE" and closed(e))]
    ltree = STRtree(leaves)
    for i, e in enumerate(x for x in msp.query('LWPOLYLINE[layer=="DOOR"]') if closed(x)):
        rect = Polygon(points(e))
        near = [leaves[j] for j in ltree.query(rect.buffer(0.05), predicate="intersects")]
        add(Node(f"door/{i}", "opening", "door", "storey", geom=rect, via="dxf"), e, unary_union([rect, *near]).envelope)
    for i, e in enumerate(x for x in msp.query('LWPOLYLINE[layer=="WINDOW"]') if closed(x)):
        add(Node(f"window/{i}", "opening", "window", "storey", geom=Polygon(points(e)), via="dxf"), e)

    # fixtures: a closed 4-point rectangle around each symbol, classified by what is drawn inside it
    rects, inner = [], []
    for e in msp.query('*[layer=="FIXTURE"]'):
        if e.dxftype() == "LWPOLYLINE" and e.closed and len(e) == 4:
            rects.append(e)
        else:
            inner.append(("C" if e.dxftype() == "CIRCLE" else f"P{len(e)}", shape(e)))
    itree = STRtree([s for _, s in inner])
    for i, e in enumerate(rects):
        rect = Polygon(points(e))
        sig = " ".join(sorted(inner[j][0] for j in itree.query(rect.buffer(0.01), predicate="contains")))
        add(Node(f"fx/{i}", "element", fixture_cat(sig), "storey", geom=rect, attrs={"symbol": sig}, via="dxf"), e)

    # views: the DXF itself (handles), the PDF render (transform) and the PNG (PDF at its pixel size)
    page = pymupdf.open(DATA / f"{rel}.pdf")[0]
    if to_pdf is None:
        (a, c, d), err = fit(msp, page)
        g.meta["pdf_fit_error_pt"] = round(err, 2)
        to_pdf = lambda x, y: (a * x + c, -a * y + d)
    W, H = page.rect.width, page.rect.height
    ext = bbox.extents(msp)
    g.views["dxf"] = View("dxf", f"{rel}.dxf", tuple(round(v, 2) for v in (ext.extmin.x, ext.extmin.y, ext.size.x, ext.size.y)))
    g.views["pdf"] = View("pdf", f"{rel}.pdf", (0, 0, W, H))
    png = DATA / f"{rel}.png"
    if png.exists():
        w, h = Image.open(png).size
        g.views["png"] = View("png", f"{rel}.png", (0, 0, w, h))
    for nid, geom in outline.items():
        g.anchor(nid, "dxf", ref=handles[nid])
        g.anchor(nid, "pdf", rings(geom, to_pdf))
        if "png" in g.views:
            g.anchor(nid, "png", rings(geom, lambda x, y: (to_pdf(x, y)[0] * w / W, to_pdf(x, y)[1] * h / H)))
    return g.derive()
