"""Drawing graph: nodes with geometry and source anchors, stored in SQLite and queried in memory."""
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import shapely
from shapely import STRtree


DATA = Path.home() / "junction-data/stoa-full/junction-dataset"
DB = Path.home() / "junction-data/graph.db"


@dataclass
class Answer:
    value: float | int | str | None
    unit: str
    ids: list[str] = field(default_factory=list)
    text: str = ""
    via: str = ""  # template name, or "llm"
    usage: str = ""  # model, calls and tokens of an llm answer
    steps: list = field(default_factory=list)  # [(what the model looked up, ids it got back)] of an llm answer
    read_as: str = ""  # the template question a reworded question was read as


@dataclass
class Node:
    id: str
    kind: str  # drawing | storey | unit | space | opening | element | annotation | table | row
    cat: str = ""
    parent: str | None = None
    name: str | None = None
    number: str | None = None
    geom: shapely.Geometry | None = None  # model coordinates
    attrs: dict = field(default_factory=dict)
    conf: float = 1.0
    via: str = ""


@dataclass
class View:
    id: str
    file: str  # dataset-relative path, e.g. data/swiss_dwellings/x.pdf
    box: tuple  # x0, y0, width, height in the view's own coordinates (y down)
    page: int = 0


@dataclass
class Anchor:
    node: str
    view: str
    shape: list | None = None  # rings [[x, y], ...] in view coordinates
    ref: str | None = None  # DXF handle, IFC GlobalId, PDF text span


@dataclass
class Edge:
    a: str
    b: str
    rel: str  # in | labels | describes


class Graph:
    def __init__(self, id, source, title="", meta=None, manifest=""):
        self.id, self.source, self.title = id, source, title
        self.meta = meta or {}  # m_per_unit, coverage, view fit residuals ...
        self.manifest = manifest
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.views: dict[str, View] = {}
        self.anchors: defaultdict[str, list[Anchor]] = defaultdict(list)

    def add(self, node):
        self.nodes[node.id] = node
        return node

    def anchor(self, node, view, shape=None, ref=None):
        self.anchors[node].append(Anchor(node, view, shape, ref))

    def index(self):
        self.by_cat = defaultdict(list)
        self.ins = defaultdict(list)
        for n in self.nodes.values():
            self.by_cat[n.cat].append(n)
        for e in self.edges:
            if e.rel == "in":
                self.ins[e.a].append(e.b)
        self._anc = {}
        return self

    # ---- derivation, run once at ingestion ----

    def derive(self):
        """Membership (element → space, space → unit, opening `in` spaces), computed areas, coverage, manifest."""
        k = self.meta.get("m_per_unit")
        spaces = [n for n in self.nodes.values() if n.kind == "space" and n.geom is not None]
        units = [n for n in self.nodes.values() if n.kind == "unit" and n.geom is not None]
        if units:
            tree = STRtree([u.geom for u in units])
            for s in spaces:
                hit = tree.query(s.geom.representative_point(), predicate="within")
                if len(hit):
                    s.parent = units[hit[0]].id
        if spaces:
            tree = STRtree([s.geom for s in spaces])
            for n in self.nodes.values():
                if n.geom is None or self.nodes.get(n.parent, Node("", "")).kind == "space":
                    continue
                if n.kind == "element":
                    hit = tree.query(n.geom.centroid, predicate="within")
                    if len(hit):
                        n.parent = spaces[hit[0]].id
                elif n.kind == "opening":
                    x0, y0, x1, y1 = n.geom.bounds
                    for i in tree.query(n.geom.buffer(0.25 * min(x1 - x0, y1 - y0)), predicate="intersects"):
                        self.edges.append(Edge(n.id, spaces[i].id, "in"))
        if k:
            for s in spaces:
                s.attrs["area_m2"] = round(s.geom.area * k * k, 2)
        self.meta["coverage"] = {
            "extractors": sorted({n.via for n in self.nodes.values() if n.via}),
            "categories": sorted({n.cat for n in self.nodes.values() if n.cat}),
            "attrs": sorted({a for n in self.nodes.values() for a in n.attrs}),
        }
        self.index()
        self.manifest = self.make_manifest()
        return self

    def make_manifest(self, max_rows=300):
        """Compact text summary of the drawing for the agent."""
        out = [f"Drawing {self.id} ({self.source}) {self.title}".strip()]
        if k := self.meta.get("m_per_unit"):
            out.append(f"Scale: 1 model unit = {k} m")
        out.append("Views: " + ", ".join(f"{v.id}={v.file}" for v in self.views.values()))
        for kind in ("storey", "unit"):
            for n in self.find(kind=kind):
                c = Counter(m.cat for m in self.find(within=n.id) if m.kind in ("space", "element", "opening"))
                area = self.measure(self.find(kind="space", within=n.id, where={"outdoor": False}))
                out.append(f"{kind} {n.id} name={n.name!r} indoor_area={area:.2f} m² counts={dict(c)}")
        rows = self.find(kind="space")
        out.append(f"Spaces ({len(rows)}): id | name | number | cat | printed m² | computed m² | parent")
        for n in rows[:max_rows]:
            a = n.attrs
            out.append(f"{n.id} | {n.name} | {n.number} | {n.cat} | {a.get('area_printed_m2')} | {a.get('area_m2')} | {n.parent}")
        c = Counter(n.cat for n in self.nodes.values() if n.kind in ("element", "opening"))
        out.append(f"Element counts: {dict(c)}")
        for t in self.find(kind="table"):
            n = len(self.find(kind="row", within=t.id))
            out.append(f"table {t.id} title={t.name!r} rows={n} columns={t.attrs['columns']}")
        out.append(f"Coverage: {self.meta['coverage']}")
        return "\n".join(out)

    # ---- queries ----

    def ancestors(self, nid):
        if nid not in self._anc:
            p = self.nodes[nid].parent
            self._anc[nid] = {nid} | (self.ancestors(p) if p in self.nodes else set())
        return self._anc[nid]

    def inside(self, n, scope):
        return scope in self.ancestors(n.id) or any(scope in self.ancestors(b) for b in self.ins.get(n.id, ()))

    def find(self, kind=None, cat=None, name=None, number=None, within=None, where=None):
        """Nodes matching all given filters. `name` matches the label case-insensitively, `where` is {attr: value}."""
        pool = self.by_cat.get(cat, []) if cat else self.nodes.values()
        key = name and name.casefold().strip()
        return [
            n for n in pool
            if (kind is None or n.kind == kind)
            and (name is None or (n.name or "").casefold() == key)
            and (number is None or n.number == number)
            and (within is None or self.inside(n, within))
            and (where is None or all(n.attrs.get(a) == v for a, v in where.items()))
        ]

    @staticmethod
    def area(n):
        """Printed area when the drawing states one, else the computed one."""
        return n.attrs.get("area_printed_m2", n.attrs.get("area_m2"))

    def measure(self, nodes):
        return round(sum(self.area(n) or 0 for n in nodes), 2)


SCHEMA = """
create table if not exists drawing (id text primary key, source text, title text, meta text, manifest text);
create table if not exists view (drawing text, id text, file text, page int, box text, primary key (drawing, id));
create index if not exists view_file on view (file);
create table if not exists node (drawing text, id text, kind text, cat text, parent text, name text, number text,
                                 attrs text, geom blob, conf real, via text, primary key (drawing, id));
create table if not exists anchor (drawing text, node text, view text, shape text, ref text);
create index if not exists anchor_drawing on anchor (drawing);
create table if not exists edge (drawing text, a text, b text, rel text);
create index if not exists edge_drawing on edge (drawing);
"""


def connect():
    con = sqlite3.connect(DB, timeout=120)  # several ingests and the web server share the file
    con.execute("pragma journal_mode = wal")
    con.executescript(SCHEMA)
    return con


def save(con, g):
    for t in ("view", "node", "anchor", "edge"):
        con.execute(f"delete from {t} where drawing = ?", (g.id,))
    con.execute("insert or replace into drawing values (?, ?, ?, ?, ?)",
                (g.id, g.source, g.title, json.dumps(g.meta), g.manifest))
    con.executemany("insert into view values (?, ?, ?, ?, ?)",
                    [(g.id, v.id, v.file, v.page, json.dumps(v.box)) for v in g.views.values()])
    nodes = list(g.nodes.values())
    wkb = shapely.to_wkb([n.geom for n in nodes])
    con.executemany("insert into node values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
        (g.id, n.id, n.kind, n.cat, n.parent, n.name, n.number, json.dumps(n.attrs), w, n.conf, n.via)
        for n, w in zip(nodes, wkb)])
    con.executemany("insert into anchor values (?, ?, ?, ?, ?)", [
        (g.id, a.node, a.view, a.shape and json.dumps(a.shape), a.ref) for al in g.anchors.values() for a in al])
    con.executemany("insert into edge values (?, ?, ?, ?)", [(g.id, e.a, e.b, e.rel) for e in g.edges])


def load(drawing_id):
    """The drawing's graph, cached in memory until graph.db changes (an ingest or merge)."""
    return cached(drawing_id, max(p.stat().st_mtime for p in (DB, DB.with_name(DB.name + "-wal")) if p.exists()))


@lru_cache(maxsize=64)
def cached(drawing_id, version):
    con = connect()
    src, title, meta, manifest = con.execute(
        "select source, title, meta, manifest from drawing where id = ?", (drawing_id,)).fetchone()
    g = Graph(drawing_id, src, title, json.loads(meta), manifest)
    for vid, file, page, box in con.execute("select id, file, page, box from view where drawing = ?", (drawing_id,)):
        g.views[vid] = View(vid, file, tuple(json.loads(box)), page)
    rows = con.execute("select id, kind, cat, parent, name, number, attrs, geom, conf, via from node where drawing = ? "
                       "order by rowid", (drawing_id,)).fetchall()
    geoms = shapely.from_wkb([r[7] for r in rows])
    for r, geom in zip(rows, geoms):
        g.add(Node(r[0], r[1], r[2], r[3], r[4], r[5], geom, json.loads(r[6]), r[8], r[9]))
    for node, view, shape, ref in con.execute("select node, view, shape, ref from anchor where drawing = ?", (drawing_id,)):
        g.anchor(node, view, shape and json.loads(shape), ref)
    g.edges = [Edge(*r) for r in con.execute("select a, b, rel from edge where drawing = ?", (drawing_id,))]
    return g.index()


def resolve(path):
    """Dataset path of a file → (drawing id, view id), or None if it was not ingested."""
    path = path.removeprefix(str(DATA) + "/")
    return connect().execute("select drawing, id from view where file = ?", (path,)).fetchone()
