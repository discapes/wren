"""LLM fallback: a model on Verda's inference API answers from the drawing summary through graph tools."""
import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf
import shapely
from PIL import Image
from shapely.geometry import Polygon

from graph import DATA, Answer, Node

API = "https://inference.verda.com/v1/chat/completions"
KEY = dict(line.split("=", 1) for line in (Path.home() / ".config/junction/verda-inference.env").read_text().split())
# The fast model answers first; the second gets the question afresh when the first errs, gives no value or runs out
# of steps or time.
MODELS = ("deepseek-4-1-flash-1017", "kimi-k3-1-pro")
DEADLINE = {"deepseek-4-1-flash-1017": 15, "kimi-k3-1-pro": 35}  # s since the question: past it, answer with what was found
LIMIT = 50  # s since the question: a call still running then is cut off
FINAL_EFFORT = {"kimi-k3-1-pro": "low"}  # the call that must answer from what was found reasons briefly
VISION = "kimi-k3-1-pro"  # `look`: the Verda model that reads images
JUDGE = "deepseek-4-1-flash-1017"
PRICE = {  # USD per 1M tokens: input, cached input, output (verda.com/pricing, Nov 2026)
    "kimi-k3-1-pro": (2.00, 0.20, 10.00),
    "deepseek-4-1-flash-1017": (0.15, 0.006, 0.60),
}
STEPS = 8
STALL = 20  # s without a streamed token
RESULT_CHARS = 8000  # each tool result and the drawing summary are cut to bound the context resent every step
SUMMARY_CHARS = 10000

SYSTEM = """You are the assistant of Wren, an app that answers questions about one construction drawing or \
building model at a time. You always answer in English, whatever language the question or the drawing is in. You \
cannot see the drawing: it has been extracted into a graph of nodes (storeys, units, spaces, openings, elements, \
schedule tables and their rows), summarized below.
If asked who or what you are, say in one sentence that you answer questions about this drawing from its extracted \
data; do not name the underlying model or its maker. For questions unrelated to the drawing, say briefly that you \
only answer questions about this drawing.

Look things up with the tools, then call `answer` exactly once.
- Count, list and sum with the tools; never estimate. For breakdowns by type, storey or room, use `tally` with `sum`.
- When a question names a classification (a category, function, occupancy or rating), look for an IFC property that \
holds it (e.g. Category, OccupancyType, IsExternal, FireRating) and filter or group by that property with `where` or `by`.
- A generic word for rooms in any language (room, space, Raum, local, ruimte, ruumi, rum, pomieszczenie, huone) \
means all spaces (kind space), not the spaces with that word in their name.
- If no property classifies the items a breakdown asks for, group them by name or type rather than giving one total.
- For areas, distances and positions no attribute holds (a building's footprint, the area inside a wall, which \
element is nearest another, where on the site something stands), measure the plan outlines with `plan`.
- If the graph lacks the information and the tools are offered, use `search_text` for words printed on PDF sheets, \
and `look` to have the drawing's picture read by a vision model (for scans and anything drawn but not extracted).

The answer:
- Write it in English, whatever the language of the question or the drawing. Keep names and labels as the drawing \
writes them (e.g. Schlafzimmer 3.01).
- `value` is the answer itself and nothing else. For counts and measures, the number, with the unit in `unit` \
(m², m, m³, SF, count ...). For lists and breakdowns, one line per item as `name: number` (with the unit for \
measures), ending with `Total: number` when the question asks for a total. Name every item asked for and only those: \
leave out related items the question did not ask about. No prose in `value`.
- `explanation` is one short sentence on where the answer comes from (which elements or attributes, or that it was \
read from the picture or the sheet text). Do not restate the question or the value. Add a caveat only when data is \
missing or uncertain.
- `ids` are the nodes the answer rests on; they are highlighted on the drawing for the reader to check.
- When no attribute holds the value but it can be derived from what the drawing holds (a gross floor area from slab \
or space areas, a footprint from the plan outlines), give the derived value and say in `explanation` how it was derived.
- A ratio or split between groups the data does not label (clinical and non-clinical space, interior and exterior \
walls when no property says which) is an assumption: say in `explanation` which grouping you assumed.
- If the drawing does not hold the information and it cannot be derived, give value null and say in `explanation` \
what the model lacks (the property, element or classification asked about) and what it holds that comes closest.
- Write in the language of the question, in plain text without Markdown.

Drawing summary:
{manifest}

File metadata: {meta}"""

TOOLS = [
    {"name": "find", "description": "Nodes matching all given filters, as compact rows. Schedule rows carry their cells as attrs.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "description": "storey | unit | space | opening | element | table | row | drawing"},
         "cat": {"type": "string", "description": "category, e.g. door, window, wall, bedroom, kitchen, toilet"},
         "name": {"type": "string", "description": "exact label, case-insensitive"},
         "number": {"type": "string", "description": "room number or mark"},
         "within": {"type": "string", "description": "id of a storey, unit, space or table to search inside"},
         "text": {"type": "string", "description": "case-insensitive substring searched in name, number and all attributes"},
         "where": {"type": "object", "description": "{field: value} that must match exactly (case-insensitive); fields as in tally's by"},
         "limit": {"type": "integer", "description": "maximum rows returned (default 25); the total is always given"},
         "sort": {"type": "string", "description": "numeric field to order rows by, largest first ('-area' for smallest "
                  "first); 'area' is the printed area, else the computed one. For the largest, smallest, top N ..."}}}},
    {"name": "tally", "description": "Count the nodes matching the filters per value of one field: cat, name, kind, parent, "
     "any attribute (type, object_type, ifc_class, materials, a schedule column ...) or any IFC property-set property "
     "(Category, OccupancyType, IsExternal ...), and with `sum` also total a numeric field per value (area, length_m, "
     "volume_m3 ...). Prefer this to listing many nodes.",
     "parameters": {"type": "object", "properties": {
         "by": {"type": "string"}, "sum": {"type": "string"}, "kind": {"type": "string"}, "cat": {"type": "string"},
         "within": {"type": "string"}, "text": {"type": "string"},
         "where": {"type": "object", "description": "{field: value} that must match exactly (case-insensitive); fields as in tally's by"},
     }, "required": ["by"]}},
    {"name": "details", "description": "All attributes of up to 10 nodes, including IFC property sets and materials.",
     "parameters": {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}, "required": ["ids"]}},
    {"name": "total", "description": "Sum of a numeric attribute over the given ids, or over the nodes matching the "
     "filters (kind, cat, within, text as in find). attr 'area' uses the printed area, else the computed one; other "
     "attrs (length_m, volume_m3, a schedule column ...) are parsed as the first number in their text ('108 SF' → 108).",
     "parameters": {"type": "object", "properties": {
         "attr": {"type": "string"}, "ids": {"type": "array", "items": {"type": "string"}},
         "kind": {"type": "string"}, "cat": {"type": "string"}, "within": {"type": "string"}, "text": {"type": "string"},
         "where": {"type": "object", "description": "{field: value} that must match exactly (case-insensitive); fields as in tally's by"},
     }, "required": ["attr"]}},
    {"name": "plan", "description": "Plan geometry in metres from the floor-plan outlines of the given ids or of the "
     "nodes matching the filters (as in find). Per node its centre (x, y; y grows downwards as on the plan sheet) and "
     "outline area; per plan view the area the outlines cover, the area they enclose with holes filled, and their convex "
     "hull. A building's footprint is the enclosed area of everything within its ground storey (walls alone leave gaps "
     "at doors); the area inside a ring wall with gaps is closer to its hull. With `to`, the distance "
     "from each node to the nearest of those ids, centre to centre.",
     "parameters": {"type": "object", "properties": {
         "ids": {"type": "array", "items": {"type": "string"}}, "to": {"type": "array", "items": {"type": "string"}},
         "kind": {"type": "string"}, "cat": {"type": "string"}, "within": {"type": "string"}, "text": {"type": "string"},
         "where": {"type": "object", "description": "{field: value} that must match exactly (case-insensitive); fields as in tally's by"},
     }}},
    {"name": "search_text", "description": "Where a word or phrase is printed on the drawing's PDF pages (notes, tags, "
     "titles, dimensions); each hit gets an id and is outlined on its page.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "look", "description": "Show the drawing's picture to a vision model with a question; it answers and boxes "
     "what it counted or read, and each box gets an id. For scans and for things drawn but not in the graph.",
     "parameters": {"type": "object", "properties": {
         "question": {"type": "string"},
         "view": {"type": "string", "description": "id of a picture view from the summary's Views line (default: the first)"}},
         "required": ["question"]}},
    {"name": "answer", "description": "The final answer, written in English whatever the language of the question or "
     "the drawing; names and labels stay as the drawing writes them.",
     "parameters": {"type": "object", "properties": {
         "value": {"type": ["number", "string", "null"]}, "unit": {"type": "string"},
         "ids": {"type": "array", "items": {"type": "string"}},
         "explanation": {"type": "string", "description": "One short sentence in English."}},
         "required": ["value", "unit", "ids", "explanation"]}},
]


def chat(model, messages, tools=None, effort=None, patient=True, until=None):
    """One completion, streamed: tokens (reasoning included) arrive every fraction of a second, so a silence of
    STALL seconds means the call stalled and is retried, however long the whole answer takes.
    Verda limits requests per minute per model (60 for DeepSeek, 40 for Kimi): over it, a patient call waits as long
    as the reply says, an impatient one raises at once so another model can answer.
    until: time after which the call is cut off with TimeoutError."""
    body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}}
    body |= ({"reasoning_effort": effort} if effort else {}) | ({"tools": [{"type": "function", "function": t} for t in tools]}
                                                               if tools else {})
    req = urllib.request.Request(API, json.dumps(body).encode(), {"Authorization": f"Bearer {KEY['VERDA_INFERENCE_KEY']}",
                                                                 "Content-Type": "application/json"})
    for wait in (2, 5, 10, 20, 40, None):  # back off on rate limits, transient server errors and stalled calls
        try:
            with urllib.request.urlopen(req, timeout=STALL) as r:
                return collect(r, until)
        except urllib.error.HTTPError as e:
            if wait is None or e.code not in (429, 500, 502, 503, 504) or (e.code == 429 and not patient):
                raise
            time.sleep(min(30, int(e.headers.get("Retry-After") or wait)) if e.code == 429 else wait)
        except (TimeoutError, urllib.error.URLError):
            if wait is None or wait > 5 or (until and time.time() + wait > until):  # two retries
                raise
            time.sleep(wait)


def collect(stream, until=None):
    """A streamed completion assembled into the shape of a non-streamed one."""
    content, calls, usage = "", {}, None
    for line in stream:
        if until and time.time() > until:
            raise TimeoutError("over the time limit")
        if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
            continue
        d = json.loads(line[5:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            content += c["delta"].get("content") or ""
            for t in c["delta"].get("tool_calls") or []:
                call = calls.setdefault(t["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                call["id"] = t.get("id") or call["id"]
                for k, v in (t.get("function") or {}).items():
                    call["function"][k] += v or ""
    message = {"role": "assistant", "content": content} | ({"tool_calls": [calls[i] for i in sorted(calls)]} if calls else {})
    return {"choices": [{"message": message}], "usage": usage}


def cost(model, usage):
    """USD of one call from its token usage."""
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    pin, pcached, pout = PRICE[model]
    return (pin * (usage["prompt_tokens"] - cached) + pcached * cached + pout * usage["completion_tokens"]) / 1e6


def short(v, n=160):
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "…"


def row(n):
    """A node as the model sees it in listings: identity plus its scalar attributes (property sets via `details`)."""
    attrs = {k: (v if n.kind == "row" else short(v)) for k, v in n.attrs.items() if n.kind == "row" or k not in ("psets", "materials", "material_layers")}
    return {"id": n.id, "kind": n.kind, "cat": n.cat, "name": n.name, "number": n.number, "parent": n.parent, **attrs}


def number(v):
    """First number in a text; a comma is a decimal separator unless it groups thousands (1,250 or 1,250.5)."""
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(v))
    if not m:
        return None
    t = m.group()
    t = t.replace(",", "") if "." in t or re.fullmatch(r"-?\d{1,3}(,\d{3})+", t) else t.replace(",", ".")
    return float(t)


def label(g, nid):
    n = g.nodes.get(nid)
    return (n.name or f"{n.cat} {n.number or ''}".strip()) if n else nid


def field(n, name):
    """A node's field, attribute, or IFC property of that name (property names match case-insensitively)."""
    if name in ("cat", "name", "kind", "parent", "number"):
        return getattr(n, name)
    if name in n.attrs:
        return n.attrs[name]
    key = name.casefold()
    return next((v for props in (n.attrs.get("psets") or {}).values() if isinstance(props, dict)
                 for k, v in props.items() if k.casefold() == key), None)


def select(g, args):
    """(nodes matching find-style filters, description of them)."""
    text = (args.get("text") or "").casefold()
    filters = {k: v for k, v in args.items() if v and k in ("kind", "cat", "name", "number", "within")}
    if "within" in filters and filters["within"] not in g.nodes:  # a storey, unit or room given by its name
        named = [n.id for n in g.nodes.values() if n.kind in ("storey", "unit", "space", "table")
                 and (n.name or "").casefold() == str(filters["within"]).casefold()]
        filters["within"] = named[0] if named else filters["within"]
    where = args.get("where") or {}
    nodes = [n for n in g.find(**filters)
             if (not text or text in f"{n.name} {n.number} {json.dumps(n.attrs, ensure_ascii=False)}".casefold())
             and all(str(field(n, k)).casefold() == str(v).casefold() for k, v in where.items())]
    what = " ".join([str(filters[k]) for k in ("cat", "kind") if k in filters]
                    + [f"with {k} = {v!r}" for k, v in where.items()]
                    + [f"{k} {filters[k]!r}" for k in ("name", "number") if k in filters]
                    + ([f"containing {text!r}"] if text else [])
                    + ([f"in {label(g, filters['within'])}"] if "within" in filters else []))
    return nodes, what or "all nodes"


PICTURE = 2000  # px, longest side of the image the vision model sees
LOOK = """{question}

The image is {w}×{h} px. Reply with JSON only: {{"answer": "<short answer>", "items": [{{"label": "<what>", \
"box": [x0, y0, x1, y1]}}]}}, with one item, boxed in image pixels, for each thing you counted or read. Write the \
answer and labels in English, keeping names printed on the drawing as written."""


def pictures(g):
    return [v for v in g.views.values() if Path(v.file).suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf")]


def add_found(g, kind, view, items):
    """Results of `look` or `search_text` as nodes outlined in the view: [(label, (x0, y0, x1, y1) in view units)]."""
    ids, start = [], sum(n.startswith(kind + "/") for n in g.nodes)
    for i, (label_, (x0, y0, x1, y1)) in enumerate(items, start):
        nid = f"{kind}/{i}"
        g.add(Node(nid, "annotation", kind, name=label_, conf=0.5 if kind == "seen" else 1.0, via=kind))
        g.anchor(nid, view.id, [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]])
        ids.append(nid)
    return ids


def look(g, question, view=None, until=None):
    """(answer, [(label, box in view units)]) from the vision model over one picture view."""
    v = g.views.get(view) or pictures(g)[0]
    if v.file.endswith(".pdf"):
        page = pymupdf.open(DATA / v.file)[v.page]
        pix = page.get_pixmap(dpi=round(72 * PICTURE / max(page.rect.width, page.rect.height)))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        im = Image.open(DATA / v.file).convert("RGB")
        im.thumbnail((PICTURE, PICTURE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    r = chat(VISION, [{"role": "user", "content": [
        {"type": "text", "text": LOOK.format(question=question, w=im.width, h=im.height)},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}]}],
        effort="low", until=until)  # full reasoning over an image takes over a minute and gave the same answers
    text = r["choices"][0]["message"]["content"]
    got = json.loads(re.search(r"\{.*\}", text, re.S).group())
    x0, y0, w, h = v.box
    sx, sy = w / im.width, h / im.height
    items = [(it.get("label", ""), (x0 + b[0] * sx, y0 + b[1] * sy, x0 + b[2] * sx, y0 + b[3] * sy))
             for it in got.get("items", []) if len(b := it.get("box") or []) == 4]
    return got.get("answer"), items, v, r["usage"]


def search_text(g, text):
    """[(view, line containing the hit, rect)] for each place the text is printed on the drawing's PDF pages."""
    hits = []
    for v in (v for v in g.views.values() if v.file.endswith(".pdf")):
        page = pymupdf.open(DATA / v.file)[v.page]
        m = page.rotation_matrix
        for r in page.search_for(text)[:50]:
            line = page.get_textbox(r + (-200, -2, 200, 2)).strip() or text
            q = r * m
            hits.append((v, " ".join(line.split())[:120], (q.x0, q.y0, q.x1, q.y1)))
    return hits[:100]


def outline(g, n, kids):
    """(view id, shape in metres with y downwards) of a node's plan outline, or of its parts' (a building's
    elements), or None. kids: {parent id: [child nodes]}."""
    if n.geom is not None and (k := g.meta.get("m_per_unit")):
        return "model", shapely.affinity.scale(n.geom, k, -k, origin=(0, 0))
    for a in g.anchors.get(n.id, ()):
        if a.shape and g.views[a.view].file.startswith("plans/"):  # ifcplans.py draws in metres
            parts = [Polygon(r).buffer(0) for r in a.shape if len(r) >= 4]
            if parts := [p for p in parts if not p.is_empty]:
                return a.view, shapely.unary_union(parts)
    parts = [o for c in kids.get(n.id, ()) if (o := outline(g, c, kids))]
    if parts:
        view = Counter(v for v, _ in parts).most_common(1)[0][0]
        return view, shapely.unary_union([s for v, s in parts if v == view])


def plan(g, args):
    """`plan`: centres, areas, enclosed areas and nearest distances from the nodes' plan outlines."""
    nodes, what = ([g.nodes[i] for i in args["ids"] if i in g.nodes], "nodes") if args.get("ids") else select(g, args)
    kids = defaultdict(list)
    for n in g.nodes.values():
        kids[n.parent].append(n)
    shapes = {n.id: s for n in nodes if (s := outline(g, n, kids))}
    per_view = defaultdict(list)
    for v, shape in shapes.values():
        per_view[v].append(shape)
    views = {}
    for v, parts in per_view.items():
        union = shapely.unary_union(parts)
        filled = shapely.unary_union([Polygon(p.exterior) for p in getattr(union, "geoms", [union]) if p.geom_type == "Polygon"])
        views[v] = {"nodes": len(parts), "covered_m2": round(union.area, 2), "enclosed_m2": round(filled.area, 2),
                    "hull_m2": round(union.convex_hull.area, 2), "bounds": [round(b, 2) for b in union.bounds]}
    targets = {i: s for i in args.get("to") or () if i in g.nodes and (s := outline(g, g.nodes[i], kids))}
    rows = []
    for nid, (v, shape) in list(shapes.items())[:40]:
        c = shape.centroid
        r = {"id": nid, "name": g.nodes[nid].name, "view": v, "x": round(c.x, 2), "y": round(c.y, 2), "area_m2": round(shape.area, 2)}
        if near := [(c.distance(t.centroid), i) for i, (tv, t) in targets.items() if tv == v and i != nid]:
            d, i = min(near)
            r |= {"nearest": i, "distance_m": round(d, 2)}
        rows.append(r)
    enclosed = ", ".join(f"{x['enclosed_m2']:g} m² enclosed" for x in views.values())
    return ({"total": len(nodes), "with_outline": len(shapes), "views": views, "rows": rows},
            f"Measured the plan outlines of {len(shapes)} × {what}" + (f": {short(enclosed, 120)}" if enclosed else ""), list(shapes))


def run(g, name, args, until=None):
    """(tool result for the model, one-line description of the step, ids of the nodes it touched).
    until: time after which a `look` still running is cut off."""
    if name in ("find", "tally"):
        limit, by = args.get("limit", 25), args.get("by")
        nodes, what = select(g, args)
        ids = [n.id for n in nodes]
        if by:
            key = {n.id: short(field(n, by), 120) for n in nodes}
            c = Counter(key.values())
            if attr := args.get("sum"):
                sums = Counter()
                for n in nodes:
                    sums[key[n.id]] += (g.area(n) if attr == "area" else number(field(n, attr))) or 0
                groups = {k: {"count": v, attr: round(sums[k], 3)} for k, v in c.most_common(60)}
                return ({"total": len(nodes), attr: round(sum(sums.values()), 3), "groups": groups},
                        f"Tallied {len(nodes)} × {what} by {by} with total {attr}: "
                        + short(", ".join(f"{k} {round(sums[k], 2):g}" for k, _ in c.most_common(8)), 200), ids)
            return ({"total": len(nodes), "counts": dict(c.most_common(60))},
                    f"Tallied {len(nodes)} × {what} by {by}: " + short(", ".join(f"{k} {v}" for k, v in c.most_common(8)), 200), ids)
        if key := (args.get("sort") or "").lstrip("-"):
            value = {n.id: g.area(n) if key == "area" else number(field(n, key)) for n in nodes}
            nodes = sorted((n for n in nodes if value[n.id] is not None), key=lambda n: value[n.id],
                           reverse=not args["sort"].startswith("-")) + [n for n in nodes if value[n.id] is None]
            what += f" by {key}"
        return {"total": len(nodes), "rows": [row(n) for n in nodes[:limit]]}, f"Found {len(nodes)} × {what}", [n.id for n in nodes]
    if name == "details":
        ids = [i for i in args["ids"][:10] if i in g.nodes]
        return ([{"id": i, **row(g.nodes[i]), "attrs": short(g.nodes[i].attrs, 6000)} for i in ids],
                "Read the properties of " + short(", ".join(label(g, i) for i in ids), 200), ids)
    if name == "total":
        nodes, what = ([g.nodes[i] for i in args["ids"] if i in g.nodes], "nodes") if args.get("ids") else select(g, args)
        values = {n.id: g.area(n) if args["attr"] == "area" else number(field(n, args["attr"])) for n in nodes}
        found = {i: v for i, v in values.items() if v is not None}
        total = round(sum(found.values()), 3)
        return ({"sum": total, "counted": len(found), "missing": [i for i in values if i not in found][:50]},
                f"Summed {args['attr']} over {len(found)} × {what}: {total:g}", list(found))
    if name == "plan":
        return plan(g, args)
    if name == "search_text":
        hits = search_text(g, args["text"])
        ids = [i for v, line, box in hits for i in add_found(g, "text", v, [(line, box)])]
        return ([{"id": i, "page": v.id, "line": line} for i, (v, line, _) in zip(ids, hits)],
                f"Searched the sheet text for {args['text']!r}: {len(ids)} places", ids)
    if name == "look":
        if not pictures(g):
            return {"error": "this drawing has no picture"}, "Looked for a picture: none", []
        said, items, v, usage = look(g, args["question"], args.get("view"), until)
        ids = add_found(g, "seen", v, items)
        return ({"answer": said, "items": [{"id": i, "label": l} for i, (l, _) in zip(ids, items)], "usage": usage},
                f"Looked at the picture ({v.id}): {short(str(said), 160)}", ids)
    return {"error": f"unknown tool {name}"}, f"Called unknown tool {name}", []


DOING = {"find": "Looking up", "tally": "Counting", "details": "Reading properties", "total": "Adding up",
         "plan": "Measuring the plan", "search_text": "Searching the drawing's text", "look": "Reading the picture"}


def answer(g, question, history=(), progress=None):
    """Answer with the models and tools in turn; the result's usage says which models, how many calls and tokens.

    history: earlier turns of the conversation, [{"q", "value", "text", "ids"}], so follow-ups can refer to them.
    progress(steps done, what is happening now) is called as the work goes on."""
    meta = {k: v for k, v in g.meta.items() if k != "coverage"}
    messages = [{"role": "system", "content": SYSTEM.format(manifest=short(g.manifest, SUMMARY_CHARS), meta=json.dumps(meta, ensure_ascii=False))}]
    for t in history:
        said = "\n".join(filter(None, (t["value"], t["text"], t["ids"] and "Nodes: " + ", ".join(t["ids"]))))
        messages += [{"role": "user", "content": t["q"]}, {"role": "assistant", "content": said}]
    messages.append({"role": "user", "content": question})
    use = Counter(calls=0, tokens=0, cached=0, usd=0.0)
    steps, started, a, error = [], time.time(), None, None
    for i, model in enumerate(MODELS):
        try:
            got, done = attempt(g, model, list(messages), steps, use, started, progress, patient=i == len(MODELS) - 1)
            a = a if a and a.value is not None and got.value is None else got  # an earlier value beats a later null
            error = None
        except Exception as e:  # the next model gets a try; the error is raised only if no model answered
            done, error = False, e
            if time.time() > started + LIMIT:  # cut off: no time left for another model
                a = a or Answer(None, "", [], f"No answer within {LIMIT} s.", "llm")
                break
        if done or i == len(MODELS) - 1:
            break
        why = "is over its rate limit" if getattr(error, "code", None) == 429 else "failed" if error else "gave no final answer"
        steps.append((f"{model} {why}; asking {MODELS[i + 1]}", []))
    if a is None:
        raise error
    a.usage = f"{' → '.join(MODELS[:i + 1])}, {use['calls']} calls, {use['tokens']} tokens " \
              f"({use['cached']} cached), ${use['usd']:.4f}"
    a.steps = steps
    return a


def attempt(g, model, messages, steps, use, started, progress, patient):
    """(Answer, whether it is final) from one model; not final when it gave no value or ran out of steps or time."""
    def report(now):
        if progress:
            progress([what for what, _ in steps], now)

    pdf, picture = any(v.file.endswith(".pdf") for v in g.views.values()), bool(pictures(g))
    tools = [t for t in TOOLS if (t["name"] != "search_text" or pdf) and (t["name"] != "look" or picture)]
    for step in range(1, STEPS + 1):
        report("Thinking" if steps else "Reading the question")
        last = step == STEPS or time.time() - started > DEADLINE[model]
        if last:  # out of lookups: answer from what was found rather than not at all
            messages.append({"role": "user", "content": "No more lookups: call `answer` now with what you have found."})
        r = chat(model, messages, [t for t in tools if not last or t["name"] == "answer"],
                 effort=last and FINAL_EFFORT.get(model), patient=patient, until=started + LIMIT)
        use.update(calls=1, tokens=r["usage"]["total_tokens"], usd=cost(model, r["usage"]),
                   cached=(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls} if calls
                        else {"role": "assistant", "content": msg.get("content") or ""})
        if not calls:
            return Answer(msg.get("content"), "", [], "", "llm"), not last
        for c in calls:
            args = json.loads(c["function"]["arguments"] or "{}")
            if c["function"]["name"] == "answer":
                v, unit = args.get("value"), args.get("unit", "")
                if isinstance(v, str) and (m := re.fullmatch(r"\s*(-?[\d,]+(?:\.\d+)?)\s*([^\d\s]{0,5})\s*", v)):
                    v, unit = number(m[1]), unit or m[2]  # "1181 SF": the unit belongs in `unit`
                if isinstance(v, float) and v.is_integer():
                    v = int(v)
                if isinstance(v, str) and ("<|" in v or v.strip().casefold() in ("", "null", "none")):  # leaked markup
                    v = None
                ids = [i for i in args.get("ids", []) if i in g.nodes]
                return Answer(v, unit, ids, args.get("explanation", ""), "llm"), v is not None and not last
            report(DOING.get(c["function"]["name"], "Looking up"))
            result, what, touched = run(g, c["function"]["name"], args, started + LIMIT)
            if isinstance(result, dict) and "usage" in result:  # the vision call inside `look`
                u = result.pop("usage")
                use.update(tokens=u["total_tokens"], usd=cost(VISION, u))
            steps.append((what, touched))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": short(result, RESULT_CHARS)})
    return Answer(None, "", [], f"No answer within {STEPS} model calls.", "llm"), False


def judge(question, reference, candidate):
    """Whether a free-text answer states the same facts as the reference (numbers within 2%), by a small model."""
    r = chat(JUDGE, [{"role": "user", "content": f"Question: {question}\nReference answer: {reference}\nCandidate answer: "
                      f"{candidate}\n\nDoes the candidate give the same facts as the reference? Numbers may differ by up "
                      "to 2% and wording may differ. Reply with only yes or no."}])
    return r["choices"][0]["message"]["content"].strip().lower().startswith("yes")
