"""Web UI: browse drawings, see what the graph extracted, ask questions and see answers drawn on the drawing.

uv run uvicorn web:app --port 8300
"""
import json
import re
import uuid
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import pymupdf
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

import ask
import graph
import scans
import uploads

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(Path(__file__).parent / "templates")
RENDERS = Path.home() / "junction-data/renders"
SHOWN = ("png", "jpg", "svg", "pdf")  # view file types the browser can display, in order of preference
SOURCES = {"uploads": "Uploads", "swiss_dwellings": "Swiss dwellings", "cubicasa5k": "CubiCasa",
           "ifc_models": "IFC models", "ifc_bench": "IFC bench", "opening_design": "Opening Design sheets",
           "scans": "Historic scans"}  # tab and list order: the user's own uploads first
GROUP = f"(case when source in {scans.SOURCES} then 'scans' else source end)"  # SQL: the source's key in SOURCES
PAGE = 60
Image.MAX_IMAGE_PIXELS = None

EXAMPLES = defaultdict(list)  # dataset path → [(question, reference answer)]
for line in open(graph.DATA / "questions.jsonl"):
    q = json.loads(line)
    EXAMPLES[q["path"]].append((q["question"], q["answer"]))


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache"
    return response


def label(title):
    """The drawing's own title, unless it is an empty placeholder like 'Project Number' or '0001'."""
    t = " ".join((title or "").split())
    return t if re.search(r"[A-Za-z]{3}", t) and t.casefold() not in {"project number", "codigo de proyecto"} else ""


templates.env.filters["label"] = label
templates.env.globals["SOURCES"] = SOURCES
templates.env.filters["group"] = lambda source: "scans" if source in scans.SOURCES else source


def suffix(v):
    return Path(v.file).suffix[1:]


def shown_views(g):
    """Displayable views, one per render (a PDF with a PNG or SVG of the same name is the same picture)."""
    one = {}
    for v in sorted((v for v in g.views.values() if suffix(v) in SHOWN), key=lambda v: SHOWN.index(suffix(v))):
        one.setdefault((Path(v.file).stem, v.page), v)
    return sorted(one.values(), key=lambda v: (v.file, v.page))


def held(g, ids):
    """{displayable view id: how many of the nodes have an outline in it}."""
    return {v.id: sum(any(a.view == v.id and a.shape for a in g.anchors.get(i, ())) for i in ids) for v in shown_views(g)}


def best_view(g, ids, current):
    """The displayable view holding most of the given nodes; the current one when it holds as many."""
    views = shown_views(g)
    if not views:
        return None
    counts = held(g, ids)
    best = max(views, key=lambda v: counts[v.id])
    return current if current in counts and counts[current] == counts[best.id] else best.id


def crop(g, view):
    """Extent of everything anchored in the view plus a margin, so the plan fills the figure instead of the sheet."""
    pts = [p for al in g.anchors.values() for a in al if a.view == view and a.shape for ring in a.shape for p in ring]
    if not pts or g.source == "opening_design":
        return g.views[view].box
    xs, ys = [x for x, _ in pts], [y for _, y in pts]
    m = 0.04 * max(max(xs) - min(xs), max(ys) - min(ys))
    return min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m


def overlay(g, view, ids):
    """[(number, node, svg path, top-left corner, smaller side)] for the nodes that have an outline in `view`."""
    out = []
    for i, nid in enumerate(ids, 1):
        for a in g.anchors.get(nid, ()):
            if a.view == view and a.shape:
                d = " ".join("M" + " L".join(f"{x},{y}" for x, y in ring) + "Z" for ring in a.shape)
                xs = [x for ring in a.shape for x, _ in ring]
                ys = [y for ring in a.shape for _, y in ring]
                out.append((i, g.nodes[nid], d, (min(xs), min(ys)), min(max(xs) - min(xs), max(ys) - min(ys))))
    return out


def examples(g):
    return [e for v in g.views.values() for e in EXAMPLES.get(v.file, ())]


def contents(g):
    """What the graph holds, as browsable groups: [(key, label, count)]."""
    c = Counter((n.kind, n.cat) for n in g.nodes.values() if n.kind in ("unit", "space", "opening", "element") and n.cat)
    out = [(f"cat:{cat}", cat.replace("_", " "), k) for (kind, cat), k in sorted(c.items(), key=lambda x: (x[0][0] != "space", -x[1]))]
    out += [(f"table:{t.id}", t.name or "table", len(g.find(kind="row", within=t.id))) for t in g.find(kind="table")]
    return out


def shown(a):
    """An answer's value as one line of text, with its unit."""
    if a.value is None:
        return ""
    v = round(a.value, 2) if isinstance(a.value, float) else a.value
    return f"{v} {a.unit}" if a.unit and a.unit != "count" and not isinstance(v, str) else str(v)


def room(g, sid):
    s = g.nodes.get(sid)
    return s and " ".join(filter(None, (s.name or (s.cat or "").replace("_", " "), s.number)))


def where(g, n):
    """Where a listed item is: the rooms a door or window joins, the room a fixture stands in, a room's apartment."""
    p = g.nodes.get(n.parent)
    if n.kind == "opening":
        rooms = [r for r in (room(g, s) for s in g.ins.get(n.id, [])[:2]) if r]
        return " ↔ ".join(rooms) if len(rooms) > 1 else f"in {rooms[0]}" if rooms else ""
    if n.kind == "space":
        return p.name or p.number if p and p.kind == "unit" else ""
    if p and p.kind == "space":
        return "in " + room(g, p.id)
    return f"on {p.name}" if p and p.kind == "storey" and p.name else ""


def outline_off(n):
    """The area computed from the room's outline when it differs from the printed area by more than 2%."""
    printed, outline = n.attrs.get("area_printed_m2"), n.attrs.get("area_m2")
    return outline if printed and outline and abs(printed - outline) > 0.02 * printed else None


templates.env.globals |= {"where": where, "outline_off": outline_off}


def recount(g, a, ids):
    """How the panel recomputes the answer when items are excluded: "count", "sum" of areas, or None."""
    nodes = [g.nodes[i] for i in ids if i in g.nodes]
    if not a or not nodes or len(set(ids)) != len(ids):
        return None
    if a.unit == "count" and a.value == len(nodes):
        return "count"
    areas = [g.area(n) for n in nodes]
    if a.unit == "m²" and isinstance(a.value, (int, float)) and None not in areas and abs(sum(areas) - a.value) <= 0.005 * abs(a.value):
        return "sum"


def context(g, view, question="", show="", history=(), progress=None):
    """Template context: the answer to a question (a follow-up when history holds earlier turns), or the nodes of one
    browsed group, and how to draw them."""
    a, table, thread = None, None, []
    if question:
        try:
            a = ask.reply(g, question, history, progress)
        except Exception as e:
            a = ask.Answer(None, "", [], f"The model call failed: {e}", "llm")
        # the conversation so far, sent back with the next question; ids let follow-ups refer to what was found
        thread = [*history, {"q": question, "value": shown(a), "text": a.text, "ids": a.ids[:50]}][-6:]
    elif show.startswith("cat:"):
        nodes = g.find(cat=show[4:])
        a = ask.Answer(len(nodes), "count", [n.id for n in nodes], f"{len(nodes)} × {show[4:].replace('_', ' ')}", "graph")
    elif show.startswith("table:"):
        t = g.nodes[show[6:]]
        rows = g.find(kind="row", within=t.id)
        a = ask.Answer(len(rows), "count", [r.id for r in rows], f"{t.name or 'table'}: {len(rows)} rows", "graph")
        table = t.attrs["columns"], rows
    ids = a.ids if a else []
    number = {nid: i for i, nid in enumerate(ids, 1)}
    steps = [(what, [number[i] for i in touched if i in number]) for what, touched in (a.steps if a else [])]
    view = best_view(g, ids, view or None)
    ref = next((r for q, r in examples(g) if q == question.strip()), None)
    marks = view and overlay(g, view, ids[:2000])
    per_view = held(g, ids) if len(ids) > 1 else {}
    outlined = {nid for nid in ids if any(an.shape for an in g.anchors.get(nid, ()))}
    # "117 × door" when all share a category; what `look` and `search_text` outlined are plain items
    cats = {g.nodes[i].cat if g.nodes[i].kind != "annotation" else None for i in ids if i in g.nodes}
    uniform = len(cats) == 1 and None not in cats
    listed = (f"{len(ids)} × {next(iter(cats)).replace('_', ' ')}" if uniform
              else f"{len(ids)} {'item' if len(ids) == 1 else 'items'}")
    listed_nodes = [g.nodes[i] for i in ids if i in g.nodes]
    return {"g": g, "a": a, "steps": steps, "table": table, "ref": ref, "per_view": per_view, "history": history, "thread": thread, "listed": listed,
            "recount": None if table else recount(g, a, ids), "uniform": uniform, "seen": any(n.via == "seen" for n in listed_nodes),
            "area_off": sum(outline_off(n) is not None for n in listed_nodes) if a and a.unit == "m²" else 0,
            "here": len({m[1].id for m in marks or ()}), "nowhere": len(set(ids) - outlined), "question": question, "show": show,
            "view": view and g.views[view], "views": shown_views(g), "marks": marks,
            "crop": view and crop(g, view), "examples": examples(g), "contents": contents(g)}


def matches(ref, value):
    """Stoa's scoring rule: counts exact, measures within 2%."""
    if not isinstance(ref, (int, float)) or not isinstance(value, (int, float)):
        return None
    return value == ref if isinstance(ref, int) else abs(value - ref) <= 0.02 * abs(ref)


templates.env.globals["matches"] = matches


@app.get("/", response_class=HTMLResponse)
def index(request: Request, q: str = "", source: str = "", offset: int = 0):
    con = graph.connect()
    counts = dict(con.execute(f"select {GROUP} g, count(*) from drawing group by g").fetchall())
    order = " ".join(f"when '{s}' then {i}" for i, s in enumerate(SOURCES))
    # every word must occur in the name or title; spaces, underscores and hyphens are the same character
    words = q.replace("_", " ").replace("-", " ").split()
    text = "replace(replace(id || ' ' || coalesce(title, ''), '_', ' '), '-', ' ')"
    match = " and ".join([f"{text} like ?"] * len(words)) or "1"
    rows = con.execute(
        f"select id, {GROUP}, title from drawing where {match} and (? = '' or {GROUP} = ?) "
        # uploads newest first: saving a graph rewrites its row, so a higher rowid is a later upload
        f"order by case {GROUP} {order} end, case when source = 'uploads' then -rowid end, id limit ? offset ?",
        [*(f"%{w}%" for w in words), source, source, PAGE + 1, offset]).fetchall()
    ctx = {"rows": rows[:PAGE], "more": len(rows) > PAGE, "q": q, "source": source, "offset": offset + PAGE,
           "counts": counts}
    return templates.TemplateResponse(request, "cards.html" if request.headers.get("HX-Request") else "index.html", ctx)


@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile):
    """Read an uploaded drawing into the graph and open it."""
    if Path(file.filename or "").suffix.lower() not in uploads.TYPES:
        return HTMLResponse("Upload a PDF, DXF, IFC model or image.")
    drawing = await run_in_threadpool(uploads.ingest, file.filename, await file.read())
    return HTMLResponse(headers={"HX-Redirect": f"/d/{drawing}"})


@app.get("/d/{drawing}", response_class=HTMLResponse)
def drawing(request: Request, drawing: str, view: str = "", q: str = "", show: str = ""):
    return templates.TemplateResponse(request, "drawing.html", context(graph.load(drawing), view, q, show))


@dataclass
class Job:
    """A question being answered in the background, with what it has done so far."""
    drawing: str
    question: str
    show: str
    turns: list
    future: Future = None
    steps: list = field(default_factory=list)
    now: str = "Reading the question"

    def progress(self, steps, now):
        self.steps, self.now = steps, now


JOBS = {}
ANSWERING = ThreadPoolExecutor(8)
WAIT = 1.5  # s: answers ready by then (templates, routed questions) come back without a progress view


@app.post("/d/{drawing}/ask", response_class=HTMLResponse)
def answer(request: Request, drawing: str, question: str = Form(""), show: str = Form(""), view: str = Form(""),
           history: str = Form("")):
    job = Job(drawing, question, show, json.loads(history) if history and question else [])
    job.future = ANSWERING.submit(context, graph.load(drawing), view, question, show, job.turns, job.progress)
    try:
        return answered(request, job, job.future.result(timeout=WAIT))
    except TimeoutError:
        key = uuid.uuid4().hex
        JOBS[key] = job
        return progress_view(request, key, job)


@app.get("/d/{drawing}/ask/{key}", response_class=HTMLResponse)
def answer_progress(request: Request, drawing: str, key: str):
    job = JOBS.get(key)
    if not job:
        return HTMLResponse('<div id="result" class="has"><p class="explain">The server restarted while answering. '
                            'Ask again.</p></div>')
    if not job.future.done():
        return progress_view(request, key, job)
    del JOBS[key]
    return answered(request, job, job.future.result())


def progress_view(request, key, job):
    return templates.TemplateResponse(request, "progress.html", {"key": key, "job": job, "history": job.turns,
                                                                 "g": graph.load(job.drawing)})


def answered(request, job, ctx):
    response = templates.TemplateResponse(request, "ask.html", ctx)
    if not job.turns:  # a follow-up's URL would re-ask it without the conversation
        response.headers["HX-Push-Url"] = f"/d/{job.drawing}?{urlencode({'q': job.question} if job.question else {'show': job.show})}"
    return response


def render(drawing, v, dpi):
    """PNG of a PDF page, cached."""
    out = RENDERS / f"{drawing}-{v.id}-{dpi}.png"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        pymupdf.open(graph.DATA / v.file)[v.page].get_pixmap(dpi=dpi).save(out)
    return out


@app.get("/view/{drawing}/{view:path}")
def view_file(drawing: str, view: str):
    v = graph.load(drawing).views[view]
    return FileResponse(render(drawing, v, 150) if suffix(v) == "pdf" else graph.DATA / v.file)


@app.get("/thumb/{drawing}")
def thumb(drawing: str):
    """Small JPEG of the drawing's first picture, cropped to the plan."""
    out = RENDERS / "thumbs" / f"{drawing}.jpg"
    if not out.exists():
        g = graph.load(drawing)
        views = sorted((v for v in g.views.values() if suffix(v) in ("png", "jpg", "pdf")),
                       key=lambda v: (suffix(v) == "pdf", v.file, v.page))
        if not views:  # IFC models with only vector plans: the fullest plan (largest file) is the thumbnail
            svgs = [graph.DATA / v.file for v in g.views.values() if suffix(v) == "svg"]
            return FileResponse(max(svgs, key=lambda p: p.stat().st_size)) if svgs else HTMLResponse(status_code=404)
        v = views[0]
        im = Image.open(render(drawing, v, 40) if suffix(v) == "pdf" else graph.DATA / v.file)
        if suffix(v) != "pdf":
            x, y, w, h = crop(g, v.id)
            im = im.crop((max(0, x), max(0, y), min(im.width, x + w), min(im.height, y + h)))
        im.thumbnail((480, 360))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.convert("RGB").save(out, quality=82)
    return FileResponse(out)
