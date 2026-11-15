"""Drawings uploaded in the web UI: stored under data/uploads and read by the reader for their file type.

PDFs get their schedule tables and their floor-plan content (rooms, apartments, doors, legend symbols). DXFs are
drawn to a PDF sheet first: one with this dataset's semantic layers is read exactly (swiss.py), any other like a PDF.
IFC models
their full semantics and generated floor plans, and pictures the symbol and room detector's boxes. The model's
`look` tool reads whatever the readers miss.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image

import ezdxf

import cubicasa
import dxfpdf
import graph
import ifc
import pdfplan
import sheets
import swiss
from graph import DATA, Graph, Node, View

DIR = DATA / "data/uploads"
PICTURES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
TYPES = (".pdf", ".dxf", ".ifc", *PICTURES)


def pdf(stem):
    """An uploaded PDF's schedule tables and floor-plan content."""
    return pdfplan.add_pages(sheets.read(f"{stem}.pdf", "uploads"), f"{stem}.pdf", "uploads")


def ingest(name, data):
    """Store an uploaded file, add its graph to graph.db and return the drawing id."""
    suffix = Path(name).suffix.lower().replace(".jpeg", ".jpg")
    stem = f"{re.sub(r'[^\w.-]+', '-', Path(name).stem).strip('-')[:60]}-{hashlib.sha1(data).hexdigest()[:8]}"
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"{stem}{suffix}").write_bytes(data)
    if suffix in PICTURES and suffix not in (".png", ".jpg"):  # browsers show PNG and JPEG
        with Image.open(DIR / f"{stem}{suffix}") as im:
            (im.convert("RGB") if im.mode == "CMYK" else im).save(DIR / f"{stem}.png")
        (DIR / f"{stem}{suffix}").unlink()
        suffix = ".png"
    if suffix == ".dxf":
        to_pdf, _ = dxfpdf.render(DIR / f"{stem}.dxf", DIR / f"{stem}.pdf")
        g = swiss.read(stem, "uploads", to_pdf) if swiss.swiss_layers(ezdxf.readfile(DIR / f"{stem}.dxf")) else pdf(stem)
    elif suffix == ".pdf":
        g = pdf(stem)
    elif suffix == ".ifc":
        # floor plans first, in a process of their own: geometry is heavy and ifcplans times out with SIGALRM
        subprocess.run([sys.executable, "-c", f"import ifcplans; ifcplans.render(('uploads', {stem!r}))"],
                       cwd=Path(__file__).parent, timeout=960)
        g = ifc.read(("uploads", stem))
    else:
        with Image.open(DIR / f"{stem}{suffix}") as im:
            size = im.size
        g = Graph(stem, "uploads", Path(name).stem)
        g.add(Node("drawing", "drawing", "upload", name=Path(name).stem, via="upload"))
        g.views["picture"] = View("picture", f"data/uploads/{stem}{suffix}", (0, 0, *size))
        # in the detector's own environment, so torch stays out of the web server
        found = json.loads(subprocess.run([cubicasa.DETECTOR / ".venv/bin/python", "detect.py", DIR / f"{stem}{suffix}"],
                                          cwd=cubicasa.DETECTOR, capture_output=True, text=True, check=True).stdout)
        cubicasa.build(g, "picture", *([(c, s, tuple(b)) for c, s, *b in found[k]] for k in ("rooms", "symbols")))
    con = graph.connect()
    graph.save(con, g)
    con.commit()
    return g.id
