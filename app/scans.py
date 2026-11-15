"""Historic scans (habs, finna, europeana, wikimedia): the picture and its archive record, no extracted content.

The record (title, credit, licence, source page) comes from the source's LICENSE.md table. Questions about what
the scan shows go to the model, which can only answer from the record and says so.
"""
import xml.etree.ElementTree as ET

from PIL import Image

from graph import DATA, Graph, Node, View

SOURCES = ("habs", "finna", "europeana", "wikimedia")
FIELDS = ("title", "licence", "credit", "source")
Image.MAX_IMAGE_PIXELS = None


def records(source):
    """file name → archive record, from the table in data/<source>/LICENSE.md."""
    out = {}
    for line in (DATA / "data" / source / "LICENSE.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) == 5 and cells[0] != "file" and not set(cells[0]) <= set("-: "):
            out[cells[0]] = dict(zip(FIELDS, cells[1:]))
    return out


def items(source):
    """One drawing per file stem: a Wikimedia SVG and its JPEG rendering are the same drawing."""
    return sorted({(source, name.rsplit(".", 1)[0]) for name in records(source)})


def read(item):
    source, stem = item
    rec = records(source)
    files = sorted((DATA / "data" / source).glob(f"{stem}.*"))
    record = next(rec[f.name] for f in files if f.name in rec)
    g = Graph(stem, source, record["title"], record)
    g.add(Node("drawing", "drawing", "scan", name=record["title"], attrs=record, via="archive"))
    for f in files:
        rel = str(f.relative_to(DATA))
        if f.suffix == ".svg":
            box = ET.parse(f).getroot().get("viewBox")
            if box:
                g.views["svg"] = View("svg", rel, tuple(float(v) for v in box.replace(",", " ").split()))
        elif f.suffix == ".jpg":
            with Image.open(f) as im:
                g.views["jpg"] = View("jpg", rel, (0, 0, *im.size))
    return g.derive()
