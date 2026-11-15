"""DXF → vector PDF sheet at a printed drawing scale, so the PDF floor-plan reader and the UI can use a DXF.

Text stays real PDF text (ezdxf's own renderer draws glyph outlines, which no text reader can find), curves are
flattened into polylines and blocks are exploded. The sheet is drawn at the scale the drawing states, or at one that
fits an A0 sheet, which is then printed on it ("1:100") as on a plan.
"""
import re
from collections import Counter

import ezdxf
import pymupdf
from ezdxf import bbox, disassemble, path

MM = 72 / 25.4  # PDF points per mm
METRES = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0}  # DXF $INSUNITS → metres per drawing unit
SCALES = (50, 100, 200, 500, 1000, 2000, 5000)
SHEET = 1189  # mm, the long side of A0: the largest scale whose sheet fits it
MARGIN = 15  # mm
TEXT = {"TEXT", "MTEXT", "ATTRIB"}


def metres_per_unit(doc, extents):
    """From $INSUNITS; unitless drawings are taken as mm when they span more than 1000 units, else metres."""
    return METRES.get(doc.units) or (0.001 if max(extents.size.x, extents.size.y) > 1000 else 1.0)


def render(src, dst):
    """Write the DXF's model space to a one-page PDF; returns (DXF → PDF point transform, scale denominator)."""
    doc = ezdxf.readfile(src)
    msp = doc.modelspace()
    ext = bbox.extents(msp)
    m = metres_per_unit(doc, ext)
    span = max(ext.size.x, ext.size.y) * m * 1000  # mm at 1:1
    # the scale the drawing states ("Massstab 1:100"), so the printed scale and the sheet agree; else one that fits A0
    stated = Counter(int(n) for e in msp.query("TEXT MTEXT") for n in re.findall(
        r"\b1\s?:\s?(\d{1,4})\b", e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text) if 10 <= int(n) <= 5000)
    denom = stated.most_common(1)[0][0] if stated else next((s for s in SCALES if span / s <= SHEET), SCALES[-1])
    k = m * 1000 / denom * MM  # PDF points per drawing unit
    x0, y1 = ext.extmin.x, ext.extmax.y
    pdf = pymupdf.open()
    page = pdf.new_page(width=ext.size.x * k + 2 * MARGIN * MM, height=ext.size.y * k + 2 * MARGIN * MM)
    to_pdf = lambda x, y: ((x - x0) * k + MARGIN * MM, (y1 - y) * k + MARGIN * MM)
    hidden = {layer.dxf.name for layer in doc.layers if layer.is_off() or layer.is_frozen()}
    shape = page.new_shape()
    for e in disassemble.recursive_decompose(msp):
        if e.dxf.get("layer") in hidden:
            continue
        if e.dxftype() in TEXT:
            text = e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text
            size = max(e.dxf.get("char_height" if e.dxftype() == "MTEXT" else "height", 1) * k, 2)
            x, y = to_pdf(*tuple(e.dxf.insert)[:2])
            for i, line in enumerate(t for t in text.splitlines() if t.strip()):
                page.insert_text((x, y + i * size * 1.2), line, fontsize=size)
            continue
        try:
            p = path.make_path(e)
        except TypeError:  # entities without a drawable outline (points, images, ...)
            continue
        for sub in p.sub_paths():
            pts = [to_pdf(v.x, v.y) for v in sub.flattening(0.5 / k)]
            if len(pts) > 1:
                shape.draw_polyline(pts)
                shape.finish(color=(0, 0, 0), width=0.3, closePath=sub.is_closed)
    shape.commit()
    if not stated:
        page.insert_text((MARGIN * MM, page.rect.height - MARGIN * MM / 2), f"1:{denom}", fontsize=8)
    pdf.save(dst)
    return to_pdf, denom
