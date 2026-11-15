"""Floor plans for IFC models that ship without vector renders: one SVG per storey, cut 1.2 m above its floor.

Every element group carries data-guid like the dataset's own renders, so ifc.py anchors answers to the plans.
uv run ifcplans.py <ifc source>    # writes DATA/plans/<source>/<model>_plan-NN-<storey>.svg
"""
import multiprocessing
import re
import signal
import sys
import traceback
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import ifcopenshell
import ifcopenshell.geom as G
import ifcopenshell.util.element as element
import numpy as np
import shapely
from svgpathtools import parse_path

from graph import DATA

SVG, IFC = "http://www.w3.org/2000/svg", "http://www.ifcopenshell.org/ns"
ET.register_namespace("", SVG)
ET.register_namespace("ifc", IFC)
# lengths in drawing units (metres): 2 cm lines and 25 cm labels, so they scale with the plan wherever it is embedded
STYLE = "path { stroke-width: 0.02px; } text { font: 0.25px sans-serif; fill: #000; }"


def plans_dir(source):
    return DATA / "plans" / source


def todo(source):
    """Models of the source without the dataset's own SVG renders."""
    directory = DATA / "data" / source
    return [p.stem for p in sorted(directory.glob("*.ifc")) if not any(directory.glob(f"{p.stem}_*.svg"))]


PLAN = {"section-height-from-storeys": True, "section-height": 1.2, "door-arcs": True, "print-space-names": True,
        "print-space-areas": True, "svg-xmlns": True}


def storey_of(el):
    """GlobalId of the building storey that contains the element, through containment and aggregation."""
    while el is not None and not el.is_a("IfcBuildingStorey"):
        el = element.get_container(el) or element.get_aggregate(el)
    return el and el.GlobalId


def own_storey_only(storey, f):
    """Drop the elements of other storeys: the serializer writes every element into every storey's group."""
    here = storey.get(f"{{{IFC}}}guid")
    for parent in list(storey.iter(f"{{{SVG}}}g")):
        for child in list(parent):
            guid = child.get(f"{{{IFC}}}guid")
            if guid and child.get("class") != "IfcBuildingStorey":
                try:
                    there = storey_of(f.by_guid(guid))
                except RuntimeError:
                    continue
                if there and there != here:
                    parent.remove(child)
    return storey


def serialize(f, path, options):
    """The serializer's SVG of the opened model for the given settings, as an element tree."""
    s = G.settings()
    for k, v in options.items():
        s.set(k, v)
    out = plans_dir("tmp") / f"{path.stem}.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    ser = G.serializers.svg(str(out), s)
    ser.setFile(f)
    ser.writeHeader()
    it = G.iterator(s, f, 1, exclude=("IfcOpeningElement",))
    if it.initialize():
        while True:
            ser.write(it.get())
            if not it.next():
                break
    ser.finalize()
    text = out.read_text().replace("&amp;#178;", "²")
    out.unlink()
    return ET.fromstring(text)


def extent(group):
    xs, ys = [], []
    for el in group.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "path" and el.get("d"):
            x0, x1, y0, y1 = parse_path(el.get("d")).bbox()
            xs += [x0, x1]
            ys += [y0, y1]
        elif tag == "text":
            xs.append(float(el.get("x")))
            ys.append(float(el.get("y")))
    return xs, ys


def footprint(shape):
    """An iterator shape's outline seen from above, in world metres (y up), or None when it has no area."""
    v = np.array(shape.geometry.verts).reshape(-1, 3)[:, :2]
    tris = shapely.polygons(v[np.array(shape.geometry.faces).reshape(-1, 3)])
    tris = tris[shapely.area(tris) > 1e-6]
    if not len(tris):
        return None
    # merged triangles follow curved roads and walls; very large meshes get their convex outline to stay fast
    foot = shapely.union_all(tris, grid_size=0.001) if len(tris) <= 5000 else shapely.convex_hull(shapely.multipoints(v))
    return None if foot.is_empty else foot


def shapes(f, threads, **kw):
    """The model's iterator shapes in world coordinates, e.g. shapes(f, 4, include=[...]) or exclude=(...)."""
    s = G.settings()
    s.set("use-world-coords", True)
    it = G.iterator(s, f, threads, **kw)
    if it.initialize():
        while True:
            yield it.get()
            if not it.next():
                break


def top_view(f):
    """Each element's footprint seen from above, as one SVG group per element."""
    groups = []
    for shape in shapes(f, 1, exclude=("IfcOpeningElement", "IfcSpace")):
        foot = footprint(shape)
        if foot is None:
            continue
        foot = shapely.transform(foot, lambda c: c * (1, -1))  # SVG y points down
        d = " ".join("M" + " L".join(f"{x:.3f},{y:.3f}" for x, y in ring.coords) + "Z"
                     for poly in getattr(foot, "geoms", [foot]) if poly.geom_type == "Polygon"
                     for ring in [poly.exterior, *poly.interiors])
        g = ET.Element(f"{{{SVG}}}g", {"class": f"projection {shape.type}", f"{{{IFC}}}guid": shape.guid,
                                       f"{{{IFC}}}name": shape.name or ""})
        ET.SubElement(g, f"{{{SVG}}}path", {"d": d, "fill": "#e8e8e8", "stroke": "#333"})
        groups.append(g)
    return groups


def write(path, head, groups):
    """One plan SVG of the given groups, framed to what they draw; False when they draw nothing."""
    xs, ys = [], []
    for group in groups:
        gx, gy = extent(group)
        xs += gx
        ys += gy
    if not xs:
        return False
    for group in groups:
        for g in group.iter(f"{{{SVG}}}g"):
            if guid := g.get(f"{{{IFC}}}guid"):
                g.set("data-guid", guid)
    m = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1)
    x0, y0, w, h = min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m
    svg = ET.Element(f"{{{SVG}}}svg", {"viewBox": f"{x0:.3f} {y0:.3f} {w:.3f} {h:.3f}",
                                       "width": f"{w * 10:.1f}mm", "height": f"{h * 10:.1f}mm"})
    svg.extend(head)
    ET.SubElement(svg, f"{{{SVG}}}style").text = STYLE
    ET.SubElement(svg, f"{{{SVG}}}rect", {"x": f"{x0:.3f}", "y": f"{y0:.3f}", "width": f"{w:.3f}",
                                         "height": f"{h:.3f}", "fill": "#ffffff"})
    svg.extend(groups)
    ET.ElementTree(svg).write(path, encoding="unicode")
    return True


def parts(root):
    head = [el for el in root if el.tag.rsplit("}", 1)[-1] in ("defs", "style")]
    return head, [el for el in root if el not in head]


def too_slow(signum, frame):
    raise TimeoutError("geometry took over 15 minutes")


def render(item):
    """Write one plan SVG per storey that has anything drawn, else a top view; returns how many."""
    source, stem = item
    path = DATA / "data" / source / f"{stem}.ifc"
    out = plans_dir(source)
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, too_slow)
    signal.alarm(900)  # a model whose geometry takes longer than 15 minutes is left without plans
    try:
        f = ifcopenshell.open(str(path))
        root = serialize(f, path, PLAN)
        head, _ = parts(root)
        n = 0
        for storey in (g for g in root.iter(f"{{{SVG}}}g") if g.get("class") == "IfcBuildingStorey"):
            name = re.sub(r"[^\w.-]+", "-", storey.get(f"{{{IFC}}}name") or "storey").strip("-").lower()
            n += write(out / f"{stem}_plan-{n + 1:02d}-{name}.svg", head, [own_storey_only(storey, f)])
        if not n:  # the plan cut only draws elements on building storeys: infrastructure and city models get a top view
            n += write(out / f"{stem}_plan-01-top-view.svg", head, top_view(f))
        return n
    except BaseException:
        print(f"{stem}: {traceback.format_exc()}", file=sys.stderr)
        return 0
    finally:
        signal.alarm(0)


def main(source):
    items = [(source, stem) for stem in todo(source)]
    # the benchmark's models are large: fewer at a time keeps their geometry in memory
    workers = max(1, multiprocessing.cpu_count() // (4 if source == "ifc_bench" else 1) - 1)
    with Pool(workers, maxtasksperchild=1) as pool:
        for i, n in enumerate(pool.imap_unordered(render, items), 1):
            if i % 25 == 0:
                print(f"{i}/{len(items)}", flush=True)
    print(f"plans for {len(items)} models in {plans_dir(source)}")


if __name__ == "__main__":
    main(sys.argv[1])
