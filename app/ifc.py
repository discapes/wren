"""IFC semantics and quantities, with GUID-linked outlines from the supplied plan renders."""
import math
import multiprocessing
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import ifcopenshell
from PIL import Image
import ifcopenshell.util.element as element
import ifcopenshell.util.unit as unit
import ifcopenshell.util.placement as placement
import numpy as np
import pymupdf
from shapely import unary_union
from shapely.geometry import Polygon
from svg.path import Arc, Move, parse_path

import vocab
import ifcplans
from graph import DATA, Edge, Graph, Node, View


CATEGORIES = {
    "IfcSpace": "space", "IfcDoor": "door", "IfcWindow": "window", "IfcWall": "wall",
    "IfcSlab": "slab", "IfcColumn": "column", "IfcBeam": "beam", "IfcStair": "stair",
    "IfcSanitaryTerminal": "sanitary", "IfcLightFixture": "light",
}


def items(source):
    return [(source, p.stem) for p in sorted((DATA / "data" / source).glob("*.ifc"))]


def clean(value):
    """Keep property values JSON-sized; discard IFC ids and exporter placeholder strings."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k != "id" and v is not None and v != "" and v != k}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, ifcopenshell.entity_instance):
        return getattr(value, "Name", None) or value.is_a()
    return value


def profile_area(profile):
    if profile.is_a("IfcRectangleProfileDef"):
        return profile.XDim * profile.YDim
    if profile.is_a("IfcCircleProfileDef"):
        return math.pi * profile.Radius ** 2
    if profile.is_a("IfcArbitraryClosedProfileDef"):
        def points(curve):
            if curve.is_a("IfcPolyline"):
                return [p.Coordinates[:2] for p in curve.Points]
            if curve.is_a("IfcIndexedPolyCurve") and not curve.Segments:
                return [p[:2] for p in curve.Points.CoordList]
        outer = points(profile.OuterCurve)
        holes = [points(c) for c in getattr(profile, "InnerCurves", ())]
        if outer and all(holes):
            return Polygon(outer, holes).area


def swept_area(solid):
    profile = solid.SweptArea
    axis = solid.Position.Axis
    if axis is None or abs(axis.DirectionRatios[2]) > 1 - 1e-6:
        return profile_area(profile)
    # Some Revit spaces are extruded sideways: their profile area is a vertical section.
    if profile.is_a("IfcArbitraryClosedProfileDef") and profile.OuterCurve.is_a("IfcPolyline"):
        coords = [p.Coordinates[:2] for p in profile.OuterCurve.Points]
    elif profile.is_a("IfcRectangleProfileDef"):
        x, y = profile.XDim / 2, profile.YDim / 2
        coords = [(-x, -y), (x, -y), (x, y), (-x, y), (-x, -y)]
    else:
        return None
    matrix = placement.get_axis2placement(solid.Position)
    points = [(matrix @ np.array([x, y, 0, 1]))[:2] for x, y in coords]
    delta = (matrix[:3, :3] @ np.array(solid.ExtrudedDirection.DirectionRatios) * solid.Depth)[:2]
    strips = [Polygon([a, b, b + delta, a + delta]) for a, b in zip(points, points[1:] + points[:1])]
    strips += [Polygon(points), Polygon([p + delta for p in points])]
    return unary_union([p for p in strips if p.area > 1e-12]).area


def space_area(obj, scale):
    """Simple swept space profiles give net area even when the exporter only provides gross Qto."""
    rep = getattr(obj, "Representation", None)
    if not rep:
        return None
    values = []
    for shape in rep.Representations:
        if shape.RepresentationIdentifier != "Body":
            continue
        for solid in shape.Items:
            if solid.is_a("IfcExtrudedAreaSolid"):
                a = swept_area(solid)
                if a is not None:
                    values.append(a * scale ** 2)
            elif solid.is_a("IfcFacetedBrep") or solid.is_a("IfcFaceBasedSurfaceModel"):
                faces = []
                shells = [solid.Outer] if solid.is_a("IfcFacetedBrep") else solid.FbsmFaces
                for face in (f for shell in shells for f in shell.CfsFaces):
                    loops = [([p.Coordinates for p in b.Bound.Polygon], b.is_a("IfcFaceOuterBound"))
                             for b in face.Bounds if b.Bound.is_a("IfcPolyLoop")]
                    if not loops:
                        continue
                    coords = [p for loop, _ in loops for p in loop]
                    if max(p[2] for p in coords) - min(p[2] for p in coords) < 1e-6:
                        a = sum(Polygon([p[:2] for p in loop]).area * (1 if outer else -1)
                                for loop, outer in loops)
                        faces.append((coords[0][2], a))
                if faces:
                    bottom = min(z for z, _ in faces)
                    values.append(sum(a for z, a in faces if abs(z - bottom) < 1e-6) * scale ** 2)
    return sum(values) if values else None


def paths(group):
    rings = []
    for child in group.iter():
        if child.tag.rsplit("}", 1)[-1] != "path":
            continue
        ring = []
        for segment in parse_path(child.get("d", "")):
            if isinstance(segment, Move):
                if len(ring) > 1:
                    rings.append(ring)
                ring = [[segment.end.real, segment.end.imag]]
            else:
                steps = 16 if isinstance(segment, Arc) else 1
                for i in range(1, steps + 1):
                    p = segment.point(i / steps)
                    ring.append([round(p.real, 4), round(p.imag, 4)])
        if len(ring) > 1:
            rings.append(ring)
    return rings


def render_views(g, directory, stem):
    """The dataset's plan renders, or the plans ifcplans.py generated where it has none; raster plans as pictures."""
    for png in sorted(directory.glob(f"{stem.removesuffix('_model')}_*.png")):
        with Image.open(png) as im:
            g.views[png.stem + "/png"] = View(png.stem + "/png", str(png.relative_to(DATA)), (0, 0, *im.size))
    svgs = sorted(directory.glob(f"{stem}_*.svg")) or sorted(ifcplans.plans_dir(directory.name).glob(f"{stem}_*.svg"))
    for svg in svgs:
        root = ET.parse(svg).getroot()
        box = tuple(float(v) for v in root.attrib["viewBox"].split())
        vid = svg.stem + "/svg"
        g.views[vid] = View(vid, str(svg.relative_to(DATA)), box)
        pdf = svg.with_suffix(".pdf")
        pdfid = None
        if pdf.exists():
            with pymupdf.open(pdf) as doc:
                width, height = doc[0].rect.width, doc[0].rect.height
            pdfid = svg.stem + "/pdf"
            g.views[pdfid] = View(pdfid, str(pdf.relative_to(DATA)), (0, 0, width, height))
        outlines = defaultdict(list)
        for group in root.iter():
            guid = group.get("data-guid")
            if guid in g.nodes:
                outlines[guid].extend(paths(group))
        # Stair assemblies often have no representation: the SVG draws their flights/treads.
        direct = set(outlines)
        for guid, shape in list(outlines.items()):
            parent = g.nodes[guid].parent
            while parent in g.nodes and g.nodes[parent].kind == "element":
                if parent not in direct:
                    outlines[parent].extend(shape)
                parent = g.nodes[parent].parent
        for guid, shape in outlines.items():
            g.anchor(guid, vid, shape or None, guid)
            if pdfid:
                # The PDF is the same SVG clipped to its viewBox, converted mm → points.
                scaled = [[[round((x - box[0]) * width / box[2], 4),
                            round((y - box[1]) * height / box[3], 4)] for x, y in ring] for ring in shape]
                g.anchor(guid, pdfid, scaled or None, guid)


def storey_of(g, nid):
    """The storey node above a node, through its parents."""
    while nid in g.nodes and g.nodes[nid].kind != "storey":
        nid = g.nodes[nid].parent
    return nid if nid in g.nodes else None


def centre(rings):
    xs, ys = [x for r in rings for x, _ in r], [y for r in rings for _, y in r]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def slug(name):
    return re.sub(r"[^\w.-]+", "-", name or "").strip("-").lower()


def place_unseen(g, model):
    """Elements no plan draws (slabs below the cut, beams above it, fixtures standing under it) get their footprint
    seen from above on their storey's plan. All plans of a model share one world → plan mapping, as a plan is the same
    drawing cropped to one storey: it is fitted axis by axis on up to 30 drawn elements per plan, dropping elements a
    plan draws only in part (long walls cut at the plan height)."""
    plans = [vid for vid in g.views if vid.endswith("/svg") and "_plan-" in vid]
    items = [n for n in g.nodes.values() if n.kind in ("element", "opening")]
    unseen = [n.id for n in items if not any(a.shape for a in g.anchors.get(n.id, ()))]
    if not plans or not unseen:
        return
    drawn = {vid: [n.id for n in items if any(a.view == vid and a.shape for a in g.anchors.get(n.id, ()))] for vid in plans}
    sample = {vid: ids[::max(1, len(ids) // 30)][:30] for vid, ids in drawn.items()}
    wanted = set(unseen) | {i for ids in sample.values() for i in ids}
    feet = {s.guid: f for s in ifcplans.shapes(model, multiprocessing.cpu_count(),
                                                include=[model.by_guid(i) for i in wanted])
            if (f := ifcplans.footprint(s)) is not None}
    pairs = [((lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))(feet[i].bounds),
              centre(next(a.shape for a in g.anchors[i] if a.view == vid and a.shape)))
             for vid in plans for i in sample[vid] if i in feet]
    if len(pairs) < 6:
        return
    world, plan = np.array([w for w, _ in pairs]), np.array([p for _, p in pairs])
    keep = np.ones(len(pairs), bool)
    for _ in range(3):
        (ax, cx), (ay, cy) = np.polyfit(world[keep, 0], plan[keep, 0], 1), np.polyfit(world[keep, 1], plan[keep, 1], 1)
        err = np.abs(np.c_[ax * world[:, 0] + cx, ay * world[:, 1] + cy] - plan).max(axis=1)
        keep = err <= max(3 * np.median(err), 1e-6)
    size = np.median([max(g.views[vid].box[2:]) for vid in plans])
    if keep.sum() < max(6, len(pairs) // 2) or err[keep].max() > 0.01 * size:
        return  # the plans are not drawn from this model's world coordinates
    storeys = {n.id: n for n in g.nodes.values() if n.kind == "storey"}
    # a plan's storey: the one its file is named after ("..._plan-02-level-1"), else the one most of its elements are on
    by_name = {vid: next((sid for sid, n in storeys.items() if slug(n.name) and vid.removesuffix("/svg").endswith("-" + slug(n.name))),
                         None) for vid in plans}
    plan_of = {}
    for vid in plans:
        sid = by_name[vid] or (drawn[vid] and Counter(storey_of(g, i) for i in drawn[vid]).most_common(1)[0][0])
        plan_of.setdefault(sid, vid)
    for nid in unseen:
        vid = plan_of.get(storey_of(g, nid)) or (plans[0] if len(plans) == 1 else None)
        if nid not in feet or not vid:
            continue
        rings = [[[round(ax * x + cx, 4), round(ay * y + cy, 4)] for x, y in poly.exterior.coords]
                 for poly in getattr(feet[nid], "geoms", [feet[nid]]) if poly.geom_type == "Polygon"]
        g.anchor(nid, vid, rings, nid)
        if pdf := g.views.get(vid.removesuffix("/svg") + "/pdf"):
            box, w, h = g.views[vid].box, pdf.box[2], pdf.box[3]
            g.anchor(nid, pdf.id, [[[round((x - box[0]) * w / box[2], 4), round((y - box[1]) * h / box[3], 4)]
                                    for x, y in r] for r in rings], nid)


def read(item):
    source, stem = item
    directory = DATA / "data" / source
    model = ifcopenshell.open(str(directory / f"{stem}.ifc"))
    scale = unit.calculate_unit_scale(model)
    area_scale = unit.calculate_unit_scale(model, "AREAUNIT")
    volume_scale = unit.calculate_unit_scale(model, "VOLUMEUNIT")
    project = model.by_type("IfcProject")[0]
    g = Graph(stem, source, project.Name or stem, {
        "m_per_unit": scale, "schema": model.schema,
        "authoring": model.header.file_name.originating_system,
        "preprocessor": model.header.file_name.preprocessor_version,
        "quantity_scales": {"length_m": scale, "area_m2": area_scale, "volume_m3": volume_scale},
    })
    g.views["ifc"] = View("ifc", f"data/{source}/{stem}.ifc", (0, 0, 1, 1))
    objects = model.by_type("IfcSpatialStructureElement") + model.by_type("IfcElement")
    for obj in objects:
        cls = obj.is_a()
        if obj.is_a("IfcOpeningElement"):
            continue
        typ = element.get_type(obj)
        cat = next((v for k, v in CATEGORIES.items()
                    if obj.is_a(k) or (typ and typ.is_a(k + "Type"))), cls.removeprefix("Ifc").lower())
        kind = "space" if obj.is_a("IfcSpace") else "storey" if obj.is_a("IfcBuildingStorey") else (
            "opening" if cat in ("door", "window") else "element")
        if obj.is_a("IfcBuilding") or obj.is_a("IfcSite"):
            kind = "drawing"
        parent = element.get_aggregate(obj) or element.get_container(obj)
        if cat == "stair" and parent and parent.is_a("IfcStair"):
            cat = "stair_part"
        attrs = {"ifc_class": cls, "psets": clean(element.get_psets(obj))}
        attrs["psets"] = {k: v for k, v in attrs["psets"].items() if v}
        if typ:
            attrs["type"] = typ.Name
            attrs["type_class"] = typ.is_a()
        materials = element.get_materials(obj)
        if materials:
            attrs["materials"] = sorted({m.Name for m in materials if m.Name})
        material = element.get_material(obj, should_skip_usage=True)
        if material and material.is_a("IfcMaterialLayerSet"):
            attrs["material_layers"] = [
                {"material": layer.Material.Name if layer.Material else None,
                 "thickness_m": layer.LayerThickness * scale} for layer in material.MaterialLayers]
            attrs["thickness_m"] = sum(layer["thickness_m"] for layer in attrs["material_layers"])
        for key in ("OverallWidth", "OverallHeight"):
            if (value := getattr(obj, key, None)) is not None:
                attrs["width_m" if key == "OverallWidth" else "height_m"] = value * scale
        systems = [r.RelatingGroup for r in getattr(obj, "HasAssignments", ())
                   if r.is_a("IfcRelAssignsToGroup")]
        if systems:
            attrs["systems"] = [{"name": s.Name, "class": s.is_a()} for s in systems]
        if getattr(obj, "Description", None):
            attrs["description"] = obj.Description
        if getattr(obj, "ObjectType", None):
            attrs["object_type"] = obj.ObjectType
        for pset in attrs["psets"].values():
            for key, dest, factor in (("NetFloorArea", "area_printed_m2", area_scale),
                                      ("NetSideArea", "side_area_m2", area_scale),
                                      ("NetVolume", "volume_m3", volume_scale),
                                      ("Length", "length_m", scale)):
                if isinstance(pset.get(key), (float, int)):
                    attrs[dest] = pset[key] * factor
        number = getattr(obj, "Tag", None)
        name = obj.Name
        if kind == "space":
            long = obj.LongName
            attrs["ifc_name"], attrs["long_name"] = obj.Name, long
            name = f"{long} ({obj.Name})" if long and obj.Name and long != obj.Name else (long or obj.Name)
            number = obj.Name
            cat = vocab.room_cat(long or obj.Name)
            attrs["outdoor"] = getattr(obj, "InteriorOrExteriorSpace", None) == "EXTERNAL"
            if "area_printed_m2" not in attrs:
                a = space_area(obj, scale)
                if a is not None:
                    attrs["area_m2"] = a
        if kind == "storey" and obj.Elevation is not None:
            attrs["elevation_m"] = obj.Elevation * scale
        g.add(Node(obj.GlobalId, kind, cat, parent.GlobalId if parent else None,
                   name, number, attrs=attrs, via="ifc"))
        g.anchor(obj.GlobalId, "ifc", ref=obj.GlobalId)
    for boundary in model.by_type("IfcRelSpaceBoundary"):
        space, obj = boundary.RelatingSpace, boundary.RelatedBuildingElement
        if obj and obj.GlobalId in g.nodes and space.GlobalId in g.nodes:
            rel = "in" if g.nodes[obj.GlobalId].kind == "opening" else "bounds"
            g.edges.append(Edge(obj.GlobalId, space.GlobalId, rel))
    render_views(g, directory, stem)
    place_unseen(g, model)
    return g.derive()
