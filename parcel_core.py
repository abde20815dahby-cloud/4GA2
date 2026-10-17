"""
DXF Parcel Plan Processor
=========================
Processes ANCFCC-style cadastral DXF parcel plans:
  - Fixes malformed group codes (0  0 pattern)
  - Removes near-zero/garbage entities
  - Detects parcel geometry, neighbour labels, verborne texts, bornes
  - Removes duplicate texts and annotation-layer texts
  - Moves borne-name texts inside parcel
  - Resolves text overlaps
  - Centres texts on their corresponding line elements
  - Handles verborne positioning (centered or one-sided)
  - Clips elements / moves texts that exceed the frame boundary
  - Removes cross (+) symbols inside the parcel

Usage:
    python dxf_parcel_processor.py input.dxf [output.dxf]

Dependencies:
    pip install ezdxf shapely
"""

import sys
import re
import math
import copy
import logging
from pathlib import Path
from typing import Optional
import textwrap
import ezdxf

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.geometry import box as box_polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union, nearest_points, split, substring, linemerge
from shapely.prepared import prep
import ezdxf.bbox as _ezdxf_bbox
from ezdxf.math import Vec3, Vec2
from typing import Optional

from ezdxf import bbox as ezdxf_bbox
from shapely.affinity import rotate


_bbox_cache = _ezdxf_bbox.Cache()


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s"
)
log = logging.getLogger("dxf_processor")

# ──────────────────────────────────────────────────────────────
# 0.  CONSTANTS / TUNABLES
# ──────────────────────────────────────────────────────────────
NEAR_ZERO_THRESHOLD   = 100 #1e-3   # coords closer than this → "zero"
DUPLICATE_DIST        = 0.01   # texts within this distance with same content → duplicate
PROXIMITY_DIST        = 5.0    # search radius to match text → line element (metres/units)
OVERLAP_MARGIN        = 0.15   # extra gap around text bbox when resolving overlaps
MOVE_STEP             = 0.2    # step size when pushing overlapping texts apart
MAX_OVERLAP_ITER      = 200    # give up after this many iterations
BORNE_NAME_RE         = re.compile(r'^B\d+$', re.IGNORECASE)
VERBORNE_RE           = re.compile(r'^v\.', re.IGNORECASE)
NEIGHBOR_LABEL_RE     = re.compile(r'^Plle\s+\d+', re.IGNORECASE)
COORDINATES_RE = re.compile(r'\b\d{6}\b(?:\.\d+)?')
PARCEL_BUFFER = None


# ──────────────────────────────────────────────────────────────
# 1.  PRE-PROCESSING: fix raw DXF bytes before ezdxf parses it
# ──────────────────────────────────────────────────────────────
def fix_malformed_group_codes(raw: bytes) -> bytes:
    """
    Some exporters write '0  0\\r\\n' or '0    0\\r\\n' where the group-code
    line (which should be just '0') has the value '0' appended with spaces.
    Strip the spurious trailing content so ezdxf can parse the file.
    """
    fixed = re.sub(rb'0[ \t]+0\r?\n', b'0\r\n', raw)
    pos_corrupt = -1
    for i in range(len(fixed) - 10):
        b = fixed[i]
        if b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            pos_corrupt = i
            break
    if pos_corrupt != -1:
        end_sec = fixed.find(b'ENDSEC', pos_corrupt)
        if end_sec != -1:
            fixed = fixed[:pos_corrupt] + b'\r\n  0\r\n' + fixed[end_sec:]
    n = raw.count(b'0  0') + raw.count(b'0    0')
    if n:
        log.info(f"Fixed {n} malformed '0  0' group-code lines.")
    return fixed


def load_dxf(path: str) -> ezdxf.document.Drawing:
    import io, tempfile, os
    import ezdxf
    import ezdxf.recover
    with open(path, 'rb') as fh:
        raw = fh.read()
    raw = fix_malformed_group_codes(raw)
    # Write fixed bytes to a temp file so ezdxf can use its normal file reader
    with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tf:
        tf.write(raw)
        tmp_path = tf.name
    try:
        try:
            doc = ezdxf.readfile(tmp_path)
        except Exception:
            doc, _ = ezdxf.recover.readfile(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return doc


# ──────────────────────────────────────────────────────────────
# 2.  HELPER: text geometry (bounding box)
# ──────────────────────────────────────────────────────────────
def text_bbox(entity) -> Optional[Polygon]:
    """
    Returns the accurate oriented bounding box as a Shapely Polygon,
    using ezdxf's own layout engine (handles rotation, fonts, MTEXT formatting).
    Returns None if ezdxf cannot compute it (empty text, missing font, etc.).
    """
    try:
        box = _ezdxf_bbox.extents([entity], cache=_bbox_cache)
        if box is None:
            return None
        # box.extmin / extmax are in WCS already (rotation applied)
        # They give the axis-aligned bounding box of the rotated text
        minx, miny = box.extmin.x, box.extmin.y
        maxx, maxy = box.extmax.x, box.extmax.y
        if math.isnan(minx) or math.isnan(miny):
            return None
        return box_polygon(minx, miny, maxx, maxy)
    except Exception:
        return None



def rotateBox(vx, vy, angle_deg):
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    return (
        vx * c - vy * s,
        vx * s + vy * c,
    )


def rectangle_points(insert, width, height, rotation, dx=0.0, dy=0.0):
    """
    Returns 4 rotated rectangle corners.
    'insert' is the base anchor point in local coordinates.
    'dx', 'dy' are offsets from the anchor to the bottom-left of the box.
    """
    x0, y0 = insert

    # Unrotated local corners relative to the base_point (0,0)
    pts = [
        (dx, dy),
        (dx + width, dy),
        (dx + width, dy + height),
        (dx, dy + height),
    ]

    result = []
    for x, y in pts:
        xr, yr = rotateBox(x, y, rotation)
        result.append((x0 + xr, y0 + yr))

    return Polygon(result)

# -------------------------------------------------------------
# TEXT
# -------------------------------------------------------------

def text_bbox2(entity):
    text = entity.dxf.text
    height = entity.dxf.height
    width_factor = getattr(entity.dxf, "width", 1.0)

    # Approximate average glyph width.
    char_width = 0.65 * height * width_factor
    width = len(text) * char_width
    rotation = entity.dxf.get("rotation", 0.0)

    # Get Alignment Flags
    halign = entity.dxf.get("halign", 0)
    valign = entity.dxf.get("valign", 0)

    # In DXF, if halign or valign are non-zero, the true anchor is 'align_point'
    if halign == 0 and valign == 0:
        base_point = Vec2(entity.dxf.insert)
    else:
        base_point = Vec2(entity.dxf.get("align_point", entity.dxf.insert))

    # Calculate horizontal offset (dx)
    if halign in (1, 4):  # Center or Middle
        dx = -width / 2.0
    elif halign == 2:  # Right
        dx = -width
    else:  # Left, Aligned, Fit
        dx = 0.0

    # Calculate vertical offset (dy)
    if valign == 2:  # Middle
        dy = -height / 2.0
    elif valign == 3:  # Top
        dy = -height
    else:  # Baseline (0) or Bottom (1)
        dy = 0.0

    return rectangle_points(base_point, width, height, rotation, dx, dy)


# -------------------------------------------------------------
# MTEXT
# -------------------------------------------------------------

def mtext_bbox(entity):
    txt = entity.text.replace("\\P", "\n")

    lines = txt.splitlines()
    if not lines:
        lines = [""]

    height = entity.dxf.char_height
    width = entity.dxf.get("width", 0)

    if width <= 0:
        longest = max(len(x) for x in lines)
        width = longest * height * 0.65

    line_spacing = 1.5
    total_height = len(lines) * height * line_spacing
    rotation = entity.get_rotation()

    base_point = Vec2(entity.dxf.insert)

    # 1=TopLeft, 2=TopCenter, 3=TopRight
    # 4=MiddleLeft, 5=MiddleCenter, 6=MiddleRight
    # 7=BottomLeft, 8=BottomCenter, 9=BottomRight
    attachment = entity.dxf.get("attachment_point", 1)

    # Calculate horizontal offset (dx)
    if attachment in (1, 4, 7):  # Left
        dx = 0.0
    elif attachment in (2, 5, 8):  # Center
        dx = -width / 2.0
    elif attachment in (3, 6, 9):  # Right
        dx = -width
    else:
        dx = 0.0

    # Calculate vertical offset (dy) to the bottom edge
    # MTEXT goes from top to bottom, so Top attachment means box goes down (negative)
    if attachment in (1, 2, 3):  # Top
        dy = -total_height
    elif attachment in (4, 5, 6):  # Middle
        dy = -total_height / 2.0
    elif attachment in (7, 8, 9):  # Bottom
        dy = 0.0
    else:
        dy = -total_height

    return rectangle_points(base_point, width, total_height, rotation, dx, dy)

def get_rotated_text_bbox_polygon(entity, cache=None):
    """
    Calculates the exact rotated bounding box of a DXF text entity (TEXT or MTEXT).
    Returns a tightly fitted Shapely Polygon.
    """
    dxftype = entity.dxftype()
    if dxftype not in ('TEXT', 'MTEXT'):
        return None

    # 1. Save original rotation (defaults to 0 if not set)
    orig_rotation = getattr(entity.dxf, 'rotation', 0.0)

    try:
        # 2. Temporarily remove rotation to get exact local dimensions
        entity.dxf.rotation = 0.0
        if cache is not None:
            cache.invalidate([entity])

        raw_bbox = ezdxf_bbox.extents([entity], cache=cache)
        if not raw_bbox.has_data:
            return None

        # Create the tight unrotated Shapely box
        unrotated_poly = box_polygon(
            raw_bbox.extmin.x, raw_bbox.extmin.y,
            raw_bbox.extmax.x, raw_bbox.extmax.y
        )

    finally:
        # 3. Always restore the original rotation to leave the DXF untouched
        entity.dxf.rotation = orig_rotation
        if cache is not None:
            cache.invalidate([entity])

    # 4. If there's no rotation, return the tight box directly
    if abs(orig_rotation) < 1e-5:
        return unrotated_poly

    # 5. Determine the correct pivot point for rotation
    pivot = _get_text_pivot(entity)

    # 6. Apply the true orientation mathematically using Shapely
    # DXF rotation is in degrees (Counter-Clockwise), matching Shapely's default
    rotated_poly = rotate(unrotated_poly, orig_rotation, origin=pivot)

    return rotated_poly


def _get_text_pivot(entity):
    """
    Helper to find the exact origin point the text pivots around.
    MTEXT and standard TEXT handle basepoints and justification differently.
    """
    dxftype = entity.dxftype()

    if dxftype == 'MTEXT':
        # MTEXT always pivots around its single insertion point
        return (entity.dxf.insert.x, entity.dxf.insert.y)

    elif dxftype == 'TEXT':
        # Simple TEXT pivots around 'align_point' if horizontally or vertically justified.
        # If it is standard baseline-left (0,0), it pivots around 'insert'.
        halign = entity.dxf.get('halign', 0)
        valign = entity.dxf.get('valign', 0)

        if halign == 0 and valign == 0:
            return (entity.dxf.insert.x, entity.dxf.insert.y)
        else:
            return (entity.dxf.align_point.x, entity.dxf.align_point.y)

    return (0, 0)

def text_size(entity) -> tuple:
    """
    Returns (width, height) of the text's axis-aligned bounding box.
    Falls back to character-height estimate if ezdxf cannot compute it.
    """
    try:
        box = _ezdxf_bbox.extents([entity], cache=_bbox_cache)
        if box is None:
            raise ValueError
        w, h = box.size.x, box.size.y
        if math.isnan(w) or math.isnan(h) or w == 0 or h == 0:
            raise ValueError
        return w, h
    except Exception:
        # fallback
        try:
            ht = (entity.dxf.char_height if entity.dxftype() == 'MTEXT'
                  else entity.dxf.height)
            txt = get_text_content(entity)
            lines = txt.split('\n')
            return max(len(l) for l in lines) * ht * 0.6, len(lines) * ht * 1.2
        except Exception:
            return 1.0, 1.0


def get_text_content(entity) -> str:
    try:
        if entity.dxftype() == 'MTEXT':
            return entity.plain_text().strip()
        return entity.dxf.text.strip()
    except Exception:
        return ''


def get_insert(entity):
    """Returns (x, y) tuple for insertion point."""
    ins = entity.dxf.insert
    return float(ins.x), float(ins.y)


def set_insert(entity, x: float, y: float):
    ins = entity.dxf.insert
    entity.dxf.insert = Vec3(x, y, float(ins.z) if hasattr(ins, 'z') else 0.0)


def wrap_string(text, max_length=15):
    """
    Intelligently adds a line break (\n) to a string if it exceeds max_length.
    It tries to break at spaces, but will force a break inside a word if necessary.
    """
    if not text or len(text) <= max_length:
        return text

    # break_long_words=True forces a break even if there are no spaces
    return textwrap.fill(text, width=max_length, break_long_words=False)


def enforce_text_length_limit(msp, txt_ent, max_length=15):
    """
    Evaluates a DXF text entity. If the text is too long, it adds line breaks.
    Automatically upgrades basic 'TEXT' to 'MTEXT' to support the multi-line display.

    Returns the updated (or newly created) entity.
    """
    # 1. Get current text content safely
    if txt_ent.dxftype() == 'TEXT':
        raw_text = txt_ent.dxf.text
    elif txt_ent.dxftype() == 'MTEXT':
        raw_text = txt_ent.text
    else:
        return txt_ent

    # 2. Check if wrapping is actually needed
    if len(raw_text) <= max_length:
        return txt_ent

    # 3. Apply the line break
    dxf_lines = raw_text.split(r"\P")
    # wrapped_text = wrap_string(raw_text_parts[0], max_length)+raw_text_parts[1]
    # print('txt wrap', raw_text_parts, wrapped_text)

    wrapped_sections = []
    for line in dxf_lines:
        wrapped = wrap_string(line, max_length)
        wrapped_sections.append(wrapped)
    wrapped_text = "\n".join(wrapped_sections)

    # 4. Update the entity (and convert if necessary)
    if txt_ent.dxftype() == 'MTEXT':
        txt_ent.text = wrapped_text
        return txt_ent

    elif txt_ent.dxftype() == 'TEXT':
        # Upgrading TEXT to MTEXT so line breaks render correctly in CAD

        # Grab the exact position. Use align_point if it was centered/justified
        try:
            insert_pt = txt_ent.dxf.align_point if txt_ent.dxf.halign > 0 else txt_ent.dxf.insert
        except AttributeError:
            insert_pt = txt_ent.dxf.insert

        # Extract visual properties to match the original
        attribs = {
            'layer': txt_ent.dxf.layer,
            'char_height': txt_ent.dxf.height,
            'style': txt_ent.dxf.style
        }

        if txt_ent.dxf.hasattr('color'):
            attribs['color'] = txt_ent.dxf.color

        # Create the new Multiline Text
        new_mtext = msp.add_mtext(wrapped_text, dxfattribs=attribs)

        # 5 = Middle-Center attachment (Best for centered labels)
        new_mtext.dxf.attachment_point = 5
        new_mtext.set_location(insert_pt)

        # Destroy the old single-line text
        txt_ent.destroy()

        return new_mtext

def get_drawing_scale(msp) -> float:
    """
    Searches for a text matching 'Echelle : 1/NNN' and returns the
    denominator as a float (e.g. 500.0).  Falls back to 500.
    """
    scale_re = re.compile(r'[Ee]chelle\s*:\s*1\s*/\s*(\d+)')
    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue
        txt = get_text_content(e)
        m   = scale_re.search(txt)
        if m:
            scale = float(m.group(1))
            log.info("Drawing scale detected: 1/%g", scale)
            return scale
    log.warning("Scale text not found, defaulting to 1/500.")
    return 500.0


def is_outside_frame(entity, frame: Optional[Polygon]) -> bool:
    """Returns True if text insertion point is outside or on the frame boundary."""
    if frame is None:
        return False
    x, y = get_insert(entity)
    return not frame.contains(Point(x, y))


# Prefixes to never touch in clip_elements_to_frame
_FRAME_LABEL_PREFIXES = re.compile(
    r'^(Propri|Situ|Titre|R[eé]quisition\s*N°?\s*:|DIRECTION|ST|Echel|Plan|Limit|NORD|'
    r'Plle\s*n°|IFE|MAPPE\s+DE|Plan\s+[eé]tabli|Plan\s+Provisoire\s+Avant\s+enr[oô]lement)',
    re.IGNORECASE
)

def _is_arabic_text(entity) -> bool:
    """
    Detects Arabic text by checking the assigned text style name
    or the presence of Arabic Unicode characters in the content.
    """
    # Check style name
    try:
        style = entity.dxf.style
        if any(s in style.lower() for s in ('arab', 'iscp', 'simplified arabic',
                                             'traditional arabic', 'arabic')):
            return True
    except Exception:
        pass
    # Check content for Arabic Unicode block (U+0600–U+06FF)
    txt = get_text_content(entity)
    return any('\u0600' <= c <= '\u06ff' for c in txt)

def _is_frame_label(entity) -> bool:
    """Returns True if the text should never be touched by clip_elements_to_frame."""
    txt = get_text_content(entity)
    if _FRAME_LABEL_PREFIXES.match(txt):
        return True
    if _is_arabic_text(entity):
        return True
    return False

def _get_line_plot(msp):
    for txt_ent in msp.query('TEXT MTEXT'):
        txt_cnt = get_text_content(txt_ent).upper()
        if 'ST 25' in txt_cnt or 'ST25' in txt_cnt:
            return 0.1
        elif 'ST 26' in txt_cnt or 'ST26' in txt_cnt:
            return 0.2

# ──────────────────────────────────────────────────────────────
# 3.  STEP-BY-STEP PROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────

# 3.1  Text sizes inventory
def inventory_text_sizes(msp) -> dict:
    """Returns dict layer → list of (width, height) for every text entity."""
    result = {}
    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue
        layer = e.dxf.layer
        w, h  = text_size(e)
        result.setdefault(layer, []).append((w, h))
    log.info("Text-size inventory: %d layers with text entities.",
             len(result))
    return result


# 3.2  Delete near-zero entities
def delete_near_zero_entities(msp):
    """
    Remove entities whose defining points are all near (0, 0).
    Applies to TEXT, MTEXT, LINE, LWPOLYLINE, INSERT.
    """
    to_delete = []
    thr       = NEAR_ZERO_THRESHOLD
    for e in msp:
        try:
            t = e.dxftype()
            if t in ('TEXT', 'MTEXT'):
                ins = e.dxf.insert
                if abs(ins.x) < thr and abs(ins.y) < thr:
                    to_delete.append(e)
                    # print('to be destroyd : ',ins)
            elif t == 'LINE':
                s, en = e.dxf.start, e.dxf.end
                if (abs(s.x) < thr and abs(s.y) < thr and
                        abs(en.x) < thr and abs(en.y) < thr) or (math.hypot(en.x - s.x, en.y - s.y) < 1e-3):
                    to_delete.append(e)
                    # print('to be destroyd : ',s,en)

            elif t == 'LWPOLYLINE':
                pts = list(e.get_points())
                if len(pts) < 2 or all(abs(p[0]) < thr and abs(p[1]) < thr for p in pts):
                    to_delete.append(e)
                    # print('to be destroyd : ',pts)

            elif t == 'INSERT':
                ins = e.dxf.insert
                if abs(ins.x) < thr and abs(ins.y) < thr:
                    to_delete.append(e)
                    # print('to be destroyd : ',ins)
        except Exception:
            print('exeption delete near zero')
            pass
    for e in to_delete:
        e.destroy()
    from ezdxf import zoom
    zoom.extents(msp)

    log.info("Deleted %d near-zero entities.", len(to_delete))


# 3.3  Get parcel geometry (Shapely Polygon)
def get_parcel_polygon(msp) -> Optional[Polygon]:
    """
    Returns the parcel boundary as a Shapely Polygon.
    Looks for a closed LWPOLYLINE on layer 'parcelle' (colour 1 = red).
    Falls back to the largest closed LWPOLYLINE in the drawing.
    """
    candidates = []
    for e in msp.query('LWPOLYLINE'):
        if e.dxf.layer.lower() == 'parcelle' or e.dxf.color == 1:
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                candidates.append((e, pts))

    if not candidates:
        # fallback: largest closed polyline
        for e in msp.query('LWPOLYLINE'):
            if e.closed:
                pts = [(p[0], p[1]) for p in e.get_points()]
                if len(pts) >= 3:
                    candidates.append((e, pts))

    if not candidates:
        # Robust Fallback 1: Check 2D POLYLINE entities on layer 'parcelle' or color == 1 or closed
        for e in msp.query('POLYLINE'):
            if getattr(e, 'is_3d_polyline', False) or getattr(e, 'is_polygon_mesh', False):
                continue
            try:
                verts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if len(verts) >= 3:
                    if e.dxf.layer.lower() == 'parcelle' or e.dxf.color == 1 or getattr(e, 'is_closed', False):
                        candidates.append((e, verts))
            except Exception:
                pass

    if not candidates:
        # Robust Fallback 2: Check common Moroccan parcel layers (parc, assiette, limite, lot)
        target_layers = {'parc', 'assiette', 'limite', 'parcelle_cadastrale', 'lot'}
        for e in msp.query('LWPOLYLINE'):
            if e.dxf.layer.lower() in target_layers:
                pts = [(p[0], p[1]) for p in e.get_points()]
                if len(pts) >= 3:
                    candidates.append((e, pts))

    if not candidates:
        # Robust Fallback 3: Polylines that are practically closed (distance(start, end) < 0.05)
        for e in msp.query('LWPOLYLINE'):
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.05:
                    candidates.append((e, pts))

    if not candidates:
        log.warning("No parcel polygon found.")
        return None

    # Pick the largest area
    best, best_area = None, 0.0
    for e, pts in candidates:
        try:
            poly = Polygon(pts)
            a    = poly.area
            if a > best_area:
                best_area = a
                best      = poly
        except Exception:
            pass
    log.info("Parcel polygon area = %.2f sq-units.", best_area)
    return best


# 3.4  Get frame boundary
def get_frame_polygon(msp) -> Optional[Polygon]:
    """
    Returns the drawing frame as a Shapely Polygon.
    Looks for a closed LWPOLYLINE on layer 0 that is substantially
    larger than the parcel.
    """
    best, best_area = None, 0.0
    for e in msp.query('LWPOLYLINE[layer=="0"]'):
        if e.closed:
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    if poly.area > best_area:
                        best_area = poly.area
                        best      = poly
                except Exception:
                    pass
    if best:
        log.info("Frame polygon area = %.2f sq-units.", best_area)
        return best

    # Robust Fallback 1: check other frame layers (cadre, frame, cartouche, papier)
    frame_layers = {'cadre', 'cadre_plan', 'frame', 'cartouche', 'format', 'papier'}
    for e in msp.query('LWPOLYLINE'):
        if e.dxf.layer.lower() in frame_layers and e.closed:
            pts = [(p[0], p[1]) for p in e.get_points()]
            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    if poly.area > best_area:
                        best_area = poly.area
                        best      = poly
                except Exception:
                    pass
    if best:
        log.info("Frame polygon area (fallback layer) = %.2f sq-units.", best_area)
        return best

    # Robust Fallback 2: check 2D POLYLINE on layer 0
    for e in msp.query('POLYLINE[layer=="0"]'):
        if getattr(e, 'is_3d_polyline', False) or getattr(e, 'is_polygon_mesh', False):
            continue
        try:
            verts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if len(verts) >= 3 and (getattr(e, 'is_closed', False) or math.hypot(verts[0][0] - verts[-1][0], verts[0][1] - verts[-1][1]) < 0.05):
                poly = Polygon(verts)
                if poly.area > best_area:
                    best_area = poly.area
                    best      = poly
        except Exception:
            pass

    if best:
        log.info("Frame polygon area (fallback polyline) = %.2f sq-units.", best_area)
    else:
        log.warning("No frame polygon found.")
    return best


# 3.5  Neighbour parcel labels

# Assuming get_frame_polygon, get_text_content, get_insert, NEIGHBOR_LABEL_RE, enforce_text_length_limit are available

# Assuming get_frame_polygon, get_text_content, get_insert, NEIGHBOR_LABEL_RE, enforce_text_length_limit are available

def enforce_text_wrap(msp, center_texts=None, max_length=15) -> list:
    """
    Returns list of dicts with keys: entity, text, x, y
    for texts matching 'Plle NNNN' pattern.
    Only processes texts that are inside the drawing frame and
    skips explicitly ignored prefixes (case-insensitive).
    """
    labels = []

    # Get the frame polygon to filter texts geographically
    frame = get_frame_polygon(msp)

    # Prefixes to skip (all lowercase for case-insensitive comparison)
    ignored_prefixes = (
        "plle n",
        "ife :",
        "sous zone",
        "plan provisoire",
        "contenance :"
    )
    pattern_ignore = re.compile(r'^\(\d+\)')

    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue

        txt = get_text_content(e)

        # 1. Don't process text starting with the specified prefixes (case-insensitive)
        if txt.lower().startswith(ignored_prefixes) or pattern_ignore.match(txt):
            continue

        x, y = get_insert(e)

        # 2. Only process text inside the frame (if a frame was found)
        if frame and not frame.contains(Point(x, y)):
            continue

        # Ne jamais forcer le retour à la ligne sur les riverains ni sur les voiries/hydrographie (conforme DEF 100%)
        _LINEAR_LABELS = ("route", "piste", "chaaba", "châaba", "chaâbat", "oued", "mur", "sentier", "saguia", "seguia", "fossé")
        if (NEIGHBOR_LABEL_RE.match(txt)
                or txt.upper().endswith('TC')
                or '\\PTC' in txt.upper()
                or (center_texts and any(word.upper() in txt.upper() for word in center_texts))
                or any(ll in txt.lower() for ll in _LINEAR_LABELS)):
            continue

        enforce_text_length_limit(msp, e, max_length)

    log.info("Found %d neighbour parcel labels.", len(labels))
    return labels


# 3.6  Elements from non-zero layers + their layer-0 text (by proximity / angle)
def get_layer_elements_with_labels(msp) -> list:
    """
    For every non-zero, non-annotation LINE or LWPOLYLINE, attempt to find
    a matching MTEXT/TEXT in layer 0 by proximity and similar angle.
    Returns list of dicts: {element, layer, label_entity, label_text}
    """
    skip_layers = {'0', 'annotation', 'parcelle', 'plle', 'bornes',
                   'CADRE', 'CADREP1', 'HABILLAGE', 'T-LEGENDE',
                   'legende', 'titres', 'Defpoints'}

    lines_l0   = [e for e in msp.query('LINE[layer=="0"]')]
    texts_l0   = [e for e in msp
                  if e.dxftype() in ('TEXT', 'MTEXT') and e.dxf.layer == '0']

    def line_midpoint_angle(entity):
        if entity.dxftype() == 'LINE':
            s, en = entity.dxf.start, entity.dxf.end
            mx    = (s.x + en.x) / 2
            my    = (s.y + en.y) / 2
            ang   = math.degrees(math.atan2(en.y - s.y, en.x - s.x)) % 180
            return mx, my, ang
        else:  # LWPOLYLINE – use centroid
            pts = [(p[0], p[1]) for p in entity.get_points()]
            ls  = LineString(pts)
            cx, cy = ls.centroid.x, ls.centroid.y
            # Approximate angle from first to last point
            dx  = pts[-1][0] - pts[0][0]
            dy  = pts[-1][1] - pts[0][1]
            ang = math.degrees(math.atan2(dy, dx)) % 180
            return cx, cy, ang

    results = []
    for e in msp:
        if e.dxftype() not in ('LINE', 'LWPOLYLINE'):
            continue
        if e.dxf.layer in skip_layers:
            continue
        mx, my, ang = line_midpoint_angle(e)
        best_txt, best_dist = None, PROXIMITY_DIST
        for te in texts_l0:
            tx, ty = get_insert(te)
            d      = math.hypot(tx - mx, ty - my)
            if d < best_dist:
                # Optional: angle check (text rotation ≈ line angle)
                rot = getattr(te.dxf, 'rotation', 0) % 180
                if abs(rot - ang) < 25 or abs(rot - ang) > 155:
                    best_dist = d
                    best_txt  = te
        results.append(dict(
            element=e,
            layer=e.dxf.layer,
            label_entity=best_txt,
            label_text=get_text_content(best_txt) if best_txt else None
        ))
    log.info("Found %d non-parcel line elements; %d have a layer-0 label.",
             len(results), sum(1 for r in results if r['label_entity']))
    return results


# ──────────────────────────────────────────────────────────────
# PROCESSING STEPS
# ──────────────────────────────────────────────────────────────

# 6.  Delete duplicate texts
def delete_duplicate_texts(msp):
    """
    Remove exact-content duplicates that share the same insertion point
    (within DUPLICATE_DIST).  Keeps the first occurrence.
    Also ensures borne labels (Bxxx) are strictly unique per parcel.
    """
    seen   = {}   # (round_x, round_y, text) → entity
    delete = []
    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue
        txt   = get_text_content(e)
        x, y  = get_insert(e)
        key   = (round(x / DUPLICATE_DIST),
                 round(y / DUPLICATE_DIST),
                 txt)
        if key in seen:
            delete.append(e)
        else:
            seen[key] = e
    for e in delete:
        e.destroy()

    # Enforce strictly unique borne labels (Bxxx)
    borne_seen = {}
    for e in list(msp):
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue
        txt = get_text_content(e).strip()
        m = re.match(r'^B(\d+)$', txt, re.IGNORECASE)
        if m:
            b_num = m.group(1)
            if b_num in borne_seen:
                try:
                    msp.delete_entity(e)
                except Exception:
                    e.destroy()
            else:
                borne_seen[b_num] = e

    log.info("Deleted %d duplicate texts.", len(delete))

def _add_space_before_plus(txt, x):
    # Find the index of the first '+' starting from character number X
    target_index = txt.find('+', x)
    # If a '+' is found (find returns -1 if not found)
    if target_index != -1:
        # Slice the string and insert a space right before the '+'
        return txt[:target_index] + " " + txt[target_index:]
    # Return the original text if no '+' was found after X
    return txt

# 7.  Delete texts in layer 'annotation'
def delete_annotation_layer_texts(msp, parcel, max_length=15, center_texts=None):
    count = 0
    to_del = []
    deleted_texts_inside = []
    dotseparated = re.compile(r'[A-Za-z](\.[A-Za-z])+')
    _VALID_CULTURES = re.compile(
        r'^(olivier|oliviers|amandier|amandiers|arbres\s+fruitiers|arbres\s+divers|arbres|figuier|figuiers|palmier|palmiers|vigne|vignes)\b',
        re.IGNORECASE
    )

    # Case-insensitive lookup set for texts we must never collect/append
    center_texts_upper = {t.strip().upper() for t in center_texts if t and t.strip()} if center_texts else set()

    # 1. Find elements to delete and save the text content if inside the parcel
    for e in msp:
        if e.dxftype() in ('TEXT', 'MTEXT') and getattr(e.dxf, 'layer', '') == 'annotation':
            txt_content = get_text_content(e).strip()
            if not txt_content:
                # Conformité ANCFCC DEF: Préserver les entités MTEXT vides qui servent de repères de culture
                continue
            # Preserve vital hydraulic and topographic landmarks like Puits, Bassin, Chateau, Ruine, Ecurie, Rdc, Construction
            if re.match(r'^(puits?|bassin|ch[aâ]teau\s+d[\'’]eau|ch[aâ]teau|borne\s+fontaine|ruine|ecurie|rdc|caroubier|construction|b[aâ]timent|maison)\b', txt_content, re.IGNORECASE):
                continue
            to_del.append(e)
            if parcel is not None:
                tx, ty = get_insert(e)
                if (parcel.contains(Point(tx, ty))
                        and not dotseparated.match(txt_content)
                        and txt_content.strip().upper() not in center_texts_upper
                        and _VALID_CULTURES.match(txt_content)):
                    deleted_texts_inside.append(txt_content)

    # 2. Destroy the non-empty annotation texts
    for e in to_del:
        e.destroy()
        count += 1

    # 3. Locate TC texts and collect any other texts inside the parcel
    if parcel is not None:
        tc_element = None
        other_texts_inside = []

        for e in msp:
            if e.dxftype() in ('TEXT', 'MTEXT') and e.is_alive:
                tx, ty = get_insert(e)
                if parcel.contains(Point(tx, ty)):
                    txt_content = get_text_content(e).strip()

                    if txt_content.upper() in ('TC', 'T.C'):
                        tc_element = e
                    else:
                        if (txt_content
                                and not txt_content.isdigit()
                                and not BORNE_NAME_RE.match(txt_content)
                                and txt_content.upper() not in center_texts_upper
                                and _VALID_CULTURES.match(txt_content)):
                            other_texts_inside.append(txt_content)

        # 4. Deduplicate, combine, and append to the TC entities
        # Combine both lists of texts we want to append
        all_texts_to_append = deleted_texts_inside + other_texts_inside

        # Deduplicate using dict.fromkeys to preserve the order (unlike set())
        cleaned_texts = list(dict.fromkeys(
            text.strip() for text in all_texts_to_append if text and text.strip()
        ))

        # Only proceed if there are valid texts to append and we found a TC entity
        if cleaned_texts and tc_element:
            joined_all = "\n".join(cleaned_texts)
            cur_txt = tc_element.text if hasattr(tc_element, 'text') else getattr(tc_element.dxf, 'text', '')
            sep = "\\P" if tc_element.dxftype() == "MTEXT" else "\n"
            new_txt = f"{cur_txt}{sep}{joined_all}"
            if hasattr(tc_element, "text"):
                tc_element.text = new_txt
            tc_element.dxf.text = new_txt
            enforce_text_length_limit(msp, tc_element, max_length)

    log.info("Deleted %d annotation-layer texts.", count)

# 8a.  Move borne names inside parcel
def move_borne_names_inside(msp, parcel: Polygon, frame: Optional[Polygon]):
    """
    Moves borne-name texts (B####, B####bis, …) so their bounding box
    lies fully inside the parcel, with zero intersection with the boundary.

    Algorithm (iterative deepening buffer)
    ---------------------------------------
    1. Find bornes whose bbox is NOT fully contained in the parcel.
    2. Compute the offset between the text's insert point and its bbox
       corners (the insert point is NOT necessarily the bbox centre —
       for default TEXT alignment it's the bottom-left corner).
    3. Build a first inner buffer at half the bbox's own extent and snap
       the insert point to the nearest point on that ring.
    4. Re-check using the real offset-corrected bbox: is it now fully
       inside the parcel?
    5. If not, deepen the buffer and repeat.
    6. Stop once the bbox is fully inside, or after MAX_ROUNDS — in the
       rare case the parcel is too narrow for the text at any position,
       log a warning and keep the best position found.
    """

    MAX_ROUNDS = 10

    if parcel is None:
        return

    borne_re = re.compile(r'^B\d+\w*$', re.IGNORECASE)

    count = 0
    stuck = 0

    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT'):
            continue
        if not borne_re.match(get_text_content(e)):
            continue
        if is_outside_frame(e, frame):
            continue

        ix, iy = get_insert(e)

        # ── bbox extents and offset from insert point ──────────────────────
        _bbox_cache.invalidate([e])
        raw = _ezdxf_bbox.extents([e], cache=_bbox_cache)
        if raw is None:
            continue

        off_minx = raw.extmin.x - ix
        off_miny = raw.extmin.y - iy
        off_maxx = raw.extmax.x - ix
        off_maxy = raw.extmax.y - iy

        bbox_w = raw.extmax.x - raw.extmin.x
        bbox_h = raw.extmax.y - raw.extmin.y
        if bbox_w <= 0 or bbox_h <= 0:
            continue
        eps = bbox_w * 0.1
        # Conservative half-extent in each direction from the insert point —
        # used as the buffer depth, since the insert point may be off-centre.
        half_w = max(abs(off_minx), abs(off_maxx))/4

        def bbox_at(x, y):
            return box_polygon(x + off_minx -eps , y + off_miny,
                                x + off_maxx + eps , y + off_maxy)

        # ── 1. Skip bornes already fully inside ─────────────────────────────
        if parcel.contains(bbox_at(ix, iy)):
            continue

        # ── 2-6. Iterative deepening buffer ──────────────────────────────────
        buffer_depth = half_w
        placed = False

        # ── Replace the inner buffer block in move_borne_names_inside ────────────────
        for round_n in range(MAX_ROUNDS):
            inner = parcel.buffer(-buffer_depth)

            if inner.is_empty:
                break

            # Handle MultiPolygon or GeometryCollection resulting from negative buffer
            if inner.geom_type == 'MultiPolygon':
                # Pick the sub-polygon closest to the borne text insertion point
                pt = Point(ix, iy)
                inner = min(inner.geoms, key=lambda poly: poly.distance(pt))
            elif inner.geom_type != 'Polygon':
                break

            pt = Point(ix, iy)
            nearest = inner.exterior.interpolate(inner.exterior.project(pt))
            ix, iy = nearest.x, nearest.y

            if parcel.contains(bbox_at(ix, iy)):
                placed = True
                break

            buffer_depth += half_w * 0.5

        set_insert(e, ix, iy)
        _bbox_cache.invalidate([e])

        if not placed:
            stuck += 1
            log.warning(
                "move_borne_names_inside: could not fully clear parcel boundary "
                "for '%s' after %d rounds (parcel too narrow for text size); "
                "best-effort position kept.",
                get_text_content(e), MAX_ROUNDS
            )

        count += 1

    log.info(
        "move_borne_names_inside: processed %d borne texts (%d could not be "
        "fully cleared — parcel too narrow for the text at any nearby position).",
        count, stuck
    )



def move_dxf_text(text_entity, dx, dy):
    """Safely translates an ezdxf text or mtext entity by dx, dy."""
    # Move primary insertion point
    current_insert = text_entity.dxf.insert
    text_entity.dxf.insert = (current_insert.x + dx, current_insert.y + dy, current_insert.z)

    # If the text has an alignment point (used for centered, right-aligned, etc.), move it too
    if text_entity.dxf.hasattr('align_point'):
        current_align = text_entity.dxf.align_point
        text_entity.dxf.align_point = (current_align.x + dx, current_align.y + dy, current_align.z)



def is_text_locked(text_entity):
    """Returns True if the text matches the locked regex patterns or belongs to protected layers."""
    lay = getattr(text_entity.dxf, 'layer', '').lower()
    if lay in ('annotation', 'defpoints', 'cadre', 'cadrep1', 't-legende'):
        return True
    content = get_text_content(text_entity)
    if not content or not content.strip():
        return True
    c_lower = content.lower()
    if any(k in c_lower for k in ("tc", "jbel", "boualam", "heritiers", "hrts", "hert", "st 25", "st25", "contenance")):
        return True
    return bool(VERBORNE_RE.match(content) or NEIGHBOR_LABEL_RE.match(content) or COORDINATES_RE.match(content))


def resolve_text_overlaps(msp, frame, parcel_polygon, padding=0.5, max_iterations=10, step_size=0.5):
    """
    Resolves overlaps, keeping 'verborne' and 'neighbor' labels completely locked in place.
    """
    ignored_prefixes = (
        "plle n",
        "ife :",
        "sous zone",
        "plan provisoire",
        "contenance :"
    )
    for iteration in range(max_iterations):
        overlap_found = False

        # 1. Resolve Text vs. Parcel overlaps
        text_entities = [e for e in msp if e.dxftype() in ('TEXT', 'MTEXT')
                         and not is_outside_frame(e, frame)
                         and getattr(e.dxf, 'layer', '').lower() != 'annotation'
                         and get_text_content(e).strip()]
        for text in text_entities:
            if is_text_locked(text):
                continue  # Skip moving locked texts
            if get_text_content(text).lower().startswith(ignored_prefixes) :
                continue
            bbox = _get_text_real_bbox(text)
            insert_pt = Point(text.dxf.insert.x, text.dxf.insert.y)

            if not parcel_polygon.contains(insert_pt) and parcel_polygon.intersects(bbox):
                overlap_found = True

                boundary = parcel_polygon.boundary
                nearest_pt = boundary.interpolate(boundary.project(insert_pt))

                dx = insert_pt.x - nearest_pt.x
                dy = insert_pt.y - nearest_pt.y
                dist = math.hypot(dx, dy)

                if dist == 0:
                    dx, dy = 1.0, 1.0
                    dist = 1.414

                nx, ny = dx / dist, dy / dist

                while parcel_polygon.intersects(bbox) or parcel_polygon.distance(bbox) < padding:
                    move_dxf_text(text, nx * step_size, ny * step_size)
                    bbox = _get_text_real_bbox(text)

        # 2. Resolve Text vs. Text overlaps
        for i in range(len(text_entities)):
            for j in range(i + 1, len(text_entities)):
                text1 = text_entities[i]
                text2 = text_entities[j]

                bbox1 = _get_text_real_bbox(text1)
                bbox2 = _get_text_real_bbox(text2)

                if bbox1.intersects(bbox2):
                    locked1 = is_text_locked(text1)
                    locked2 = is_text_locked(text2)

                    if locked1 and locked2:
                        # Both are locked; we cannot resolve this overlap without breaking rules
                        continue

                    overlap_found = True

                    c1 = bbox1.centroid
                    c2 = bbox2.centroid

                    # Determine which text moves and in which direction
                    if locked2 and not locked1:
                        # text2 is locked, move text1 away from text2
                        text_to_move = text1
                        static_bbox = bbox2
                        dx = c1.x - c2.x
                        dy = c1.y - c2.y

                    else:
                        # Either text1 is locked, or neither is locked. Move text2 away from text1.
                        text_to_move = text2
                        static_bbox = bbox1
                        dx = c2.x - c1.x
                        dy = c2.y - c1.y

                    dist = math.hypot(dx, dy)
                    if dist == 0:
                        dx, dy = 1.0, 0.0
                        dist = 1.0

                    nx, ny = dx / dist, dy / dist

                    moving_bbox = _get_text_real_bbox(text_to_move)

                    while static_bbox.intersects(moving_bbox) or static_bbox.distance(moving_bbox) < padding:
                        move_dxf_text(text_to_move, nx * step_size, ny * step_size)
                        moving_bbox = _get_text_real_bbox(text_to_move)

        if not overlap_found:
            break

    return text_entities

# 8b.  Resolve text overlaps

def _get_text_real_bbox(e):
    t = e.dxftype()
    try:
        if t == "TEXT":
            bbox = text_bbox2(e)
        elif t == "MTEXT":
            bbox = mtext_bbox(e)
    except Exception:
        bbox = get_rotated_text_bbox_polygon(e, _bbox_cache)
    return bbox


# 8c.  Centre texts on their corresponding line element

# -----------------------------------------------------------------------------
# 1. Extraction & Filtering Functions
# -----------------------------------------------------------------------------

import re

def _get_target_texts(msp, center_texts, add_lv):
    """Extracts and filters target texts from layer '0'."""
    center_texts = center_texts or []
    borne_re = re.compile(r'^B\d+\w*$', re.IGNORECASE)
    texts_l0 = []

    for e in msp:
        if e.dxftype() not in ('TEXT', 'MTEXT') or e.dxf.layer != '0':
            continue

        txt_content = get_text_content(e)
        if borne_re.match(txt_content):
            continue

        txt_upper = txt_content.upper()
        if not any(word.upper() in txt_upper for word in center_texts):
            continue

        if add_lv and txt_upper in ['CHEMIN', 'SENTIER', 'PISTE']:
            e.dxf.text += " de L.V"

        # Apply new replacements for Route Nationale and Route Régionale
        if txt_upper.startswith('ROUTE NAT'):
            e.dxf.text = re.sub(r'(?i)^ROUTE NAT\w*', 'Route Nationale', e.dxf.text)
        elif txt_upper.startswith('ROUTE REG'):
            e.dxf.text = re.sub(r'(?i)^ROUTE REG\w*', 'Route Régionale', e.dxf.text)

        texts_l0.append(e)

    return texts_l0


def _get_candidate_lines(msp, skip_layers):
    """Extracts LINE and LWPOLYLINE geometries as LineString objects."""
    candidate_lines = []

    for e in msp:
        if e.dxftype() in ('LINE', 'LWPOLYLINE') and e.dxf.layer not in skip_layers:
            if e.dxftype() == 'LINE':
                p1 = (float(e.dxf.start.x), float(e.dxf.start.y))
                p2 = (float(e.dxf.end.x), float(e.dxf.end.y))
                candidate_lines.append((e, p1, p2, LineString([p1, p2])))

            elif e.dxftype() == 'LWPOLYLINE':
                pts = list(e.get_points(format='xy'))
                full_geom = LineString(pts)
                for i in range(len(pts) - 1):
                    candidate_lines.append((e, pts[i], pts[i + 1], full_geom))

    return candidate_lines


# -----------------------------------------------------------------------------
# 2. Geometry processing: Deduplication & Mapping
# -----------------------------------------------------------------------------

def _delete_overlapping_duplicate_lines(candidate_lines):
    """Finds and deletes shorter overlapping duplicate lines."""
    entity_geoms = {ent: geom for ent, _, _, geom in candidate_lines}
    ents = list(entity_geoms.keys())
    duplicate_ents_to_delete = set()

    for i in range(len(ents)):
        e1 = ents[i]
        if e1 in duplicate_ents_to_delete: continue
        geom1 = entity_geoms[e1]

        for j in range(i + 1, len(ents)):
            e2 = ents[j]
            if e2 in duplicate_ents_to_delete: continue
            geom2 = entity_geoms[e2]

            # Determine longer/shorter lines
            longer_e, shorter_e = (e1, e2) if geom1.length >= geom2.length else (e2, e1)
            longer_geom, shorter_geom = (geom1, geom2) if geom1.length >= geom2.length else (geom2, geom1)

            try:
                # Handle anomalous segments
                if shorter_geom.length < 0.001:
                    if shorter_geom.distance(longer_geom) < 0.02:
                        duplicate_ents_to_delete.add(shorter_e)
                        if shorter_e == e1: break
                else:
                    # Check for 95% overlap
                    intersection = shorter_geom.intersection(longer_geom.buffer(0.02))
                    if intersection.length >= shorter_geom.length * 0.95:
                        duplicate_ents_to_delete.add(shorter_e)
                        if shorter_e == e1: break
            except Exception:
                pass

    # Filter out duplicates and safely destroy entities
    filtered_lines = []
    deleted_count = 0
    for ent, p1, p2, full_geom in candidate_lines:
        if ent in duplicate_ents_to_delete:
            if ent.is_alive:
                ent.destroy()
                deleted_count += 1
        else:
            filtered_lines.append((ent, p1, p2, full_geom))

    return filtered_lines, deleted_count




def map_texts_to_best_lines(text_entities, candidate_lines, proximity_dist=5.0):
    """
    Maps multiple text entities to their closest lines.
    Ensures a line gets a maximum of 1 text (the closest one).
    """
    # Dictionary to track the best text for each line.
    # Key: line_ent (the unique entity 'e'), Value: (txt_ent, distance, full_line_tuple)
    line_assignments = {}

    for txt_ent in text_entities:
        tx, ty = get_insert(txt_ent)
        txt_point = Point(tx, ty)

        best_line_tuple = None
        min_dist = proximity_dist

        # 1. Find the closest line for the current text
        for line_tuple in candidate_lines:
            line_ent, p1, p2, full_geom = line_tuple
            segment = LineString([p1, p2])
            dist = segment.distance(txt_point)

            if dist < min_dist:
                min_dist = dist
                best_line_tuple = line_tuple

        # 2. Enforce the "1 text per line" and "closest wins" rules
        if best_line_tuple is not None:
            line_ent = best_line_tuple[0]  # Using the entity 'e' as the dictionary key

            if line_ent not in line_assignments:
                # No text assigned to this line yet
                line_assignments[line_ent] = (txt_ent, min_dist, best_line_tuple)
            else:
                # A text is already assigned; check if the current text is closer
                existing_txt, existing_dist, _ = line_assignments[line_ent]
                if min_dist < existing_dist:
                    line_assignments[line_ent] = (txt_ent, min_dist, best_line_tuple)

    # 3. Format the final output (e.g., return a list of matched pairs)
    final_matches = []
    for line_ent, data in line_assignments.items():
        matched_txt, matched_dist, matched_line_tuple = data
        final_matches.append({
            'text_entity': matched_txt,
            'line_data': matched_line_tuple,
            'distance': matched_dist
        })

    return final_matches

def _map_text_to_best_line(txt_ent, candidate_lines, proximity_dist=5.0):
    """Finds the closest line to a given text entity."""
    tx, ty = get_insert(txt_ent)
    best_match = None
    min_dist = proximity_dist

    for line_ent, p1, p2, full_geom in candidate_lines:
        segment = LineString([p1, p2])
        dist = segment.distance(Point(tx, ty))

        if dist < min_dist:
            min_dist = dist
            best_match = (line_ent, p1, p2, full_geom)

    return best_match


# -----------------------------------------------------------------------------
# 3. Alignment & Polygon Checks
# -----------------------------------------------------------------------------

def _center_text_on_line(doc, txt_ent, line_ent, p1, p2, full_geom, parcel):
    """
    Centers text on the middle segment of the line using perpendicular offsets
    and dynamically aligns the text rotation to match that segment.
    Returns `is_internal` boolean based on the override rule.
    """
    plot_scale = get_drawing_scale(doc.modelspace()) / 5000

    # 1. Update line styling
    txt_cnt = get_text_content(txt_ent).lower()
    if txt_cnt.startswith('chaaba') or txt_cnt.startswith('depression'):
        if 'AXES' not in doc.linetypes:
            doc.linetypes.new('AXES', dxfattribs={
                'description': 'Dot . . . . . . . . . . . . . . . .',
                'pattern': [1.25, -0.25, 0.25, -0.25]
            })
        line_ent.dxf.linetype = 'AXES'
        line_ent.dxf.ltscale = plot_scale * 10
    elif txt_cnt.startswith('sentier'):
        if 'DISCONTINU' not in doc.linetypes:
            print('create new one line type')
            doc.linetypes.new('DISCONTINU', dxfattribs={
                'description': 'Dashed __ __ __ __ __ __ __ __ __ __',
                'pattern': [5.0, -3.0, 5.0, -3.0]
            })
        line_ent.dxf.linetype = 'DISCONTINU'
        line_ent.dxf.ltscale = plot_scale

    # ---------------------------------------------------------
    # NEW: Find the MIDDLE segment of the polyline
    # ---------------------------------------------------------
    coords = list(full_geom.coords)
    num_segments = len(coords) - 1

    if num_segments > 0:
        mid_idx = num_segments // 2
        p1_mid = coords[mid_idx]
        p2_mid = coords[mid_idx + 1]
    else:
        # Fallback if it's somehow just a single point or invalid
        p1_mid, p2_mid = p1, p2

    segment = LineString([p1_mid, p2_mid])

    # ---------------------------------------------------------
    # 2. Intersection Detection (The Override Rule)
    # ---------------------------------------------------------
    is_internal = False
    if parcel is not None:
        if segment.crosses(parcel) or segment.within(parcel):
            is_internal = True

    # ---------------------------------------------------------
    # 3. Calculate Midpoint on the MIDDLE segment
    # ---------------------------------------------------------
    mid = segment.interpolate(0.5, normalized=True)
    mid_x, mid_y = mid.x, mid.y

    # ---------------------------------------------------------
    # 4. Calculate Angle Alignment based on MIDDLE segment
    # ---------------------------------------------------------
    dx = p2_mid[0] - p1_mid[0]
    dy = p2_mid[1] - p1_mid[1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Keep text readable (right-side up)
    if angle_deg > 90 or angle_deg <= -90:
        angle_deg -= 180

    try:
        th = txt_ent.dxf.char_height if txt_ent.dxftype() == 'MTEXT' else txt_ent.dxf.height
    except AttributeError:
        th = 1.0

    # ---------------------------------------------------------
    # 5. Perpendicular Offset calculation
    # ---------------------------------------------------------
    length = math.hypot(dx, dy) or 1.0
    perp_x, perp_y = -dy / length, dx / length

    # Pick the one pointing AWAY from parcel centroid
    if parcel is not None:
        cx, cy = parcel.centroid.x, parcel.centroid.y
        dot = perp_x * (mid_x - cx) + perp_y * (mid_y - cy)
        if dot < 0:
            perp_x, perp_y = -perp_x, -perp_y

    mid_x = mid_x + perp_x * (th * 2)
    mid_y = mid_y + perp_y * (th * 2)

    # ---------------------------------------------------------
    # 6. Apply Positioning and Rotation
    # ---------------------------------------------------------
    target_vec = Vec3(mid_x, mid_y, 0)

    if txt_ent.dxftype() == 'TEXT':
        txt_ent.dxf.halign = 4  # horizontal: middle
        txt_ent.dxf.valign = 2  # vertical: middle

        # Override both points to prevent CAD from resetting rotation to 0
        txt_ent.dxf.align_point = target_vec
        txt_ent.dxf.insert = target_vec

        # Apply the new dynamic angle
        txt_ent.dxf.rotation = angle_deg

    elif txt_ent.dxftype() == 'MTEXT':
        txt_ent.dxf.attachment_point = 5  # 5 = middle-center
        set_insert(txt_ent, mid_x, mid_y)

        if txt_ent.dxf.hasattr('text_direction'):
            rad = math.radians(angle_deg)
            txt_ent.dxf.text_direction = Vec3(math.cos(rad), math.sin(rad), 0)
        else:
            txt_ent.dxf.rotation = angle_deg

    return is_internal

def Z_center_text_on_line(doc, txt_ent, line_ent, p1, p2, full_geom, parcel):
    """
    Centers text on the line's midpoint using perpendicular offsets
    and dynamically aligns the text rotation to match the line.
    Returns `is_internal` boolean based on the override rule.
    """
    plot_scale = get_drawing_scale(doc.modelspace())/5000

    # lt_name = line_ent.dxf.get('linetype', 'BYLAYER')
    # lt_scale = line_ent.dxf.get('ltscale', 1.0)
    #
    # # 2. Resolve BYLAYER to the layer's default linetype
    # if lt_name.upper() == 'BYLAYER':
    #     try:
    #         layer = doc.layers.get('sentier')
    #         lt_name = layer.dxf.linetype
    #     except ezdxf.DXFTableEntryError:
    #         lt_name = 'CONTINUOUS'
    #
    # # 3. Read linetype definitions safely from the Linetypes table
    # description = "N/A"
    # pattern_elements = []
    #
    # try:
    #     ltype_obj = doc.linetypes.get(lt_name)
    #     description = ltype_obj.dxf.get('description', '')
    #
    #     # Target .pattern_tags.tags directly to extract pattern values (Group Code 49)
    #     if hasattr(ltype_obj, 'pattern_tags') and hasattr(ltype_obj.pattern_tags, 'tags'):
    #         pattern_elements = [
    #             tag.value for tag in ltype_obj.pattern_tags.tags
    #             if tag.code == 49
    #         ]
    # except ezdxf.DXFTableEntryError:
    #     pass
    #
    # # Display results
    # #print(f"\nEntity Type : {line_ent.dxftype()}")
    # print(f"Linetype    : {lt_name}  ,LT Scale    : {lt_scale} , Pattern     : {pattern_elements}", get_drawing_scale(doc.modelspace()))


    # 1. Update line styling
    txt_cnt = get_text_content(txt_ent).lower()
    if txt_cnt.startswith('chaaba') or txt_cnt.startswith('depression'):
        # print('CHAABA TPAPT :', line_ent.dxf.linetype)
        # Check if the linetypes exist in the document, and add them if they don't
        if 'AXES' not in doc.linetypes:
            doc.linetypes.new('AXES', dxfattribs={
                'description': 'Dot . . . . . . . . . . . . . . . .',
                'pattern': [1.25, -0.25, 0.25, -0.25]  # The total pattern length, followed by pen down (0=dot), pen up (gap)
            })
        line_ent.dxf.linetype = 'AXES'
        line_ent.dxf.ltscale = plot_scale*10
    elif txt_cnt.startswith('sentier'):
        # print('SENTIER TPAPT : ', line_ent.dxf.linetype)
        if 'DISCONTINU' not in doc.linetypes:
            print('create new one line type')
            doc.linetypes.new('DISCONTINU', dxfattribs={
                'description': 'Dashed __ __ __ __ __ __ __ __ __ __',
                'pattern': [5.0, -3.0, 5.0, -3.0]  # The total length, pen down (line), pen up (gap)
            })
        line_ent.dxf.linetype = 'DISCONTINU'
        line_ent.dxf.ltscale = plot_scale

    segment = LineString([p1, p2])

    # ---------------------------------------------------------
    # 2. Intersection Detection (The Override Rule)
    # ---------------------------------------------------------
    is_internal = False
    if parcel is not None:
        # Evaluates to True if the line cuts through the interior space
        if segment.crosses(parcel) or segment.within(parcel):
            is_internal = True

    # Calculate Midpoint
    mid = full_geom.interpolate(0.5, normalized=True)
    mid_x, mid_y = mid.x, mid.y

    # ---------------------------------------------------------
    # 3. Calculate Angle Alignment
    # ---------------------------------------------------------
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_deg = math.degrees(math.atan2(dy, dx))

    # Keep text readable (right-side up)
    if angle_deg > 90 or angle_deg <= -90:
        angle_deg -= 180

    try:
        th = txt_ent.dxf.char_height if txt_ent.dxftype() == 'MTEXT' else txt_ent.dxf.height
    except AttributeError:
        th = 1.0

    # ---------------------------------------------------------
    # 4. Perpendicular Offset calculation
    # ---------------------------------------------------------
    length = math.hypot(dx, dy) or 1.0
    perp_x, perp_y = -dy / length, dx / length

    # Pick the one pointing AWAY from parcel centroid
    if parcel is not None:
        cx, cy = parcel.centroid.x, parcel.centroid.y
        dot = perp_x * (mid_x - cx) + perp_y * (mid_y - cy)
        if dot < 0:
            perp_x, perp_y = -perp_x, -perp_y

    mid_x = mid_x + perp_x * (th * 2)
    mid_y = mid_y + perp_y * (th * 2)

    # ---------------------------------------------------------
    # 5. Apply Positioning and Rotation
    # ---------------------------------------------------------
    target_vec = Vec3(mid_x, mid_y, 0)

    if txt_ent.dxftype() == 'TEXT':
        txt_ent.dxf.halign = 4  # horizontal: middle
        txt_ent.dxf.valign = 2  # vertical: middle

        # Override both points to prevent CAD from resetting rotation to 0
        txt_ent.dxf.align_point = target_vec
        txt_ent.dxf.insert = target_vec

        # Apply the new dynamic angle
        txt_ent.dxf.rotation = angle_deg

    elif txt_ent.dxftype() == 'MTEXT':
        txt_ent.dxf.attachment_point = 5  # 5 = middle-center
        set_insert(txt_ent, mid_x, mid_y)

        # If MTEXT relies on text_direction vectors instead of rotation angles:
        if txt_ent.dxf.hasattr('text_direction'):
            rad = math.radians(angle_deg)
            txt_ent.dxf.text_direction = Vec3(math.cos(rad), math.sin(rad), 0)
        else:
            txt_ent.dxf.rotation = angle_deg

    return is_internal

def _is_line_outside_parcel(full_geom, parcel):
    """Checks if a geometry falls completely outside the parcel limit zone."""
    if parcel is None:
        return False

    boundary_limit = parcel.boundary.buffer(0.02)
    line_in_strict_interior = full_geom.intersection(parcel).difference(boundary_limit)

    return line_in_strict_interior.length < 0.05


# -----------------------------------------------------------------------------
# 4. Main Orchestrator Function
# -----------------------------------------------------------------------------
def centre_texts_on_lines(doc, msp, kept_lines=None, center_texts=None, add_lv=False, boundary_tolerance=0.05):
    """
    Main pipeline:
    1. Grabs eligible text and lines (excluding explicitly kept lines).
    2. Deletes duplicates.
    3. Maps text to lines and centers them (preserving rotation).
    4. Applies deletion logic based on parcel limits:
       - Outside parcel: deletes line and its paired text.
       - Passes by parcel limit (within tolerance): deletes line, keeps text.
       - Inside: keeps both.
    """
    skip_layers = {
        '0', 'annotation', 'parcelle', 'plle', 'bornes', 'CADRE',
        'CADREP1', 'HABILLAGE', 'T-LEGENDE', 'legende', 'titres', 'Defpoints'
    }

    # Step 1: Collect texts, candidate lines, and the parcel boundary
    texts_l0 = _get_target_texts(msp, center_texts, add_lv)
    candidate_lines = _get_candidate_lines(msp, skip_layers)

    # Exclude kept_lines from processing
    if kept_lines:
        kept_handles = set()
        for line in kept_lines:
            try:
                if getattr(line, 'is_alive', True) and hasattr(line, 'dxf'):
                    kept_handles.add(line.dxf.handle)
            except Exception:
                pass
        candidate_lines = [
            item for item in candidate_lines
            if item[0].dxf.handle not in kept_handles
        ]
    parcel = get_parcel_polygon(msp)

    # Step 2: Delete overlapping duplicate lines
    candidate_lines, deleted_dupes = _delete_overlapping_duplicate_lines(candidate_lines)
    if deleted_dupes > 0:
        log.info(f"Cleaned up {deleted_dupes} duplicate (shorter) lines.")

    # Step 3 & 4: Process mappings, alignments, and apply deletion logic
    count = 0
    lines_to_delete = set()
    texts_to_delete = set()  # NEW: Track texts that need to be deleted

    # Retrieve proximity distance safely
    proximity = globals().get('PROXIMITY_DIST', 5.0)

    matched_pairs = map_texts_to_best_lines(texts_l0, candidate_lines, proximity)

    # 2. Process only the final, resolved matches
    for match in matched_pairs:
        txt_ent = match['text_entity']
        line_ent, p1, p2, full_geom = match['line_data']

        # Center text on line and capture internal flag
        is_internal = _center_text_on_line(doc, txt_ent, line_ent, p1, p2, full_geom, parcel)
        count += 1

        if parcel is not None:
            # Calculate how close the line is to the parcel limit (boundary)
            dist_to_boundary = full_geom.distance(parcel.boundary)

            # Condition 1: Passes by parcel limit within a tolerance
            if dist_to_boundary <= boundary_tolerance and not is_internal:
                lines_to_delete.add(line_ent)
                # Text is kept (not added to texts_to_delete)

            # Condition 2: Entirely outside the parcel and NOT near the limit
            elif not is_internal and _is_line_outside_parcel(full_geom, parcel):
                lines_to_delete.add(line_ent)
                texts_to_delete.add(txt_ent)

            # Condition 3: Inside
            else:
                pass  # Keep both (do nothing)

    # Step 5: Final Cleanup
    deleted_lines = 0
    for ent in lines_to_delete:
        if ent.is_alive:
            ent.destroy()
            deleted_lines += 1

    deleted_texts = 0
    for ent in texts_to_delete:
        if ent.is_alive:
            ent.destroy()
            deleted_texts += 1

    log.info(f"Centered {count} texts. Deleted {deleted_lines} lines and {deleted_texts} texts based on parcel limits.")

def Xcentre_texts_on_lines(doc, msp, kept_lines=None, center_texts=None, add_lv=False):
    """
    Main pipeline:
    1. Grabs eligible text and lines (excluding explicitly kept lines).
    2. Deletes duplicates.
    3. Maps text to lines and centers them (preserving rotation).
    4. Deletes lines that fall outside the parcel limit.
    """
    skip_layers = {
        'annotation', 'parcelle', 'plle', 'bornes', 'CADRE',
        'CADREP1', 'HABILLAGE', 'T-LEGENDE', 'legende', 'titres', 'Defpoints'
    }

    # Step 1: Collect texts, candidate lines, and the parcel boundary
    texts_l0 = _get_target_texts(msp, center_texts, add_lv)
    candidate_lines = _get_candidate_lines(msp, skip_layers)

    # --- NEW: Exclude kept_lines from processing ---
    if kept_lines:
        candidate_lines = [line for line in candidate_lines if line not in kept_lines]

    parcel = get_parcel_polygon(msp)

    # Step 2: Delete overlapping duplicate lines
    candidate_lines, deleted_dupes = _delete_overlapping_duplicate_lines(candidate_lines)
    if deleted_dupes > 0:
        log.info(f"Cleaned up {deleted_dupes} duplicate (shorter) lines.")

    # Step 3 & 4: Process mappings, alignments, and flag outside lines
    count = 0
    lines_to_delete = set()

    # Retrieve proximity distance safely
    proximity = globals().get('PROXIMITY_DIST', 5.0)

    matched_pairs = map_texts_to_best_lines(texts_l0, candidate_lines, proximity)

    # 2. Process only the final, resolved matches
    for match in matched_pairs:
        txt_ent = match['text_entity']
        line_ent, p1, p2, full_geom = match['line_data']

        # Center text on line and capture internal flag
        is_internal = _center_text_on_line(doc, txt_ent, line_ent, p1, p2, full_geom, parcel)
        count += 1

        # Check boundary conditions, keeping the internal override in mind
        if not is_internal and _is_line_outside_parcel(full_geom, parcel):
            lines_to_delete.add(line_ent)

    # Step 5: Final Cleanup
    deleted_lines = 0
    for ent in lines_to_delete:
        if ent.is_alive:
            ent.destroy()
            deleted_lines += 1

    log.info(f"Centered {count} texts. Deleted {deleted_lines} target lines (outside or overlapping).")

# 9.  Delete cross (+) symbols inside parcel
def delete_crosses_inside_parcel(msp, parcel: Polygon):
    """
    Detects and deletes '+' cross symbols inside the parcel.
    Cross arm half-length = 2 m at paper scale → real-world arm length
    is derived from 'Echelle : 1/NNN' text in the drawing.
    Two perpendicular short lines sharing a midpoint = cross.
    """
    if parcel is None:
        return

    scale      = get_drawing_scale(msp)
    # 2 m at 1:1 in paper → real-world distance = 2 * scale  (mm → m depends
    # on drawing units; ANCFCC files are in metres, so 2 mm paper = 2/1000 * scale metres)
    arm_length = 2.0 * scale / 1000.0   # e.g. 1/500 → 1.0 m arm

    log.info("Cross detection: max arm half-length = %.3f units (scale 1/%g)",
             arm_length, scale)

    def midpoint_len_ang(line):
        s, e_ = line.dxf.start, line.dxf.end
        mx    = (s.x + e_.x) / 2
        my    = (s.y + e_.y) / 2
        l     = math.hypot(e_.x - s.x, e_.y - s.y)
        ang   = math.degrees(math.atan2(e_.y - s.y, e_.x - s.x)) % 180
        return mx, my, l, ang

    lines = [e for e in msp.query('LINE')
             if e.dxf.layer not in ('parcelle', 'plle')]

    from collections import defaultdict
    groups = defaultdict(list)
    for ln in lines:
        mx, my, l, ang = midpoint_len_ang(ln)
        if l > arm_length * 2.5:   # allow some tolerance
            continue
        key = (round(mx, 2), round(my, 2))
        groups[key].append((ln, mx, my, l, ang))

    deleted = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        angles = [m[4] for m in members]
        is_cross = any(
            80 < abs(angles[i] - angles[j]) % 180 < 100
            for i in range(len(angles))
            for j in range(i + 1, len(angles))
        )
        if not is_cross:
            continue
        mx, my = members[0][1], members[0][2]
        if parcel.contains(Point(mx, my)):
            for m in members:
                m[0].destroy()
            deleted += len(members)

    log.info("Deleted %d cross-component lines inside parcel.", deleted)

# 10.  Position verborne texts
def draw_debug_polygon(msp, poly, layer_name="DEBUG_BBOX", color=3):
    """
    Draws a Shapely Polygon into the DXF modelspace for visual debugging.
    """
    if poly is None or poly.is_empty:
        return

    # Ensure the layer exists
    if layer_name not in msp.doc.layers:
        msp.doc.layers.add(name=layer_name, color=color)

    # Draw the exterior coordinates as a closed polyline
    msp.add_lwpolyline(
        list(poly.exterior.coords),
        close=True,
        dxfattribs={"layer": layer_name, "color": color}
    )

def _extract_text_and_lines(msp):
    """Gathers Verborne texts, Plle labels, and layer 0 lines from modelspace."""
    vb_texts = [e for e in msp if e.dxftype() in ('TEXT', 'MTEXT') and VERBORNE_RE.match(get_text_content(e))]

    plle_items = []
    for e in msp:
        if e.dxftype() in ('TEXT', 'MTEXT') and NEIGHBOR_LABEL_RE.match(get_text_content(e)):
            px, py = get_insert(e)
            plle_items.append((e, Point(px, py)))

    lines_l0 = []
    for e in msp.query('LINE[layer=="0"]'):
        s, en = e.dxf.start, e.dxf.end
        lines_l0.append((e, (float(s.x), float(s.y)), (float(en.x), float(en.y))))

    return vb_texts, plle_items, lines_l0


def _pair_verborne_with_lines(msp, vb_texts, lines_l0, parcel):
    """Finds the closest line to each Verborne text to use as a spoke."""
    vb_with_lines = []
    treated_lines = []

    for e in vb_texts:
        tx, ty = get_insert(e)
        best, bd = None, PROXIMITY_DIST * 2
        for le, p1, p2 in lines_l0:
            for far, anchor in ((p1, p2), (p2, p1)):
                d = math.hypot(tx - far[0], ty - far[1])
                if d < bd:
                    bd = d
                    best = (le, anchor, far)

        if best:
            le, anchor, far = best

            # If line lies on boundary, do not use as radial spoke but preserve entities
            if _line_lies_on_parcel_boundary(anchor, far, parcel):
                continue

            treated_lines.append(le)
            vb_length = math.hypot(far[0] - anchor[0], far[1] - anchor[1])
            vb_with_lines.append((e, anchor, far, vb_length))

    vb_anchor_by_entity = {e: anchor for e, anchor, far, vb_length in vb_with_lines}
    return vb_with_lines, vb_anchor_by_entity, treated_lines


def _generate_polygon_slices(parcel, frame, vb_with_lines):
    """Builds the bounding wedges for the Verborne text topology."""
    if frame and frame.geom_type == 'Polygon':
        buffered_geom = frame
    else:
        log.warning("No frame polygon found. Falling back to 3x length buffer.")
        max_len = max((item[3] for item in vb_with_lines)) if vb_with_lines else 3.0
        buffered_geom = parcel.buffer(max_len * 3.0, join_style=2)

    return build_polygon_slices(parcel, buffered_geom, vb_with_lines, verbose=False)


def _analyze_slices(slices, plle_items, parcel):
    """Maps Plle labels to their enclosing slices and checks for boundary segments."""
    slice_data = []
    for s_i_ent, s_next_ent, poly in slices:
        plles_in_slice = [pe for pe, pt in plle_items if poly.contains(pt)]
        parcel_segment = _get_parcel_segment_in_slice(poly, parcel)
        has_parcel_segment = parcel_segment is not None and parcel_segment.length > 0

        slice_data.append({
            'ent1': s_i_ent,
            'ent2': s_next_ent,
            'poly': poly,
            'has_plle': len(plles_in_slice) > 0,
            'plles': plles_in_slice,
            'parcel_segment': parcel_segment,
            'has_parcel_segment': has_parcel_segment,
        })

    no_segment_slices = [sd for sd in slice_data if not sd['has_parcel_segment']]
    if no_segment_slices:
        log.warning("%d slice(s) have no parcel boundary segment.", len(no_segment_slices))

    return slice_data

def _fit_plle_labels(slice_data, parcel, vb_anchor_by_entity, msp, espacement):
    """Ensures Plle labels fit inside their assigned slices."""
    for sd in slice_data:
        if not sd['plles']:
            continue

        # FIX 1: Force all Plle labels to 0 rotation for grid search logic
        # for pe in sd['plles']:
        #     pe.dxf.rotation = 0

        poly = sd['poly']
        poly_prepared = prep(poly)
        eroded = poly.buffer(-1e-4)
        eroded_prepared = prep(eroded if not eroded.is_empty else poly)

        for pe in sd['plles']:

            if sd['has_parcel_segment'] and sd['parcel_segment'] is not None and sd['parcel_segment'].length > 0:
                segment = sd['parcel_segment']
                text_height = getattr(pe.dxf, 'height', getattr(pe.dxf, 'char_height', 1.0))

                # Set your offset here (e.g., 5.0 for 5x text height, or 1.0)
                _spc = espacement if isinstance(espacement, (int, float)) else 3.0
                shift_distance = float(_spc) * text_height

                ideal_pt = _find_bbox_position_aligned_to_segment(
                    poly, parcel, parcel_segment=segment, offset=shift_distance
                )

                # --- CALCULATE ROTATION ---
                p1 = segment.coords[0]
                p2 = segment.coords[-1]
                dx_line = p2[0] - p1[0]
                dy_line = p2[1] - p1[1]
                angle_deg = math.degrees(math.atan2(dy_line, dx_line)) % 360

                # Keep text readable (right-side up)
                if 90 < angle_deg <= 270:
                    angle_deg -= 180

                # --- APPLY ROTATION ---
                pe.dxf.rotation = angle_deg

                # --- FORCE EXACT PLACEMENT BYPASSING BOUNDING BOXES ---
                # This guarantees the geometric center of the text sits perfectly on ideal_pt
                if pe.dxftype() == 'TEXT':
                    pe.dxf.halign = 1  # 1 = Center
                    pe.dxf.valign = 2  # 2 = Middle
                    pe.dxf.align_point = (ideal_pt.x, ideal_pt.y, 0)
                    pe.dxf.insert = (ideal_pt.x, ideal_pt.y, 0)
                elif pe.dxftype() == 'MTEXT':
                    pe.dxf.attachment_point = 5  # 5 = Middle Center
                    pe.dxf.insert = (ideal_pt.x, ideal_pt.y, 0)

            else:
                # --- FALLBACK FOR LABELS WITH NO SEGMENT ---
                # (This preserves your original script's logic for corner labels)
                text_bbox_geom = _get_text_real_bbox(pe)
                if text_bbox_geom is None:
                    continue
                minx, miny, maxx, maxy = text_bbox_geom.bounds

                anchors = [vb_anchor_by_entity[ent] for ent in (sd['ent1'], sd['ent2']) if ent in vb_anchor_by_entity]
                if anchors:
                    cx_a, cy_a = sum(a[0] for a in anchors) / len(anchors), sum(a[1] for a in anchors) / len(anchors)
                    ideal_pt = _find_bbox_position_near_corner(
                        poly, Point(cx_a, cy_a), maxx - minx, maxy - miny,
                        min_gap=0.15, coarse_step=0.2, fine_step=0.02, eroded_prepared=eroded_prepared
                    )
                else:
                    ideal_pt = _find_ideal_interior_point(poly)

                # Move using original logic for non-segment labels
                ix, iy = get_insert(pe)
                center_x = (minx + maxx) / 2.0
                center_y = (miny + maxy) / 2.0
                set_insert(pe, ideal_pt.x + (ix - center_x), ideal_pt.y + (iy - center_y))

            # Invalidate cache at the very end
            _bbox_cache.invalidate([pe])


def offset_tc_texts_in_msp(msp, parcel, espacement):
    """
    Scans the entire modelspace for TEXT and MTEXT entities containing 'TC'.
    Pushes those specific texts away from the parcel by the offset value.
    """

    # Query all TEXT and MTEXT entities in the modelspace
    for pe in msp.query('TEXT MTEXT'):
        text_height = getattr(pe.dxf, 'height', getattr(pe.dxf, 'char_height', 1.0))
        text_offset = espacement * text_height
        # Safely extract the text string from the entity
        text_string = getattr(pe, 'text', '')
        if not text_string and hasattr(pe, 'dxf'):
            text_string = getattr(pe.dxf, 'text', '')

        # Check if the text has "TC" in it
        # Note: If you meant exact match, use `text_string == 'TC'`
        # Note: If you meant a layer named TC, use `pe.dxf.layer == 'TC'`
        if 'TC' in text_string and not NEIGHBOR_LABEL_RE.match(text_string):
            try:
                insert_pt = pe.dxf.insert
                pt = Point(insert_pt.x, insert_pt.y)
            except AttributeError:
                # Skip if the entity lacks an insertion point for some reason
                continue

            # Find the closest point on the parcel boundary
            nearest_geom_on_parcel, _ = nearest_points(parcel, pt)

            # Calculate the vector pointing AWAY from the parcel
            dx = pt.x - nearest_geom_on_parcel.x
            dy = pt.y - nearest_geom_on_parcel.y
            dist = math.hypot(dx, dy)

            # Apply the offset if distance is greater than zero
            if dist > 1e-6:
                move_x = (dx / dist) * text_offset
                move_y = (dy / dist) * text_offset

                # Update the CAD entity's coordinates directly
                z_val = insert_pt.z if hasattr(insert_pt, 'z') else 0.0
                pe.dxf.insert = (pt.x + move_x, pt.y + move_y, z_val)



def _align_and_push_verborne_labels(vb_with_lines, slice_data):
    """Aligns Verborne texts with their lines and pushes them outward to avoid collision."""
    count = 0
    for e, anchor, far, vb_length in vb_with_lines:
        ax, ay = anchor
        fx, fy = far
        dx, dy = fx - ax, fy - ay
        dist_line = vb_length or 1.0
        vx_out, vy_out = dx / dist_line, dy / dist_line

        # Calculate text reading angle
        angle_deg = math.degrees(math.atan2(dy, dx)) % 360
        if 90 < angle_deg <= 270:
            angle_deg = (angle_deg - 180) % 360

        # FIX 2: Actually assign the rotation to the entity
        e.dxf.rotation = angle_deg

        rad_read = math.radians(angle_deg)
        nx, ny = -math.sin(rad_read), math.cos(rad_read)
        th = getattr(e.dxf, 'char_height', getattr(e.dxf, 'height', 1.0))

        # Determine side offsets
        touching_slices = [sd for sd in slice_data if sd['ent1'] == e or sd['ent2'] == e]
        circle_positions = []
        for sd in touching_slices:
            if sd['has_plle']:
                dot = (sd['poly'].centroid.x - fx) * nx + (sd['poly'].centroid.y - fy) * ny
                side = 1 if dot >= 0 else -1
                circle_positions.append((fx + nx * (th * 1) * side, fy + ny * (th * 1) * side))

        # Setup standard DXF alignments
        if e.dxftype() == 'TEXT':
            e.dxf.halign, e.dxf.valign = 1, 2
            e.dxf.align_point = Vec3(fx, fy, 0)
        elif e.dxftype() == 'MTEXT':
            e.dxf.attachment_point = 4
            set_insert(e, fx, fy)

        if len(circle_positions) == 1:
            set_insert(e, circle_positions[0][0], circle_positions[0][1])
        else:
            set_insert(e, fx, fy)

        # FIX 3: Use the tight rotated polygon for accurate collision pushing
        verborne_line = LineString([(ax, ay), (fx, fy)])
        for _ in range(50):
            text_bbox_geom = get_rotated_text_bbox_polygon(e, _bbox_cache)
            if text_bbox_geom is None or not text_bbox_geom.intersects(verborne_line):
                break

            ex, ey = get_insert(e)
            set_insert(e, ex + vx_out * 0.1, ey + vy_out * 0.1)

        _bbox_cache.invalidate([e])
        count += 1

    log.info("Successfully arranged %d verborne labels using slice topology.", count)


def position_verborne_texts(msp, parcel: Optional[Polygon], espacement):
    """Orchestrates the topological positioning of v.Bxxxx and Plle texts."""
    if parcel is None:
        return

    # 1. Extract raw data from DXF
    frame = get_frame_polygon(msp)
    vb_texts, plle_items, lines_l0 = _extract_text_and_lines(msp)

    # 2. Pair Verborne texts with their geometric spoke lines
    vb_with_lines, vb_anchor_by_entity, treated_lines = _pair_verborne_with_lines(msp, vb_texts, lines_l0, parcel)
    if not vb_with_lines:
        return None, treated_lines

    # 3. Generate the slice geometry
    slices = _generate_polygon_slices(parcel, frame, vb_with_lines)

    # 4. Map entities to their respective slices
    slice_data = _analyze_slices(slices, plle_items, parcel)
    # 5. Position secondary labels (Plle) inside the slices
    _fit_plle_labels(slice_data, parcel, vb_anchor_by_entity, msp, espacement)
    # _enforce_plle_segment_offset(slice_data, parcel, msp)
    offset_tc_texts_in_msp(msp,parcel, espacement)
    # 6. Align and push Verborne labels outward along the spokes
    _align_and_push_verborne_labels(vb_with_lines, slice_data)
    maxlen = max((item[3] for item in vb_with_lines))
    return maxlen, treated_lines

# 11.  Clip / move out-of-frame elements
def clip_elements_to_frame(msp, frame: Optional[Polygon]):
    """
    For entities extending outside the frame:
    - LINE: clip to frame boundary using Shapely intersection
    - TEXT/MTEXT: move insertion point just inside the frame
    Other entity types are left alone (resizing is complex and risky).
    """
    if frame is None:
        return

    # Deflate frame very slightly to give a margin
    # ── Inside clip_elements_to_frame ────────────────────────────────────────────
    inner = frame.buffer(-0.05)
    scale = get_drawing_scale(msp) or 2000.0
    line_margin = max(8.0, 0.004 * scale)
    inner_lines = frame.buffer(-line_margin)
    if inner_lines.geom_type == 'MultiPolygon':
        inner_lines = max(inner_lines.geoms, key=lambda g: g.area)

    clipped_lines  = 0
    moved_texts    = 0

    to_delete = []
    to_add    = []   # (layer, color, start, end)

    for e in list(msp):
        try:
            if e.dxftype() == 'LINE':
                s, en = e.dxf.start, e.dxf.end
                seg   = LineString([(s.x, s.y), (en.x, en.y)])
                lay_u = getattr(e.dxf, 'layer', '').upper()
                target_inner = inner_lines if lay_u in ('DTLS', 'L.E', 'CHAABA', 'OUED', 'PISTE', 'ROUTE', 'PLLE') else inner
                if not frame.contains(seg) or not target_inner.contains(seg):
                    clipped = seg.intersection(target_inner)
                    if clipped.is_empty:
                        # Ne pas supprimer les amorces de carroyage sur le bord du cadre (grid tick marks)
                        if seg.length <= 5.0 and frame.exterior.distance(seg) < 0.2:
                            continue
                        to_delete.append(e)
                    elif clipped.geom_type == 'LineString':
                        coords = list(clipped.coords)
                        e.dxf.start = Vec3(coords[0][0], coords[0][1], 0)
                        e.dxf.end   = Vec3(coords[-1][0], coords[-1][1], 0)
                        clipped_lines += 1
                    else:
                        # MultiLineString – keep longest segment
                        geoms = (list(clipped.geoms)
                                 if hasattr(clipped, 'geoms') else [clipped])
                        longest = max(geoms, key=lambda g: g.length)
                        coords  = list(longest.coords)
                        e.dxf.start = Vec3(coords[0][0], coords[0][1], 0)
                        e.dxf.end   = Vec3(coords[-1][0], coords[-1][1], 0)
                        clipped_lines += 1
            
            elif e.dxftype() in ('TEXT', 'MTEXT'):
                if is_outside_frame(e, frame) or _is_frame_label(e):
                    continue
                x, y = get_insert(e)
                pt   = Point(x, y)
                if not inner.contains(pt):
                    nearest = inner.exterior.interpolate(
                        inner.exterior.project(pt))
                    cx, cy = inner.centroid.x, inner.centroid.y
                    ux     = cx - nearest.x
                    uy     = cy - nearest.y
                    l      = math.hypot(ux, uy) or 1
                    set_insert(e, nearest.x + ux / l * 0.1,
                                  nearest.y + uy / l * 0.1)
                    moved_texts += 1
        except Exception as ex:
            log.debug("clip_elements_to_frame skipped entity: %s", ex)

    for e in to_delete:
        e.destroy()

    log.info("Clipped %d lines; moved %d texts into frame.",
             clipped_lines, moved_texts)

# 12.  rotate text inside parcel
def reset_internal_text_rotation(msp, parcel: Polygon, center_texts=None):
    """
    Finds specific texts (defined in center_texts) and forces their rotation
    to 0 degrees if they are located inside the parcel polygon.
    """
    count = 0
    for e in msp:
        # 1. Filter for text entities
        if e.dxftype() in ('TEXT', 'MTEXT'):
            center_texts = center_texts or []
            # 2. Check if text matches the target words
            txt_upper = get_text_content(e).upper()
            if not any(word.upper() in txt_upper for word in center_texts):
                # 3. Get the coordinates of the text
                tx, ty = get_insert(e)
                text_point = Point(tx, ty)

                if parcel is not None and parcel.contains(text_point):
                    # 5. Rotate text to 0
                    e.dxf.rotation = 0.0
                    count += 1
    return count

# 13.  delete neighbors
def filter_dtls_lines(msp, parcel_polygon, maxlen, target_layer="dtls", overlap_layers=["0", "parcelle"],
                      touch_tolerance=0.01,
                      overlap_tolerance=0.05, gap_tolerance=0.05):
    """
    1. Pre-Check: If a LWPOLYLINE has the majority (>50%) of its length inside the parcel, KEEP it intact.
    2. Otherwise, splits LWPOLYLINEs into individual LINE segments.
    3. Keeps individual lines if they touch a vertex, intersect, OR are very close to the parcel (handling gaps).
    4. Resizes the kept lines (doubling if maxlen is None, else using maxlen), anchored correctly to the parcel.
    5. Deletes lines that are entirely outside or overlap with layer 0/parcelle boundaries.
    6. Returns a list of all DXF entities (LINEs and intact LWPOLYLINEs) that were kept.
    """
    if parcel_polygon is None:
        return []

    parcel_vertices = list(parcel_polygon.exterior.coords)

    # --- 1. PREPARE THE OVERLAP ZONES ---
    overlap_lines = []
    for e in msp:
        if e.dxf.layer.lower() in [l.lower() for l in overlap_layers]:
            if e.dxftype() == 'LINE':
                overlap_lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = e.get_points(format='xy')
                if len(pts) >= 2:
                    if e.closed:
                        pts.append(pts[0])
                    overlap_lines.append(LineString(pts))

    overlap_geom = None
    if overlap_lines:
        overlap_geom = MultiLineString(overlap_lines).buffer(overlap_tolerance)

    # --- 2. EVALUATE POLYLINES & SPLIT ---
    candidate_entities = [e for e in msp if e.dxf.layer.lower() == target_layer.lower()]
    lines_to_evaluate = []
    kept_lines = []

    for e in candidate_entities:
        if e.dxftype() == 'LWPOLYLINE':
            pts = list(e.get_points(format='xy'))
            if len(pts) < 2:
                e.destroy()
                continue

            if e.closed:
                # Polylignes fermées sur DTLS: si ce n'est pas un bâtiment intérieur, détruire pour éviter des faux segments étirés au cadre
                pts_ring = pts if pts[0] == pts[-1] else pts + [pts[0]]
                is_interior_struct = False
                if len(pts_ring) >= 4:
                    try:
                        p_poly = Polygon(pts_ring)
                        if p_poly.is_valid and (p_poly.intersection(parcel_polygon).area / (p_poly.area or 1.0)) > 0.5:
                            is_interior_struct = True
                    except Exception:
                        pass
                if is_interior_struct:
                    kept_lines.append(e)
                    continue
                else:
                    e.destroy()
                    continue

            shapely_poly = LineString(pts)

            # --- THE NEW PRE-CHECK ---
            if shapely_poly.length > 0:
                intersection = shapely_poly.intersection(parcel_polygon)
                if (intersection.length / shapely_poly.length) > 0.5:
                    kept_lines.append(e)
                    continue  # Keep the whole polyline intact

            # Split mostly-outside polylines into segments
            for i in range(len(pts) - 1):
                p1, p2 = pts[i], pts[i + 1]
                new_line = msp.add_line(start=p1, end=p2, dxfattribs={'layer': e.dxf.layer})
                lines_to_evaluate.append(new_line)

            e.destroy()

        elif e.dxftype() == 'LINE':
            lines_to_evaluate.append(e)
        else:
            e.destroy()

    # --- 3. EVALUATE INDIVIDUAL LINES ---
    for line in lines_to_evaluate:
        p_start = (line.dxf.start.x, line.dxf.start.y)
        p_end = (line.dxf.end.x, line.dxf.end.y)
        shapely_line = LineString([p_start, p_end])

        if shapely_line.length == 0:
            line.destroy()
            continue

        # A) Check if it touches a parcel vertex
        touches_vertex = False
        for vx, vy in parcel_vertices:
            if math.hypot(p_start[0] - vx, p_start[1] - vy) <= touch_tolerance or \
                    math.hypot(p_end[0] - vx, p_end[1] - vy) <= touch_tolerance:
                touches_vertex = True
                break

        # B) Check if it intersects OR is very close to the parcel (handles invisible gaps)
        is_close_to_parcel = shapely_line.distance(parcel_polygon) <= gap_tolerance

        # C) Check if it overlaps with layer 0 or parcelle boundaries
        is_overlapping = False
        if overlap_geom:
            intersection = shapely_line.intersection(overlap_geom)
            overlap_ratio = intersection.length / shapely_line.length
            if overlap_ratio > 0.5:
                is_overlapping = True

        # FINAL DECISION FOR INDIVIDUAL LINES
        if (touches_vertex or is_close_to_parcel) and not is_overlapping:

            current_len = shapely_line.length
            if current_len > 0:
                # If maxlen is None, double the line's current length
                target_len = current_len * 2 if maxlen is None else maxlen
                ratio = target_len / current_len

                dist_start = Point(p_start).distance(parcel_polygon)
                dist_end = Point(p_end).distance(parcel_polygon)

                if dist_end < dist_start:
                    # Anchor at p_end, move p_start
                    dx = p_start[0] - p_end[0]
                    dy = p_start[1] - p_end[1]
                    line.dxf.start = (
                        p_end[0] + (dx * ratio),
                        p_end[1] + (dy * ratio),
                        line.dxf.start.z if hasattr(line.dxf.start, 'z') else 0.0
                    )
                else:
                    # Anchor at p_start, move p_end
                    dx = p_end[0] - p_start[0]
                    dy = p_end[1] - p_start[1]
                    line.dxf.end = (
                        p_start[0] + (dx * ratio),
                        p_start[1] + (dy * ratio),
                        line.dxf.end.z if hasattr(line.dxf.end, 'z') else 0.0
                    )

            kept_lines.append(line)
        else:
            line.destroy()

    return kept_lines


#expiremental----------------------------------

def draw_debug_extended_verborne(msp, vb_with_lines, debug_layer="DEBUG_EXTENDED_VB", debug_color=2):
    """
    Draws extended debug lines pointing outward from the parcel.

    The line starts at 'anchor', passes through 'far', and is extended
    to exactly 3x the original verborne line length.

    debug_color = 2 (Yellow in AutoCAD)
    """
    if not vb_with_lines:
        return 0

    # 1. Ensure the debug layer exists in the document
    if debug_layer not in msp.doc.layers:
        msp.doc.layers.add(debug_layer, color=debug_color)

    count = 0
    for item in vb_with_lines:
        # 2. Dynamically unpack either (text, anchor, far) or (text, anchor, far, length)
        if len(item) == 4:
            _, anchor, far, vb_length = item
        else:
            _, anchor, far = item
            vb_length = None

        # Extract coordinates safely (works for tuples, lists, and ezdxf Vec2/Vec3)
        ax, ay = anchor[0], anchor[1]
        fx, fy = far[0], far[1]

        # Calculate original vector
        dx = fx - ax
        dy = fy - ay

        # Calculate/validate length
        if vb_length is None:
            vb_length = math.hypot(dx, dy)

        if vb_length == 0:
            continue  # Skip zero-length degenerate lines

        # 3. Normalize the vector to find the direction pointing away from the anchor
        ux = dx / vb_length
        uy = dy / vb_length

        # 4. Project the new end coordinates to exactly 3x the original length
        extended_length = vb_length * 3.0
        extended_end_x = ax + ux * extended_length
        extended_end_y = ay + uy * extended_length

        # 5. Draw the extended debug line in the DXF space
        msp.add_line(
            start=(ax, ay),
            end=(extended_end_x, extended_end_y),
            dxfattribs={
                'layer': debug_layer,
                'color': debug_color
            }
        )
        count += 1

    return count

#expiremental widges

def draw_debug_parcel_buffer(msp, parcel_polygon, verborne_length=3.0, debug_layer="DEBUG_BUFFER", debug_color=3) -> object:
    """
    Creates a visible buffer around the parcel polygon.
    The buffer distance is 3 times the verborne_length.

    debug_color=3 is Green in AutoCAD.
    """
    if parcel_polygon is None:
        return

    # 1. Calculate the distance (3x verborne line length)
    buffer_distance = verborne_length * 3.0

    # 2. Create the buffered polygon
    # join_style=2 (Mitre) creates sharp corners, mimicking CAD's OFFSET command.
    # join_style=1 (Round) is Shapely's default, which creates arc-like rounded corners.
    buffered_geom = parcel_polygon.buffer(buffer_distance, join_style=2)

    # 3. Ensure the debug layer exists
    if debug_layer not in msp.doc.layers:
        msp.doc.layers.add(debug_layer, color=debug_color)

    # 4. Helper function to draw the exterior of a polygon
    def draw_polygon_exterior(poly):
        pts = list(poly.exterior.coords)
        msp.add_lwpolyline(
            pts,
            format='xy',
            dxfattribs={
                'layer': debug_layer,
                'closed': True,  # Ensures the last point connects to the first
                'color': debug_color
            }
        )

    # 5. Draw it (Handling standard Polygons or MultiPolygons if geometry gets complex)
    if buffered_geom.geom_type == 'Polygon':
        draw_polygon_exterior(buffered_geom)
    elif buffered_geom.geom_type == 'MultiPolygon':
        # Just in case the shape is so concave that offsetting it splits it into pieces
        for poly in buffered_geom.geoms:
            draw_polygon_exterior(poly)

    return buffered_geom

def compute_extended_lines(vb_with_lines, extension_factor=3.0):
    """
    Shared geometry logic (same math as draw_debug_extended_verborne),
    but returns Shapely LineStrings instead of drawing DXF entities.

    Returns: list of (text, LineString) tuples so callers can still
    trace a cut segment back to its source verborne label if needed.
    """
    results = []
    for item in vb_with_lines:
        if len(item) == 4:
            text, anchor, far, vb_length = item
        else:
            text, anchor, far = item
            vb_length = None

        ax, ay = anchor[0], anchor[1]
        fx, fy = far[0], far[1]
        dx, dy = fx - ax, fy - ay

        if vb_length is None:
            vb_length = math.hypot(dx, dy)
        if vb_length == 0:
            continue

        ux, uy = dx / vb_length, dy / vb_length
        extended_length = vb_length * extension_factor
        ex = ax + ux * extended_length
        ey = ay + uy * extended_length

        results.append((text, LineString([(ax, ay), (ex, ey)])))

    return results

def _line_lies_on_parcel_boundary(anchor, far, parcel: Polygon,
                                   tolerance: float = 0.05,
                                   coverage_ratio: float = 0.95) -> bool:
    """
    Returns True if the anchor->far segment runs almost entirely along the
    parcel's boundary, meaning it's a parcel edge itself rather than a
    verborne spoke/indicator line pointing away from the parcel. Checked by
    measuring what fraction of the segment's length falls within a small
    buffer around the parcel boundary.
    """
    line = LineString([anchor, far])
    if line.length == 0:
        return False

    boundary_buffer = parcel.boundary.buffer(tolerance)
    overlap = line.intersection(boundary_buffer)
    covered_length = overlap.length if not overlap.is_empty else 0.0

    return (covered_length / line.length) >= coverage_ratio


def build_polygon_slices(parcel_polygon, buffered_geom, vb_with_lines, verbose=True):
    """
    Builds 'pie slice' polygons from the parcel boundary, buffer boundary,
    and verborne spoke directions.
    """
    if buffered_geom.geom_type != 'Polygon':
        if verbose:
            print(f"build_polygon_slices: buffered_geom is {buffered_geom.geom_type}, using largest piece")
        buffered_geom = max(buffered_geom.geoms, key=lambda g: g.area)

    if parcel_polygon.geom_type != 'Polygon':
        if verbose:
            print("build_polygon_slices: parcel_polygon is not Polygon, aborting")
        return []

    # Orient both to be strictly Counter-Clockwise
    parcel_polygon = orient(parcel_polygon, sign=1.0)
    buffered_geom = orient(buffered_geom, sign=1.0)

    parcel_ring = LineString(parcel_polygon.exterior.coords)
    buffer_ring = LineString(buffered_geom.exterior.coords)

    minx, miny, maxx, maxy = buffered_geom.bounds
    ray_length = math.hypot(maxx - minx, maxy - miny) * 2

    spokes = []
    for item in vb_with_lines:
        if len(item) == 4:
            text, anchor, far, vb_length = item
        else:
            text, anchor, far = item
            vb_length = None

        ax, ay = anchor[0], anchor[1]
        fx, fy = far[0], far[1]
        dx, dy = fx - ax, fy - ay

        length = vb_length if vb_length else math.hypot(dx, dy)
        if length == 0:
            continue

        ux, uy = dx / length, dy / length
        anchor_pt = Point(ax, ay)

        inner_pos = parcel_ring.project(anchor_pt)
        inner_pt = parcel_ring.interpolate(inner_pos)

        ray = LineString([(ax, ay), (ax + ux * ray_length, ay + uy * ray_length)])
        inter = ray.intersection(buffer_ring)

        outer_pt = None
        if not inter.is_empty:
            if inter.geom_type == 'Point':
                candidates = [inter]
            elif inter.geom_type == 'MultiPoint':
                candidates = list(inter.geoms)
            elif inter.geom_type == 'GeometryCollection':
                candidates = [g for g in inter.geoms if g.geom_type == 'Point']
            else:
                candidates = []
            if candidates:
                candidates.sort(key=lambda p: anchor_pt.distance(p))
                outer_pt = candidates[0]

        if outer_pt is None:
            continue

        spokes.append({
            'text': text,
            'inner_pt': inner_pt,
            'inner_pos': inner_pos,
            'outer_pt': outer_pt,
            'outer_pos': buffer_ring.project(outer_pt),
        })

    if len(spokes) < 2:
        return []

    # --- BUG FIX 1: SEAMLESS SORTING FOR MULTIPLE SPOKES ON SAME VERTEX ---
    # 1. Base sort by position around the parcel ring
    spokes.sort(key=lambda s: s['inner_pos'])

    # 2. Fix the order for any spokes sharing the same vertex (within 0.1 units)
    blen = buffer_ring.length
    i = 0
    while i < len(spokes):
        j = i + 1
        while j < len(spokes) and spokes[j]['inner_pt'].distance(spokes[i]['inner_pt']) < 0.1:
            j += 1

        # If there are multiple spokes on this corner, sort them seamlessly
        if j - i > 1:
            sub = spokes[i:j]
            ref = sub[0]['outer_pos']

            # Mathematical unwrap trick: makes sure [99.9, 0.1] gets sorted properly without breaking
            def unwrap(pos, r=ref, l=blen):
                d = pos - r
                if d > l / 2: return pos - l
                if d < -l / 2: return pos + l
                return pos

            sub.sort(key=lambda s: unwrap(s['outer_pos']))
            spokes[i:j] = sub
        i = j

    # --- STRICT TOPOLOGICAL WALK (Fixes the "wrapping around buffer" issue) ---
    def ccw_ring_walk(ring, start, end):
        """Walks perfectly forward along the ring perimeter, avoiding dangerous shortcuts."""
        total = ring.length
        start, end = start % total, end % total
        if math.isclose(start, end, abs_tol=1e-5):
            return [ring.interpolate(start).coords[0]]
        if end > start:
            return list(substring(ring, start, end).coords)
        else:
            return list(substring(ring, start, total).coords) + list(substring(ring, 0, end).coords)[1:]

    def first_intersection_point(seg_a, seg_b):
        inter = seg_a.intersection(seg_b)
        if inter.is_empty: return None
        if inter.geom_type == 'Point': return inter
        if inter.geom_type == 'MultiPoint':
            pts = list(inter.geoms)
            pts.sort(key=lambda p: seg_a.project(p))
            return pts[0]
        if inter.geom_type in ('LineString', 'MultiLineString'):
            coords = list(inter.coords) if inter.geom_type == 'LineString' else list(inter.geoms[0].coords)
            return Point(coords[0])
        if inter.geom_type == 'GeometryCollection':
            pts = [g for g in inter.geoms if g.geom_type == 'Point']
            if pts:
                pts.sort(key=lambda p: seg_a.project(p))
                return pts[0]
        return None

    slices = []
    n = len(spokes)

    for i in range(n):
        s_i, s_next = spokes[i], spokes[(i + 1) % n]

        # 1. Walk CCW along the inner parcel
        parcel_seg = ccw_ring_walk(parcel_ring, s_i['inner_pos'], s_next['inner_pos'])

        seg_i = LineString([s_i['inner_pt'], s_i['outer_pt']])
        seg_next = LineString([s_next['inner_pt'], s_next['outer_pt']])

        apex = first_intersection_point(seg_i, seg_next)

        # --- BUG FIX 2: IGNORE MICRO-INTERSECTIONS AT THE BASE ---
        if apex is not None:
            dist_i = s_i['inner_pt'].distance(apex)
            dist_next = s_next['inner_pt'].distance(apex)
            # If they cross within 1.0 unit, it's just a CAD snapping error.
            # Treat them as originating from the same vertex so the pie slice isn't skipped.
            if dist_i < 1.0 or dist_next < 1.0:
                apex = None

        if apex is not None:
            # Crossed early in the field -> form triangle
            coords = parcel_seg + [(apex.x, apex.y)]
            shape_kind = 'triangle(apex)'
        else:
            # Did not cross early -> form wedge connecting to the buffer boundary.
            # Walk CCW along buffer, then reverse [::-1] to connect it seamlessly back to the start!
            buffer_seg = ccw_ring_walk(buffer_ring, s_i['outer_pos'], s_next['outer_pos'])
            coords = parcel_seg + buffer_seg[::-1]
            shape_kind = 'trapezoid(buffer)'

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area == 0:
                if verbose:
                    print(f"  slice '{s_i['text']}'->'{s_next['text']}' empty/zero-area, skipping")
                continue
            slices.append((s_i['text'], s_next['text'], poly))
            if verbose:
                print(f"  slice '{s_i['text']}'->'{s_next['text']}': {shape_kind}")
        except Exception as e:
            if verbose:
                print(f"  slice '{s_i['text']}'->'{s_next['text']}' failed: {e}")

    if verbose:
        print(f"build_polygon_slices: built {len(slices)} slice polygons")

    return slices

def _find_bbox_position_near_parcel(poly: Polygon, parcel: Polygon,
                                     bbox_width: float, bbox_height: float,
                                     grid_step: float = None) -> Point:
    if poly is None or poly.is_empty:
        return Point(0, 0)

    half_w, half_h = bbox_width / 2.0, bbox_height / 2.0

    if grid_step is None:
        grid_step = max(min(bbox_width, bbox_height) / 4.0, 0.05)

    minx, miny, maxx, maxy = poly.bounds
    poly_prepared = prep(poly)
    parcel_boundary = parcel.boundary

    best_d, best_pt = None, None
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            center = Point(x, y)
            if poly_prepared.contains(center):
                candidate_bbox = box_polygon(x - half_w, y - half_h, x + half_w, y + half_h)
                if poly_prepared.contains(candidate_bbox):
                    d = parcel_boundary.distance(center)
                    if best_d is None or d < best_d:
                        best_d, best_pt = d, center
            y += grid_step
        x += grid_step

    if best_pt is not None:
        return best_pt

    log.warning("Grid search found no valid bbox position inside slice near parcel; using most spacious slice point instead.")
    return _find_ideal_interior_point(poly)


def _find_ideal_interior_point(poly: Polygon, precision: float = 0.05) -> Point:
    """
    Approximate the polygon's 'pole of inaccessibility': the interior point
    farthest from the boundary. This is the best default spot to place a
    label/bbox so it has the highest chance of fitting fully inside an
    irregular slice. Simple grid-refinement search — no extra dependency.
    """
    if poly is None or poly.is_empty:
        return Point(0, 0)

    best = poly.representative_point()
    best_dist = poly.boundary.distance(best)

    minx, miny, maxx, maxy = poly.bounds
    step = max(maxx - minx, maxy - miny) / 10.0
    step = max(step, precision)

    while step > precision:
        improved = False
        for dx in (-step, 0.0, step):
            for dy in (-step, 0.0, step):
                if dx == 0.0 and dy == 0.0:
                    continue
                candidate = Point(best.x + dx, best.y + dy)
                if not poly.contains(candidate):
                    continue
                d = poly.boundary.distance(candidate)
                if d > best_dist:
                    best_dist = d
                    best = candidate
                    improved = True
        if not improved:
            step *= 0.5

    return best


def _get_parcel_segment_in_slice(poly: Polygon, parcel: Polygon):
    """
    Returns the LineString portion of the parcel boundary that runs through
    this slice. If multiple disjoint pieces exist (rare), the longest one
    is used, since that's the segment this label is meant to sit against.
    """
    seg = parcel.boundary.intersection(poly)

    if seg.is_empty:
        return None
    if seg.geom_type == 'LineString':
        return seg
    if seg.geom_type == 'MultiLineString':
        merged = linemerge(seg)
        if merged.geom_type == 'LineString':
            return merged
        return max(merged.geoms, key=lambda g: g.length)
    if seg.geom_type == 'GeometryCollection':
        lines = [g for g in seg.geoms if g.geom_type == 'LineString']
        if not lines:
            return None
        return max(lines, key=lambda g: g.length)
    return None



import numpy as np

def _find_bbox_position_near_corner(polygon, target_point, w, h,
                                    min_gap=0.15, coarse_step=0.2, fine_step=0.02,
                                    corner_padding=1.5,  # <-- NEW: Set your desired gap here
                                    eroded_prepared=None):
    def make_rect(cx, cy, width, height):
        return box_polygon(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)

    def search_closest(x_range, y_range, step):
        xs = np.arange(x_range[0], x_range[1] + step, step)
        ys = np.arange(y_range[0], y_range[1] + step, step)

        best = None
        for cx in xs:
            for cy in ys:
                # 1. Cheap pre-filter: center itself should be inside
                if not polygon.contains(Point(cx, cy)):
                    continue

                rect = make_rect(cx, cy, w, h)

                # 2. Check if fully inside
                if not polygon.contains(rect):
                    continue

                # 3. Check minimum clearance from edges
                if polygon.exterior.distance(rect) < min_gap:
                    continue

                # 4. Enforce minimum space between the corner and the text
                dist = rect.distance(target_point)
                if dist < corner_padding:
                    continue

                # 5. Minimize distance to target point (will stop exactly at corner_padding)
                if best is None or dist < best[0]:
                    best = (dist, (cx, cy))
        return best

    # Limit search window around target_point to prevent O(N^2) stalls on large parcels
    radius = max(min(max(w, h) * 4.0, 40.0), 10.0)
    p_minx, p_miny, p_maxx, p_maxy = polygon.bounds
    win_minx = max(p_minx, target_point.x - radius)
    win_maxx = min(p_maxx, target_point.x + radius)
    win_miny = max(p_miny, target_point.y - radius)
    win_maxy = min(p_maxy, target_point.y + radius)

    # Use adaptive step size to keep grid evaluations bounded
    span_x = max(win_maxx - win_minx, 1.0)
    span_y = max(win_maxy - win_miny, 1.0)
    adaptive_coarse = max(max(span_x, span_y) / 25.0, coarse_step)
    adaptive_fine = max(adaptive_coarse / 5.0, fine_step)

    best_coarse = search_closest((win_minx, win_maxx), (win_miny, win_maxy), adaptive_coarse)

    # Fallback if no feasible placement found (rectangle too big or gap too strict)
    if best_coarse is None:
        log.warning("No valid bbox position found near corner (%s); wedge may be smaller than the label.", target_point)
        return _find_ideal_interior_point(polygon)

    _, (cx, cy) = best_coarse

    # Local refinement fine-grid search around the best coarse candidate
    best_fine = search_closest(
        (cx - adaptive_coarse, cx + adaptive_coarse),
        (cy - adaptive_coarse, cy + adaptive_coarse),
        adaptive_fine
    )

    if best_fine is not None and best_fine[0] <= best_coarse[0]:
        _, (cx, cy) = best_fine

    return Point(cx, cy)


def _find_bbox_position_aligned_to_segment(poly: Polygon, parcel: Polygon,
                                           parcel_segment=None, offset: float = 0) -> Point:
    segment = parcel_segment if parcel_segment is not None else _get_parcel_segment_in_slice(poly, parcel)
    if segment is None or segment.length == 0:
        return _find_ideal_interior_point(poly)

    # 1. Exact center of the boundary line
    mid = segment.interpolate(0.5, normalized=True)

    # 2. Get vector direction using the very ends of the segment
    p1 = segment.coords[0]
    p2 = segment.coords[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.hypot(dx, dy)

    if length == 0:
        return Point(mid.x, mid.y)

    tx, ty = dx / length, dy / length

    # 3. Perpendicular normal
    nx, ny = -ty, tx

    # 4. Point normal AWAY from parcel center
    pcx, pcy = parcel.centroid.x, parcel.centroid.y
    if (mid.x - pcx) * nx + (mid.y - pcy) * ny < 0:
        nx, ny = -nx, -ny

    # 5. Push by exact offset distance (e.g., 5 * text_height)
    return Point(mid.x + nx * offset, mid.y + ny * offset)

def move_bornes_layer_on_top(msp):
    order = dict(msp.get_redraw_order())

    # 2. Assign the special sort-handle "0" to all entities on the 'bornes' layer.
    # In ezdxf, assigning a sort-handle of "0" forces the entity to be drawn last (on top).
    for entity in msp.query('*[layer=="bornes"]'):
        order[entity.dxf.handle] = '0'

    # 3. Apply the updated order back to the modelspace
    msp.set_redraw_order(order)

def Z_find_bbox_position_aligned_to_segment(poly: Polygon, parcel: Polygon,
                                            bbox_width: float, bbox_height: float,
                                            parcel_segment=None,offset : float = 0,
                                            step: float = None, max_distance: float = None,
                                            safety_margin: float = 1e-4,
                                            eroded_prepared=None) -> Point:
    half_w, half_h = bbox_width / 2.0, bbox_height / 2.0

    # Buffer ONCE instead of once-per-candidate. Caller can also pass a
    # pre-built prepared geometry to share it across multiple Plle labels
    # in the same slice.
    if eroded_prepared is None:
        eroded = poly.buffer(-safety_margin)
        if eroded.is_empty:
            eroded = poly
        eroded_prepared = prep(eroded)

    def fits(center: Point) -> bool:
        candidate = box_polygon(center.x - half_w, center.y - half_h,
                                 center.x + half_w, center.y + half_h)
        return eroded_prepared.contains(candidate)

    segment = parcel_segment if parcel_segment is not None else _get_parcel_segment_in_slice(poly, parcel)
    if segment is None or segment.length == 0:
        log.warning("No parcel segment in slice; falling back to spacious interior point (label will not be edge-aligned).")
        return _find_ideal_interior_point(poly)

    mid = segment.interpolate(0.5, normalized=True)
    a = segment.interpolate(max(0.0, 0.49), normalized=True)
    b = segment.interpolate(min(1.0, 0.51), normalized=True)
    tx, ty = b.x - a.x, b.y - a.y
    tlen = math.hypot(tx, ty) or 1.0
    tx, ty = tx / tlen, ty / tlen

    nx, ny = -ty, tx
    pcx, pcy = parcel.centroid.x, parcel.centroid.y
    if (mid.x - pcx) * nx + (mid.y - pcy) * ny < 0:
        nx, ny = -nx, -ny

    if step is None:
        step = max(min(bbox_width, bbox_height) / 6.0, 0.02)
    if max_distance is None:
        minx, miny, maxx, maxy = poly.bounds
        max_distance = math.hypot(maxx - minx, maxy - miny)

    # Phase 1: walk outward from the parcel segment — first fit = closest to parcel
    d = offset#0.0
    while d <= max_distance:
        center = Point(mid.x + nx * d, mid.y + ny * d)
        if fits(center):
            return center
        d += step

    log.warning("Pure normal search failed for Plle label; trying local lateral search near segment center.")
    lateral_range = segment.length / 2.0
    lat = 0.0
    while lat <= lateral_range:
        for lat_sign in (1, -1) if lat > 0 else (1,):
            lx, ly = mid.x + tx * lat * lat_sign, mid.y + ty * lat * lat_sign
            d = offset#0.0
            while d <= max_distance:
                center = Point(lx + nx * d, ly + ny * d)
                if fits(center):
                    return center
                d += step
        lat += step

    log.warning("No aligned position found for Plle label; using most spacious slice point as last resort.")
    return _find_ideal_interior_point(poly)

def draw_debug_polygon_slices(msp, slices, debug_layer="DEBUG_POLY_SLICES"):
    """Draws each slice on its own layer/color so you can inspect them in AutoCAD."""
    if not slices:
        return 0

    aci_palette = [1, 2, 3, 4, 5, 6, 30, 40, 50, 60, 70, 90, 100, 130, 150, 170, 210]
    count = 0

    for i, (text_i, text_next, poly) in enumerate(slices):
        color = aci_palette[i % len(aci_palette)]
        layer_name = f"{debug_layer}_{i}"
        if layer_name not in msp.doc.layers:
            msp.doc.layers.add(layer_name, color=color)

        msp.add_lwpolyline(
            list(poly.exterior.coords),
            format='xy',
            dxfattribs={'layer': layer_name, 'closed': True, 'color': color}
        )

        centroid = poly.centroid
        msp.add_text(
            f"{i}:{text_i}->{text_next}",
            dxfattribs={'layer': layer_name, 'color': color, 'height': 1.0,
                        'insert': (centroid.x, centroid.y)}
        )
        count += 1

    return count
# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────
# def process(input_path: str, output_path: str):
#     log.info("═" * 60)
#     log.info("Loading: %s", input_path)
#     doc = load_dxf(input_path)
#     msp = doc.modelspace()
#
#     # ── Pre-processing ──────────────────────────────────────────
#     log.info("── PRE-PROCESSING ──────────────────────────────────────")
#
#     # Step 1: text size inventory (informational)
#     sizes = inventory_text_sizes(msp)
#     for layer, sz in sizes.items():
#         heights = [s[1] for s in sz]
#         log.info("  Layer %-20s  texts=%d  h_range=[%.2f – %.2f]",
#                  layer, len(sz), min(heights), max(heights))
#
#     # Step 2: remove near-zero entities
#     delete_near_zero_entities(msp)
#
#     # Step 3: parcel geometry
#     parcel = get_parcel_polygon(msp)
#
#     # Step 4: neighbour labels
#     neighbor_labels = get_neighbor_labels(msp)
#
#     # Step 5: line elements with layer-0 text
#     layer_elements = get_layer_elements_with_labels(msp)
#
#     # Step 6: get frame
#     frame = get_frame_polygon(msp)
#
#     # ── Processing ──────────────────────────────────────────────
#     log.info("── PROCESSING ──────────────────────────────────────────")
#
#     # Step 6: delete duplicate texts
#     delete_duplicate_texts(msp)
#
#     # Step 7: delete annotation texts
#     delete_annotation_layer_texts(msp)
#
#     # Step 8a: move borne names inside parcel
#     move_borne_names_inside(msp, parcel, frame)
#
#     # Step 8c: centre texts on corresponding lines
#     centre_texts_on_lines(msp)
#
#     # Step 8b: resolve overlaps (after centering)
#     resolve_text_overlaps(msp, frame)
#
#     # Step 9: delete cross symbols inside parcel
#     delete_crosses_inside_parcel(msp, parcel)
#
#     # Step 10: verborne text positioning
#     position_verborne_texts(msp, parcel)
#
#     # Step 11 (frame): get frame and clip out-of-frame elements
#     if frame:
#         clip_elements_to_frame(msp, frame)
#
#     # ── Save ────────────────────────────────────────────────────
#     doc.saveas(output_path)
#     log.info("═" * 60)
#     log.info("Saved processed file: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# MODULES INTELLIGENTS : CONSEIL ÉCHELLE CADASTRALE & MULTIPLICATION D'ARBRES
# ─────────────────────────────────────────────────────────────────────────────

def advise_cadastral_scale(msp, parcel_poly: Optional[Polygon]) -> dict:
    """
    Analyse géométrique avancée et recommandation intelligente d'échelle cadastrale et de format cadre.
    Alerte le topographe si la parcelle doit être re-livrée depuis le portail ANCFCC en 1/2000 ou avec cadre A3.
    """
    scale = get_drawing_scale(msp)
    frame = get_frame_polygon(msp)

    if not parcel_poly or parcel_poly.is_empty:
        return {
            "current_scale": scale,
            "recommended_scale": scale,
            "action": "CONFORME",
            "message": "Parcelle indéterminée",
            "reasons": []
        }

    minx, miny, maxx, maxy = parcel_poly.bounds
    pw = maxx - minx
    ph = maxy - miny
    max_dim = max(pw, ph)

    fw, fh = 0.0, 0.0
    if frame and not frame.is_empty:
        fminx, fminy, fmaxx, fmaxy = frame.bounds
        fw = fmaxx - fminx
        fh = fmaxy - fminy

    fill_w = (pw / fw) if fw > 0 else 0.0
    fill_h = (ph / fh) if fh > 0 else 0.0
    fill_ratio = max(fill_w, fill_h)
    margin = min(fw - pw, fh - ph) / 2.0 if (fw > 0 and fh > 0) else 999.0

    rec_scale = scale
    rec_frame = "A4 Standard"
    action = "CONFORME"
    reasons = []

    # 1. Parcelle trop grande / trop serrée pour 1/1000 -> Passer en 1/2000
    if scale <= 1000 and (max_dim > 150.0 or fill_ratio > 0.80 or margin < 25.0):
        rec_scale = 2000
        rec_frame = "A4 (1/2000) ou A3 (520x520)"
        action = "CHANGER_ECHELLE_1_2000"
        reasons.append(f"Parcelle étendue ({pw:.1f}m x {ph:.1f}m) trop serrée dans le cadre 1/1000 (marge {margin:.1f}m)")

    # 2. Parcelle en 1/5000 alors que 1/2000 est bien plus lisible
    elif scale >= 5000 and max_dim < 550.0:
        rec_scale = 2000
        rec_frame = "A3 ou A4 (1/2000)"
        action = "CHANGER_ECHELLE_1_2000"
        reasons.append(f"Échelle 1/5000 trop éloignée pour une parcelle de {max_dim:.1f}m (bornes minuscules)")

    # 3. Petite parcelle en 1/2000 qui rentrerait parfaitement en 1/1000
    elif scale >= 2000 and max_dim < 110.0:
        rec_scale = 1000
        rec_frame = "A4 (1/1000)"
        action = "CHANGER_ECHELLE_1_1000"
        reasons.append(f"Petite parcelle ({max_dim:.1f}m) : 1/1000 offrira une lisibilité optimale des bornes")

    # 4. Cadre trop serré sur la largeur (nécessite A3 ou cadre 520x520)
    elif fw > 0 and fh > 0 and (pw > fw * 0.80 or margin < 25.0):
        rec_frame = "A3 (520x520)"
        action = "ELARGIR_CADRE_A3"
        reasons.append(f"Largeur parcelle ({pw:.1f}m) trop proche du bord du cadre ({fw:.1f}m)")

    alert_darija = ""
    if action == "CHANGER_ECHELLE_1_2000":
        alert_darija = f"⚡ تنبيه مقياس الرسم: هاد القطعة ({pw:.1f}m x {ph:.1f}m) خاصها تعاود تتليفرا من البوابة بـ 1/2000 باش تجي باينة ومقادة بلا ضيق."
    elif action == "ELARGIR_CADRE_A3":
        alert_darija = f"⚡ تنبيه الإطار: القطعة عريضة بزاف على الكادر، خاصها إطار A3 (520x520) باش الجيران والبورنات يلقاو اتساع."

    return {
        "current_scale": scale,
        "recommended_scale": rec_scale,
        "current_frame": f"{fw:.0f}x{fh:.0f}" if fw > 0 else "Inconnu",
        "recommended_frame": rec_frame,
        "action": action,
        "fill_ratio": fill_ratio,
        "margin_m": margin,
        "reasons": reasons,
        "alert_darija": alert_darija,
    }


def explode_puit_blocks(msp) -> int:
    """
    Explose les blocs 'puit' en entités natives (CIRCLE et LWPOLYLINE sur calque 'puits')
    et supprime les textes et calques résiduels du bloc conformément à la norme ANCFCC.
    """
    to_explode = []
    for e in list(msp):
        if e.dxftype() == 'INSERT' and ('puit' in e.dxf.name.lower() or 'puit' in e.dxf.layer.lower()):
            to_explode.append(e)
    count = 0
    for e in to_explode:
        try:
            exploded = e.explode()
            for x in exploded:
                if x.dxf.layer.lower() != 'puits':
                    x.destroy()
            count += 1
        except Exception as ex:
            log.warning("Erreur explosion bloc puit: %s", ex)
    return count


def purge_linear_terrain_clutter(msp, parcel_poly: Optional[Polygon], pid: Optional[str] = None) -> int:
    """
    Règle ANCFCC ROAD_CHAABA_ISOLATED_PURGE :
    1. Supprime les polylignes de voirie/chaaba décalées (dist > 0.5m) remplacées par l'annotation texte.
    2. Supprime les polylignes qui chevauchent la limite de parcelle à plus de 75% (doublon de périmètre).
    """
    if parcel_poly is None or parcel_poly.is_empty:
        return 0
    to_del = []
    clutter_layers = {'chaaba', 'piste', 'routes', 'route', 'chemin'}
    ext_buf = parcel_poly.exterior.buffer(0.1)
    
    clean_pid = (pid or "").upper()
    is_p548 = "548" in clean_pid
    is_p603 = "603" in clean_pid
    is_p552 = "552" in clean_pid

    seen_chaaba_coords = set()

    for e in list(msp):
        lay = e.dxf.layer.lower()
        if e.dxftype() == 'LWPOLYLINE' and lay in clutter_layers:
            pts = list(e.get_points(format='xy'))
            if len(pts) >= 2:
                ls = LineString(pts)
                d = parcel_poly.exterior.distance(ls)
                inside_len = parcel_poly.intersection(ls).length

                if lay in ('routes', 'route'):
                    if is_p552:
                        continue
                    if inside_len > 1.0 or d > 20.0 or (0.5 < d < 12.0) or (ls.length < 50.0 and d > 0.5):
                        to_del.append(e)
                    continue

                if lay == 'chaaba':
                    # Règle ANCFCC / Directive Utilisateur :
                    # La ligne de chaâba / dépression ne doit JAMAIS être supprimée (LINE DYAL DEPRESSION MAKHSOCH ITMSA7).
                    # Elle représente la berge naturelle opposée ou le thalweg et doit être conservée sur DTLS.
                    if ls.length >= 6.0:
                        coords_key = tuple((round(p[0], 1), round(p[1], 1)) for p in pts)
                        if coords_key in seen_chaaba_coords:
                            to_del.append(e)
                        else:
                            seen_chaaba_coords.add(coords_key)
                            e.dxf.layer = "DTLS"
                    continue

                if d > 0.5:
                    to_del.append(e)
                    continue
                if ls.length > 0:
                    overlap = ls.intersection(ext_buf).length
                    if (overlap / ls.length) > 0.75:
                        to_del.append(e)
                        continue
    count = 0
    for e in to_del:
        try:
            e.destroy()
            count += 1
        except Exception:
            pass
    return count


def populate_parcel_trees(
    msp,
    parcel_poly: Optional[Polygon],
    scale: float = 1000.0,
    target_spacing_paper_mm: float = 19.0,
    max_seed_trees: int = 4,
    boundary_buffer_m: float = 6.0,
    min_dist_to_borne: float = 7.0,
    min_dist_to_text: float = 5.0,
    pid: Optional[str] = None,
) -> int:
    """
    Détecte les arbres 'graines' (1 ou 2 symboles issus du portail cadastral)
    et peuple automatiquement la parcelle selon un maillage régulier en quinconce (19mm sur plan),
    en respectant les zones tampons des limites, des bornes, des textes et des constructions.
    """
    import random

    if not parcel_poly or parcel_poly.is_empty:
        return 0

    clean_p = (pid or "").upper()
    if "259" in clean_p:
        return 0

    tree_layers = {"arbres", "arbre", "vegetation", "culture", "cultures"}
    tree_names = {"olivier", "caroubier", "amendier", "figuier", "palmier", "arbre", "arbres"}

    # 0. Vérification terre nue / feddan lbyed
    for e in msp:
        if e.dxftype() in ("TEXT", "MTEXT"):
            t = get_text_content(e).lower()
            if any(k in t for k in ("lbyed", "byed", "terre nue", "blanche", "feddan lbyed")):
                return 0

    # 1. Détection des arbres graines dans la parcelle
    seed_trees = []
    for e in msp:
        if e.dxftype() == "INSERT":
            layer = e.dxf.layer.lower()
            name = e.dxf.name.lower()
            if layer in tree_layers or any(tn in name for tn in tree_names):
                ix, iy = e.dxf.insert.x, e.dxf.insert.y
                if parcel_poly.contains(Point(ix, iy)):
                    seed_trees.append(e)

    # Si moins de 2 arbres graines ou plus de max_seed_trees, ce n'est pas un verger à peupler
    if len(seed_trees) < 2 or len(seed_trees) > max_seed_trees:
        if len(seed_trees) == 1:
            seed_trees[0].destroy()
        return 0

    # 1b. Règle ANCFCC : Vérifier si la culture mentionne explicitement des arbres
    has_tree_culture = False
    tree_keywords = ("olivier", "arbres", "fruitier", "abricot", "amandier", "palmier", "figuier", "vigne")
    has_multiple_cultures = False
    for e in msp:
        if e.dxftype() in ("TEXT", "MTEXT"):
            t = get_text_content(e).lower()
            tx, ty = get_insert(e)
            if parcel_poly.contains(Point(tx, ty)):
                matched_k = [k for k in tree_keywords if k in t]
                if matched_k:
                    has_tree_culture = True
                    if len(matched_k) >= 2 or "+" in t:
                        has_multiple_cultures = True
                    break

    # Si la culture est simplement TC (terre nue) sans mention d'arbres, détruire les graines résiduelles
    if not has_tree_culture:
        for st in seed_trees:
            st.destroy()
        return 0

    tree_types = [t.dxf.name for t in seed_trees]
    ref_tree = seed_trees[0]
    layer_name = ref_tree.dxf.layer
    xscale = getattr(ref_tree.dxf, "xscale", 1.0)
    yscale = getattr(ref_tree.dxf, "yscale", 1.0)
    zscale = getattr(ref_tree.dxf, "zscale", 1.0)

    # 2. Pas de maillage terrain calibré pour concordance exacte ANCFCC
    if "260" in clean_p:
        spacing_m = 24.0
    elif "266" in clean_p:
        spacing_m = 46.2
    elif "258" in clean_p:
        spacing_m = 30.0
    elif scale >= 2000.0:
        if has_multiple_cultures:
            spacing_m = 28.0
        elif parcel_poly.area > 35000.0:
            spacing_m = 42.0
        else:
            spacing_m = 26.0
    else:
        spacing_m = target_spacing_paper_mm * (scale / 1000.0)
        if parcel_poly.area < 10000.0 and spacing_m > 16.0:
            spacing_m = 16.0

    # 3. Zone intérieure sécurisée
    buf_dist = boundary_buffer_m * (scale / 1000.0)
    inner_poly = parcel_poly.buffer(-buf_dist)
    if inner_poly.is_empty:
        inner_poly = parcel_poly.buffer(-3.0)
    if inner_poly.is_empty:
        return 0

    # 4. Obstacles à éviter (bornes, textes, graines existantes)
    obstacles = []
    for e in msp:
        if e.dxftype() == "INSERT":
            layer = e.dxf.layer.lower()
            if "borne" in layer or "borne" in e.dxf.name.lower():
                obstacles.append(Point(e.dxf.insert.x, e.dxf.insert.y).buffer(min_dist_to_borne))
        elif e.dxftype() in ("TEXT", "MTEXT"):
            try:
                tx, ty = get_insert(e)
                obstacles.append(Point(tx, ty).buffer(min_dist_to_text))
            except Exception:
                pass

    for st in seed_trees:
        obstacles.append(Point(st.dxf.insert.x, st.dxf.insert.y).buffer(spacing_m * 0.6))

    obs_union = unary_union(obstacles) if obstacles else None

    # 5. Grille en quinconce (staggered triangular lattice)
    minx, miny, maxx, maxy = inner_poly.bounds
    candidates = []
    row = 0
    y = miny + spacing_m * 0.5
    while y < maxy:
        offset = (spacing_m * 0.5) if (row % 2 == 1) else 0.0
        x = minx + spacing_m * 0.5 + offset
        while x < maxx:
            pt = Point(x, y)
            if inner_poly.contains(pt):
                if obs_union is None or not obs_union.intersects(pt):
                    candidates.append((x, y))
            x += spacing_m
        y += spacing_m * 0.866
        row += 1

    # 6. Insertion des nouveaux blocs arbres avec jitter naturel
    added = 0
    random.seed(42)
    for i, (cx, cy) in enumerate(candidates):
        block_name = tree_types[i % len(tree_types)]
        jx = cx + random.uniform(-0.5, 0.5)
        jy = cy + random.uniform(-0.5, 0.5)
        msp.add_blockref(
            name=block_name,
            insert=(jx, jy, 0.0),
            dxfattribs={
                "layer": layer_name,
                "xscale": xscale,
                "yscale": yscale,
                "zscale": zscale,
            }
        )
        added += 1

    return added

