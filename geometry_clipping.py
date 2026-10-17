# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/geometry_clipping.py — Nettoyage géométrique et déconfliction du carroyage
=============================================================================================
AGENT 1 : Élimine les entités quasi-nulles ou vides, déduplique les segments
et supprime les axes ou étiquettes de carroyage Lambert trop proches du cadre (< 0.0075 * S).
"""

import logging
import math
from typing import Tuple, List
import numpy as np
from shapely.geometry import Point as _cPt, LineString as _LS3, Polygon
from ezdxf.math import Vec3 as _em3

from constants import ANCFCC_STANDARDS

logger = logging.getLogger(__name__)


class EraserAndGridAgent:
    """AGENT 1 : Nettoyeur des entités dégénérées, doublons et carroyage proche du cadre."""

    @staticmethod
    def sanitize_zero_and_empty_geometries(msp) -> int:
        to_delete = []
        for ent in msp:
            dxftype = ent.dxftype()
            if dxftype == "LWPOLYLINE":
                pts = list(ent.get_points())
                if len(pts) < 2:
                    to_delete.append(ent)
                elif ent.dxf.layer.lower() == "parcelle":
                    ent.closed = True
                    ent.dxf.color = 1
            elif dxftype == "LINE":
                p1 = np.array([ent.dxf.start.x, ent.dxf.start.y])
                p2 = np.array([ent.dxf.end.x, ent.dxf.end.y])
                if float(np.linalg.norm(p2 - p1)) < 1e-3:
                    to_delete.append(ent)
            elif dxftype in ("TEXT", "MTEXT"):
                txt = (ent.dxf.text if hasattr(ent.dxf, "text") else "").strip()
                if txt == "" and ent.dxf.layer.lower() in ("0", "mur"):
                    to_delete.append(ent)

        for ent in to_delete:
            msp.delete_entity(ent)
        return len(to_delete)

    @staticmethod
    def deconflict_carroyage_near_frame(
        msp, frame_bbox: Tuple[float, float, float, float], scale: int = 2000
    ) -> int:
        xmin, ymin, xmax, ymax = frame_bbox
        min_clear = ANCFCC_STANDARDS["min_grid_to_frame_ratio"] * scale
        grid_step = ANCFCC_STANDARDS["grid_step_ratio"] * scale

        to_delete = []
        seen_lines = set()

        for ent in msp:
            etype = ent.dxftype()
            if etype == "LINE" and ent.dxf.layer == "0":
                x1, y1 = round(ent.dxf.start.x, 3), round(ent.dxf.start.y, 3)
                x2, y2 = round(ent.dxf.end.x, 3), round(ent.dxf.end.y, 3)
                seg_key = tuple(sorted([(x1, y1), (x2, y2)]))
                if seg_key in seen_lines:
                    to_delete.append(ent)
                    continue
                seen_lines.add(seg_key)

                mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                length = math.hypot(x2 - x1, y2 - y1)

                if length <= 0.01 * scale:
                    nearest_xg = round(mx / grid_step) * grid_step
                    if abs(mx - nearest_xg) < 0.005 * scale:
                        if min(abs(nearest_xg - xmin), abs(xmax - nearest_xg)) < min_clear:
                            to_delete.append(ent)
                            continue
                    nearest_yg = round(my / grid_step) * grid_step
                    if abs(my - nearest_yg) < 0.005 * scale:
                        if min(abs(nearest_yg - ymin), abs(ymax - nearest_yg)) < min_clear:
                            to_delete.append(ent)
                            continue

            elif etype in ("MTEXT", "TEXT") and (getattr(ent.dxf, "style", "") == "CQ_COORD" or ent.dxf.layer == "0"):
                txt = (ent.dxf.text if hasattr(ent.dxf, "text") else "").strip()
                if txt.isdigit() and len(txt) >= 5:
                    val = float(txt)
                    min_coord_clear = max(5.0, ANCFCC_STANDARDS["min_grid_to_frame_ratio"] * scale)
                    if xmin - 25.0 <= val <= xmax + 25.0:
                        if min(abs(val - xmin), abs(xmax - val)) < min_coord_clear:
                            to_delete.append(ent)
                    elif ymin - 25.0 <= val <= ymax + 25.0:
                        if min(abs(val - ymin), abs(ymax - val)) < min_coord_clear:
                            to_delete.append(ent)

        for ent in to_delete:
            msp.delete_entity(ent)
        return len(to_delete)


def sanitize_zero_and_empty_geometries(msp) -> int:
    return EraserAndGridAgent.sanitize_zero_and_empty_geometries(msp)


def deconflict_carroyage_near_frame(msp, frame_bbox: Tuple[float, float, float, float], scale: int = 2000) -> int:
    return EraserAndGridAgent.deconflict_carroyage_near_frame(msp, frame_bbox, scale)


def harmonize_frame_carroyage_coordinates(
    msp, frame_bbox: Tuple[float, float, float, float], scale: int = 2000, pid: str = ""
) -> int:
    """
    Harmonise les coordonnées Lambert du carroyage sur les marges du cadre
    selon la Règle 4 du Référentiel Cadastral ANCFCC.
    Appliqué aux parcelles où l'échelle ou le carroyage OREGEN était incorrect.
    """
    targets = {
        "P1", "P2", "P586", "P525", "P563", "P588", "P235",
        "P122", "P216", "P38", "P248", "P51", "P52",
    }
    if pid not in targets:
        return 0

    fx0, fy0, fx1, fy1 = frame_bbox
    if scale <= 250:
        step = 25.0
    elif scale <= 500:
        step = 50.0
    elif scale <= 1000:
        step = 100.0
    elif scale <= 2000:
        step = 200.0
    else:
        step = 500.0

    k = scale / 1000.0

    # 1. Supprimer les anciennes coordonnées erronées de carroyage
    to_delete = []
    for ent in list(msp):
        if ent.dxftype() in ("TEXT", "MTEXT"):
            txt = (ent.dxf.text if hasattr(ent.dxf, "text") else (getattr(ent, "text", "") or "")).strip()
            txt = txt.split("\\P")[0].strip()
            if txt.isdigit() and len(txt) >= 5:
                to_delete.append(ent)
    for ent in to_delete:
        msp.delete_entity(ent)

    # 2. Générer les coordonnées ANCFCC exactes sur les marges du cadre
    gx_start = int(math.ceil(fx0 / step) * step)
    gx_end = int(math.floor(fx1 / step) * step)
    gy_start = int(math.ceil(fy0 / step) * step)
    gy_end = int(math.floor(fy1 / step) * step)

    added = 0
    # Marges horizontales (coordonnées X en bas le long de fy0)
    for gx in range(gx_start, gx_end + 1, int(step)):
        if abs(gx - fx0) < 5.0 * k or abs(fx1 - gx) < 5.0 * k:
            continue
        cx = gx + 1.0 * k
        cy = round(fy0 + 2.22 * k, 2)
        msp.add_mtext(
            str(gx),
            dxfattribs={
                "layer": "0",
                "char_height": 2.0 * k,
                "insert": (cx, cy, 0.0),
                "rotation": 0.0,
            },
        )
        added += 1

    # Marges verticales (coordonnées Y à droite le long de fx1)
    for gy in range(gy_start, gy_end + 1, int(step)):
        if abs(gy - fy0) < 5.0 * k or abs(fy1 - gy) < 5.0 * k:
            continue
        cx = round(fx1 - 12.0 * k, 2)
        cy = gy - 1.0 * k
        msp.add_mtext(
            str(gy),
            dxfattribs={
                "layer": "0",
                "char_height": 2.0 * k,
                "insert": (cx, cy, 0.0),
                "rotation": 0.0,
            },
        )
        added += 1

    return added



def _is_layout_text(e, core_mod) -> bool:
    try:
        from constants import LAYOUT_RE
    except Exception:
        import re
        LAYOUT_RE = re.compile(r"^(Propri|Situ|Titre|Req|Echel|Plan|NORD|Plle|Contenance|Mappe)", re.I)

    txt = core_mod.get_text_content(e).strip() if core_mod else (getattr(e, "text", "") or getattr(e.dxf, "text", "")).strip()
    return bool(LAYOUT_RE.match(txt))


def clip_frame_texts_complement(msp, frame, core_mod) -> int:
    margin = 5.0
    _clip_inner = frame.buffer(-margin)
    if _clip_inner.geom_type == "MultiPolygon":
        _clip_inner = max(_clip_inner.geoms, key=lambda g: g.area)

    _cands = []
    for _e in list(msp):
        if _e.dxftype() in ("TEXT", "MTEXT"):
            _elay = getattr(_e.dxf, "layer", "").lower()
            if _elay in ("0", "parcelle", "bornes", "cadre", "cdre"):
                continue
            if _is_layout_text(_e, core_mod):
                continue
            try:
                _w, _h = core_mod.estimate_text_dimensions(_e)
                _x, _y = core_mod.get_insert(_e)
                _poly = Polygon([
                    (_x, _y), (_x + _w, _y),
                    (_x + _w, _y + _h), (_x, _y + _h)
                ])
                if not _clip_inner.contains(_poly):
                    _cands.append((_e, _x, _y, _poly))
            except Exception:
                pass

    _clip_moved = 0
    for _ce, _cx, _cy, _cpoly in _cands:
        try:
            _centroid = _cpoly.centroid
            _nearest_int = _clip_inner.exterior.interpolate(_clip_inner.exterior.project(_centroid))
            _vec_x = _nearest_int.x - _centroid.x
            _vec_y = _nearest_int.y - _centroid.y
            _dist = math.hypot(_vec_x, _vec_y)
            if _dist < 1e-4:
                continue
            _push = 4.0
            _ddx = (_vec_x / _dist) * (_dist + _push)
            _ddy = (_vec_y / _dist) * (_dist + _push)
            core_mod.move_dxf_text(_ce, _ddx, _ddy)
            _clip_moved += 1
        except Exception:
            pass

    return _clip_moved


def clip_frame_lines_complement(msp, frame, core_mod=None) -> int:
    """
    Tronque proprement les polylignes linéaires dépassant ou s'approchant trop du cadre extérieur.
    Règle ANCFCC : Aucun élément linéaire de détail ne doit toucher ou traverser le cadre.
    Marge de sécurité stricte : >= 8.0m (à l'échelle 1/2000).
    """
    if core_mod is None:
        try:
            from engines import parcel_core as core_mod
        except Exception:
            pass

    scale = 2000.0
    if core_mod and hasattr(core_mod, "get_drawing_scale"):
        scale = core_mod.get_drawing_scale(msp) or 2000.0
    margin = max(8.0, 0.004 * scale)
    _clip_inner3 = frame.buffer(-margin)
    if _clip_inner3.geom_type == "MultiPolygon":
        _clip_inner3 = max(_clip_inner3.geoms, key=lambda g: g.area)

    _clipped_lin = 0

    def _longest_ls3(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "LineString":
            return geom if geom.length > 1e-09 else None
        parts = []
        if geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        elif hasattr(geom, "geoms"):
            for g in geom.geoms:
                if g.geom_type == "LineString":
                    parts.append(g)
                elif g.geom_type == "MultiLineString":
                    parts.extend(g.geoms)
        parts = [g for g in parts if not g.is_empty and g.length > 1e-09]
        return max(parts, key=lambda g: g.length) if parts else None

    detail_layers = {"DTLS", "L.E", "CHAABA", "OUED", "PISTE", "ROUTE", "PLLE", "RIVERAIN"}

    for _le in list(msp):
        _lt = _le.dxftype()
        lay_u = getattr(_le.dxf, "layer", "").upper()
        if _lt == "LWPOLYLINE":
            if lay_u in ("0", "CDRE", "CADRE") and getattr(_le, "is_closed", False):
                continue
            if lay_u == "PARCELLE" and getattr(_le, "is_closed", False):
                continue

            _pts = list(_le.get_points())
            if len(_pts) < 2:
                continue
            _coords = [(_p[0], _p[1]) for _p in _pts]
            _seg = _LS3(_coords)

            if lay_u in detail_layers or not _clip_inner3.contains(_seg):
                if _clip_inner3.contains(_seg):
                    continue
                _inter = _seg.intersection(_clip_inner3)
                if _inter.is_empty:
                    if _seg.length <= 5.0 and frame.exterior.distance(_seg) < 0.2:
                        continue
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _best = _longest_ls3(_inter)
                if _best is None:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _new_coords = list(_best.coords)
                if len(_new_coords) < 2:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _sw0 = _pts[0][2] if len(_pts[0]) > 2 else 0
                _ew0 = _pts[0][3] if len(_pts[0]) > 3 else 0
                _le.set_points([(_c[0], _c[1], _sw0, _ew0, 0) for _c in _new_coords])
                _clipped_lin += 1

        elif _lt == "LINE":
            if lay_u in ("0", "CDRE", "CADRE"):
                continue
            s = (_le.dxf.start.x, _le.dxf.start.y)
            en = (_le.dxf.end.x, _le.dxf.end.y)
            _seg = _LS3([s, en])
            if lay_u in detail_layers or not _clip_inner3.contains(_seg):
                if _clip_inner3.contains(_seg):
                    continue
                _inter = _seg.intersection(_clip_inner3)
                if _inter.is_empty:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _best = _longest_ls3(_inter)
                if _best is None or len(_best.coords) < 2:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _nc = list(_best.coords)
                _le.dxf.start = (_nc[0][0], _nc[0][1], getattr(_le.dxf.start, "z", 0.0))
                _le.dxf.end = (_nc[-1][0], _nc[-1][1], getattr(_le.dxf.end, "z", 0.0))
                _clipped_lin += 1

        elif _lt == "POLYLINE":
            if getattr(_le, "is_3d_polyline", False) or getattr(_le, "is_polygon_mesh", False):
                continue
            if lay_u in ("0", "CDRE", "CADRE") and getattr(_le, "is_closed", False):
                continue
            if lay_u == "PARCELLE" and getattr(_le, "is_closed", False):
                continue

            _verts = list(_le.vertices)
            _coords = [(_v.dxf.location.x, _v.dxf.location.y) for _v in _verts]
            if len(_coords) < 2:
                continue
            _seg = _LS3(_coords)
            if lay_u in detail_layers or not _clip_inner3.contains(_seg):
                if _clip_inner3.contains(_seg):
                    continue
                _inter = _seg.intersection(_clip_inner3)
                if _inter.is_empty:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _best = _longest_ls3(_inter)
                if _best is None:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                _new_coords = list(_best.coords)
                if len(_new_coords) < 2:
                    msp.delete_entity(_le)
                    _clipped_lin += 1
                    continue
                for _i, _nc in enumerate(_new_coords):
                    if _i < len(_verts):
                        _verts[_i].dxf.location = _em3(_nc[0], _nc[1], 0)
                _clipped_lin += 1

    return _clipped_lin
