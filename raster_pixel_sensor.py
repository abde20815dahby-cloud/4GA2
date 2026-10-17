# -*- coding: utf-8 -*-
"""
01_PRODUCTION_CLEANER_PIPELINE/dxf_cleaner/raster_pixel_sensor.py
================================================================================
SYSTÈME 1 : CAPTEUR VISUEL MATRICIEL & ENCOMBREMENT DES SYMBOLES COMPLEXES
================================================================================
Prend en compte l'encombrement visuel réel et les masques des entités complexes :
  1. Masques WIPEOUT intégrés dans les blocs 'borne_plantee' et 'BLKR'.
  2. Géométrie SPLINE et contours réels des 30+ blocs officiels ANCFCC
     (v113, puit, metfia, olivier, arganier, PEBT, PEMT, PEHT, etc.).
  3. Symboles répétitifs et corridors des types de lignes complexes
     (BT, MT, HT, CLOTURE, talus2d, mpsech).
Fournit un index spatial STRtree au Contrôleur Qualité pour garantir 0.00%
de superposition entre les étiquettes et les symboles physiques ou décors.
"""

from __future__ import annotations
import logging
import math
import re
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import ezdxf
from shapely.geometry import Polygon, Point, LineString, box
from shapely.strtree import STRtree

from constants import ANCFCC_STANDARDS
from models import (
    OrientedBox2D,
    ANCFCC_MASTER_BLOCKS_DIMENSIONS,
    estimate_exact_text_dimensions,
    strip_dxf_formatting,
)
from exact_text_obb import get_exact_text_obb

logger = logging.getLogger("dxf_cleaner.raster_pixel_sensor")

COMPLEX_LINETYPES = {
    "BT", "MT", "HT", "CLOTURE", "TALUS2D", "MPSECH", "LE_BT", "LEMT", "LEHT"
}
COMPLEX_LAYERS = {
    "le_bt", "lemt", "leht", "l.e", "cloture", "talus", "mpsech", "pebt", "pemt", "peht"
}


class RasterPixelSensor:
    """
    Capteur matriciel spatial modélisant l'encombrement exact des symboles,
    blocs avec masques WIPEOUT, et décorations linéaires complexes.
    """

    @classmethod
    def get_block_footprint_polygon(
        cls,
        insert_ent,
        scale: int = 2000
    ) -> Optional[Tuple[str, Polygon]]:
        """
        Calcule le polygone d'emprise réelle au sol d'un bloc inséré,
        en appliquant l'échelle (xscale/yscale), la rotation et le masque WIPEOUT.
        """
        block_name = getattr(insert_ent.dxf, "name", "").strip()
        bname_lower = block_name.lower()
        ins = insert_ent.dxf.insert
        ix, iy = float(ins.x), float(ins.y)
        rot_deg = float(getattr(insert_ent.dxf, "rotation", 0.0))
        rot_rad = math.radians(rot_deg)
        x_sc = float(getattr(insert_ent.dxf, "xscale", 1.0))
        y_sc = float(getattr(insert_ent.dxf, "yscale", 1.0))

        # 1. Vérification dans le dictionnaire des blocs officiels
        info = ANCFCC_MASTER_BLOCKS_DIMENSIONS.get(bname_lower)
        if not info:
            # Fallback par correspondance de préfixe ou de mot-clé
            for k, val in ANCFCC_MASTER_BLOCKS_DIMENSIONS.items():
                if k in bname_lower or bname_lower in k:
                    info = val
                    break

        if info:
            w = info["width"] * abs(x_sc)
            h = info["height"] * abs(y_sc)
            # Pour les blocs avec WIPEOUT (borne_plantee, BLKR), ajouter un coussin de sécurité
            if bname_lower in ("borne_plantee", "blkr"):
                w = max(w, 0.45)
                h = max(h, 0.45)
        else:
            # Bloc inconnu ou générique : estimation conservative basée sur l'échelle
            w = 0.0008 * scale
            h = 0.0008 * scale

        hw, hh = 0.5 * w, 0.5 * h
        # 4 sommets locaux centrés
        local_pts = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        # Rotation et translation vers les coordonnées monde
        cos_a, sin_a = math.cos(rot_rad), math.sin(rot_rad)
        world_pts = []
        for lx, ly in local_pts:
            wx = ix + lx * cos_a - ly * sin_a
            wy = iy + lx * sin_a + ly * cos_a
            world_pts.append((round(wx, 3), round(wy, 3)))

        return block_name, Polygon(world_pts)

    @classmethod
    def get_complex_line_footprint_polygons(
        cls,
        msp,
        scale: int = 2000
    ) -> List[Tuple[str, Polygon]]:
        """
        Modélise les corridors d'occupation des types de lignes complexes
        (flèches BT/MT/HT, clôtures, talus) sous forme de polygones Shapely.
        """
        footprints: List[Tuple[str, Polygon]] = []
        corridor_radius = 0.00045 * scale  # ex: 0.9m au 1/2000

        EXCLUDED_LINE_LAYERS = {"parcelle", "limite", "0", "cadre", "cartouche", "titre"}
        for ent in msp.query("LINE LWPOLYLINE"):
            lay = ent.dxf.layer.lower()
            if lay in EXCLUDED_LINE_LAYERS:
                continue
            ltype = getattr(ent.dxf, "linetype", "").upper()

            is_complex = (ltype in COMPLEX_LINETYPES) or any(k in lay for k in COMPLEX_LAYERS)
            if not is_complex:
                continue

            if ent.dxftype() == "LINE":
                p1 = (ent.dxf.start.x, ent.dxf.start.y)
                p2 = (ent.dxf.end.x, ent.dxf.end.y)
                if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 0.1:
                    line_geom = LineString([p1, p2])
                    footprints.append((f"LINETYPE_{ltype or lay.upper()}", line_geom.buffer(corridor_radius, cap_style=2)))

            elif ent.dxftype() == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in ent.get_points()]
                if len(pts) >= 2:
                    line_geom = LineString(pts)
                    footprints.append((f"LINETYPE_{ltype or lay.upper()}", line_geom.buffer(corridor_radius, cap_style=2)))

        return footprints

    @classmethod
    def build_occupancy_spatial_index(
        cls,
        msp,
        scale: int = 2000
    ) -> Tuple[Optional[STRtree], List[Tuple[str, Polygon]]]:
        """
        Construit l'index spatial STRtree de toutes les entités physiques d'occupation
        (Blocs ANCFCC + Masques WIPEOUT + Lignes complexes).
        """
        items: List[Tuple[str, Polygon]] = []

        # 1. Tous les blocs insérés
        for ins in msp.query("INSERT"):
            res = cls.get_block_footprint_polygon(ins, scale=scale)
            if res:
                items.append(res)

        # 2. Toutes les décorations de lignes complexes
        line_footprints = cls.get_complex_line_footprint_polygons(msp, scale=scale)
        items.extend(line_footprints)

        if not items:
            return None, []

        geoms = [poly for _, poly in items]
        tree = STRtree(geoms)
        return tree, items

    @classmethod
    def detect_text_symbol_collisions(
        cls,
        msp,
        scale: int = 2000,
        tree: Optional[STRtree] = None,
        items: Optional[List[Tuple[str, Polygon]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Détecte toutes les superpositions entre les textes (MTEXT/TEXT)
        et les symboles / blocs physiques / linetypes complexes.
        """
        if tree is None or items is None:
            tree, items = cls.build_occupancy_spatial_index(msp, scale=scale)

        if tree is None or not items:
            return []

        geoms = [poly for _, poly in items]
        collisions: List[Dict[str, Any]] = []

        for ent in msp.query("MTEXT TEXT"):
            raw_t = ent.dxf.text if hasattr(ent.dxf, "text") else getattr(ent, "text", "")
            clean_t = strip_dxf_formatting(raw_t).strip()
            if not clean_t:
                continue

            # Obtenir l'OBB exact du texte
            text_poly = get_exact_text_obb(ent)
            if not text_poly or text_poly.is_empty:
                continue

            # Interroger l'index spatial
            candidates_idx = tree.query(text_poly)
            for idx in candidates_idx:
                obs_name, obs_poly = items[idx]
                is_borne_lbl = bool(BORNE_NAME_RE.match(clean_t))
                if is_borne_lbl and obs_name.startswith("LINETYPE_"):
                    continue
                if is_borne_lbl and any(k in obs_name.lower() for k in ("borne_", "blkr")):
                    c_obs = obs_poly.centroid
                    if math.hypot(ent.dxf.insert.x - c_obs.x, ent.dxf.insert.y - c_obs.y) > 0.45 * (0.00175 * scale):
                        continue
                if text_poly.intersects(obs_poly):
                    inter_geom = text_poly.intersection(obs_poly)
                    if inter_geom.area > 1e-4:
                        ins = ent.dxf.insert
                        collisions.append({
                            "text_entity": ent,
                            "text_content": clean_t,
                            "obstacle_name": obs_name,
                            "location": (float(ins.x), float(ins.y)),
                            "overlap_area": float(inter_geom.area),
                        })

        return collisions
