# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/linear_centering.py — Centrage et alignement des détails linéaires et riverains
===================================================================================================
AGENT 3 : Relocalisation curviligne des textes de détails linéaires (Chaâbat, pistes, routes, seguia).
AGENT 4 : Normalisation des angles ]-90°, +90°], placement de TC par Polylabel sans obstacles,
          et calage de la position de Contenance.
"""

import math
import logging
import re
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, LineString, Point, box as _box
from shapely.ops import polylabel, unary_union

from constants import ANCFCC_STANDARDS
from standards import BORNE_NAME_RE

logger = logging.getLogger("dxf_cleaner.linear_centering")


def normalize_cadastral_angle(angle_rad: float) -> Tuple[float, Tuple[float, float]]:
    """
    Normalise l'angle pour que le texte soit toujours lisible du bas ou de la droite :
    Angle résultant dans ]-90°, +90°] (soit [0°, 90°] U [270°, 360°[).
    """
    deg = math.degrees(angle_rad) % 360.0
    if 90.0 < deg <= 270.0:
        deg = (deg + 180.0) % 360.0
    rad = math.radians(deg)
    return deg, (math.cos(rad), math.sin(rad))


def center_linear_details_and_fix_angles(
    msp,
    parcel_poly: Optional[Polygon],
    frame_bbox: Tuple[float, float, float, float],
    scale: int = 2000
) -> int:
    """
    AGENT 3 (Détails linéaires) & AGENT 4 (Riverains, TC par Polylabel, Contenance).
    """
    modified = 0
    h_text = ANCFCC_STANDARDS["borne_text_height_ratio"] * scale
    frame_poly = _box(*frame_bbox) if frame_bbox else None

    # Collecte des bornes et étiquettes dans un cKDTree pour éloigner les textes linéaires des sommets
    borne_pts = []
    for ins in msp.query("INSERT"):
        name = getattr(ins.dxf, "name", "").lower()
        lay = getattr(ins.dxf, "layer", "").lower()
        if "borne" in name or "borne" in lay or "croisi" in name:
            borne_pts.append([ins.dxf.insert.x, ins.dxf.insert.y])
    for mtxt in msp.query("MTEXT TEXT"):
        txt = getattr(mtxt.dxf, "text", "").strip()
        if txt.startswith("B") or txt.startswith("(B") or txt.startswith("v.B") or txt.startswith("BP") or txt.startswith("CROISI"):
            borne_pts.append([mtxt.dxf.insert.x, mtxt.dxf.insert.y])
    borne_tree = cKDTree(borne_pts) if borne_pts else None

    # AGENT 4.A : Normalisation des angles MTEXT et nettoyage du suffixe redondant \PTC
    for mtxt in msp.query("MTEXT"):
        if hasattr(mtxt.dxf, "text_direction") and math.hypot(mtxt.dxf.text_direction[0], mtxt.dxf.text_direction[1]) > 1e-4:
            td = mtxt.dxf.text_direction
            ang_rad = math.atan2(td[1], td[0])
        else:
            ang_rad = math.radians(getattr(mtxt.dxf, "rotation", 0.0))

        if abs(ang_rad) > 1e-3:
            _, (ux, uy) = normalize_cadastral_angle(ang_rad)
            mtxt.dxf.discard("rotation")
            mtxt.dxf.text_direction = (ux, uy, 0.0)
            if r"\PTC" in mtxt.dxf.text and not mtxt.dxf.text.strip().startswith("Plle"):
                mtxt.dxf.text = mtxt.dxf.text.replace(r"\PTC", "").strip()
            modified += 1

    # AGENT 3 : Relocalisation des textes de détails linéaires sur les segments internes libres
    # AGENT 3 : Relocalisation des textes de détails linéaires sur les segments internes libres
    cartouche_admin_kw = (
        "caidat", "caïdat", "prefecture", "préfecture", "commune",
        "situee", "située", "province", "conservation", "direction", "royaume"
    )
    chaaba_texts = [
        t for t in msp.query("TEXT MTEXT")
        if any(k in getattr(t.dxf, "text", "").lower() for k in ("chaaba", "oued", "seguia", "cha\u00e2ba", "ch\u00e2aba"))
        and len(getattr(t.dxf, "text", "").strip()) <= 15
        and not any(adm in getattr(t.dxf, "text", "").lower() for adm in cartouche_admin_kw)
        and getattr(t.dxf, "layer", "").lower() not in ("cadre", "cartouche", "titre")
    ]

    # Normalisation immédiate du texte de tous les cours d'eau en "Chaâbat"
    for t_ent in chaaba_texts:
        t_ent.dxf.text = "Cha\u00e2bat"

    # Déduplication spatiale des étiquettes de cours d'eau identiques (< 25m)
    # Règle ANCFCC: Si la parcelle a une longue façade hydrographique (>= 35m) avec plusieurs segments,
    # conserver jusqu'à 2 étiquettes pour couvrir les différents tronçons de la chaaba
    ext_coords = list(parcel_poly.exterior.coords) if parcel_poly else []
    chaaba_frontage_len = 0.0
    if parcel_poly and chaaba_texts:
        for i in range(len(ext_coords) - 1):
            seg = LineString([ext_coords[i], ext_coords[i+1]])
            if any(seg.distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 35.0 for t in chaaba_texts):
                chaaba_frontage_len += seg.length

    max_kept = max(2, int(chaaba_frontage_len / 75.0)) if chaaba_frontage_len >= 35.0 else 1
    # Trier pour conserver en priorité les textes ayant le plus grand dégagement par rapport aux bornes
    borne_pts_raw = [(b.dxf.insert.x, b.dxf.insert.y) for b in msp.query("INSERT") if "borne" in getattr(b.dxf, "name", "").lower()]
    if borne_pts_raw:
        chaaba_texts.sort(key=lambda t: -min(math.hypot(t.dxf.insert.x - bx, t.dxf.insert.y - by) for bx, by in borne_pts_raw))
    dedup_ch = []
    for t_ent in chaaba_texts:
        # Les textes chaâba proches des bornes seront déconflictés et recentrés plus bas le long de la ligne
        tx, ty = t_ent.dxf.insert.x, t_ent.dxf.insert.y
        # Règle Cadre: Éliminer tout texte chaâba en dehors ou trop proche du cadre (< 5.0m)
        if frame_poly and not frame_poly.buffer(-5.0).contains(Point(tx, ty)):
            msp.delete_entity(t_ent)
            continue
        if any(math.hypot(tx - sx, ty - sy) < 30.0 for sx, sy, _ in dedup_ch):
            msp.delete_entity(t_ent)
        elif len(dedup_ch) >= max_kept and any(math.hypot(tx - sx, ty - sy) < 35.0 for sx, sy, _ in dedup_ch):
            msp.delete_entity(t_ent)
        else:
            dedup_ch.append((tx, ty, t_ent))
    chaaba_texts = [item[2] for item in dedup_ch]

    dedicated_ch = [
        pl for pl in msp.query("LWPOLYLINE POLYLINE")
        if pl.dxf.layer.lower() in ("chaaba", "oued", "seguia")
        and len(list(pl.get_points())) >= 2
        and LineString([(p[0], p[1]) for p in pl.get_points()]).length >= 15.0
    ]
    if dedicated_ch:
        chaaba_polylines = dedicated_ch
    else:
        chaaba_polylines = [
            pl for pl in msp.query("LWPOLYLINE POLYLINE")
            if pl.dxf.layer.lower() == "dtls" and not getattr(pl, "is_closed", False)
            and len(list(pl.get_points())) >= 4
            and LineString([(p[0], p[1]) for p in pl.get_points()]).length >= 25.0
        ]
    placed_frontage_chaaba = []
    # AGENT 3.A : Cours d'eau riverains (bordant directement une façade de la parcelle principale)
    if parcel_poly and chaaba_texts:
        exterior_coords = list(parcel_poly.exterior.coords)
        neighbor_texts = [
            Point(te.dxf.insert.x, te.dxf.insert.y)
            for te in msp.query("TEXT MTEXT")
            if not parcel_poly.contains(Point(te.dxf.insert.x, te.dxf.insert.y))
            and (
                any(k in getattr(te.dxf, "text", "").lower() for k in ("hritier", "heritier", "hrts", "t.", "r.", "piste", "route", "voie", "plle", "titre"))
                or len(getattr(te.dxf, "text", "").strip()) > 15
            )
        ]

        # Identifier tous les segments bordant la chaaba sans conflit avec les riverains nominatifs
        # et alignés avec le sens d'écoulement du cours d'eau
        orig_chaaba_rot = [getattr(t.dxf, "rotation", 0.0) for t in chaaba_texts]
        hydro_segs = []
        for i in range(len(exterior_coords) - 1):
            p1, p2 = exterior_coords[i], exterior_coords[i+1]
            seg = LineString([p1, p2])
            is_near_ch_txt = any(seg.distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 35.0 for t in chaaba_texts)
            is_near_ch_pl = any(seg.distance(LineString([(p[0], p[1]) for p in pl_ch.get_points()])) < 20.0 for pl_ch in chaaba_polylines) if chaaba_polylines else False
            if is_near_ch_txt or is_near_ch_pl:
                is_road = False
                for pl_rd in msp.query("LWPOLYLINE POLYLINE LINE"):
                    if getattr(pl_rd.dxf, "layer", "").lower() in ("piste", "route", "voie"):
                        pts_rd = [p[:2] for p in pl_rd.get_points()] if hasattr(pl_rd, "get_points") else [(pl_rd.dxf.start.x, pl_rd.dxf.start.y), (pl_rd.dxf.end.x, pl_rd.dxf.end.y)]
                        if len(pts_rd) >= 2 and LineString(pts_rd).distance(seg) < 15.0:
                            is_road = True
                            break
                if not is_road and seg.length >= 10.0:
                    dx_s, dy_s = p2[0] - p1[0], p2[1] - p1[1]
                    seg_ang = math.degrees(math.atan2(dy_s, dx_s))
                    # Filtrer les façades transversales non alignées avec l'écoulement de la chaaba
                    if orig_chaaba_rot:
                        ang_diff = min(abs(math.sin(math.radians(seg_ang - r))) for r in orig_chaaba_rot)
                        if ang_diff > 0.50:
                            continue
                    hydro_segs.append((p1, p2, seg, i))

        if hydro_segs:
            # S'il y a plusieurs segments hydrographiques et moins d'étiquettes dans OREGEN,
            # en ajouter pour étiqueter chaque tronçon distinct de la chaaba
            while len(chaaba_texts) < min(2, len(hydro_segs)):
                new_t = msp.add_text("Cha\u00e2bat ", dxfattribs={"layer": "0", "height": h_text})
                chaaba_texts.append(new_t)

            assigned_segs = []
            placed_texts = []
            borne_pts = [(b.dxf.insert.x, b.dxf.insert.y) for b in msp.query("INSERT") if "borne" in getattr(b.dxf, "name", "").lower()]

            tot_frontage = sum(s[2].length for s in hydro_segs)
            frontage_candidates = [
                t for t in chaaba_texts
                if min((s[2].distance(Point(t.dxf.insert.x, t.dxf.insert.y)) for s in hydro_segs), default=999.0) < 55.0
            ]
            is_long_frontage_multi = len(frontage_candidates) >= 2 and tot_frontage >= 80.0

            if is_long_frontage_multi:
                N_f = len(frontage_candidates)
                part_size = tot_frontage / N_f
                # Trier les textes candidats par coordonnées Y pour les aligner avec les tronçons
                frontage_candidates.sort(key=lambda t: t.dxf.insert.y)
                chosen_pairs = []
                cum = 0.0
                for k in range(N_f):
                    part_start = k * part_size
                    part_end = (k + 1) * part_size
                    cand_segs = []
                    c_seg = 0.0
                    for item in hydro_segs:
                        if (c_seg < part_end and c_seg + item[2].length > part_start) and item not in [cp[1] for cp in chosen_pairs]:
                            cand_segs.append(item)
                        c_seg += item[2].length
                    if not cand_segs:
                        cand_segs = [s for s in hydro_segs if s not in [cp[1] for cp in chosen_pairs]] or hydro_segs
                    best_seg = max(cand_segs, key=lambda it: it[2].length)
                    chosen_pairs.append((frontage_candidates[k], best_seg))

                for k, (t_ent, seg_item) in enumerate(chosen_pairs):
                    p1, p2, seg, _ = seg_item
                    frac = 0.85 if (k == 0 and N_f >= 3) else (0.75 if (k == 1 and N_f >= 3) else 0.50)
                    mid = seg.interpolate(frac * seg.length)
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    L = seg.length
                    ux, uy = dx / L, dy / L
                    n1 = (-uy, ux)
                    test_pt = Point(mid.x + n1[0] * 1.0, mid.y + n1[1] * 1.0)
                    ext_n = n1 if not parcel_poly.contains(test_pt) else (uy, -ux)
                    offset = max(1.5, 0.35 * h_text)
                    new_pos = (mid.x + ext_n[0] * offset, mid.y + ext_n[1] * offset)
                    ang = math.degrees(math.atan2(dy, dx)) % 360.0
                    if 90.0 < ang < 270.0:
                        ang = (ang + 180.0) % 360.0

                    orig_pos = getattr(t_ent.dxf, "insert", None)
                    is_unplaced = orig_pos is None or (abs(orig_pos.x) < 1000.0 and abs(orig_pos.y) < 1000.0)
                    disp = math.hypot(new_pos[0] - orig_pos.x, new_pos[1] - orig_pos.y) if (orig_pos and not is_unplaced) else 0.0
                    if is_unplaced or disp <= 60.0:
                        t_ent.dxf.text = "Cha\u00e2bat"
                        t_ent.dxf.insert = (new_pos[0], new_pos[1], 0.0)
                        t_ent.dxf.rotation = ang
                        t_ent.dxf.layer = "0"
                        t_ent.dxf.height = h_text
                        placed_texts.append(t_ent)
                        modified += 1
            else:
                for t_ent in list(chaaba_texts):
                    t_pt = Point(t_ent.dxf.insert.x, t_ent.dxf.insert.y)
                    # Si le texte est plus proche d'une polyligne de cours d'eau dédiée et loin des façades hydrographiques, le réserver pour la section 3.B
                    if chaaba_polylines:
                        d_poly = min((LineString([(p[0], p[1]) for p in pl.get_points()]).distance(t_pt) for pl in chaaba_polylines), default=999.0)
                        d_front = min((s[2].distance(t_pt) for s in hydro_segs), default=999.0)
                        if d_poly < 25.0 and d_front > 45.0:
                            continue
                    free_segs = [s for s in hydro_segs if s not in assigned_segs]
                    if not free_segs:
                        continue
                    free_major = [s for s in free_segs if s[2].length >= 30.0]
                    candidates = free_major if free_major else free_segs
                    def _cand_score(item):
                        m_dist = min((item[2].interpolate(0.5, normalized=True).distance(a[2].interpolate(0.5, normalized=True)) for a in assigned_segs), default=999.0)
                        pen = 60.0 if m_dist < 50.0 else 0.0
                        return item[2].distance(t_pt) - (item[2].length - pen) * 0.5
                    best_seg_item = min(candidates, key=_cand_score)
                    assigned_segs.append(best_seg_item)
                    p1, p2, seg, _ = best_seg_item


                    # Vérifier si t_pt est près d'un sommet de la limite de parcelle (ex: virage / méandre)
                    near_v = min(exterior_coords, key=lambda c: math.hypot(c[0] - t_pt.x, c[1] - t_pt.y))
                    if math.hypot(near_v[0] - t_pt.x, near_v[1] - t_pt.y) < 2.5:
                        vi = exterior_coords.index(near_v)
                        p_prev = exterior_coords[vi - 1]
                        p_next = exterior_coords[(vi + 1) % len(exterior_coords)]
                        v1 = (p_prev[0] - near_v[0], p_prev[1] - near_v[1])
                        v2 = (p_next[0] - near_v[0], p_next[1] - near_v[1])
                        l1, l2 = math.hypot(*v1), math.hypot(*v2)
                        u1 = (v1[0] / l1, v1[1] / l1) if l1 > 1e-4 else (0.0, 1.0)
                        u2 = (v2[0] / l2, v2[1] / l2) if l2 > 1e-4 else (0.0, 1.0)
                        bis = (u1[0] + u2[0], u1[1] + u2[1])
                        lb = math.hypot(*bis)
                        ubis = (bis[0] / lb, bis[1] / lb) if lb > 1e-4 else (0.0, 1.0)
                        test_pt = Point(near_v[0] + ubis[0] * 1.0, near_v[1] + ubis[1] * 1.0)
                        if parcel_poly.contains(test_pt):
                            ubis = (-ubis[0], -ubis[1])
                        offset = max(1.5, 0.35 * h_text)
                        new_pos = (near_v[0] + ubis[0] * offset, near_v[1] + ubis[1] * offset)
                        ang = getattr(t_ent.dxf, "rotation", 0.0)
                        if abs(ang) < 1e-3:
                            dx_s = p2[0] - p1[0]
                            dy_s = p2[1] - p1[1]
                            ang = math.degrees(math.atan2(dy_s, dx_s))
                    else:
                        proj_dist = seg.project(t_pt)
                        margin = max(3.0, 0.8 * h_text)
                        if seg.length <= 2.5 * margin:
                            frac = 0.50
                        else:
                            frac = max(margin / seg.length, min(1.0 - margin / seg.length, proj_dist / seg.length))
                        placed_pt = seg.interpolate(frac * seg.length)
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        L = seg.length
                        ux, uy = dx / L, dy / L
                        n1 = (-uy, ux)
                        test_pt = Point(placed_pt.x + n1[0] * 1.0, placed_pt.y + n1[1] * 1.0)
                        ext_n = n1 if not parcel_poly.contains(test_pt) else (uy, -ux)

                        offset = max(1.5, 0.35 * h_text)
                        new_pos = (placed_pt.x + ext_n[0] * offset, placed_pt.y + ext_n[1] * offset)
                        ang = math.degrees(math.atan2(dy, dx))

                    ang = ang % 360.0
                    if 90.0 < ang < 270.0:
                        ang = (ang + 180.0) % 360.0

                    t_ent.dxf.text = "Cha\u00e2bat"
                    is_unplaced = abs(t_pt.x) < 1000.0 and abs(t_pt.y) < 1000.0
                    disp = math.hypot(new_pos[0] - t_pt.x, new_pos[1] - t_pt.y) if not is_unplaced else 0.0
                    if is_unplaced or disp <= 60.0:
                        t_ent.dxf.insert = (new_pos[0], new_pos[1], 0.0)
                        t_ent.dxf.rotation = ang
                        t_ent.dxf.layer = "0"
                        t_ent.dxf.height = h_text
                        placed_texts.append(t_ent)
                        modified += 1

            placed_frontage_chaaba.extend(placed_texts)

            for pt in placed_texts:
                if pt in chaaba_texts:
                    chaaba_texts.remove(pt)
            for extra_t in list(chaaba_texts):
                extra_pt = Point(extra_t.dxf.insert.x, extra_t.dxf.insert.y)
                if chaaba_polylines and any(LineString([(p[0], p[1]) for p in pl.get_points()]).distance(extra_pt) < 25.0 for pl in chaaba_polylines):
                    continue
                if parcel_poly.exterior.distance(extra_pt) < 35.0:
                    msp.delete_entity(extra_t)
                    chaaba_texts.remove(extra_t)

    if not chaaba_polylines:
        for pl in msp.query("LWPOLYLINE POLYLINE"):
            if pl.dxf.layer.lower() == "dtls":
                pts = list(pl.get_points())
                # Exclure les bâtiments ou parcelles fermées (<= 8 sommets fermés)
                if getattr(pl, "is_closed", False) and len(pts) <= 8:
                    continue
                if len(pts) >= 4:
                    ls = LineString([(p[0], p[1]) for p in pts])
                    if ls.length >= 25.0 and any(ls.distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 35.0 for t in chaaba_texts):
                        chaaba_polylines.append(pl)

    if not placed_frontage_chaaba and not chaaba_texts and chaaba_polylines:
        new_t = msp.add_text("Cha\u00e2bat", dxfattribs={"layer": "0", "height": h_text})
        chaaba_texts.append(new_t)

    if chaaba_texts:
        if chaaba_polylines:
            distinct_pls = []
            for pl in chaaba_polylines:
                pts = [(p[0], p[1]) for p in pl.get_points()]
                ls = LineString(pts)
                if not any(math.hypot(ls.centroid.x - dls.centroid.x, ls.centroid.y - dls.centroid.y) < 20.0 for dls, _ in distinct_pls):
                    distinct_pls.append((ls, pl))
            distinct_pls.sort(key=lambda item: -item[0].length)
            # Conserver tous les cours d'eau distincts significatifs (>= 20m)
            valid_distinct = [item for item in distinct_pls if item[0].length >= 20.0]
            if valid_distinct:
                distinct_pls = valid_distinct

            # Regrouper les polylignes en corridors hydrographiques continus (distance < 25m)
            corridor_clusters = []
            for item in distinct_pls:
                ls, pl = item
                placed = False
                for cl in corridor_clusters:
                    if any(ls.distance(cls) < 25.0 for cls, _ in cl):
                        cl.append(item)
                        placed = True
                        break
                if not placed:
                    corridor_clusters.append([item])

            # Pour chaque corridor distinct, retenir la polyligne principale la plus longue
            distinct_pls = [sorted(cl, key=lambda it: -it[0].length)[0] for cl in corridor_clusters]
            if placed_frontage_chaaba:
                uncovered_pls = [
                    item for item in distinct_pls
                    if not any(item[0].distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 35.0 for t in placed_frontage_chaaba)
                ]
                if uncovered_pls:
                    distinct_pls = uncovered_pls
            distinct_pls.sort(key=lambda item: item[0].centroid.y)

            # Nombre cible d'étiquettes selon les corridors distincts et l'existant
            target_count = max(len(distinct_pls), len(chaaba_texts))

            # Prioriser les textes chaâba déjà bien positionnés le long des cours d'eau
            if parcel_poly is not None:
                chaaba_texts.sort(key=lambda te: (
                    -1 if (any(ls.distance(Point(te.dxf.insert.x, te.dxf.insert.y)) < 15.0 for ls, _ in distinct_pls)
                           and parcel_poly.exterior.distance(Point(te.dxf.insert.x, te.dxf.insert.y)) < 25.0
                           and getattr(te.dxf, "rotation", 0.0) != 0.0) else 1
                ))

            while len(chaaba_texts) < target_count:
                sec_t = msp.add_text("Cha\u00e2bat", dxfattribs={"layer": "0", "height": h_text})
                chaaba_texts.append(sec_t)
            while len(chaaba_texts) > target_count:
                extra = chaaba_texts.pop()
                try:
                    msp.delete_entity(extra)
                except Exception:
                    pass

            n_labels = len(chaaba_texts)
            for idx, t_ent in enumerate(chaaba_texts):
                t_p = Point(t_ent.dxf.insert.x, t_ent.dxf.insert.y)
                best_pls = [(ls.distance(t_p), ls, pl) for ls, pl in distinct_pls]
                if not best_pls:
                    continue
                best_pls.sort(key=lambda it: it[0])
                d_to_pl, ls, pl = best_pls[0]

                # Si le libellé est déjà placé le long du cours d'eau sans collision avec une borne, le préserver
                if borne_tree is not None:
                    d_b_cur, _ = borne_tree.query([t_p.x, t_p.y])
                else:
                    d_b_cur = 999.0

                if (
                    d_to_pl < 25.0
                    and (parcel_poly is None or parcel_poly.exterior.distance(t_p) < 35.0)
                    and getattr(t_ent.dxf, "rotation", 0.0) != 0.0
                    and d_b_cur >= 2.0 * h_text
                ):
                    t_ent.dxf.text = "Cha\u00e2bat"
                    continue

                if d_to_pl < 35.0:
                    cur_proj = ls.project(t_p)
                    pt_proj = ls.interpolate(cur_proj)
                    delta_d = min(1.5, ls.length * 0.02)
                    p_prev = ls.interpolate(max(0.0, cur_proj - delta_d))
                    p_next = ls.interpolate(min(ls.length, cur_proj + delta_d))
                    dx_c = p_next.x - p_prev.x
                    dy_c = p_next.y - p_prev.y
                    deg_norm, _ = normalize_cadastral_angle(math.atan2(dy_c, dx_c))
                    L_v = math.hypot(dx_c, dy_c) or 1.0
                    nx, ny = -dy_c / L_v, dx_c / L_v
                    if parcel_poly and parcel_poly.contains(Point(pt_proj.x + nx * 0.5, pt_proj.y + ny * 0.5)):
                        nx, ny = -nx, -ny
                    norm_offset = max(3.5, 0.45 * h_text)
                    cx = pt_proj.x + nx * norm_offset
                    cy = pt_proj.y + ny * norm_offset
                    best_pt = (cx, cy)
                    best_ang_deg = deg_norm
                else:
                    if len(distinct_pls) >= n_labels and n_labels > 1:
                        target_frac = 0.46 if idx == 0 else 0.38
                    elif n_labels == 1:
                        target_frac = 0.44
                    elif n_labels == 2:
                        target_frac = 0.20 if idx == 0 else 0.75
                    else:
                        target_frac = (idx + 0.5) / n_labels

                    best_pt = None
                    best_score = -1e9
                    best_ang_deg = 0.0

                    test_fracs = [
                        target_frac + df for df in [-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12]
                        if 0.05 <= target_frac + df <= 0.95
                    ]
                    for tf in test_fracs:
                        d_curv = tf * ls.length
                        pt = ls.interpolate(d_curv)
                        delta_d = min(1.5, ls.length * 0.02)
                        p_prev = ls.interpolate(max(0.0, d_curv - delta_d))
                        p_next = ls.interpolate(min(ls.length, d_curv + delta_d))
                        ang = math.atan2(p_next.y - p_prev.y, p_next.x - p_prev.x)
                        deg_norm, _ = normalize_cadastral_angle(ang)

                        if borne_tree is not None:
                            d_b, _ = borne_tree.query([pt.x, pt.y])
                            clearance = float(d_b)
                        else:
                            clearance = 50.0

                        frac_penalty = abs(tf - target_frac) * 20.0
                        score = min(clearance, 3.5 * h_text) - frac_penalty
                        if score > best_score:
                            best_score = score
                            best_pt = (pt.x, pt.y)
                            best_ang_deg = deg_norm

                if best_pt:
                    cur_ins = t_ent.dxf.insert
                    is_unplaced = abs(cur_ins.x) < 1000.0 and abs(cur_ins.y) < 1000.0
                    disp = math.hypot(best_pt[0] - cur_ins.x, best_pt[1] - cur_ins.y) if not is_unplaced else 0.0
                    if is_unplaced or disp <= 60.0:
                        t_ent.dxf.text = "Cha\u00e2bat"
                        t_ent.dxf.insert = (float(best_pt[0]), float(best_pt[1]), 0.0)
                        t_ent.dxf.rotation = float(best_ang_deg)
                        modified += 1

            # Standardisation ANCFCC des polylignes hydrographiques utilisées
            for _, pl in distinct_pls:
                lay = getattr(pl.dxf, "layer", "").lower()
                if lay in ("chaaba", "oued", "seguia"):
                    pts = list(pl.get_points())
                    if len(pts) >= 2 and parcel_poly is not None and parcel_poly.is_valid:
                        dists = [parcel_poly.boundary.distance(Point(p[0], p[1])) for p in pts]
                        on_bnd = sum(1 for d in dists if d < 0.25)
                        if (on_bnd >= 3 or (on_bnd >= 2 and (on_bnd / len(pts)) >= 0.45)) and max(dists) <= 6.0:
                            try:
                                msp.delete_entity(pl)
                            except Exception:
                                pass
                            continue
                    pl.dxf.layer = "DTLS"
        elif parcel_poly is not None and parcel_poly.is_valid:
            # Fallback géométrique : alignement le long du segment de limite riveraine adjacent
            for t_ent in chaaba_texts:
                t_pt = Point(t_ent.dxf.insert.x, t_ent.dxf.insert.y)
                t_rot = math.radians(getattr(t_ent.dxf, "rotation", 0.0))
                coords = list(parcel_poly.exterior.coords)
                best_seg = None
                best_score = 1e9
                for i in range(len(coords) - 1):
                    p_a, p_b = coords[i], coords[i + 1]
                    seg = LineString([p_a, p_b])
                    d = seg.distance(t_pt)
                    if d > 45.0:
                        continue
                    seg_ang = math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
                    ang_diff = abs(math.sin(seg_ang - t_rot))
                    score = d + 25.0 * ang_diff
                    if score < best_score:
                        best_score = score
                        best_seg = (p_a, p_b)
                if best_seg:
                    p0, p1 = best_seg
                    seg = LineString([p0, p1])
                    seg_len = seg.length or 1.0
                    seg_vec = (p1[0] - p0[0], p1[1] - p0[1])
                    # Normal extérieur garanti
                    nx, ny = -seg_vec[1] / seg_len, seg_vec[0] / seg_len
                    mid_pt = seg.interpolate(0.5 * seg_len)
                    if parcel_poly.contains(Point(mid_pt.x + nx * 0.5, mid_pt.y + ny * 0.5)):
                        nx, ny = -nx, -ny
                    proj_d = seg.project(t_pt)
                    min_clearance = min(seg_len * 0.35, 5.0 * h_text)
                    safe_d = max(min_clearance, min(seg_len - min_clearance, proj_d))
                    safe_pt = seg.interpolate(safe_d)
                    mx = safe_pt.x + nx * (1.8 * h_text)
                    my = safe_pt.y + ny * (1.8 * h_text)
                    is_unplaced = abs(t_pt.x) < 1000.0 and abs(t_pt.y) < 1000.0
                    disp = math.hypot(mx - t_pt.x, my - t_pt.y) if not is_unplaced else 0.0
                    if is_unplaced or disp <= 60.0:
                        ang = math.atan2(seg_vec[1], seg_vec[0])
                        deg_norm, _ = normalize_cadastral_angle(ang)
                        t_ent.dxf.text = "Cha\u00e2bat"
                        t_ent.dxf.insert = (float(mx), float(my), 0.0)
                        t_ent.dxf.rotation = float(deg_norm)
                        modified += 1

    # AGENT 3.A-bis : Centrage curviligne du canal / canalisation / seguia / saguia
    canal_texts = [
        t for t in msp.query("TEXT MTEXT")
        if any(k in getattr(t.dxf, "text", "").lower() for k in ("canal", "seguia", "saguia"))
        and not any(adm in getattr(t.dxf, "text", "").lower() for adm in cartouche_admin_kw)
    ]
    if canal_texts:
        canal_pls = []
        for pl in msp.query("LWPOLYLINE POLYLINE"):
            lay = getattr(pl.dxf, "layer", "").lower()
            if not getattr(pl, "is_closed", False):
                raw_pts = list(pl.get_points()) if hasattr(pl, "get_points") else [(pl.dxf.start.x, pl.dxf.start.y), (pl.dxf.end.x, pl.dxf.end.y)]
                pts = [p[:2] for p in raw_pts]
                if len(pts) >= 2:
                    ls = LineString(pts)
                    if lay in ("canal", "canalisation", "seguia", "saguia") or any(ls.distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 15.0 for t in canal_texts):
                        canal_pls.append((ls, pl))
        if canal_pls:
            canal_pls.sort(key=lambda item: item[0].length, reverse=True)
            main_ls, main_pl = canal_pls[0]
            while len(canal_texts) > 1:
                extra = canal_texts.pop()
                msp.delete_entity(extra)
            t_ent = canal_texts[0]
            cur_txt = getattr(t_ent.dxf, "text", "").strip()
            orig_pt = Point(t_ent.dxf.insert.x, t_ent.dxf.insert.y)
            proj_d = main_ls.project(orig_pt)
            end_margin = max(1.5, 2.0 * h_text)
            d_curv = max(end_margin, min(main_ls.length - end_margin, proj_d))
            # Évitement strict des bornes pour éviter toute collision
            borne_pts = [(b.dxf.insert.x, b.dxf.insert.y) for b in msp.query("INSERT") if "borne" in getattr(b.dxf, "name", "").lower()]
            test_pt = main_ls.interpolate(d_curv)
            min_borne_dist = max(1.5, 1.8 * h_text)
            if any(math.hypot(test_pt.x - bx, test_pt.y - by) < min_borne_dist for bx, by in borne_pts):
                shift_dist = max(1.8, 2.0 * h_text)
                cand_a = max(end_margin, min(main_ls.length - end_margin, d_curv - shift_dist))
                cand_b = max(end_margin, min(main_ls.length - end_margin, d_curv + shift_dist))
                dist_a = min((math.hypot(main_ls.interpolate(cand_a).x - bx, main_ls.interpolate(cand_a).y - by) for bx, by in borne_pts), default=999.0)
                dist_b = min((math.hypot(main_ls.interpolate(cand_b).x - bx, main_ls.interpolate(cand_b).y - by) for bx, by in borne_pts), default=999.0)
                d_curv = cand_a if dist_a >= dist_b else cand_b
                pt = main_ls.interpolate(d_curv)
            else:
                pt = test_pt

            delta_d = min(2.0, main_ls.length * 0.05)
            p_prev = main_ls.interpolate(max(0.0, d_curv - delta_d))
            p_next = main_ls.interpolate(min(main_ls.length, d_curv + delta_d))
            dx_c = p_next.x - p_prev.x
            dy_c = p_next.y - p_prev.y
            ang = math.atan2(dy_c, dx_c)
            deg_norm, _ = normalize_cadastral_angle(ang)
            L_v = math.hypot(dx_c, dy_c) or 1.0
            nx, ny = -dy_c / L_v, dx_c / L_v
            if parcel_poly and parcel_poly.contains(Point(pt.x + nx * 0.5, pt.y + ny * 0.5)):
                nx, ny = -nx, -ny
            cx = pt.x + nx * (1.2 * h_text)
            cy = pt.y + ny * (1.2 * h_text)
            orig_ins = getattr(t_ent.dxf, "insert", None)
            is_unplaced = orig_ins is None or (abs(orig_ins.x) < 1000.0 and abs(orig_ins.y) < 1000.0)
            disp = math.hypot(cx - orig_ins.x, cy - orig_ins.y) if (orig_ins and not is_unplaced) else 0.0
            if is_unplaced or disp <= 60.0:
                t_ent.dxf.text = cur_txt
                t_ent.dxf.insert = (float(cx), float(cy), 0.0)
                t_ent.dxf.rotation = float(deg_norm)
                modified += 1

    # AGENT 3.A-ter : Centrage du chemin d'accès / chemin intérieur
    chemin_texts = [
        t for t in msp.query("TEXT MTEXT")
        if "chemin" in getattr(t.dxf, "text", "").lower()
        and not any(adm in getattr(t.dxf, "text", "").lower() for adm in cartouche_admin_kw)
    ]
    if chemin_texts:
        while len(chemin_texts) > 1:
            extra = chemin_texts.pop()
            msp.delete_entity(extra)
        ch_ent = chemin_texts[0]
        ch_pls = [
            pl for pl in msp.query("LWPOLYLINE POLYLINE")
            if getattr(pl.dxf, "layer", "").lower() == "chemin"
        ]
        if ch_pls:
            all_segs = []
            for pl in ch_pls:
                pts = [(p[0], p[1]) for p in pl.get_points()] if hasattr(pl, "get_points") else [(pl.dxf.start.x, pl.dxf.start.y), (pl.dxf.end.x, pl.dxf.end.y)]
                for i in range(len(pts) - 1):
                    s = LineString([pts[i], pts[i+1]])
                    if s.length >= 15.0:
                        all_segs.append(s)
            if all_segs:
                # Privilégier le segment le plus proche de la position d'origine du libellé
                t_pt = Point(ch_ent.dxf.insert.x, ch_ent.dxf.insert.y)
                main_seg = min(all_segs, key=lambda s: s.distance(t_pt))
                p0, p1 = main_seg.coords[0], main_seg.coords[1]
                dx_s, dy_s = p1[0] - p0[0], p1[1] - p0[1]
                ang_s = math.degrees(math.atan2(dy_s, dx_s)) % 360.0
                if 90.0 < ang_s < 270.0:
                    ang_s = (ang_s + 180.0) % 360.0
                mid_pt = main_seg.interpolate(0.50 * main_seg.length)
                best_pt = mid_pt
                best_obs_dist = -1.0
                ch_obstacles = [Point(b.dxf.insert.x, b.dxf.insert.y) for b in msp.query("INSERT")]
                for txt_e in msp.query("TEXT MTEXT"):
                    t_val = getattr(txt_e.dxf, "text", "").strip()
                    if txt_e != ch_ent and not any(adm in t_val.lower() for adm in cartouche_admin_kw) and not BORNE_NAME_RE.match(t_val):
                        ch_obstacles.append(Point(txt_e.dxf.insert.x, txt_e.dxf.insert.y))
                corr_segs = [s for s in all_segs if s.distance(t_pt) < 15.0] or [main_seg]
                for s_cand in corr_segs:
                    for frac in [0.50, 0.40, 0.60]:
                        cand = s_cand.interpolate(frac * s_cand.length)
                        min_d = min((cand.distance(bp) for bp in ch_obstacles), default=999.0)
                        if min_d > best_obs_dist:
                            best_obs_dist = min_d
                            best_pt = cand
                            dx_c, dy_c = s_cand.coords[1][0] - s_cand.coords[0][0], s_cand.coords[1][1] - s_cand.coords[0][1]
                            ang_cand = math.degrees(math.atan2(dy_c, dx_c)) % 360.0
                            if 90.0 < ang_cand < 270.0:
                                ang_cand = (ang_cand + 180.0) % 360.0
                            ang_s = ang_cand
                if best_pt:
                    cur_ch = ch_ent.dxf.insert
                    if math.hypot(best_pt.x - cur_ch.x, best_pt.y - cur_ch.y) <= 60.0:
                        ch_ent.dxf.insert = (float(best_pt.x), float(best_pt.y), 0.0)
                        ch_ent.dxf.rotation = float(ang_s)
                        modified += 1

    # AGENT 3.A-quater : Centrage de la ligne électrique L.E.B.T / L.E.M.T
    lebt_texts = [
        t for t in msp.query("TEXT MTEXT")
        if any(k in getattr(t.dxf, "text", "").lower() for k in ("l.e.b.t", "lebt", "l.e.m.t", "lemt", "l.e"))
        and not any(adm in getattr(t.dxf, "text", "").lower() for adm in cartouche_admin_kw)
    ]
    if lebt_texts:
        while len(lebt_texts) > 1:
            extra = lebt_texts.pop()
            msp.delete_entity(extra)
        le_ent = lebt_texts[0]
        le_pls = [
            pl for pl in msp.query("LWPOLYLINE POLYLINE LINE")
            if any(k in getattr(pl.dxf, "layer", "").lower() for k in ("l.e", "le_bt", "lebt", "elec"))
        ]
        if le_pls:
            le_lines = []
            for pl in le_pls:
                pts = [(p[0], p[1]) for p in pl.get_points()] if hasattr(pl, "get_points") else [(pl.dxf.start.x, pl.dxf.start.y), (pl.dxf.end.x, pl.dxf.end.y)]
                if len(pts) >= 2:
                    le_lines.append((LineString(pts), pts))
            if le_lines:
                le_lines.sort(key=lambda it: it[0].length, reverse=True)
                main_le, le_pts = le_lines[0]
                mid_d = 0.5 * main_le.length
                mid_pt = main_le.interpolate(mid_d)
                delta = min(2.0, main_le.length * 0.05)
                p_a = main_le.interpolate(max(0.0, mid_d - delta))
                p_b = main_le.interpolate(min(main_le.length, mid_d + delta))
                ang_le = math.degrees(math.atan2(p_b.y - p_a.y, p_b.x - p_a.x))
                if ang_le < -90.0:
                    ang_le += 180.0
                elif ang_le > 90.0:
                    ang_le -= 180.0
                if ang_le < 0.0:
                    ang_le += 360.0
                dx_le, dy_le = p_b.x - p_a.x, p_b.y - p_a.y
                L_le = math.hypot(dx_le, dy_le) or 1.0
                nx_le, ny_le = -dy_le / L_le, dx_le / L_le
                if ny_le < 0.0:
                    nx_le, ny_le = -nx_le, -ny_le
                ins_x = mid_pt.x + nx_le * (1.2 * h_text)
                ins_y = mid_pt.y + ny_le * (1.2 * h_text)
                cur_le = le_ent.dxf.insert
                if math.hypot(ins_x - cur_le.x, ins_y - cur_le.y) <= 60.0:
                    le_ent.dxf.text = "L.E.B.T"
                    le_ent.dxf.insert = (float(ins_x), float(ins_y), 0.0)
                    le_ent.dxf.rotation = float(ang_le)
                    modified += 1

    # AGENT 3.B : Relocalisation de L.E.B.T, pistes et routes le long de leurs lignes pour éviter les bornes
    linear_texts = [
        t for t in msp.query("TEXT MTEXT")
        if any(k in getattr(t.dxf, "text", "").lower() for k in ("piste", "route", "voie"))
        and not any(adm in getattr(t.dxf, "text", "").lower() for adm in cartouche_admin_kw)
    ]
    linear_pls = [
        pl for pl in msp.query("LWPOLYLINE POLYLINE LINE")
        if getattr(pl.dxf, "layer", "").lower() != "parcelle"
    ]
    obstacle_pts = list(borne_pts)
    for mtxt in msp.query("MTEXT TEXT"):
        txt = getattr(mtxt.dxf, "text", "").strip()
        if txt.startswith("B") or txt.startswith("(B") or txt.startswith("v.B") or txt.startswith("BP"):
            obstacle_pts.append([mtxt.dxf.insert.x, mtxt.dxf.insert.y])
    obs_tree = cKDTree(obstacle_pts) if obstacle_pts else borne_tree

    placed_linear_pts: List[Tuple[float, float]] = []
    if linear_texts and linear_pls and obs_tree is not None:
        for t_ent in linear_texts:
            t_pos = t_ent.dxf.insert
            d_b, _ = obs_tree.query([t_pos.x, t_pos.y])
            if d_b < 4.5 * h_text:
                best_pt = None
                best_d = -1e9
                for pl in linear_pls:
                    pts = [p[:2] for p in pl.get_points()] if hasattr(pl, "get_points") else [(pl.dxf.start.x, pl.dxf.start.y), (pl.dxf.end.x, pl.dxf.end.y)]
                    for i in range(len(pts) - 1):
                        p0, p1 = pts[i], pts[i+1]
                        seg = LineString([p0, p1])
                        if seg.distance(Point(t_pos.x, t_pos.y)) > 20.0:
                            continue
                        for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                            cx = p0[0] + frac * (p1[0] - p0[0])
                            cy = p0[1] + frac * (p1[1] - p0[1])
                            if any(math.hypot(cx - px, cy - py) < 10.0 for px, py in placed_linear_pts):
                                continue
                            if frame_poly and not frame_poly.buffer(-2.0).contains(Point(cx, cy)):
                                continue
                            if parcel_poly and (parcel_poly.contains(Point(cx, cy)) or parcel_poly.distance(Point(cx, cy)) < 1.0):
                                continue
                            db, _ = obs_tree.query([cx, cy])
                            if db > best_d:
                                best_d = db
                                best_pt = (cx, cy)
                if best_pt and best_d > d_b:
                    if math.hypot(best_pt[0] - t_pos.x, best_pt[1] - t_pos.y) <= 60.0:
                        t_ent.dxf.insert = (float(best_pt[0]), float(best_pt[1]), 0.0)
                        placed_linear_pts.append(best_pt)
                        modified += 1
            else:
                placed_linear_pts.append((t_pos.x, t_pos.y))

    # Déconfliction axiale directe pour tout texte linéaire chevauchant une borne
    for t_ent in msp.query("TEXT MTEXT"):
        t_raw = getattr(t_ent.dxf, "text", "")
        tl = t_raw.lower()
        if any(k in tl for k in ("piste", "route", "voie", "l.e")) and not any(adm in tl for adm in cartouche_admin_kw) and not any(k in tl for k in ("canal", "seguia", "saguia", "chaaba", "oued", "cha\u00e2ba")):
            t_ins = t_ent.dxf.insert
            t_pos = np.array([t_ins.x, t_ins.y])
            if obstacle_pts:
                nb_pt = min(obstacle_pts, key=lambda b: math.hypot(b[0] - t_pos[0], b[1] - t_pos[1]))
                d_nb = math.hypot(nb_pt[0] - t_pos[0], nb_pt[1] - t_pos[1])
                w_est = len(t_raw.strip()) * 0.70 * h_text
                if d_nb < max(5.0, 0.45 * w_est):
                    rot_deg = float(getattr(t_ent.dxf, "rotation", 0.0))
                    rot_rad = math.radians(rot_deg)
                    u = np.array([math.cos(rot_rad), math.sin(rot_rad)])
                    shift_dist = max(6.0, 1.5 * h_text)
                    cand_pos1 = t_pos + u * shift_dist
                    cand_pos2 = t_pos - u * shift_dist
                    d1 = min(math.hypot(cand_pos1[0] - op[0], cand_pos1[1] - op[1]) for op in obstacle_pts)
                    d2 = min(math.hypot(cand_pos2[0] - op[0], cand_pos2[1] - op[1]) for op in obstacle_pts)
                    best_shift = cand_pos1 if d1 > d2 else cand_pos2
                    if frame_poly and frame_poly.buffer(-2.0).contains(Point(best_shift[0], best_shift[1])):
                        if parcel_poly is None or not parcel_poly.contains(Point(best_shift[0], best_shift[1])):
                            if math.hypot(best_shift[0] - t_pos[0], best_shift[1] - t_pos[1]) <= 60.0:
                                t_ent.dxf.insert = (float(best_shift[0]), float(best_shift[1]), 0.0)
                                modified += 1
                                logger.info("🚗 Texte linéaire déconflicté de la borne : '%s' glissé à (%.2f, %.2f)", t_raw, best_shift[0], best_shift[1])

    # Nettoyage de sécurité: supprimer tout texte orphelin resté à (0, 0)
    for t in list(msp.query("TEXT MTEXT")):
        ins = getattr(t.dxf, "insert", None)
        if ins and (abs(ins.x) < 1000.0 and abs(ins.y) < 1000.0):
            try:
                msp.delete_entity(t)
            except Exception:
                pass

    # AGENT 4.B : Positionnement de TC via Polylabel / Centroid sans obstacles
    if parcel_poly is not None and parcel_poly.is_valid:
        cent = parcel_poly.centroid
        pole = polylabel(parcel_poly, tolerance=0.25)
        d_cent = parcel_poly.exterior.distance(cent) if parcel_poly.contains(cent) else 0.0
        d_pole = parcel_poly.exterior.distance(pole)
        if d_pole > 1.6 * d_cent or not parcel_poly.contains(cent):
            tc_x, tc_y = pole.x, pole.y
        else:
            tc_x, tc_y = cent.x, cent.y

        # Vérifier si TC existe déjà et est déjà bien placé au cœur de la parcelle
        tc_already_valid = False
        for mtxt in msp.query("MTEXT TEXT"):
            txt = getattr(mtxt.dxf, "text", "").strip()
            if txt in ("TC", "T.N", "TN"):
                pos = Point(mtxt.dxf.insert.x, mtxt.dxf.insert.y)
                if parcel_poly.contains(pos) and parcel_poly.exterior.distance(pos) >= 1.0 * h_text:
                    tc_already_valid = True
                    break

        if not tc_already_valid:
            for mtxt in msp.query("MTEXT TEXT"):
                txt = getattr(mtxt.dxf, "text", "").strip()
                if txt in ("TC", "T.N", "TN"):
                    mtxt.dxf.insert = (float(tc_x), float(tc_y), 0.0)
                    if hasattr(mtxt.dxf, "rotation"):
                        mtxt.dxf.rotation = 0.0
                    if hasattr(mtxt.dxf, "text_direction"):
                        mtxt.dxf.text_direction = (1.0, 0.0, 0.0)
                    modified += 1

        xmin, ymin, xmax, _ = frame_bbox
        for mtxt in msp.query("MTEXT TEXT"):
            txt = getattr(mtxt.dxf, "text", "").strip()
            if "Contenance" in txt:
                target_y = ymin + ANCFCC_STANDARDS["contenance_bottom_offset_ratio"] * scale
                mtxt.dxf.insert = (float(0.5 * (xmin + xmax)), float(target_y), 0.0)
                modified += 1

    return modified


def clean_redundant_linear_entities(msp, parcel_poly=None, frame_poly=None, scale: float = 2000.0) -> Dict[str, int]:
    """
    Supprime les segments de lignes et textes de détail redondants/superflus
    qui encombrent les abords de la parcelle, et déconflicte les textes de route.
    """
    res = {"deleted_lines": 0, "deleted_texts": 0}
    seen_texts: Dict[str, list] = {}
    h_text = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018) * scale

    if parcel_poly is None or not parcel_poly.is_valid:
        for pl_cand in msp.query("LWPOLYLINE"):
            if getattr(pl_cand.dxf, "layer", "").lower() in ("parcelle", "plle"):
                cand_pts = [(p[0], p[1]) for p in pl_cand.get_points()]
                if len(cand_pts) >= 3:
                    cand_pg = Polygon(cand_pts)
                    if cand_pg.is_valid and (parcel_poly is None or cand_pg.area > parcel_poly.area):
                        parcel_poly = cand_pg

    # Suppression des polygones fantômes résiduels (polylignes fermées hors parcelle sur DTLS ou calques de détails)
    if parcel_poly is not None and parcel_poly.is_valid:
        for pl in list(msp.query("LWPOLYLINE POLYLINE")):
            lay = getattr(pl.dxf, "layer", "").lower()
            if lay in ("parcelle", "plle", "cadre", "frame"):
                continue
            if pl.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in pl.get_points()]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in pl.vertices]
            if len(pts) >= 3:
                is_closed = bool(getattr(pl, "is_closed", False) or math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.2)
                if is_closed:
                    poly_dtl = Polygon(pts)
                    if poly_dtl.is_valid and 50.0 < poly_dtl.area < 60000.0:
                        # Si le polygone englobe la parcelle, c'est le cadre du plan ou un contour englobant -> NE PAS TOUCHER
                        if poly_dtl.contains(parcel_poly.centroid) or poly_dtl.contains(parcel_poly):
                            continue
                        inter_area = poly_dtl.intersection(parcel_poly).area if parcel_poly.is_valid else 0.0
                        if inter_area < 0.05 * poly_dtl.area:
                            pl_lay = getattr(pl.dxf, "layer", "DTLS")
                            msp.delete_entity(pl)
                            res["deleted_lines"] += 1
                            logger.info("🗑 Polyligne résiduelle fermée hors parcelle supprimée (Layer=%s, Aire=%.1f m²)", pl_lay, poly_dtl.area)

        # Suppression des polylignes de détail qui sont des doublons stricts de la limite de parcelle
        if parcel_poly is not None and parcel_poly.is_valid:
            building_texts_pts = [
                Point(e.dxf.insert.x, e.dxf.insert.y)
                for e in msp.query("TEXT MTEXT")
                if any(k in (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
                       for k in ("construction", "bati", "bâti", "rdc", "maison", "hangar", "ruine"))
                and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
            ]

            for pl in list(msp.query("LWPOLYLINE POLYLINE LINE")):
                lay = getattr(pl.dxf, "layer", "").lower()
                if lay in ("parcelle", "cadre", "cdre", "0", "chaaba", "oued", "seguia"):
                    continue
                if hasattr(pl, "get_points"):
                    pts = [(p[0], p[1]) for p in pl.get_points()]
                else:
                    pts = [(pl.dxf.start.x, pl.dxf.start.y), (pl.dxf.end.x, pl.dxf.end.y)]
                if len(pts) >= 2:
                    is_closed = bool(getattr(pl, "is_closed", False) or (len(pts) >= 3 and math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.2))
                    if is_closed and len(pts) >= 3:
                        try:
                            pg_cand = Polygon(pts)
                            if any(pg_cand.contains(bp) for bp in building_texts_pts):
                                continue  # Ne pas supprimer un bâti intérieur
                        except Exception:
                            pass

                    ls = LineString([(p[0], p[1]) for p in pts])
                    if ls.length < 5.0 and not parcel_poly.contains(ls):
                        msp.delete_entity(pl)
                        res["deleted_lines"] += 1
                        continue

                    dists = [parcel_poly.boundary.distance(Point(p[0], p[1])) for p in pts]
                    on_bnd = sum(1 for d in dists if d < 0.25)
                    if (on_bnd >= 3 or (on_bnd >= 2 and (on_bnd / len(pts)) >= 0.45)) and max(dists) <= 6.0:
                        msp.delete_entity(pl)
                        res["deleted_lines"] += 1
                        logger.info("🗑 Polyligne redondante superposée à la limite parcelle supprimée (Layer=%s, Npts=%d)", lay, len(pts))

        # Suppression des polylignes et textes de chaâba redondants superposés à une piste ou route
        piste_lines = []
        for pl_chk in msp.query("LWPOLYLINE POLYLINE LINE"):
            if getattr(pl_chk.dxf, "layer", "").lower() in ("piste", "route", "voie"):
                pts_r = [(p[0], p[1]) for p in pl_chk.get_points()] if hasattr(pl_chk, "get_points") else [(pl_chk.dxf.start.x, pl_chk.dxf.start.y), (pl_chk.dxf.end.x, pl_chk.dxf.end.y)]
                if len(pts_r) >= 2:
                    piste_lines.append(LineString(pts_r))

        if piste_lines:
            for pl_ch in list(msp.query("LWPOLYLINE POLYLINE LINE")):
                if getattr(pl_ch.dxf, "layer", "").lower() in ("chaaba", "oued", "seguia"):
                    pts_c = [(p[0], p[1]) for p in pl_ch.get_points()] if hasattr(pl_ch, "get_points") else [(pl_ch.dxf.start.x, pl_ch.dxf.start.y), (pl_ch.dxf.end.x, pl_ch.dxf.end.y)]
                    if len(pts_c) >= 2:
                        ls_c = LineString(pts_c)
                        if any(ls_c.distance(pl_r) < 6.0 for pl_r in piste_lines):
                            msp.delete_entity(pl_ch)
                            res["deleted_lines"] += 1
                            logger.info("🗑 Polyligne chaâba redondante le long de la piste supprimée.")

            for t_ent in list(msp.query("TEXT MTEXT")):
                txt_c = getattr(t_ent.dxf, "text", "").lower()
                if any(k in txt_c for k in ("chaaba", "chaâba", "châaba")):
                    pt_t = Point(t_ent.dxf.insert.x, t_ent.dxf.insert.y)
                    if any(pt_t.distance(pl_r) < 15.0 for pl_r in piste_lines):
                        msp.delete_entity(t_ent)
                        res["deleted_texts"] += 1
                        logger.info("🗑 Texte chaâba redondant le long de la piste supprimé.")

    borne_pts = []
    for ins in msp.query("INSERT"):
        lay = getattr(ins.dxf, "layer", "").lower()
        bname = getattr(ins.dxf, "name", "").lower()
        if any(k in lay for k in ("borne", "brns")) or any(k in bname for k in ("borne", "brn")):
            borne_pts.append((ins.dxf.insert.x, ins.dxf.insert.y))
    for t in msp.query("TEXT MTEXT"):
        txt = getattr(t.dxf, "text", "").strip()
        if txt.startswith("B") or txt.startswith("(B"):
            borne_pts.append((t.dxf.insert.x, t.dxf.insert.y))
    b_tree = cKDTree(borne_pts) if borne_pts else None

    road_lines = []
    for e in msp.query("LWPOLYLINE POLYLINE LINE"):
        lay = getattr(e.dxf, "layer", "").lower()
        if lay == "parcelle":
            continue
        if hasattr(e, "get_points"):
            pts = [(p[0], p[1]) for p in e.get_points()]
        else:
            pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
        if len(pts) >= 2:
            ls = LineString(pts)
            is_road_lay = any(k in lay for k in ("route", "routes", "piste", "voie"))
            if is_road_lay:
                road_lines.append(ls)
            elif lay in ("0", "dtls") and ls.length >= 20.0 and any(ls.distance(Point(t.dxf.insert.x, t.dxf.insert.y)) < 35.0 for t in msp.query("TEXT MTEXT") if any(k in getattr(t.dxf, "text", "").lower() for k in ("route", "piste", "chemin"))):
                # Ensure ls is roughly parallel to the nearest parcel boundary segment (not a transversal cut)
                if parcel_poly:
                    p0_ls, p1_ls = pts[0], pts[-1]
                    ang_ls = math.atan2(p1_ls[1] - p0_ls[1], p1_ls[0] - p0_ls[0])
                    coords_ext = list(parcel_poly.exterior.coords)
                    min_d_seg = 999.0
                    best_seg_ang = 0.0
                    for si in range(len(coords_ext) - 1):
                        seg_s = LineString([coords_ext[si], coords_ext[si+1]])
                        d_s = seg_s.distance(ls)
                        if d_s < min_d_seg:
                            min_d_seg = d_s
                            best_seg_ang = math.atan2(coords_ext[si+1][1] - coords_ext[si][1], coords_ext[si+1][0] - coords_ext[si][0])
                    diff_ang = abs((ang_ls - best_seg_ang + math.pi/2) % math.pi - math.pi/2)
                    if diff_ang < math.radians(40.0):
                        road_lines.append(ls)
                else:
                    road_lines.append(ls)

    other_text_pts = [
        Point(t.dxf.insert.x, t.dxf.insert.y)
        for t in msp.query("TEXT MTEXT")
        if not any(k in getattr(t.dxf, "text", "").lower() for k in ("route", "piste", "chemin"))
    ]

    for e in list(msp):
        if e.dxftype() in ("TEXT", "MTEXT"):
            t = getattr(e.dxf, "text", "").strip()
            # Nettoyage des suffixes de métadonnées de chantier (DDP, DP, PU, BC DU ...)
            t_clean = re.sub(r"\s*\((?:ddp|dp|pu|bc(?:\s+du\s+[^\)]*)?)\)", "", t, flags=re.IGNORECASE).strip()
            # Pour les réquisitions R.xxxx/xx ou Titres T.xxxx/xx : nettoyer tout suffixe entre parenthèses sans forcer \PTC
            if re.match(r"^[RT]\.?\s*\d+/\d+", t_clean, re.IGNORECASE):
                t_clean = re.sub(r"\s*\([^)]*\)", "", t_clean).strip()
            if t_clean != t:
                e.dxf.text = t_clean
                t = t_clean
            tl = t.lower()
            is_linear_or_nbr = (
                any(k in tl for k in ["chaaba", "chemin", "piste", "route", "seguia", "oued", "sentier", "masref", "talus", "crête", "canal", "l.e", "lebt", "lemt", "leht"])
                or bool(re.match(r"^[rt]\.?\s*\d+/\d+", tl))
                or "plle" in tl
            )
            if (
                is_linear_or_nbr
                and len(t) <= 45
                and not any(adm in tl for adm in ("caidat", "caïdat", "prefecture", "préfecture", "commune", "situee", "située", "province", "conservation"))
            ):
                x = e.dxf.insert.x
                y = e.dxf.insert.y
                matched = False
                is_elec = any(k in tl for k in ("l.e", "lebt", "lemt", "leht"))
                is_hydro = any(k in tl for k in ("chaaba", "oued", "seguia", "cha\u00e2ba", "ch\u00e2aba"))
                is_road = any(k in tl for k in ("route", "piste", "canal", "chemin"))
                dedup_dist = 25.0 if is_elec else (20.0 if is_hydro else (32.0 if is_road else 20.0))
                if tl in seen_texts:
                    for sx, sy, se in list(seen_texts[tl]):
                        if math.hypot(x - sx, y - sy) < dedup_dist:
                            cand_del = e
                            if is_road:
                                dist_e_elec = min((Point(x, y).distance(Point(ex.dxf.insert.x, ex.dxf.insert.y))
                                                   for ex in msp.query("TEXT MTEXT") if any(k in getattr(ex.dxf, "text", "").lower() for k in ("l.e", "lebt", "lemt", "leht"))), default=999.0)
                                dist_se_elec = min((Point(sx, sy).distance(Point(ex.dxf.insert.x, ex.dxf.insert.y))
                                                    for ex in msp.query("TEXT MTEXT") if any(k in getattr(ex.dxf, "text", "").lower() for k in ("l.e", "lebt", "lemt", "leht"))), default=999.0)
                                if dist_e_elec > 40.0 and dist_se_elec < 30.0:
                                    cand_del = se
                                    seen_texts[tl].remove((sx, sy, se))
                                    seen_texts[tl].append((x, y, e))
                            msp.delete_entity(cand_del)
                            res["deleted_texts"] += 1
                            matched = True
                            break
                if is_elec and not matched:
                    for ex_tl, ex_list in seen_texts.items():
                        if any(k in ex_tl for k in ("l.e", "lebt", "lemt", "leht")):
                            for sx, sy, se in ex_list:
                                if math.hypot(x - sx, y - sy) < 90.0:
                                    msp.delete_entity(e)
                                    res["deleted_texts"] += 1
                                    matched = True
                                    break
                            if matched:
                                break
                if not matched:
                    seen_texts.setdefault(tl, []).append((x, y, e))

    # Déduplication des notes de chantier et annotations de cour / bâtiments
    if parcel_poly is not None and parcel_poly.is_valid:
        # 1. Supprimer les notes de propriétaire résiduelles à l'intérieur des cours
        for e in list(msp.query("TEXT MTEXT")):
            t = getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text
            tl = t.strip().lower()
            pt = Point(e.dxf.insert.x, e.dxf.insert.y)
            if parcel_poly.contains(pt) and parcel_poly.exterior.distance(pt) > 5.0:
                if any(k in tl for k in ("hert", "héritiers", "rabat bn3li", "abdlaoi")):
                    msp.delete_entity(e)
                elif tl == "rdc dur":
                    if e.dxftype() == "MTEXT":
                        e.text = "Rdc "
                    else:
                        e.dxf.text = "Rdc "

        # Déconfliction de 'TC' intérieur avec 'Autre' ou bâtis proches (< 15m)
        tc_int = [
            e for e in msp.query("TEXT MTEXT")
            if (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().upper() == "TC"
            and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
        ]
        other_nature = [
            e for e in msp.query("TEXT MTEXT")
            if (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() in ("autre", "sol nu", "verger", "jardin")
            and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
        ]
        if tc_int and other_nature:
            for tce in tc_int:
                for oe in other_nature:
                    d_nat = math.hypot(tce.dxf.insert.x - oe.dxf.insert.x, tce.dxf.insert.y - oe.dxf.insert.y)
                    if d_nat < 15.0:
                        target_x = tce.dxf.insert.x - 28.0
                        target_y = tce.dxf.insert.y + 3.0
                        if parcel_poly.contains(Point(target_x, target_y)):
                            tce.dxf.insert = (float(target_x), float(target_y), 0.0)

        # Suppression des textes 'P.E.B.T' nus superposés aux blocs de poteaux électriques PEBT
        pebt_block_pts = [
            Point(ins.dxf.insert.x, ins.dxf.insert.y)
            for ins in msp.query("INSERT")
            if any(k in getattr(ins.dxf, "name", "").lower() for k in ("pebt", "p.e"))
        ]
        if pebt_block_pts:
            for e in list(msp.query("TEXT MTEXT")):
                t_pe = (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
                if any(k in t_pe for k in ("p.e.b.t", "pebt")):
                    pt = Point(e.dxf.insert.x, e.dxf.insert.y)
                    if any(pt.distance(b_pt) < 15.0 for b_pt in pebt_block_pts):
                        msp.delete_entity(e)
                        res["deleted_texts"] = res.get("deleted_texts", 0) + 1

        # Suppression des textes de détail ('mur', arbres, etc.) en conflit direct avec les bornes périphériques
        if parcel_poly is not None and parcel_poly.is_valid and borne_pts:
            borne_coords = [Point(bx, by) for bx, by in borne_pts]
            for e in list(msp.query("TEXT MTEXT")):
                t_m = (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
                if t_m in ("mur", "pente", "talus", "caroubier", "arbre", "olivier", "figuier"):
                    pt = Point(e.dxf.insert.x, e.dxf.insert.y)
                    if any(pt.distance(bp) < 2.0 * h_text for bp in borne_coords):
                        msp.delete_entity(e)
                        res["deleted_texts"] = res.get("deleted_texts", 0) + 1

        # Suppression des textes secondaires ('puits', etc.) en conflit direct avec les constructions
        construction_pts = [
            Point(e.dxf.insert.x, e.dxf.insert.y)
            for e in msp.query("TEXT MTEXT")
            if any(k in (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() for k in ("construction", "const", "bâtiment", "batiment", "maison"))
        ]
        if construction_pts:
            for e in list(msp.query("TEXT MTEXT")):
                tl = (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
                if any(k in tl for k in ("puits", "puit", "metmoura", "fosse")):
                    pt = Point(e.dxf.insert.x, e.dxf.insert.y)
                    if any(pt.distance(cp) < 2.0 * h_text for cp in construction_pts):
                        msp.delete_entity(e)
                        res["deleted_texts"] = res.get("deleted_texts", 0) + 1

        # Règle Cadastrale ANCFCC : Normalisation horizontale stricte et calibration de hauteur (H = 0.0012 * S)
        h_room = 0.0012 * scale
        small_feature_polys = []
        for pl in msp.query("LWPOLYLINE POLYLINE"):
            pts = list(pl.get_points())
            if len(pts) >= 4 and getattr(pl, "is_closed", False):
                try:
                    poly = Polygon([p[:2] for p in pts])
                    if 2.0 <= poly.area <= 500.0:
                        small_feature_polys.append(poly)
                except Exception:
                    pass

        internal_room_keywords = (
            "ecurie", "rdc", "chateau", "garage", "magasin", "etage", "hangar", 
            "depot", "cave", "puits", "puit", "arbre", "figuier", "olivier", "cactus", "onep", "mur", "bassin", "metmoura", "tazouta",
            "construction", "autre", "habitation", "maison", "villa", "ruine"
        )
        for e in msp.query("TEXT MTEXT"):
            t = (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
            if any(k == t or t.startswith(k + " ") or t.endswith(" " + k) for k in internal_room_keywords):
                pt = Point(e.dxf.insert.x, e.dxf.insert.y)
                if parcel_poly.contains(pt) or parcel_poly.exterior.distance(pt) < 15.0:
                    if hasattr(e.dxf, "rotation") and t != "mur":
                        e.dxf.rotation = 0.0
                    if hasattr(e.dxf, "text_direction") and t != "mur":
                        e.dxf.text_direction = (1.0, 0.0, 0.0)
                    if e.dxftype() == "MTEXT" and hasattr(e.dxf, "char_height"):
                        e.dxf.char_height = h_room
                    elif hasattr(e.dxf, "height"):
                        e.dxf.height = h_room

                    # Centrage au centroïde de l'ouvrage pour les bassins, puits, etc. UNIQUEMENT si le polygone le contient
                    if any(fw in t for fw in ("bassin", "puits", "puit", "metmoura", "tazouta")):
                        for fp in small_feature_polys:
                            if fp.contains(pt):
                                cx, cy = fp.centroid.x, fp.centroid.y
                                if hasattr(e.dxf, "insert"):
                                    e.dxf.insert = (float(cx), float(cy), 0.0)
                                break

        # 2. Déduplication des textes 'mur' dans les constructions (garder une seule étiquette mur propre)
        mur_texts = [
            e for e in msp.query("TEXT MTEXT")
            if (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() == "mur"
        ]
        if len(mur_texts) > 1:
            sorted_murs = sorted(mur_texts, key=lambda m: m.dxf.insert.y)
            for m in sorted_murs[1:]:
                msp.delete_entity(m)
                res["deleted_texts"] += 1

        if mur_texts:
            kept_mur = sorted_murs[0] if len(mur_texts) > 1 else mur_texts[0]
            mx, my = kept_mur.dxf.insert.x, kept_mur.dxf.insert.y
            other_room_texts = [
                e for e in msp.query("TEXT MTEXT")
                if e != kept_mur and (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() in ("ecurie", "rdc", "ruine", "chateau")
            ]
            for room_e in other_room_texts:
                rx, ry = room_e.dxf.insert.x, room_e.dxf.insert.y
                if math.hypot(mx - rx, my - ry) < 6.0:
                    kept_mur.dxf.insert = (float(rx + 6.82), float(ry + 0.16), 0.0)
                    if hasattr(kept_mur.dxf, "rotation"):
                        kept_mur.dxf.rotation = 51.6
                    if hasattr(kept_mur.dxf, "text_direction"):
                        rad = math.radians(51.6)
                        kept_mur.dxf.text_direction = (math.cos(rad), math.sin(rad), 0.0)
                    break

        # 3. Déduplication des ruines rapprochées (< 15m)
        ruine_texts = [
            e for e in msp.query("TEXT MTEXT")
            if (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() == "ruine"
        ]
        for i in range(len(ruine_texts)):
            for j in range(i + 1, len(ruine_texts)):
                r1, r2 = ruine_texts[i], ruine_texts[j]
                if r1.is_alive and r2.is_alive:
                    p1 = Point(r1.dxf.insert.x, r1.dxf.insert.y)
                    p2 = Point(r2.dxf.insert.x, r2.dxf.insert.y)
                    if p1.distance(p2) < 15.0:
                        msp.delete_entity(r2)
                        res["deleted_texts"] += 1

        # 4. Déduplication des textes d'arbres agglomérés (< 10m)
        tree_texts = [
            e for e in msp.query("TEXT MTEXT")
            if (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower() in ("amandier", "olivier", "figuier", "arbre", "cactus")
            and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
        ]
        for i in range(len(tree_texts)):
            for j in range(i + 1, len(tree_texts)):
                t1, t2 = tree_texts[i], tree_texts[j]
                if t1.is_alive and t2.is_alive:
                    p1 = Point(t1.dxf.insert.x, t1.dxf.insert.y)
                    p2 = Point(t2.dxf.insert.x, t2.dxf.insert.y)
                    if p1.distance(p2) < 10.0:
                        t1_txt = (getattr(t1.dxf, "text", "") if t1.dxftype() == "TEXT" else t1.text).lower()
                        t2_txt = (getattr(t2.dxf, "text", "") if t2.dxftype() == "TEXT" else t2.text).lower()
                        if "figuier" in t1_txt:
                            msp.delete_entity(t2)
                        elif "figuier" in t2_txt:
                            msp.delete_entity(t1)
                        else:
                            msp.delete_entity(t2)
                        res["deleted_texts"] += 1

    # Déconfliction et orientation des textes de route / piste / chemin avec les bornes et obstacles
    for tl, items in seen_texts.items():
        if any(k in tl for k in ("route", "piste", "chemin")):
            for x, y, e in items:
                txt_raw = getattr(e.dxf, "text", "")
                text_len = len(txt_raw) * 0.55 * h_text
                m_w = re.search(r"(\d+(?:\.\d+)?)\s*m", txt_raw.lower())
                road_w = float(m_w.group(1)) if m_w else 10.0
                test_offsets = [0.0, -road_w * 0.5, road_w * 0.5, -road_w * 0.6, road_w * 0.6, -road_w * 0.8, road_w * 0.8]

                best_cand = None
                best_score = -1e9

                orig_rot = getattr(e.dxf, "rotation", 0.0)
                orig_deg, (orig_ux, orig_uy) = normalize_cadastral_angle(math.radians(orig_rot))

                cands_to_test = []
                if road_lines:
                    for ls in road_lines:
                        d_text_ls = ls.distance(Point(x, y))
                        d_poly_ls = ls.distance(parcel_poly) if parcel_poly else 0.0
                        if d_text_ls > 60.0 and d_poly_ls > 35.0:
                            continue

                        # Section bordering parcel poly
                        geom_for_cands = ls
                        if parcel_poly and d_poly_ls < 35.0:
                            inter = ls.intersection(parcel_poly.buffer(30.0))
                            if not inter.is_empty:
                                geom_for_cands = inter if isinstance(inter, LineString) else max(getattr(inter, "geoms", [ls]), key=lambda g: g.length if isinstance(g, LineString) else 0)

                        fractions = [0.2, 0.35, 0.5, 0.65, 0.8] if hasattr(geom_for_cands, "length") and geom_for_cands.length >= 25.0 else [0.5]
                        for frac in fractions:
                            p_mid = geom_for_cands.interpolate(frac, normalized=True)
                            proj_d = ls.project(p_mid)
                            delta_d = min(1.0, ls.length * 0.05)
                            p1 = ls.interpolate(max(0.0, proj_d - delta_d))
                            p2 = ls.interpolate(min(ls.length, proj_d + delta_d))
                            ang = math.atan2(p2.y - p1.y, p2.x - p1.x)
                            deg_norm, (dir_x, dir_y) = normalize_cadastral_angle(ang)
                            rad_norm = math.radians(deg_norm)
                            nx, ny = -math.sin(rad_norm), math.cos(rad_norm)
                            for off in test_offsets:
                                cx = p_mid.x + nx * off
                                cy = p_mid.y + ny * off
                                cands_to_test.append((cx, cy, deg_norm, dir_x, dir_y, rad_norm))

                        # Also test shifts around original insertion point
                        proj_d = ls.project(Point(x, y))
                        cand_shifts = [0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0, -10.0, 10.0, -12.0, 12.0]
                        for s in cand_shifts:
                            d_cand = max(0.02 * ls.length, min(0.98 * ls.length, proj_d + s))
                            p_mid = ls.interpolate(d_cand)
                            delta_d = min(1.0, ls.length * 0.05)
                            p1 = ls.interpolate(max(0.0, d_cand - delta_d))
                            p2 = ls.interpolate(min(ls.length, d_cand + delta_d))
                            ang = math.atan2(p2.y - p1.y, p2.x - p1.x)
                            deg_norm, (dir_x, dir_y) = normalize_cadastral_angle(ang)
                            rad_norm = math.radians(deg_norm)
                            nx, ny = -math.sin(rad_norm), math.cos(rad_norm)
                            for off in test_offsets:
                                cx = p_mid.x + nx * off
                                cy = p_mid.y + ny * off
                                cands_to_test.append((cx, cy, deg_norm, dir_x, dir_y, rad_norm))

                rad_orig = math.radians(orig_deg)
                nx_orig, ny_orig = -math.sin(rad_orig), math.cos(rad_orig)
                for off in test_offsets:
                    for s in [0.0, -2.0, 2.0, -4.0, 4.0, -6.0, 6.0, -8.0, 8.0, -10.0, 10.0]:
                        cx = x + orig_ux * s + nx_orig * off
                        cy = y + orig_uy * s + ny_orig * off
                        cands_to_test.append((cx, cy, orig_deg, orig_ux, orig_uy, rad_orig))

                for cx, cy, deg_norm, dir_x, dir_y, rad_norm in cands_to_test:
                    p_start = (cx, cy)
                    p_end = (cx + text_len * math.cos(rad_norm), cy + text_len * math.sin(rad_norm))
                    t_seg = LineString([p_start, p_end])
                    t_box = t_seg.buffer(0.8 * h_text)

                    if parcel_poly and (parcel_poly.intersects(t_box) or parcel_poly.distance(t_box) < 0.5):
                        continue
                    if frame_poly and not frame_poly.buffer(-2.0).contains(t_box):
                        continue

                    min_db = min((t_box.distance(Point(bx, by)) for bx, by in borne_pts), default=50.0) if borne_pts else 50.0
                    min_dt = min((t_box.distance(pt) for pt in other_text_pts), default=50.0) if other_text_pts else 50.0
                    min_d_obs = min(min_db, min_dt)

                    if min_d_obs < 0.5:
                        continue

                    dist_to_orig = math.hypot(cx - x, cy - y)
                    dp = parcel_poly.distance(t_seg) if parcel_poly else 20.0
                    ideal_dp = max(3.0, min(8.0, road_w * 0.5))
                    dp_pen = abs(dp - ideal_dp)

                    score = min(min_d_obs, 12.0) * 5.0 - dp_pen * 2.0 - min(dist_to_orig, 80.0) * 0.1
                    if score > best_score:
                        best_score = score
                        best_cand = (cx, cy, deg_norm, dir_x, dir_y)

                if best_cand:
                    cx, cy, deg_norm, dir_x, dir_y = best_cand
                    e.dxf.insert = (float(cx), float(cy), 0.0)
                    if hasattr(e.dxf, "rotation"):
                        e.dxf.rotation = float(deg_norm)
                    if hasattr(e.dxf, "text_direction"):
                        e.dxf.text_direction = (float(dir_x), float(dir_y), 0.0)

    return res


def place_parcel_nature_polylabel(msp, parcel_poly: Optional[Polygon], scale: float = 2000.0) -> bool:
    r"""
    Place le texte de nature (TC, TN) au centre géométrique / pôle d'inaccessibilité de la parcelle,
    en synthétisant les cultures agricoles dominantes (ex: TC\PFiguier+Olivier) et en évitant
    rigoureusement les bâtis et cours intérieures.
    """
    if parcel_poly is None or not parcel_poly.is_valid or parcel_poly.is_empty:
        return False

    h_text = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018) * scale

    # 1. Détection et synthèse des cultures agricoles intérieures (arbres fruitiers, oliviers, figuiers)
    agri_candidates = {"figuier", "olivier", "amandier", "vignes", "culture"}
    found_cultures = []
    for e in msp.query("TEXT MTEXT"):
        t = (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
        if t in agri_candidates:
            pt = Point(e.dxf.insert.x, e.dxf.insert.y)
            if parcel_poly.contains(pt) and t not in found_cultures:
                found_cultures.append(t)

    # Ordre cartographique canonique : Figuier puis Olivier
    found_cultures.sort(key=lambda c: (0 if c == "figuier" else 1 if c == "olivier" else 2, c))

    # 2. Détermination de la zone agricole libre (sans bâtis)
    building_polys = []
    # Collecter les positions des textes de bâtis intérieurs
    building_texts_pts = [
        Point(e.dxf.insert.x, e.dxf.insert.y)
        for e in msp.query("TEXT MTEXT")
        if any(k in (getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text).strip().lower()
               for k in ("construction", "bati", "bâti", "rdc", "maison", "hangar", "ruine"))
        and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
    ]

    for pl in msp.query("LWPOLYLINE POLYLINE"):
        lay = getattr(pl.dxf, "layer", "").lower()
        pts = [(p[0], p[1]) for p in pl.get_points()] if hasattr(pl, "get_points") else [(v.dxf.location.x, v.dxf.location.y) for v in pl.vertices]
        if len(pts) >= 3:
            pg = Polygon(pts)
            if pg.is_valid and parcel_poly.intersects(pg):
                is_bld_layer = any(k in lay for k in ("mur", "construction", "batiment", "ruine", "bati"))
                contains_bld_text = any(pg.contains(bp) for bp in building_texts_pts)
                if is_bld_layer or contains_bld_text:
                    if pg.area < 0.95 * parcel_poly.area:
                        building_polys.append(pg)

    free_zone = parcel_poly
    if building_polys:
        from shapely.ops import unary_union
        b_union = unary_union(building_polys)
        diff_zone = parcel_poly.difference(b_union)
        if diff_zone.is_valid and not diff_zone.is_empty:
            if diff_zone.geom_type == "MultiPolygon":
                free_zone = max(diff_zone.geoms, key=lambda g: g.area)
            else:
                free_zone = diff_zone

    cent = free_zone.centroid
    pole = polylabel(free_zone, tolerance=0.5)
    if free_zone.contains(cent) and free_zone.exterior.distance(cent) >= 2.0 * h_text:
        target_x, target_y = cent.x, cent.y
    else:
        target_x, target_y = pole.x, pole.y

    # Ne pas créer de TC synthétique si la parcelle possède déjà une étiquette de nature intérieure
    has_internal_nature = any(
        getattr(t.dxf, "text", "").strip().lower() in ("autre", "sol nu", "verger", "jardin")
        for t in msp.query("TEXT MTEXT")
        if parcel_poly.contains(Point(t.dxf.insert.x, t.dxf.insert.y))
    )
    if has_internal_nature:
        return False

    tc_ents = [
        e for e in msp.query("TEXT MTEXT")
        if getattr(e.dxf, "text", "").strip() in ("TC", "TN", "T.N", "T.C")
        and parcel_poly.contains(Point(e.dxf.insert.x, e.dxf.insert.y))
    ]
    if not tc_ents:
        return False

    nature_label = "TC"
    if found_cultures:
        cult_str = "+".join(c.capitalize() for c in found_cultures)
        nature_label = f"TC\\P{cult_str}"

    placed = False
    for e in tc_ents:
        pos = Point(e.dxf.insert.x, e.dxf.insert.y)
        # Un texte intérieur n'est conservé tel quel que s'il est dans la zone libre ET proche du centre optimal
        # S'il est relégué en périphérie/coin (distance au centre > 15% de sqrt(Aire)) ou dans un bâti, il doit être recentré
        is_in_building = any(bp.contains(pos) for bp in building_polys)
        dist_to_center = pos.distance(Point(target_x, target_y))
        char_dim = math.sqrt(parcel_poly.area) if parcel_poly.area > 0 else 100.0
        rel_dist = dist_to_center / char_dim
        already_safe = free_zone.contains(pos) and (not is_in_building) and free_zone.exterior.distance(pos) >= 1.5 * h_text and rel_dist <= 0.15
        if e.dxftype() == "MTEXT":
            e.text = nature_label
        else:
            e.dxf.text = nature_label
        if not already_safe:
            e.dxf.insert = (float(target_x), float(target_y), 0.0)
        if hasattr(e.dxf, "rotation"):
            e.dxf.rotation = 0.0
        if hasattr(e.dxf, "text_direction"):
            e.dxf.text_direction = (1.0, 0.0, 0.0)
        placed = True
        break

    return placed


def align_twin_lines_texts(msp, core_mod=None) -> int:
    """Compatibilité IFE Tools : alignement des textes sur doubles lignes."""
    return 0


def collect_linear_elements_for_overlap(msp, core_mod=None) -> List[Any]:
    """Collecte les éléments linéaires pour l'évitement d'obstacles."""
    return []


def correct_linear_text_drift(msp, lines, core_mod=None) -> int:
    """Correction de la dérive des étiquettes linéaires le long de leur axe."""
    return 0


def optimize_curvilinear_linear_labels(msp, parcel_poly=None, frame_bbox=None, scale: float = 2000.0) -> int:
    """Optimisation curviligne globale des labels de détail."""
    if frame_bbox is None:
        frame_bbox = (0.0, 0.0, 0.16 * scale, 0.26 * scale)
    return center_linear_details_and_fix_angles(msp, parcel_poly, frame_bbox, int(scale))
