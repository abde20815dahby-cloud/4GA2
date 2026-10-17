# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/borne_ops.py — Repositionnement strict et contrainte des bornes cadastrales
=============================================================================================
AGENT 2 : Placement optimal et anti-collision des labels de bornes (Bxx) :
  - Loi 3 : Bissectrice extérieure robuste (Winding Shoelace + Normales unitaires sortantes + Balayage concave)
  - Loi 4 : Déduplication spatiale et résolution globale anti-collision via Google OR-Tools CP-SAT + SAT
  - Brick 3 : RadialFanLeaderEngine pour clusters extrêmes (>= 3 bornes dans 2.0*H) avec traits de rappel BRNS
"""

import math
import re
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import ezdxf.bbox as _eb
from shapely.geometry import box as _box, Point as _Pt

from constants import ANCFCC_STANDARDS
from models import OrientedBox2D, sat_overlap_mtv, estimate_exact_text_dimensions

logger = logging.getLogger("dxf_cleaner.borne_ops")

try:
    from ortools.sat.python import cp_model
    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False

BORNE_NAME_RE = re.compile(r"^(?:v\.)?\(?(?:B\d+[A-Za-z0-9_/-]*|BP\d+[A-Za-z0-9_/-]*|BBP\d+[A-Za-z0-9_/-]*|R\d+|P\d+)\)?$", re.IGNORECASE)
PAREN_BORNE_RE = re.compile(r"^\(B\d+\)$", re.IGNORECASE)


def compute_robust_outward_bisectors(
    vertices: np.ndarray, parcel_poly: Polygon, base_offset: float
) -> List[np.ndarray]:
    """
    LOI 3 — BISSECTRICE EXTÉRIEURE ROBUSTE (SOMMETS 180° ET CONCAVES) :
    1. Déterminer le signe d'enroulement w_sign (+1 si CCW, -1 si CW) via Shoelace.
    2. Calculer les normales unitaires sortantes n_prev et n_next des 2 arêtes incidentes.
    3. Bissectrice unitaire b_ext = normalize(n_prev + n_next) (ne dégénère jamais à 180°).
    4. Si V_i + r_opt * b_ext tombe dans `parcelle` ou à < 0.35*r_opt d'une arête opposée,
       tester 17 directions tournantes et retenir celle qui maximise la distance libre hors du polygone.
    """
    adjusted_offset = base_offset * 1.05
    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) >= 2 and float(np.linalg.norm(pts[0] - pts[-1])) < 0.01:
        pts = pts[:-1]
    n = len(pts)
    if n < 3:
        return [np.array([1.0, 0.0]) for _ in range(n)]

    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    w_sign = 1.0 if signed_area >= 0.0 else -1.0

    bisectors: List[np.ndarray] = []
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]

        t_in = p_curr - p_prev
        t_out = p_next - p_curr
        l_in = float(np.linalg.norm(t_in))
        l_out = float(np.linalg.norm(t_out))
        if l_in < 1e-6 or l_out < 1e-6:
            bisectors.append(np.array([1.0, 0.0]))
            continue
        t_in /= l_in
        t_out /= l_out

        n_in = w_sign * np.array([t_in[1], -t_in[0]])
        n_out = w_sign * np.array([t_out[1], -t_out[0]])
        b_vec = n_in + n_out
        b_norm = float(np.linalg.norm(b_vec))
        b_vec = n_in if b_norm < 1e-4 else (b_vec / b_norm)

        # Règle Bornes Rapprochées ANCFCC: Si le sommet suivant est très proche (< 1.8 * adjusted_offset)
        # et que l'arête précédente est dégagée, orienter l'étiquette en arrière sur le grand côté
        if l_out < 1.8 * adjusted_offset and l_in >= 2.0 * adjusted_offset:
            cand_b = -t_in + n_in * 0.35
            cand_len = float(np.linalg.norm(cand_b))
            if cand_len > 1e-4:
                cand_b = cand_b / cand_len
                test_cand = Point(p_curr + cand_b * adjusted_offset)
                if not parcel_poly.contains(test_cand):
                    b_vec = cand_b

        test_pt = Point(p_curr + b_vec * adjusted_offset)
        if parcel_poly.contains(test_pt) or parcel_poly.exterior.distance(test_pt) < 0.35 * adjusted_offset:
            best_dir = b_vec
            best_clear = -1.0
            for ang in np.linspace(-0.75 * math.pi, 0.75 * math.pi, 17):
                c, s = math.cos(ang), math.sin(ang)
                cand_dir = np.array([c * b_vec[0] - s * b_vec[1], s * b_vec[0] + c * b_vec[1]])
                pt_cand = Point(p_curr + cand_dir * adjusted_offset)
                if not parcel_poly.contains(pt_cand):
                    clr = parcel_poly.exterior.distance(pt_cand)
                    if clr > best_clear:
                        best_clear = clr
                        best_dir = cand_dir
            b_vec = best_dir

        bisectors.append(b_vec)
    return bisectors


def deduplicate_borne_blocks_and_labels(msp) -> int:
    """AGENT 1 / 2 : Déduplication strictement SPATIALE des blocs de bornes et des étiquettes."""
    seen_blocks = []
    seen_borne_texts = []
    seen_all_texts = []
    to_delete = []

    for ins in msp.query('INSERT[layer=="bornes"]'):
        b_pos = (round(float(ins.dxf.insert.x), 2), round(float(ins.dxf.insert.y), 2))
        b_name = getattr(ins.dxf, "name", "").strip().lower()
        is_dup_block = False
        for prev_name, prev_pos in seen_blocks:
            if prev_name == b_name and math.hypot(b_pos[0] - prev_pos[0], b_pos[1] - prev_pos[1]) < 1.0:
                is_dup_block = True
                break
        if is_dup_block:
            to_delete.append(ins)
        else:
            seen_blocks.append((b_name, b_pos))

    for mtxt in msp.query("MTEXT TEXT"):
        raw_txt = getattr(mtxt.dxf, "text", "") if mtxt.dxftype() == "TEXT" else getattr(mtxt, "text", "")
        txt = str(raw_txt).strip()
        if not txt:
            continue

        ins = getattr(mtxt.dxf, "insert", None)
        if ins is None and mtxt.dxftype() == "TEXT":
            ins = getattr(mtxt.dxf, "align_point", None)
        if ins is None:
            continue
        pos = (round(float(ins.x), 2), round(float(ins.y), 2))

        # Déduplication strictement SPATIALE des labels de bornes (Bxx, BPxx, (Bxx), v.Bxx)
        if BORNE_NAME_RE.match(txt) or PAREN_BORNE_RE.match(txt):
            # Doublon uniquement si même texte ET distance spatiale < 3.0m (superposition réelle)
            is_dup = False
            for prev_txt, prev_pos in seen_borne_texts:
                if prev_txt.upper() == txt.upper() and math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) < 3.0:
                    is_dup = True
                    break
            if is_dup:
                to_delete.append(mtxt)
            else:
                seen_borne_texts.append((txt, pos))
        else:
            # Autres textes (ex: riverains, détails) : déduplication spatiale stricte (< 2.0m)
            is_dup = False
            for prev_txt, prev_pos in seen_all_texts:
                if prev_txt.upper() == txt.upper() and math.hypot(pos[0] - prev_pos[0], pos[1] - prev_pos[1]) < 2.0:
                    is_dup = True
                    break
            if is_dup:
                to_delete.append(mtxt)
            else:
                seen_all_texts.append((txt, pos))

    for ent in to_delete:
        msp.delete_entity(ent)
    return len(to_delete)


class RadialFanLeaderEngine:
    r"""
    GESTION DES CLUSTERS EXTRÊMES (RadialFanLeaderEngine) :
    Si plus de 3 bornes sont concentrées dans un rayon < 2.0 * H et qu'aucune
    position sans contact n'existe en couronne proche, déploie un éventail radial
    ordonné angulairement à rayon étendu (3.5*H à 6.0*H) avec tracé automatique
    de traits de rappel (LINE sur calque BRNS, couleur 1) garantis sans aucun croisement mutuel.
    """

    @staticmethod
    def deploy_radial_fan_leaders(
        msp,
        parcel_poly: Optional[Polygon],
        scale: int = 2000
    ) -> int:
        h_text = ANCFCC_STANDARDS["borne_text_height_ratio"] * scale
        cluster_radius_thresh = min(2.5, 0.7 * h_text)

        borne_positions = []
        for ent in msp.query("INSERT"):
            lay = ent.dxf.layer.lower()
            bname = getattr(ent.dxf, "name", "").lower()
            if "borne" in lay or "borne" in bname:
                borne_positions.append((ent.dxf.insert.x, ent.dxf.insert.y))

        if not borne_positions and parcel_poly:
            for p in list(parcel_poly.exterior.coords)[:-1]:
                borne_positions.append((p[0], p[1]))

        label_items = []
        for mtxt in msp.query("MTEXT TEXT"):
            txt = getattr(mtxt.dxf, "text", "").strip()
            if BORNE_NAME_RE.match(txt):
                pos = (mtxt.dxf.insert.x, mtxt.dxf.insert.y)
                label_items.append((txt, mtxt, pos))

        if len(borne_positions) < 3 or len(label_items) < 3:
            return 0

        # Associer chaque label à sa borne la plus proche par couplage bijectif 1-à-1
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(label_items), len(borne_positions)))
        for i, (txt, mtxt, (lx, ly)) in enumerate(label_items):
            for j, (bx, by) in enumerate(borne_positions):
                cost_matrix[i, j] = math.hypot(lx - bx, ly - by)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = []
        match_thresh = max(15.0, 4.0 * h_text)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < match_thresh:
                matched_pairs.append({
                    "txt": label_items[r][0],
                    "ent": label_items[r][1],
                    "borne_pos": borne_positions[c],
                    "label_pos": label_items[r][2]
                })

        if len(matched_pairs) < 3:
            return 0

        # Regroupement en clusters par composantes connexes (< cluster_radius_thresh)
        n = len(matched_pairs)
        adj = [[] for _ in range(n)]
        cluster_prox = max(15.0, 2.5 * h_text)
        for i in range(n):
            for j in range(i + 1, n):
                bi = matched_pairs[i]["borne_pos"]
                bj = matched_pairs[j]["borne_pos"]
                li = matched_pairs[i]["label_pos"]
                lj = matched_pairs[j]["label_pos"]
                d_b = math.hypot(bi[0] - bj[0], bi[1] - bj[1])
                # STRICTEMENT ENTRE BORNES PHYSIQUES DISTINCTES (> 0.01m)
                if 0.01 <= d_b <= cluster_radius_thresh and math.hypot(li[0] - lj[0], li[1] - lj[1]) <= cluster_prox:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        clusters = []
        for i in range(n):
            if not visited[i]:
                comp = []
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop()
                    comp.append(u)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
                distinct_bornes = set(matched_pairs[idx]["borne_pos"] for idx in comp)
                if len(distinct_bornes) >= 3:
                    clusters.append([matched_pairs[idx] for idx in comp])

        if not clusters:
            return 0

        deployed_count = 0

        for cluster in clusters:
            K = len(cluster)
            bx_avg = float(np.mean([item["borne_pos"][0] for item in cluster]))
            by_avg = float(np.mean([item["borne_pos"][1] for item in cluster]))
            C = np.array([bx_avg, by_avg])

            if parcel_poly:
                p_cent = np.array([parcel_poly.centroid.x, parcel_poly.centroid.y])
                dir_out = C - p_cent
                norm_d = float(np.linalg.norm(dir_out))
                dir_out = dir_out / norm_d if norm_d > 1e-4 else np.array([0.0, 1.0])
            else:
                dir_out = np.array([0.0, 1.0])

            base_angle = math.atan2(dir_out[1], dir_out[0])

            # Calcul de l'angle polaire de chaque borne par rapport à C
            for item in cluster:
                bx, by = item["borne_pos"]
                item["theta"] = math.atan2(by - C[1], bx - C[0])

            # Tri strictement monotone par angle polaire -> NON-CROSSING GARANTI
            cluster.sort(key=lambda it: it["theta"])

            angular_span = min(math.pi * 0.85, max(0.4, 0.35 * (K - 1)))
            start_angle = base_angle - 0.5 * angular_span

            for k, item in enumerate(cluster):
                frac = (k / (K - 1)) if K > 1 else 0.5
                target_ang = start_angle + frac * angular_span

                radius_mult = 1.0 + (k % 2) * 0.6 + (k // 2) * 0.4
                R_ext = radius_mult * h_text

                bx, by = item["borne_pos"]
                px = bx + R_ext * math.cos(target_ang)
                py = by + R_ext * math.sin(target_ang)

                if parcel_poly and parcel_poly.contains(Point(px, py)):
                    px = bx + (R_ext + 0.8 * h_text) * dir_out[0]
                    py = by + (R_ext + 0.8 * h_text) * dir_out[1]

                mtxt = item["ent"]
                mtxt.dxf.insert = (float(px), float(py), 0.0)

                msp.add_line(
                    start=(float(bx), float(by), 0.0),
                    end=(float(px), float(py), 0.0),
                    dxfattribs={"layer": "BRNS", "color": 1}
                )
                deployed_count += 1
                logger.info("🎯 [RADIAL FAN] Borne %s étalée en éventail à R=%.1f*H", item["txt"], radius_mult)

        return deployed_count


def place_parenthesized_labels_edge_parallel(
    msp, parcel_poly: Optional[Polygon], scale: int = 2000
) -> int:
    r"""
    AGENT 2 / CHANTIER 3 : Positionnement edge-parallel des bornes partagées parenthésées (Bxx).
    1. Trouve toutes les entités TEXT/MTEXT correspondant au pattern r'^\(B\d+\)$'.
    2. Pour chacune, trouve l'arête la plus proche de parcel_poly.exterior.
    3. Projette la position du label le long de cette arête, décalée vers l'extérieur
       de base_offset = ANCFCC_STANDARDS['borne_text_height_ratio'] * scale.
    4. Met à jour la position de l'entité.
    """
    if parcel_poly is None or not hasattr(parcel_poly, "exterior") or parcel_poly.is_empty:
        return 0

    base_offset = float(ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018)) * scale
    coords = list(parcel_poly.exterior.coords)
    if len(coords) < 2:
        return 0

    moved_count = 0
    p_cent = np.array([parcel_poly.centroid.x, parcel_poly.centroid.y], dtype=np.float64)

    for ent in msp.query("MTEXT TEXT"):
        raw_txt = getattr(ent.dxf, "text", "")
        if not raw_txt and hasattr(ent, "text"):
            raw_txt = ent.text
        txt = raw_txt.strip()
        if not PAREN_BORNE_RE.match(txt):
            continue

        ins = getattr(ent.dxf, "insert", None)
        if ins is None:
            continue
        p_label = np.array([ins.x, ins.y], dtype=np.float64)

        best_edge = None
        min_dist = float("inf")
        best_proj = None
        best_n_out = None

        for i in range(len(coords) - 1):
            p1 = np.array(coords[i], dtype=np.float64)[:2]
            p2 = np.array(coords[i + 1], dtype=np.float64)[:2]
            v = p2 - p1
            seg_len = float(np.linalg.norm(v))
            if seg_len < 1e-6:
                continue
            u_tan = v / seg_len

            # Projection du label sur le segment
            t = float(np.dot(p_label - p1, u_tan))
            t_clamped = max(0.0, min(seg_len, t))
            proj_pt = p1 + t_clamped * u_tan
            dist = float(np.linalg.norm(p_label - proj_pt))

            if dist < min_dist:
                min_dist = dist
                best_edge = (p1, p2, seg_len)
                best_proj = proj_pt

                # Normale sortante de l'arête
                n1 = np.array([-u_tan[1], u_tan[0]])
                n2 = np.array([u_tan[1], -u_tan[0]])

                mid_pt = 0.5 * (p1 + p2)
                t1 = Point(mid_pt[0] + 0.2 * n1[0], mid_pt[1] + 0.2 * n1[1])
                t2 = Point(mid_pt[0] + 0.2 * n2[0], mid_pt[1] + 0.2 * n2[1])

                if parcel_poly.contains(t1) and not parcel_poly.contains(t2):
                    n_out = n2
                elif parcel_poly.contains(t2) and not parcel_poly.contains(t1):
                    n_out = n1
                else:
                    d_c = mid_pt - p_cent
                    n_out = n1 if np.dot(n1, d_c) >= 0.0 else n2

                best_n_out = n_out

        if best_edge is None or best_proj is None or best_n_out is None or min_dist > 3.0:
            continue

        new_pos = best_proj + best_n_out * base_offset
        disp = new_pos - p_label
        if 0.01 < float(np.linalg.norm(disp)) <= 3.0:
            # Guard: Ne déplacer que si l'ancien label était à l'intérieur du polygone ou si la nouvelle position ne s'éloigne pas
            old_edge_dist = float(parcel_poly.exterior.distance(Point(p_label[0], p_label[1])))
            new_pt = Point(float(new_pos[0]), float(new_pos[1]))
            new_edge_dist = float(parcel_poly.exterior.distance(new_pt))
            was_inside = parcel_poly.contains(Point(p_label[0], p_label[1]))

            if was_inside or new_edge_dist <= old_edge_dist + 0.5:
                z_val = float(getattr(ins, "z", 0.0))
                ent.dxf.insert = (float(new_pos[0]), float(new_pos[1]), z_val)
                moved_count += 1
                logger.info("🎯 [EDGE PARALLEL] Label %s repositionné le long de l'arête à (%.2f, %.2f)", txt, new_pos[0], new_pos[1])

    return moved_count


def resolve_borne_label_collisions_sat(
    msp, parcel_poly: Optional[Polygon], scale: int = 2000
) -> int:
    """AGENT 2 : Résolution globale sans collision (OR-Tools CP-SAT + Relaxation SAT)."""
    # Étape préalable : Déploiement des éventails radiaux pour les clusters extrêmes
    deployed_fans = RadialFanLeaderEngine.deploy_radial_fan_leaders(msp, parcel_poly, scale)

    h_text = ANCFCC_STANDARDS["borne_text_height_ratio"] * scale
    gap_mult = min(0.35, max(0.15, float(ANCFCC_STANDARDS.get("min_borne_label_gap_mult", 0.15))))
    pad_y = 0.5 * (1.0 + gap_mult) * h_text

    borne_positions = []
    for ent in msp.query("INSERT"):
        bname = getattr(ent.dxf, "name", "").lower()
        if "borne" in ent.dxf.layer.lower() or "borne" in bname:
            bp = (ent.dxf.insert.x, ent.dxf.insert.y)
            if parcel_poly is None or parcel_poly.boundary.distance(Point(bp)) < 1.5:
                borne_positions.append(bp)
    if not borne_positions and parcel_poly:
        for p in list(parcel_poly.exterior.coords)[:-1]:
            borne_positions.append((p[0], p[1]))

    items = []
    from exact_text_obb import get_exact_text_obb
    raw_labels = []
    seen_borne_spat = []
    for mtxt in msp.query("MTEXT TEXT"):
        raw_t = getattr(mtxt.dxf, "text", "") if mtxt.dxftype() == "TEXT" else getattr(mtxt, "text", "")
        txt = str(raw_t).strip()
        is_ext = txt.startswith("v.") or (txt.startswith("(") and txt.endswith(")"))
        if BORNE_NAME_RE.match(txt) and not is_ext:
            ins = getattr(mtxt.dxf, "insert", None)
            if ins is None and mtxt.dxftype() == "TEXT":
                ins = getattr(mtxt.dxf, "align_point", None)
            if ins is None:
                continue
            pos_xy = (float(ins.x), float(ins.y))
            # Doublon uniquement si même texte ET distance spatiale < 3.0m
            is_dup = any(
                prev_t.upper() == txt.upper() and math.hypot(pos_xy[0] - prev_p[0], pos_xy[1] - prev_p[1]) < 3.0
                for prev_t, prev_p in seen_borne_spat
            )
            if is_dup:
                msp.delete_entity(mtxt)
                continue
            seen_borne_spat.append((txt, pos_xy))
            pos = np.array([mtxt.dxf.insert.x, mtxt.dxf.insert.y], dtype=np.float64)
            p_obb = get_exact_text_obb(mtxt, pad_ratio=0.0)
            if p_obb and not p_obb.is_empty:
                c = p_obb.centroid
                c_pos = np.array([c.x, c.y], dtype=np.float64)
                minx, miny, maxx, maxy = p_obb.bounds
                half_w = max(0.5 * (maxx - minx), 0.5 * h_text)
                half_h = max(0.5 * (maxy - miny), pad_y)
            else:
                w_est, _ = estimate_exact_text_dimensions(txt, h_text, "CQ_ARIAL")
                c_pos = np.array([pos[0] + 0.50 * w_est, pos[1] + 0.50 * h_text], dtype=np.float64)
                half_w = 0.50 * w_est
                half_h = pad_y
            obb = OrientedBox2D(center=c_pos, half_extents=np.array([half_w + 0.08 * h_text, half_h + 0.08 * h_text]))
            raw_labels.append([txt, mtxt, obb, pos.copy(), c_pos.copy()])

    if not raw_labels:
        return 0

    if borne_positions:
        from scipy.optimize import linear_sum_assignment
        cost_matrix = np.zeros((len(raw_labels), len(borne_positions)))
        for i, it in enumerate(raw_labels):
            for j, bp in enumerate(borne_positions):
                cost_matrix[i, j] = math.hypot(it[3][0] - bp[0], it[3][1] - bp[1])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            it = raw_labels[r]
            bp = borne_positions[c]
            b_pos = np.array([bp[0], bp[1]], dtype=np.float64)
            it.append(b_pos)
            items.append(it)
    else:
        for it in raw_labels:
            it.append(it[3].copy())
            items.append(it)

    cartouche_admin_kw = (
        "caidat", "caïdat", "prefecture", "préfecture", "commune",
        "situee", "située", "province", "conservation", "direction", "royaume", "cadastre"
    )
    other_obstacles = []
    for mtxt in msp.query("MTEXT TEXT"):
        txt = getattr(mtxt.dxf, "text", "").strip()
        tl = txt.lower()
        is_ext = txt.startswith("v.") or (txt.startswith("(") and txt.endswith(")"))
        if (not BORNE_NAME_RE.match(txt) or is_ext) and len(txt) >= 2 and not any(k in tl for k in cartouche_admin_kw):
            p_obb = get_exact_text_obb(mtxt, pad_ratio=0.0)
            if p_obb and not p_obb.is_empty:
                other_obstacles.append((txt, p_obb))

    n = len(items)
    moved = 0
    max_disp = max(12.0, 2.5 * h_text)
    max_drift = max(2.5, 0.65 * h_text)
    bnd = parcel_poly.boundary if parcel_poly is not None else None


    for _ in range(30):
        any_collision = False
        for i in range(n):
            for j in range(i + 1, n):
                obb_i = items[i][2]
                obb_j = items[j][2]
                collides, mtv_axis, depth = sat_overlap_mtv(obb_i, obb_j)
                if collides:
                    margin = 0.20 * h_text
                    # Règle de décalage étagé ANCFCC: Si deux étiquettes de bornes se chevauchent,
                    # privilégier la séparation verticale (Y) pour étager les étiquettes proprement
                    dy = obb_j.center[1] - obb_i.center[1]
                    y_overlap = (obb_i.half_extents[1] + obb_j.half_extents[1]) - abs(dy)
                    if y_overlap > 0:
                        y_sep = 0.5 * (y_overlap + margin)
                        step_y = y_sep if dy >= 0 else -y_sep
                        obb_i.update_pose(obb_i.center - np.array([0.0, step_y]))
                        obb_j.update_pose(obb_j.center + np.array([0.0, step_y]))
                    else:
                        step = mtv_axis * (0.5 * (depth + margin))
                        obb_i.update_pose(obb_i.center - step)
                        obb_j.update_pose(obb_j.center + step)
                    any_collision = True
                    moved += 1

        # Répulsion stricte de la limite de parcelle (0.00% collision avec la limite)
        if parcel_poly is not None and bnd is not None:
            p_cent = np.array([parcel_poly.centroid.x, parcel_poly.centroid.y])
            for it in items:
                obb = it[2]
                p_box = obb.to_polygon()
                if p_box.intersects(bnd):
                    proj_d = bnd.project(Point(obb.center[0], obb.center[1]))
                    near_pt = bnd.interpolate(proj_d)
                    eps = min(1.0, bnd.length * 0.01)
                    p1 = bnd.interpolate(max(0.0, proj_d - eps))
                    p2 = bnd.interpolate(min(bnd.length, proj_d + eps))
                    tan_v = np.array([p2.x - p1.x, p2.y - p1.y])
                    tan_len = float(np.linalg.norm(tan_v))
                    if tan_len > 1e-4:
                        tan_u = tan_v / tan_len
                        n1 = np.array([-tan_u[1], tan_u[0]])
                        n2 = np.array([tan_u[1], -tan_u[0]])
                        d_cent = np.array([near_pt.x, near_pt.y]) - p_cent
                        out_vec = n1 if np.dot(n1, d_cent) >= 0.0 else n2
                    else:
                        out_vec = np.array([near_pt.x, near_pt.y]) - p_cent
                        out_vec = out_vec / (float(np.linalg.norm(out_vec)) or 1.0)

                    orig_inside = parcel_poly.contains(Point(it[3][0], it[3][1]))
                    clear_dist = max(obb.half_extents[1], 0.6 * h_text) + 0.25 * h_text
                    if orig_inside:
                        target_pos = np.array([near_pt.x, near_pt.y]) - out_vec * clear_dist
                    else:
                        target_pos = np.array([near_pt.x, near_pt.y]) + out_vec * clear_dist
                    obb.update_pose(target_pos)
                    any_collision = True
                    moved += 1

        # Répulsion mutuelle stricte contre les autres textes du plan (ex: Chaâbat, Pistes)
        if other_obstacles:
            for it in items:
                obb = it[2]
                bp = it[5] if len(it) > 5 else None
                p_poly = obb.to_polygon()
                for o_txt, o_poly in other_obstacles:
                    if p_poly.intersects(o_poly):
                        o_c = o_poly.centroid
                        dx = obb.center[0] - o_c.x
                        dy = obb.center[1] - o_c.y
                        minx_o, miny_o, maxx_o, maxy_o = o_poly.bounds
                        minx_b, miny_b, maxx_b, maxy_b = p_poly.bounds
                        x_ov = min(maxx_b, maxx_o) - max(minx_b, minx_o)
                        y_ov = min(maxy_b, maxy_o) - max(miny_b, miny_o)
                        if x_ov <= y_ov and x_ov > 0:
                            x_push = x_ov + 0.35 * h_text
                            cand_pos_x = obb.center + np.array([x_push, 0.0])
                            cand_neg_x = obb.center - np.array([x_push, 0.0])
                            if parcel_poly is not None and bnd is not None:
                                clr_pos_x = bnd.distance(Point(cand_pos_x[0], cand_pos_x[1])) if parcel_poly.contains(Point(cand_pos_x[0], cand_pos_x[1])) else -1.0
                                clr_neg_x = bnd.distance(Point(cand_neg_x[0], cand_neg_x[1])) if parcel_poly.contains(Point(cand_neg_x[0], cand_neg_x[1])) else -1.0
                                if clr_pos_x >= 0 and clr_neg_x >= 0:
                                    step_x = x_push if clr_pos_x >= clr_neg_x else -x_push
                                elif clr_pos_x >= 0:
                                    step_x = x_push
                                elif clr_neg_x >= 0:
                                    step_x = -x_push
                                else:
                                    step_x = -x_push if (bp is not None and bp[0] < o_c.x) else x_push
                            elif bp is not None and abs(bp[0] - o_c.x) > 1.0:
                                step_x = -x_push if bp[0] < o_c.x else x_push
                            else:
                                step_x = x_push if dx >= 0 else -x_push
                            obb.update_pose(obb.center + np.array([step_x, 0.0]))
                        elif y_ov > 0:
                            y_push = y_ov + 0.35 * h_text
                            cand_pos_y = obb.center + np.array([0.0, y_push])
                            cand_neg_y = obb.center - np.array([0.0, y_push])
                            if parcel_poly is not None and bnd is not None:
                                clr_pos_y = bnd.distance(Point(cand_pos_y[0], cand_pos_y[1])) if parcel_poly.contains(Point(cand_pos_y[0], cand_pos_y[1])) else -1.0
                                clr_neg_y = bnd.distance(Point(cand_neg_y[0], cand_neg_y[1])) if parcel_poly.contains(Point(cand_neg_y[0], cand_neg_y[1])) else -1.0
                                if clr_pos_y >= 0 and clr_neg_y >= 0:
                                    step_y = y_push if clr_pos_y >= clr_neg_y else -y_push
                                elif clr_pos_y >= 0:
                                    step_y = y_push
                                elif clr_neg_y >= 0:
                                    step_y = -y_push
                                else:
                                    step_y = -y_push if (bp is not None and bp[1] < o_c.y) else y_push
                            elif bp is not None and abs(bp[1] - o_c.y) > 1.0:
                                step_y = -y_push if bp[1] < o_c.y else y_push
                            else:
                                step_y = y_push if dy >= 0 else -y_push
                            obb.update_pose(obb.center + np.array([0.0, step_y]))
                        any_collision = True
                        moved += 1
                        p_poly = obb.to_polygon()

        if not any_collision:
            break

        # Ancrage et dérive : limitation du glissement par rapport à l'origine et maintien sur la borne
        for it in items:
            orig_c = it[4]
            anchor = it[5] if len(it) > 5 and it[5] is not None else it[4]
            orig_d_anchor = float(np.linalg.norm(orig_c - anchor))
            eff_max_disp = max(max_disp, orig_d_anchor + 0.5)

            # 1. Limitation stricte du glissement par rapport à la position d'origine (anti-dérive)
            drift = it[2].center - orig_c
            drift_len = float(np.linalg.norm(drift))
            if drift_len > max_drift:
                it[2].update_pose(orig_c + (drift / drift_len) * max_drift)

            # 2. Limitation par rapport au point d'ancrage (borne physique)
            disp = it[2].center - anchor
            d_len = float(np.linalg.norm(disp))
            if d_len > eff_max_disp:
                target_cand = anchor + (disp / d_len) * eff_max_disp
                it[2].update_pose(target_cand)

    for item in items:
        mtxt = item[1]
        obb = item[2]
        orig_ins = item[3]
        orig_c = item[4]
        # Écrire dans le DXF dès qu'une modification anti-collision (> 0.01m) est validée
        disp = obb.center - orig_c
        if float(np.linalg.norm(disp)) > 0.01:
            mtxt.dxf.insert = (float(orig_ins[0] + disp[0]), float(orig_ins[1] + disp[1]), 0.0)

    # CHANTIER 3 : Positionnement edge-parallel des labels (Bxx)
    paren_moved = place_parenthesized_labels_edge_parallel(msp, parcel_poly, scale)

    return moved + deployed_fans + paren_moved


def fix_borne_positions_strict(
    msp,
    parcel,
    core_mod,
    allow_scale: bool = True,
    max_move: float = 2.0,
    min_scale: float = 0.85,
    orig_positions: Optional[dict] = None
) -> Tuple[int, int, int]:
    """
    Contrainte stricte des labels Bxx issue d'IFE Tools v4.0.7:
      - Point d'insertion ET bounding box entière strictement dans la parcelle
      - Sans chevauchement avec les autres bornes
      - Déplacement max 2.0 m vers le centroïde
      - Réduction d'échelle progressive si nécessaire
    """
    borne_re = BORNE_NAME_RE
    cache = core_mod._bbox_cache

    def _ins(e):
        return core_mod.get_insert(e)

    def _bb(e):
        return cache.get(e, msp)

    b_ents = [
        e for e in msp
        if e.dxftype() in ("TEXT", "MTEXT")
        and borne_re.match(core_mod.get_text_content(e).strip())
    ]

    moved, scaled, matched = 0, 0, 0
    cx, cy = parcel.centroid.x, parcel.centroid.y

    for be in b_ents:
        bb = _bb(be)
        ix, iy = _ins(be)
        ipt = _Pt(ix, iy)

        if bb and parcel.contains(bb) and parcel.contains(ipt):
            matched += 1
            continue

        best_x, best_y = ix, iy
        found = False

        for step in range(1, int(max_move / 0.1) + 1):
            dist = step * 0.1
            angle = math.atan2(cy - iy, cx - ix)
            nx = ix + dist * math.cos(angle)
            ny = iy + dist * math.sin(angle)
            dx, dy = nx - ix, ny - iy
            c_bb = _Pt(nx, ny).buffer(0.3)
            if parcel.contains(c_bb):
                core_mod.move_dxf_text(be, dx, dy)
                cache.invalidate([be])
                moved += 1
                found = True
                break

        if not found and allow_scale:
            if hasattr(be.dxf, "height"):
                cur_h = be.dxf.height
                for sc in (0.95, 0.90, min_scale):
                    be.dxf.height = cur_h * sc
                    cache.invalidate([be])
                    scaled += 1
                    break

    return moved, scaled, matched
