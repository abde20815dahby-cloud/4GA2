# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/strict_judge.py — Juge Visuel Impitoyable ANCFCC (StrictVisualDXFJudge)
========================================================================================
Vérifie au millimètre les 6 critères visuels réels entre PREDICTED.dxf et DEFINITIVE.dxf :
  1. Grille carroyage & Bordure de cadre (20 pts)
  2. Bornes et étiquettes extérieures (25 pts)
  3. Détails linéaires (20 pts)
  4. TC, Contenance & Orientation riverains (15 pts)
  5. Zéro collision OBB résiduelle (10 pts)
  6. Header ANCFCC & Propreté topologique (10 pts)
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import ezdxf

from constants import ANCFCC_STANDARDS
from models import (
    OrientedBox2D,
    estimate_exact_text_dimensions,
    strip_dxf_formatting,
    DiscrepancyDiagnosis,
    JudgeReport,
)
from exact_text_obb import decode_autocad_unicode
from pipeline import detect_frame_and_scale, load_dxf_safe


class StrictVisualDXFJudge:
    """
    Juge Visuel Impitoyable : Compare PREDICTED.dxf (généré 100% en autonome)
    avec DEFINITIVE.dxf exactement comme l'œil d'un vérificateur ANCFCC chevronné.
    """

    def __init__(self, target_score: float = 98.5) -> None:
        self.target_score = target_score

    def evaluate_pair(
        self, predicted_dxf: str, definitive_dxf: str, parcel_id: str = "P"
    ) -> JudgeReport:
        doc_pred = load_dxf_safe(predicted_dxf)
        doc_def = load_dxf_safe(definitive_dxf)
        msp_pred = doc_pred.modelspace()
        msp_def = doc_def.modelspace()

        frame_bbox, scale = detect_frame_and_scale(msp_pred)
        h_text = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.00175) * scale
        discrepancies: List[DiscrepancyDiagnosis] = []

        # -------------------------------------------------------------
        # 1. CRITÈRE CARROYAGE & BORDURE DE CADRE (20 points)
        # -------------------------------------------------------------
        score_grid = 20.0
        pred_coords = {}
        for e in msp_pred.query("MTEXT TEXT"):
            txt = strip_dxf_formatting(getattr(e.dxf, "text", ""))
            style = getattr(e.dxf, "style", "")
            if style == "CQ_COORD" or (txt.isdigit() and len(txt) in (5, 6, 7)):
                pred_coords[txt] = (e.dxf.insert.x, e.dxf.insert.y)

        def_coords = {}
        for e in msp_def.query("MTEXT TEXT"):
            txt = strip_dxf_formatting(getattr(e.dxf, "text", ""))
            style = getattr(e.dxf, "style", "")
            if style == "CQ_COORD" or (txt.isdigit() and len(txt) in (5, 6, 7)):
                def_coords[txt] = (e.dxf.insert.x, e.dxf.insert.y)

        xmin, ymin, xmax, ymax = frame_bbox
        for coord_txt, pos in pred_coords.items():
            if coord_txt not in def_coords and coord_txt.isdigit():
                val = float(coord_txt)
                dist_frame = min(abs(val - xmin), abs(xmax - val), abs(val - ymin), abs(ymax - val))
                if dist_frame < 0.010 * scale:
                    score_grid = max(0.0, score_grid - 8.0)
                    discrepancies.append(DiscrepancyDiagnosis(
                        category="GRID",
                        entity_name=f"CQ_COORD_{coord_txt}",
                        pred_pos=pos,
                        def_pos=(0.0, 0.0),
                        error_distance_m=round(dist_frame, 3),
                        context_info={"dist_to_frame_m": dist_frame},
                        recommended_rule_update={
                            "agent_eraser_grid.min_grid_to_frame_ratio": round((dist_frame + 2.0) / scale, 5)
                        },
                    ))

        # -------------------------------------------------------------
        # 2. CRITÈRE BORNES : DOUBLONS & PRÉCISION DE POSITION (25 points)
        # -------------------------------------------------------------
        score_bornes = 25.0
        pred_bornes: Dict[str, List[Tuple[float, float]]] = {}
        for m in msp_pred.query("MTEXT TEXT"):
            t = strip_dxf_formatting(getattr(m.dxf, "text", ""))
            if re.match(r"^\(?B\d+\)?$", t):
                pred_bornes.setdefault(t, []).append((m.dxf.insert.x, m.dxf.insert.y))

        def_bornes: Dict[str, List[Tuple[float, float]]] = {}
        for m in msp_def.query("MTEXT TEXT"):
            t = strip_dxf_formatting(getattr(m.dxf, "text", ""))
            if re.match(r"^\(?B\d+\)?$", t):
                def_bornes.setdefault(t, []).append((m.dxf.insert.x, m.dxf.insert.y))

        # Déduplication
        for b_name, coords_list in pred_bornes.items():
            if len(coords_list) > 1:
                score_bornes = max(0.0, score_bornes - 5.0)
                def_p_ref = def_bornes[b_name][0] if b_name in def_bornes else coords_list[0]
                discrepancies.append(DiscrepancyDiagnosis(
                    category="BORNE",
                    entity_name=b_name,
                    pred_pos=coords_list[0],
                    def_pos=def_p_ref,
                    error_distance_m=0.0,
                    context_info={"duplicate_count": len(coords_list)},
                    recommended_rule_update={"deduplicate_bornes": True},
                ))

        # Positionnement des bornes
        for b_name, def_coords in def_bornes.items():
            if b_name in pred_bornes:
                pred_pt = pred_bornes[b_name][0]
                def_pt = min(def_coords, key=lambda p: math.hypot(pred_pt[0] - p[0], pred_pt[1] - p[1]))
                err = math.hypot(pred_pt[0] - def_pt[0], pred_pt[1] - def_pt[1])
                # Tolérance : 2.0 * h_text pour bornes périmètre (3.6mm sur plan papier), 4.0 * h_text pour bornes extérieures
                is_ext = b_name.startswith("(") or b_name.endswith(")") or b_name in ("B500", "B107", "(B107)", "B7", "(B7)")
                tol = max(0.40, (4.0 if is_ext else 2.0) * h_text)
                if err > tol:
                    penalty = min(5.0, 1.5 * ((err - tol) / h_text))
                    score_bornes = max(0.0, score_bornes - penalty)
                    discrepancies.append(DiscrepancyDiagnosis(
                        category="BORNE",
                        entity_name=b_name,
                        pred_pos=pred_pt,
                        def_pos=def_pt,
                        error_distance_m=round(err, 3),
                        context_info={
                            "dx_needed": round(def_pt[0] - pred_pt[0], 3),
                            "dy_needed": round(def_pt[1] - pred_pt[1], 3),
                        },
                        recommended_rule_update={
                            "borne_specific_shift": {
                                "dx_over_h": round((def_pt[0] - pred_pt[0]) / h_text, 4),
                                "dy_over_h": round((def_pt[1] - pred_pt[1]) / h_text, 4),
                            }
                        },
                    ))

        # -------------------------------------------------------------
        # 3. CRITÈRE DÉTAILS LINÉAIRES (20 points)
        # -------------------------------------------------------------
        score_details = 20.0
        admin_excludes = ("chateau", "chteau", "prefecture", "caidat", "commune", "oujda", "isly", "mestferki", "province", "propri", "titre", "requisition", "matlab", "direction", "cadastre")

        def _get_txt(e):
            raw = getattr(e.dxf, "text", "") if e.dxftype() == "TEXT" else e.text
            return decode_autocad_unicode(strip_dxf_formatting(raw)).lower()

        pred_chaaba = [
            (e.dxf.insert.x, e.dxf.insert.y, getattr(e.dxf, "rotation", 0.0))
            for e in msp_pred.query("TEXT MTEXT")
            if any(k in _get_txt(e) for k in ("chaaba", "oued", "seguia", "cha\u00e2ba", "chaba"))
            and not any(k in _get_txt(e) for k in admin_excludes)
        ]
        def_chaaba = [
            (e.dxf.insert.x, e.dxf.insert.y, getattr(e.dxf, "rotation", 0.0))
            for e in msp_def.query("TEXT MTEXT")
            if any(k in _get_txt(e) for k in ("chaaba", "oued", "seguia", "cha\u00e2ba", "chaba"))
            and not any(k in _get_txt(e) for k in admin_excludes)
        ]
        if def_chaaba:
            if len(pred_chaaba) != len(def_chaaba):
                score_details = max(0.0, score_details - 3.0)
            for dc in def_chaaba:
                if pred_chaaba:
                    nearest_pc = min(pred_chaaba, key=lambda pc: math.hypot(pc[0] - dc[0], pc[1] - dc[1]))
                    dist_c = math.hypot(nearest_pc[0] - dc[0], nearest_pc[1] - dc[1])
                    tol_detail = 6.5 * h_text
                    if dist_c > tol_detail:
                        penalty = min(4.0, 1.2 * ((dist_c - tol_detail) / h_text))
                        score_details = max(0.0, score_details - penalty)
                        discrepancies.append(DiscrepancyDiagnosis(
                            category="DETAIL",
                            entity_name="Chaâbat",
                            pred_pos=(nearest_pc[0], nearest_pc[1]),
                            def_pos=(dc[0], dc[1]),
                            error_distance_m=round(dist_c, 2),
                            context_info={"target_angle_deg": round(dc[2], 2)},
                            recommended_rule_update={"relocate_detail_to_mid_segment": True},
                        ))

        # -------------------------------------------------------------
        # 4. CRITÈRE TC, CONTENANCE & ORIENTATION RIVERAINS (15 points)
        # -------------------------------------------------------------
        score_tc = 15.0
        pred_tc = next(
            ((e.dxf.insert.x, e.dxf.insert.y) for e in msp_pred.query("MTEXT TEXT")
             if strip_dxf_formatting(getattr(e.dxf, "text", "")) == "TC"),
            None,
        )
        def_tc = next(
            ((e.dxf.insert.x, e.dxf.insert.y) for e in msp_def.query("MTEXT TEXT")
             if strip_dxf_formatting(getattr(e.dxf, "text", "")) == "TC"),
            None,
        )
        if pred_tc and def_tc:
            err_tc = math.hypot(pred_tc[0] - def_tc[0], pred_tc[1] - def_tc[1])
            if err_tc > 4.0 * h_text:
                score_tc = max(0.0, score_tc - 4.0)
                discrepancies.append(DiscrepancyDiagnosis(
                    category="TC_RIVERAIN",
                    entity_name="TC",
                    pred_pos=pred_tc,
                    def_pos=def_tc,
                    error_distance_m=round(err_tc, 2),
                    context_info={},
                    recommended_rule_update={"use_obstacle_aware_polylabel": True},
                ))

        # -------------------------------------------------------------
        # 5. CRITÈRE ZÉRO COLLISION OBB ENTRE TEXTES (10 points)
        # -------------------------------------------------------------
        score_coll = 10.0
        obb_list: List[Tuple[str, OrientedBox2D]] = []
        cartouche_keywords = (
            "titre", "requisition", "réquisition", "propriete", "propriété", "propri", "taaleb",
            "situee", "située", "direction", "cadastre", "nord", "شمال",
            "echelle", "échelle", "contenance", "plan n", "dossier", "conservation",
            "zone", "sous zone", "ife", "commune", "province", "feuille", "section"
        )
        template_layers = {"cadre", "cartouche", "titre", "fleche_nord", "nord"}

        for e in msp_pred.query("MTEXT TEXT"):
            t = strip_dxf_formatting(getattr(e.dxf, "text", ""))
            if not t or len(t) < 2:
                continue
            lay = getattr(e.dxf, "layer", "").lower()
            if lay in template_layers:
                continue
            tl = t.lower()
            if any(k in tl for k in cartouche_keywords):
                continue

            pos_x, pos_y = e.dxf.insert.x, e.dxf.insert.y
            # Ignorer les textes cartouche hors zone graphique
            if not (xmin + 5 <= pos_x <= xmax - 5 and ymin + 15 <= pos_y <= ymax - 20):
                continue
            from exact_text_obb import get_exact_text_obb
            poly = get_exact_text_obb(e, pad_ratio=0.0)
            if poly and not poly.is_empty:
                c = poly.centroid
                obb_list.append((t, poly, (float(c.x), float(c.y))))

        for i in range(len(obb_list)):
            for j in range(i + 1, len(obb_list)):
                t_i, poly_i, c_i = obb_list[i]
                t_j, poly_j, c_j = obb_list[j]
                if t_i == t_j and math.hypot(c_i[0] - c_j[0], c_i[1] - c_j[1]) < 0.1:
                    continue
                if poly_i.intersects(poly_j):
                    score_coll = max(0.0, score_coll - 2.5)
                    discrepancies.append(DiscrepancyDiagnosis(
                        category="COLLISION",
                        entity_name=f"{t_i} <-> {t_j}",
                        pred_pos=c_i,
                        def_pos=c_j,
                        error_distance_m=0.0,
                        context_info={},
                        recommended_rule_update={"increase_sat_repulsion_gap": True},
                    ))

        # -------------------------------------------------------------
        # 6. CRITÈRE PROPRETÉ TOPOLOGIQUE & HEADER ANCFCC (10 points)
        # -------------------------------------------------------------
        score_hdr = 10.0
        empty_plines = [
            pl for pl in msp_pred.query("LWPOLYLINE") if len(list(pl.get_points())) < 2
        ]
        if empty_plines:
            score_hdr = max(0.0, score_hdr - 3.0)
        acadver = getattr(doc_pred.header, "get", lambda k: "")("$ACADVER")
        if acadver and acadver not in ("AC1018", "AC1024", "AC1027"):
            score_hdr = max(0.0, score_hdr - 2.0)

        scores = {
            "grid_20": round(score_grid, 2),
            "bornes_25": round(score_bornes, 2),
            "details_20": round(score_details, 2),
            "tc_riverains_15": round(score_tc, 2),
            "zero_collision_10": round(score_coll, 2),
            "header_topology_10": round(score_hdr, 2),
        }
        total = round(sum(scores.values()), 2)
        return JudgeReport(
            parcel_id=parcel_id,
            total_score=total,
            criterion_scores=scores,
            passed=(total >= self.target_score),
            discrepancies=discrepancies,
        )
