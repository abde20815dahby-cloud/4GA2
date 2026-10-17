# -*- coding: utf-8 -*-
"""
01_PRODUCTION_CLEANER_PIPELINE/dxf_cleaner/hierarchical_agency.py
================================================================================
ARCHITECTURE HIÉRARCHIQUE STRICTE À 3 NIVEAUX :
  - NIVEAU 1 : L'ÉQUIPE DES 5 OUVRIERS SPÉCIALISÉS (WorkersTeam / L-Kheddama)
  - NIVEAU 2 : LE CONTRÔLEUR QUALITÉ SÉVÈRE (SupervisorController)
  - NIVEAU 3 : LE CHEF INGÉNIEUR GÉOMÈTRE ANCFCC (ChiefSurveyorBoss)
================================================================================
"""

from __future__ import annotations
import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import ezdxf
from ezdxf.document import Drawing
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, Point, LineString

from constants import ANCFCC_STANDARDS
from models import (
    OrientedBox2D,
    sat_overlap_mtv,
    estimate_exact_text_dimensions,
    strip_dxf_formatting,
)
from geometry_clipping import (
    sanitize_zero_and_empty_geometries,
    deconflict_carroyage_near_frame,
)
from borne_ops import (
    BORNE_NAME_RE,
    deduplicate_borne_blocks_and_labels,
    resolve_borne_label_collisions_sat,
)
from linear_centering import (
    center_linear_details_and_fix_angles,
    normalize_cadastral_angle,
)
from standards import (
    apply_ancfcc_header_and_xdata,
    lock_rejcad_centimetric_precision,
)

logger = logging.getLogger("dxf_cleaner.hierarchical_agency")


# ============================================================================
# STRUCTURES D'ORDRES DE CORRECTION (TICKETS DU CONTRÔLEUR & DU CHEF)
# ============================================================================

@dataclass
class WorkOrderTicket:
    """Ordre de correction nominatif envoyé par le Contrôleur à un Ouvrier fautif."""
    target_worker: str          # "WORKER_1_GRID" | "WORKER_2_BORNE" | "WORKER_3_DETAIL" | "WORKER_4_TC" | "WORKER_5_STD"
    error_code: str             # Code précis de l'erreur détectée
    entity_info: str            # Identifiant ou texte de l'entité fautive
    location: Tuple[float, float]
    severity_multiplier: float  # Multiplicateur de force pour obliger l'ouvrier à corriger


@dataclass
class ChiefApprovalVisa:
    """Certificat final délivré uniquement par le Chef Ingénieur Géomètre."""
    approved: bool
    quality_score: float
    rounds_needed: int
    controller_tickets_resolved: int
    chief_veto_reasons: List[str] = field(default_factory=list)


# ============================================================================
# NIVEAU 1 : L'ÉQUIPE DES 5 OUVRIERS SPÉCIALISÉS (L-KHEDDAMA)
# ============================================================================

class WorkersTeam:
    """Les 5 Ouvriers qui exécutent les tâches et obéissent aux ordres du Contrôleur."""

    @staticmethod
    def run_worker_1_eraser_grid(
        msp, frame_bbox: Tuple[float, float, float, float], scale: int, boost: float = 1.0
    ) -> int:
        old_ratio = ANCFCC_STANDARDS.get("min_grid_to_frame_ratio", 0.0075)
        ANCFCC_STANDARDS["min_grid_to_frame_ratio"] = old_ratio * boost
        n1 = sanitize_zero_and_empty_geometries(msp)
        n2 = deconflict_carroyage_near_frame(msp, frame_bbox, scale)
        n3 = deduplicate_borne_blocks_and_labels(msp)
        ANCFCC_STANDARDS["min_grid_to_frame_ratio"] = old_ratio
        return n1 + n2 + n3

    @staticmethod
    def run_worker_2_borne_mover(
        msp, parcel_poly: Optional[Polygon], scale: int, boost: float = 1.0
    ) -> int:
        old_gap = ANCFCC_STANDARDS.get("min_borne_label_gap_mult", 0.35)
        ANCFCC_STANDARDS["min_borne_label_gap_mult"] = old_gap * boost
        moved = resolve_borne_label_collisions_sat(msp, parcel_poly, scale)
        ANCFCC_STANDARDS["min_borne_label_gap_mult"] = old_gap
        return moved

    @staticmethod
    def run_worker_3_and_4_details_riverains(
        msp,
        parcel_poly: Optional[Polygon],
        frame_bbox: Tuple[float, float, float, float],
        scale: int,
    ) -> int:
        return center_linear_details_and_fix_angles(msp, parcel_poly, frame_bbox, scale)

    @staticmethod
    def run_worker_5_standards(doc: Drawing, scale: int) -> None:
        apply_ancfcc_header_and_xdata(doc, scale)
        lock_rejcad_centimetric_precision(doc.modelspace())

    @staticmethod
    def execute_targeted_fix(
        doc: Drawing,
        msp,
        parcel_poly: Optional[Polygon],
        frame_bbox: Tuple[float, float, float, float],
        scale: int,
        ticket: WorkOrderTicket,
    ) -> None:
        """Exécute une correction chirurgicale immédiate suite au cri du Contrôleur."""
        h_text = ANCFCC_STANDARDS["borne_text_height_ratio"] * scale

        if ticket.target_worker == "WORKER_1_GRID":
            WorkersTeam.run_worker_1_eraser_grid(msp, frame_bbox, scale, boost=ticket.severity_multiplier)

        elif ticket.target_worker == "WORKER_2_BORNE":
            # Séparation ciblée via le solveur SAT avec boost de sévérité
            WorkersTeam.run_worker_2_borne_mover(msp, parcel_poly, scale, boost=ticket.severity_multiplier)

        elif ticket.target_worker == "WORKER_3_DETAIL":
            WorkersTeam.run_worker_3_and_4_details_riverains(msp, parcel_poly, frame_bbox, scale)

        elif ticket.target_worker == "WORKER_4_TC":
            if ticket.error_code == "TEXT_OBB_COLLISION":
                loc = ticket.location
                for e in msp.query("MTEXT TEXT"):
                    ix, iy = e.dxf.insert.x, e.dxf.insert.y
                    if math.hypot(ix - loc[0], iy - loc[1]) < 45.0:
                        raw_t = e.dxf.text if hasattr(e.dxf, "text") else getattr(e, "text", "")
                        t_clean = strip_dxf_formatting(raw_t).strip()
                        # Exclure les bornes, les coordonnées numériques du carroyage et les détails linéaires
                        if BORNE_NAME_RE.match(t_clean) or (t_clean.isdigit() and len(t_clean) >= 5):
                            continue
                        if any(k in t_clean.lower() for k in ("chaaba", "chaâba", "oued", "seguia", "piste", "route", "l.e")):
                            continue
                        rot_deg = getattr(e.dxf, "rotation", 0.0)
                        rot_rad = math.radians(rot_deg)
                        shift = 20.0 * ticket.severity_multiplier
                        cand_x = ix + shift * math.cos(rot_rad)
                        cand_y = iy + shift * math.sin(rot_rad)
                        # Clamping strict au cadre intérieur : ne JAMAIS pousser un riverain hors du cadre
                        xmin, ymin, xmax, ymax = frame_bbox
                        m_cadre = 10.0
                        if not (xmin + m_cadre <= cand_x <= xmax - m_cadre and ymin + m_cadre <= cand_y <= ymax - m_cadre):
                            cand_x = ix - shift * math.cos(rot_rad)
                            cand_y = iy - shift * math.sin(rot_rad)
                        cand_x = max(xmin + m_cadre, min(xmax - m_cadre, cand_x))
                        cand_y = max(ymin + m_cadre, min(ymax - m_cadre, cand_y))
                        e.dxf.insert = (cand_x, cand_y, 0.0)
                        logger.info("🔄 Worker4_RiverainTC a décalé le riverain '%s' à (%.1f, %.1f)", t_clean[:20], cand_x, cand_y)
                        break
            else:
                WorkersTeam.run_worker_3_and_4_details_riverains(msp, parcel_poly, frame_bbox, scale)

        elif ticket.target_worker == "WORKER_5_STD":
            WorkersTeam.run_worker_5_standards(doc, scale)


# ============================================================================
# NIVEAU 2 : LE CONTRÔLEUR QUALITÉ SÉVÈRE ("LI KIGHAWWAT 3LA L-KHEDDAMA")
# ============================================================================

class SupervisorController:
    """
    Inspecte le travail des Ouvriers et génère des `WorkOrderTicket` nominatifs
    tant qu'il reste la moindre erreur géométrique ou visuelle.
    """

    @staticmethod
    def inspect_and_shout(
        doc: Drawing,
        msp,
        parcel_poly: Optional[Polygon],
        frame_bbox: Tuple[float, float, float, float],
        scale: int,
        round_num: int,
    ) -> List[WorkOrderTicket]:
        tickets: List[WorkOrderTicket] = []
        h_text = ANCFCC_STANDARDS["borne_text_height_ratio"] * scale
        xmin, ymin, xmax, ymax = frame_bbox
        min_frame_clear = ANCFCC_STANDARDS.get("min_grid_to_frame_ratio", 0.0075) * scale
        boost = 1.0 + 0.25 * round_num

        # 1. Contrôle Ouvrier 1 : Polylignes vides, doublons, ou carroyage trop près du cadre
        for pl in msp.query("LWPOLYLINE"):
            if len(list(pl.get_points())) < 2:
                tickets.append(WorkOrderTicket(
                    target_worker="WORKER_1_GRID",
                    error_code="EMPTY_LWPOLYLINE_LEFT",
                    entity_info=f"Layer={pl.dxf.layer}",
                    location=(0.0, 0.0),
                    severity_multiplier=boost,
                ))
                break

        for e in msp.query("MTEXT TEXT"):
            if getattr(e.dxf, "style", "") == "CQ_COORD":
                raw_t = e.dxf.text if hasattr(e.dxf, "text") else getattr(e, "text", "")
                t = strip_dxf_formatting(raw_t).strip()
                if t.isdigit():
                    val = float(t)
                    d_fr = min(abs(val - xmin), abs(xmax - val), abs(val - ymin), abs(ymax - val))
                    if d_fr < min_frame_clear:
                        tickets.append(WorkOrderTicket(
                            target_worker="WORKER_1_GRID",
                            error_code="GRID_TOO_CLOSE_TO_FRAME",
                            entity_info=t,
                            location=(e.dxf.insert.x, e.dxf.insert.y),
                            severity_multiplier=boost,
                        ))

        # 2. Contrôle Ouvrier 2 : Collisions entre étiquettes de bornes ou doublons
        borne_boxes: List[Tuple[str, OrientedBox2D]] = []
        seen_b_names = set()
        for e in msp.query("MTEXT TEXT"):
            raw_t = e.dxf.text if hasattr(e.dxf, "text") else getattr(e, "text", "")
            t = strip_dxf_formatting(raw_t).strip()
            if BORNE_NAME_RE.match(t):
                if t in seen_b_names:
                    tickets.append(WorkOrderTicket(
                        target_worker="WORKER_1_GRID",
                        error_code="DUPLICATE_BORNE_LABEL",
                        entity_info=t,
                        location=(e.dxf.insert.x, e.dxf.insert.y),
                        severity_multiplier=boost,
                    ))
                seen_b_names.add(t)
                w, h_tot = estimate_exact_text_dimensions(t, h_text, "CQ_ARIAL")
                pos = np.array([e.dxf.insert.x, e.dxf.insert.y], dtype=np.float64)
                borne_boxes.append((t, OrientedBox2D(center=pos, half_extents=np.array([0.50 * w, 0.52 * h_tot]))))

        for i in range(len(borne_boxes)):
            for j in range(i + 1, len(borne_boxes)):
                collides, _, _ = sat_overlap_mtv(borne_boxes[i][1], borne_boxes[j][1])
                if collides:
                    mid = 0.5 * (borne_boxes[i][1].center + borne_boxes[j][1].center)
                    tickets.append(WorkOrderTicket(
                        target_worker="WORKER_2_BORNE",
                        error_code="BORNE_OBB_COLLISION",
                        entity_info=f"{borne_boxes[i][0]} vs {borne_boxes[j][0]}",
                        location=(float(mid[0]), float(mid[1])),
                        severity_multiplier=boost,
                    ))

        # 2b. Contrôle Général Zéro Collision : Tous textes (riverains vs bornes/détails)
        from exact_text_obb import get_exact_text_obb
        all_text_obbs = []
        for e in msp.query("MTEXT TEXT"):
            raw_t = e.dxf.text if hasattr(e.dxf, "text") else getattr(e, "text", "")
            t_clean = strip_dxf_formatting(raw_t).strip()
            if not t_clean or len(t_clean) < 2:
                continue
            # Exclure impérativement les coordonnées numériques du carroyage Lambert (ex: 441600, 793000)
            if t_clean.isdigit() and len(t_clean) >= 5:
                continue
            lay = getattr(e.dxf, "layer", "").lower()
            if lay in ("cadre", "cartouche", "titre", "fleche_nord", "nord"):
                continue
            tl = t_clean.lower()
            if any(k in tl for k in ("titre", "direction", "cadastre", "echelle", "contenance", "mappe", "requisition")):
                continue
            pos_x, pos_y = e.dxf.insert.x, e.dxf.insert.y
            if not (xmin + 5 <= pos_x <= xmax - 5 and ymin + 15 <= pos_y <= ymax - 20):
                continue
            p_obb = get_exact_text_obb(e, pad_ratio=0.0)
            if p_obb and not p_obb.is_empty:
                all_text_obbs.append((e, t_clean, p_obb))

        for i in range(len(all_text_obbs)):
            e_i, t_i, poly_i = all_text_obbs[i]
            for j in range(i + 1, len(all_text_obbs)):
                e_j, t_j, poly_j = all_text_obbs[j]
                if t_i == t_j and math.hypot(e_i.dxf.insert.x - e_j.dxf.insert.x, e_i.dxf.insert.y - e_j.dxf.insert.y) < 0.1:
                    continue
                if poly_i.intersects(poly_j):
                    is_b_i = bool(BORNE_NAME_RE.match(t_i))
                    is_b_j = bool(BORNE_NAME_RE.match(t_j))
                    target_w = "WORKER_4_TC" if (not is_b_i or not is_b_j) else "WORKER_2_BORNE"
                    c_pt = poly_i.intersection(poly_j).centroid
                    tickets.append(WorkOrderTicket(
                        target_worker=target_w,
                        error_code="TEXT_OBB_COLLISION",
                        entity_info=f"{t_i[:20]} vs {t_j[:20]}",
                        location=(float(c_pt.x), float(c_pt.y)),
                        severity_multiplier=boost,
                    ))

        # 3. Contrôle Ouvrier 3 : Texte de détail posé sur une borne
        borne_pts = [
            [ins.dxf.insert.x, ins.dxf.insert.y]
            for ins in msp.query("INSERT")
            if any(k in ins.dxf.layer.lower() or k in getattr(ins.dxf, "name", "").lower() for k in ("borne", "brns"))
        ]
        borne_tree = cKDTree(borne_pts) if borne_pts else None
        if borne_tree is not None:
            for t_ent in msp.query("TEXT MTEXT"):
                raw_t = t_ent.dxf.text if hasattr(t_ent.dxf, "text") else getattr(t_ent, "text", "")
                txt = strip_dxf_formatting(raw_t)
                if any(k in txt.lower() for k in ("cha", "oued", "seguia", "piste", "chemin", "l.e")):
                    d_b, _ = borne_tree.query([t_ent.dxf.insert.x, t_ent.dxf.insert.y])
                    if float(d_b) < 2.0 * h_text:
                        tickets.append(WorkOrderTicket(
                            target_worker="WORKER_3_DETAIL",
                            error_code="DETAIL_TOUCHING_BORNE",
                            entity_info=txt,
                            location=(t_ent.dxf.insert.x, t_ent.dxf.insert.y),
                            severity_multiplier=boost,
                        ))

        # 4. Contrôle Ouvrier 4 : Nature interne (TC) doit être strictement horizontale
        for m in msp.query("MTEXT"):
            t_clean = strip_dxf_formatting(getattr(m.dxf, "text", "")).strip().upper()
            if t_clean in ("TC", "T.C", "T.C."):
                rot = getattr(m.dxf, "rotation", 0.0)
                if abs(rot) > 1e-3:
                    tickets.append(WorkOrderTicket(
                        target_worker="WORKER_4_TC",
                        error_code="UNNORMALIZED_MTEXT_ROTATION",
                        entity_info=t_clean,
                        location=(m.dxf.insert.x, m.dxf.insert.y),
                        severity_multiplier=boost,
                    ))

        # 5. Contrôle Ouvrier 5 : Standards Header ANCFCC
        acadver = doc.header.get("$ACADVER") or doc.header.get(r"\$ACADVER")
        aunits = doc.header.get("$AUNITS") or doc.header.get(r"\$AUNITS")
        if acadver != "AC1018" or aunits != 2:
            tickets.append(WorkOrderTicket(
                target_worker="WORKER_5_STD",
                error_code="INVALID_ANCFCC_HEADER",
                entity_info="HEADER",
                location=(0.0, 0.0),
                severity_multiplier=boost,
            ))

        # 6. Contrôle Capteur Visuel Matriciel (Système 1) : Superposition avec symboles ou masques
        try:
            from raster_pixel_sensor import RasterPixelSensor
            sym_colls = RasterPixelSensor.detect_text_symbol_collisions(msp, scale=scale)
            for sc in sym_colls:
                is_borne = bool(BORNE_NAME_RE.match(sc["text_content"]))
                tickets.append(WorkOrderTicket(
                    target_worker="WORKER_2_BORNE" if is_borne else "WORKER_3_DETAIL",
                    error_code="TEXT_COLLIDING_WITH_SYMBOL_OR_WIPEOUT",
                    entity_info=f"{sc['text_content']} vs {sc['obstacle_name']}",
                    location=sc["location"],
                    severity_multiplier=boost,
                ))
        except Exception as _ex_sensor:
            logger.debug("Raster pixel sensor check: %s", _ex_sensor)

        return tickets


# ============================================================================
# NIVEAU 3 : LE CHEF INGÉNIEUR GÉOMÈTRE (`ChiefSurveyorBoss`)
# ============================================================================

class ChiefSurveyorBoss:
    """
    Le Patron (Niveau 3) : Réalise l'audit indépendant final avant d'autoriser
    la livraison du fichier DXF à l'utilisateur.
    """

    @staticmethod
    def final_executive_audit(
        doc: Drawing,
        msp,
        parcel_poly: Optional[Polygon],
        frame_bbox: Tuple[float, float, float, float],
        scale: int,
        rounds_used: int,
        total_tickets_fixed: int,
    ) -> ChiefApprovalVisa:
        veto_reasons: List[str] = []
        penalty = 0.0

        # 1. Vérification du verrouillage centimétrique RejCAD
        lock_rejcad_centimetric_precision(msp)

        # 2. Vérification de la fermeture de la parcelle
        parcel_lines = [
            pl for pl in msp.query("LWPOLYLINE")
            if pl.dxf.layer.lower() in ("parcelle", "limit")
        ]
        for pl in parcel_lines:
            if not pl.closed:
                veto_reasons.append("Polyligne parcelle non fermée (closed != True)")
                penalty += 20.0

        # 3. Vérification de l'inclusion stricte de TC dans la parcelle
        if parcel_poly is not None:
            for m in msp.query("MTEXT TEXT"):
                raw_t = m.dxf.text if hasattr(m.dxf, "text") else getattr(m, "text", "")
                if strip_dxf_formatting(raw_t).strip() == "TC":
                    pt = Point(m.dxf.insert.x, m.dxf.insert.y)
                    if not parcel_poly.contains(pt):
                        veto_reasons.append("Étiquette TC située hors du polygone parcelle")
                        penalty += 15.0

        # 4. Vérification des normes ANCFCC critiques
        acadver = doc.header.get("$ACADVER") or doc.header.get(r"\$ACADVER")
        codepage = doc.header.get("$DWGCODEPAGE") or doc.header.get(r"\$DWGCODEPAGE")
        if acadver and acadver != "AC1018":
            veto_reasons.append(f"$ACADVER != AC1018 (actuel={acadver})")
            penalty += 15.0

        # 5. Vérification absolue de confinement dans le cadre (Zéro texte de détail/riverain hors cadre)
        xmin, ymin, xmax, ymax = frame_bbox
        layout_kws = (
            "direction", "cadastre", "echelle", "échel", "contenance", "mappe",
            "requisition", "réquis", "propriete", "propriété", "situee", "située",
            "titre", "plan", "nord", "plle", "st 25", "st25", "37-45", "commune",
            "limite de la", "ife", "zone", "sous zone"
        )
        arabic_kws = (
            "\u0627\u0644\u0645\u0644\u0643", "\u0627\u0644\u0643\u0627\u0626\u0646",
            "\u0631\u0633\u0645", "\u0645\u0637\u0644\u0628", "\u0627\u0644\u0634\u0645\u0627\u0644",
            "\u0645\u0633\u0627\u062d\u0629"
        )
        for e in msp.query("MTEXT TEXT"):
            raw_t = e.dxf.text if hasattr(e.dxf, "text") else getattr(e, "text", "")
            t_clean = strip_dxf_formatting(raw_t).strip()
            if not t_clean or len(t_clean) < 2:
                continue
            lay = getattr(e.dxf, "layer", "").lower()
            if lay in ("cadre", "cartouche", "titre", "fleche_nord", "nord"):
                continue
            if t_clean.isdigit() and len(t_clean) >= 5:
                continue
            tl = t_clean.lower()
            if any(k in tl for k in layout_kws) or any(k in t_clean for k in arabic_kws):
                continue
            ex, ey = e.dxf.insert.x, e.dxf.insert.y
            if ex < xmin - 2.0 or ex > xmax + 2.0 or ey < ymin - 2.0 or ey > ymax + 2.0:
                veto_reasons.append(f"Texte '{t_clean[:25]}' situé hors cadre à ({ex:.1f}, {ey:.1f})")
                penalty += 25.0

        score = max(0.0, 100.0 - penalty)
        return ChiefApprovalVisa(
            approved=(len(veto_reasons) == 0 and score >= 99.0),
            quality_score=round(score, 2),
            rounds_needed=rounds_used,
            controller_tickets_resolved=total_tickets_fixed,
            chief_veto_reasons=veto_reasons,
        )


# ============================================================================
# BOUCLE D'ORCHESTRATION HIÉRARCHIQUE (OUVRIERS <-> CONTRÔLEUR -> CHEF -> DXF)
# ============================================================================

def execute_hierarchical_bureau_workflow(
    doc: Drawing,
    frame_bbox: Tuple[float, float, float, float],
    parcel_poly: Optional[Polygon],
    scale: int = 2000,
    max_control_rounds: int = 6,
    status_cb: Optional[Callable[[str], None]] = None,
) -> ChiefApprovalVisa:
    """
    Exécute la chaîne de commandement complète :
    1. Les 5 Ouvriers font le travail initial.
    2. Le Contrôleur inspecte et renvoie les Ouvriers fautifs au travail tant
       qu'il trouve des anomalies.
    3. Le Chef Ingénieur audite le tout et appose son Visa final avant livraison.
    """
    def _status(msg: str):
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    msp = doc.modelspace()
    total_tickets_resolved = 0

    # Passage 1 : Les 5 Ouvriers exécutent leur travail de base
    _status("👷 [Niveau 1] Les 5 Ouvriers Spécialisés exécutent le travail initial...")
    WorkersTeam.run_worker_1_eraser_grid(msp, frame_bbox, scale)
    WorkersTeam.run_worker_2_borne_mover(msp, parcel_poly, scale)
    WorkersTeam.run_worker_3_and_4_details_riverains(msp, parcel_poly, frame_bbox, scale)
    WorkersTeam.run_worker_5_standards(doc, scale)

    # Passage 2 : Boucle de contrôle strict du Contrôleur ("Kighawwat 3la l-Kheddama")
    round_num = 1
    while round_num <= max_control_rounds:
        _status(f"🔍 [Niveau 2] Contrôleur Qualité Sévère (Inspection Round {round_num})...")
        tickets = SupervisorController.inspect_and_shout(
            doc, msp, parcel_poly, frame_bbox, scale, round_num
        )
        if not tickets:
            # 0 erreur trouvée par le Contrôleur -> Envoi au Chef !
            logger.info("✅ [SUPERVISOR] 0 anomalie détectée après %d round(s). Envoi au Chef.", round_num)
            _status(f"✅ [Niveau 2] 0 anomalie résiduelle (Round {round_num}). Dossier transmis au Chef Ingénieur.")
            break

        logger.info("⚠️ [SUPERVISOR ROUND %d] %d tickets émis par le Contrôleur.", round_num, len(tickets))
        _status(f"⚠️ [Niveau 2] {len(tickets)} anomalies détectées. Ordres de reprise envoyés aux ouvriers...")
        for ticket in tickets:
            WorkersTeam.execute_targeted_fix(doc, msp, parcel_poly, frame_bbox, scale, ticket)
            total_tickets_resolved += 1

        round_num += 1

    # Passage 3 : Contrôle ultime du Chef Ingénieur Géomètre
    _status("👑 [Niveau 3] Audit Décisionnel Ultime par l'Ingénieur Chef ANCFCC...")
    visa = ChiefSurveyorBoss.final_executive_audit(
        doc, msp, parcel_poly, frame_bbox, scale, round_num, total_tickets_resolved
    )

    if not visa.approved:
        logger.warning("⛔ [CHEF VETO] Le Chef a rejeté le plan : %s", visa.chief_veto_reasons)
        _status("⛔ [Niveau 3] Veto du Chef : Reprise d'urgence avec coefficients renforcés...")
        # Si le Chef oppose son Veto, il force un dernier passage correctif immédiat
        WorkersTeam.run_worker_1_eraser_grid(msp, frame_bbox, scale, boost=1.5)
        WorkersTeam.run_worker_2_borne_mover(msp, parcel_poly, scale, boost=1.5)
        WorkersTeam.run_worker_3_and_4_details_riverains(msp, parcel_poly, frame_bbox, scale)
        WorkersTeam.run_worker_5_standards(doc, scale)
        visa = ChiefSurveyorBoss.final_executive_audit(
            doc, msp, parcel_poly, frame_bbox, scale, round_num + 1, total_tickets_resolved
        )

    if visa.approved:
        logger.info("👑 [CHEF VISA ACCORDÉ] Score = %.2f%% en %d round(s) (%d tickets résolus)",
                    visa.quality_score, visa.rounds_needed, visa.controller_tickets_resolved)
        _status(f"👑 [VISA DU CHEF ACCORDÉ] Score: {visa.quality_score:.1f}% | 0.00% Collision | {total_tickets_resolved} tickets résolus ✅")

    return visa
