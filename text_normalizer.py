# -*- coding: utf-8 -*-
r"""
engines/dxf_cleaner/text_normalizer.py — Application des Règles Extraites par mine_rules.py
============================================================================================
Module qui applique les règles déterministes extraites automatiquement depuis les paires
DEF/OREGEN sur les textes d'un modelspace DXF.

Règles implémentées (triées par impact) :
  RULE_012 : coord       → rot=0°  (100% conf, 135 parcelles)
  RULE_015 : verborne    → rot=0°  (99.4% conf, 90 parcelles)
  RULE_019 : tc_nature   → rot=0°  (99.6% conf, 70 parcelles)
  RULE_007 : riverain_name → rot=0° (98.0% conf, 136 parcelles)
  RULE_025 : riverain_RT → rot=0°  (93.5% conf, 26 parcelles)
  RULE_018 : verborne    → dedup   (99.4% conf, 34 parcelles)
  RULE_024 : linear_detail → dedup (16.6% conf, 46 parcelles)
  RULE_005 : borne       → dedup   (3.4% conf, 51 parcelles)
  RULE_002 : paren_borne → dedup   (4.4% conf, 4 parcelles)
  FIX_TC   : tc_nature   → preserve OREGEN position if inside parcel
  FIX_ANGLE: linear_detail → normalize angle to [-90°, +90°]
"""

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from shapely.geometry import Polygon, Point

logger = logging.getLogger("dxf_cleaner.text_normalizer")

# ── Regex patterns pour classification ─────────────────────────────────
RE_BORNE_LABEL = re.compile(r"^B\d+$", re.IGNORECASE)
RE_PAREN_BORNE = re.compile(r"^\(B\d+\)$", re.IGNORECASE)
RE_RIVERAIN_RT = re.compile(r"^[RT]\d+", re.IGNORECASE)
RE_VERBORNE = re.compile(r"^v\.?\s*B\d+", re.IGNORECASE)
RE_LINEAR_DETAIL = re.compile(
    r"(piste|chemin|route|oued|chaaba|cha[âa]bat|depression|d[ée]pression|L\.?E\.?B\.?T|L\.?E\.?M\.?T|L\.?E\.?H\.?T|sentier|canal|mur|foss[eé]|s[eé]guia|cloture|cl[oô]ture)",
    re.IGNORECASE,
)
RE_COORD_NUMBER = re.compile(r"^\d{5,6}(\.\d+)?$")
RE_TC_NATURE = re.compile(
    r"(T\.?C\.?|terrain|culture|argan|amand|oliv|vigne|figuier|cactus|jachère|labour|irrig)",
    re.IGNORECASE,
)
RE_CARTOUCHE = re.compile(
    r"(مطلب|رسم|ملك|الملك|كائن|الكائن|جماعة|قيادة|عمالة|دائرة|إقليم|بقعة|تحديد|ST\s*\d+|ZN\s*\d+|caidat|prefecture|préfecture|commune|province|royaume|contenance|superficie|conservation|maroc|marro?c)",
    re.IGNORECASE,
)


def classify_text(text: str) -> str:
    """
    Classifie un texte DXF en catégorie.

    Args:
        text: Le texte brut de l'entité.

    Returns:
        La catégorie du texte.
    """
    nt = text.strip()
    nt = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), nt)
    nt = re.sub(r"\s+", " ", nt)

    if RE_BORNE_LABEL.match(nt):
        return "borne"
    if RE_PAREN_BORNE.match(nt):
        return "paren_borne"
    if RE_RIVERAIN_RT.match(nt):
        return "riverain_RT"
    if RE_VERBORNE.match(nt):
        return "verborne"
    if RE_LINEAR_DETAIL.search(nt):
        return "linear_detail"
    if RE_COORD_NUMBER.match(nt):
        return "coord"
    if RE_TC_NATURE.search(nt):
        return "tc_nature"
    if RE_CARTOUCHE.search(nt) or re.search(r"[\u0600-\u06FF]", nt):
        return "cartouche"
    if any(kw in nt.lower() for kw in ("héritier", "heritier", "plle", "titre", "hér.")):
        return "riverain_name"
    return "other"


def _normalize_angle_90(angle_deg: float) -> float:
    """
    Normalise un angle en degrés dans la plage [-90°, +90°]
    pour garantir la lisibilité du texte.

    Args:
        angle_deg: L'angle en degrés.

    Returns:
        L'angle normalisé dans [-90°, +90°].
    """
    a = angle_deg % 360.0
    if a > 270.0:
        a -= 360.0
    elif a > 90.0:
        a -= 180.0
    return a


def _get_entity_text(ent) -> str:
    """
    Extrait le texte brut d'une entité TEXT ou MTEXT.

    Args:
        ent: L'entité DXF.

    Returns:
        Le texte brut.
    """
    try:
        if ent is None or not hasattr(ent, "dxf"):
            return ""
        if hasattr(ent, "is_alive") and not ent.is_alive:
            return ""
        dxf_type = ent.dxftype()
        if dxf_type == "TEXT":
            return getattr(ent.dxf, "text", "")
        elif dxf_type == "MTEXT":
            return getattr(ent, "text", "")
    except Exception:
        return ""
    return ""


# ── RÈGLE 1 : Forcer la rotation à 0° ─────────────────────────────────

# Catégories où DEF a toujours rot=0° (confidence ≥ 93%)
FORCE_ROT_ZERO_CATEGORIES = {
    "coord",          # RULE_012: 100% confidence, 135 parcels
    "verborne",       # RULE_015: 99.4% confidence, 90 parcels
    "tc_nature",      # RULE_019: 99.6% confidence, 70 parcels
    "riverain_RT",    # RULE_025: 93.5% confidence, 26 parcels
    "riverain_name",  # RULE_007: 98.8% confidence, 3280/3320 texts rot=0 in DEF
    "cartouche",      # Cartouche & Arabic titles: 99.8% confidence rot=0 in DEF
    "borne",          # Borne labels: 100% rot=0 in DEF
    "paren_borne",    # (Bxx) labels: 100% rot=0 in DEF
    "other",          # General notes, stamps, unclassified: 98% rot=0 in DEF
}


def apply_force_rotation_zero(msp, parcel_poly: Optional[Polygon] = None) -> int:
    """
    Force la rotation à 0° pour les catégories de textes où l'expert
    le fait systématiquement (≥93% des cas dans les 137 paires analysées).

    Catégories : coord, verborne, tc_nature, riverain_RT

    Args:
        msp: Le modelspace DXF.
        parcel_poly: Le polygone de la parcelle (optionnel).

    Returns:
        Le nombre de textes dont la rotation a été forcée.
    """
    n_fixed = 0

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue

            category = classify_text(text)
            if category not in FORCE_ROT_ZERO_CATEGORIES:
                continue

            current_rot = getattr(ent.dxf, "rotation", 0.0)
            if abs(current_rot) > 1.0 and abs(current_rot - 360.0) > 1.0:
                ent.dxf.rotation = 0.0
                n_fixed += 1
                logger.debug(
                    "ROT→0: [%s] '%s' rot=%.1f° → 0.0°",
                    category, text[:30], current_rot,
                )
        except Exception:
            continue

    if n_fixed > 0:
        logger.info("RULE force_rot_zero: %d textes corrigés", n_fixed)
    return n_fixed


# ── RÈGLE 2 : Normaliser les angles linéaires [-90°, +90°] ────────────

def apply_angle_normalization(msp) -> int:
    """
    Normalise les angles des détails linéaires dans [-90°, +90°]
    pour garantir la lisibilité (pas de texte à l'envers).

    Applicable à : linear_detail (piste, chemin, route, oued, chaaba...)

    Args:
        msp: Le modelspace DXF.

    Returns:
        Le nombre de textes dont l'angle a été normalisé.
    """
    n_fixed = 0

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue

            category = classify_text(text)
            if category != "linear_detail":
                continue

            current_rot = getattr(ent.dxf, "rotation", 0.0)
            normalized = _normalize_angle_90(current_rot)

            if abs(current_rot - normalized) > 1.0:
                ent.dxf.rotation = normalized
                n_fixed += 1
                logger.debug(
                    "ANGLE_NORM: '%s' rot=%.1f° → %.1f°",
                    text[:30], current_rot, normalized,
                )
        except Exception:
            continue

    if n_fixed > 0:
        logger.info("RULE angle_normalize: %d textes corrigés", n_fixed)
    return n_fixed


# ── RÈGLE 3 : Déduplication des textes ─────────────────────────────────

# Catégories à dédupliquer (expert ne garde qu'une copie)
DEDUP_CATEGORIES = {
    "verborne",       # RULE_018: 34 parcels, avg 1.7 copies removed
    "linear_detail",  # RULE_024: 46 parcels, avg 2.6 copies removed
    "paren_borne",    # RULE_002: 4 parcels, avg 3.2 copies removed
    "borne",          # Spatial deduplication (<8m) for stacked identical borne names
    "riverain_name",  # Spatial deduplication (<8m) for stacked identical riverain names (e.g. P32)
    "riverain_RT",    # Spatial deduplication (<8m) for stacked identical title numbers
    "other",          # Spatial deduplication (<8m) for stacked duplicate entities (e.g. Chateau d'eau in P586)
}


def apply_deduplication(msp, parcel_poly: Optional[Polygon] = None) -> int:
    """
    Déduplique les textes : pour chaque contenu identique dans les
    catégories ciblées, ne garde que l'entité la plus proche du
    polygone parcelle (ou la première si pas de polygone).

    Args:
        msp: Le modelspace DXF.
        parcel_poly: Le polygone de la parcelle (optionnel).

    Returns:
        Le nombre d'entités supprimées.
    """
    n_deleted = 0

    # Collecter tous les textes par catégorie et contenu normalisé
    groups: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue

            category = classify_text(text)
            if category not in DEDUP_CATEGORIES:
                continue

            key = text.strip().lower()
            ins = ent.dxf.insert
            groups[category][key].append({
                "entity": ent,
                "x": ins.x,
                "y": ins.y,
            })
        except Exception:
            continue

    deleted_set = set()

    # Pour chaque groupe de contenu identique, dédupliquer UNIQUEMENT les entités proches spatialement (< 8.0m ou 35m pour détails linéaires)
    for category, content_groups in groups.items():
        prox_thresh = 20.0 if category == "linear_detail" else 8.0
        for content_key, entities in content_groups.items():
            if len(entities) <= 1:
                continue

            # Regrouper en clusters spatiaux selon la proximité
            clusters: List[List[dict]] = []
            for ent_info in entities:
                assigned = False
                for cl in clusters:
                    if math.hypot(ent_info["x"] - cl[0]["x"], ent_info["y"] - cl[0]["y"]) <= prox_thresh:
                        cl.append(ent_info)
                        assigned = True
                        break
                if not assigned:
                    clusters.append([ent_info])

            # Pour chaque cluster ayant plusieurs doublons empilés, en garder un seul
            for cl in clusters:
                if len(cl) <= 1:
                    continue
                if parcel_poly is not None:
                    cl.sort(key=lambda e: parcel_poly.exterior.distance(Point(e["x"], e["y"])))
                for dup in cl[1:]:
                    try:
                        msp.delete_entity(dup["entity"])
                        deleted_set.add(id(dup["entity"]))
                        n_deleted += 1
                        logger.debug(
                            "DEDUP: [%s] '%s' doublon empilé supprimé à (%.0f, %.0f)",
                            category, content_key[:20], dup["x"], dup["y"],
                        )
                    except Exception:
                        continue

    # Déduplication croisée pour entités linéaires empilées (< 5.0m) ayant des noms différents
    linear_ents = [
        ent_info
        for content_key, elist in groups.get("linear_detail", {}).items()
        for ent_info in elist
        if id(ent_info["entity"]) not in deleted_set
    ]
    for i in range(len(linear_ents)):
        e1 = linear_ents[i]
        if id(e1["entity"]) in deleted_set:
            continue
        for j in range(i + 1, len(linear_ents)):
            e2 = linear_ents[j]
            if id(e2["entity"]) in deleted_set:
                continue
            if math.hypot(e1["x"] - e2["x"], e1["y"] - e2["y"]) < 5.0:
                t1 = _get_entity_text(e1["entity"]).strip()
                t2 = _get_entity_text(e2["entity"]).strip()
                cand_to_del = e2 if len(t1) >= len(t2) else e1
                try:
                    msp.delete_entity(cand_to_del["entity"])
                    deleted_set.add(id(cand_to_del["entity"]))
                    n_deleted += 1
                    logger.debug(
                        "DEDUP CROISÉE: '%s' vs '%s' superposés à < 5m",
                        t1[:20], t2[:20],
                    )
                except Exception:
                    pass

    if n_deleted > 0:
        logger.info("RULE deduplication: %d entités supprimées", n_deleted)
    return n_deleted


# ── RÈGLE 4 : Préserver la position TC originale de l'OREGEN ──────────

def apply_tc_position_preserve(
    msp,
    parcel_poly: Optional[Polygon],
    original_tc_positions: Optional[List[Tuple[float, float]]] = None,
) -> int:
    """
    Si un texte TC existe déjà dans l'OREGEN et qu'il est à l'intérieur
    du polygone parcelle, le conserver à sa position originale
    (ne pas le déplacer par polylabel).

    Note: Cette fonction doit être appelée AVANT place_parcel_nature_polylabel()
    pour sauvegarder les positions originales, puis APRÈS pour les restaurer
    si polylabel les a déplacées.

    Args:
        msp: Le modelspace DXF.
        parcel_poly: Le polygone de la parcelle.
        original_tc_positions: Positions originales des TC depuis l'OREGEN.

    Returns:
        Le nombre de TC dont la position a été restaurée.
    """
    if parcel_poly is None or original_tc_positions is None:
        return 0

    n_restored = 0

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue

            category = classify_text(text)
            if category != "tc_nature":
                continue

            upper_t = text.upper()
            if "\\P" in text or "\\p" in text or "\n" in text or "PLLE" in upper_t or "TITRE" in upper_t:
                continue

            current_x = ent.dxf.insert.x
            current_y = ent.dxf.insert.y

            # Chercher la position originale la plus proche
            best_orig = None
            best_dist = float("inf")
            for ox, oy in original_tc_positions:
                dist = math.sqrt((current_x - ox) ** 2 + (current_y - oy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_orig = (ox, oy)

            # Si on a trouvé un original et qu'il est dans le polygone (seuil max 250m pour couvrir les grandes parcelles comme P216)
            if best_orig is not None and 5.0 < best_dist <= 250.0:
                orig_pt = Point(best_orig[0], best_orig[1])
                if parcel_poly.contains(orig_pt) or parcel_poly.distance(orig_pt) < 2.0:
                    ent.dxf.insert = (best_orig[0], best_orig[1], 0.0)
                    n_restored += 1
                    logger.debug(
                        "TC_PRESERVE: '%s' restauré à (%.0f, %.0f) au lieu de (%.0f, %.0f)",
                        text[:20], best_orig[0], best_orig[1], current_x, current_y,
                    )
        except Exception:
            continue

    if n_restored > 0:
        logger.info("RULE tc_preserve: %d TC restaurés", n_restored)
    return n_restored


def save_tc_positions(msp) -> List[Tuple[float, float]]:
    """
    Sauvegarde les positions actuelles des textes TC avant que polylabel
    ne les déplace.

    Args:
        msp: Le modelspace DXF.

    Returns:
        Liste des positions (x, y) des textes TC.
    """
    positions: List[Tuple[float, float]] = []

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue

            category = classify_text(text)
            if category != "tc_nature":
                continue

            # Ne sauvegarder que les vrais labels de nature interne (exclure les voisins riverains avec \PTC)
            upper_t = text.upper()
            if "\\P" in text or "\\p" in text or "\n" in text or "PLLE" in upper_t or "TITRE" in upper_t:
                continue
            if not (upper_t in ("TC", "T.C", "TN", "T.N", "T.C.", "TERRAIN DE CULTURE", "TERRAIN", "CULTURE") or
                    any(k in text.lower() for k in ("argan", "oliv", "figu", "vign", "amand"))):
                continue

            positions.append((ent.dxf.insert.x, ent.dxf.insert.y))
        except Exception:
            continue

    return positions


# ── RÈGLE 5 : Déduplication verbornes + force rot=0 ───────────────────

def apply_verborne_cleanup(msp) -> int:
    """
    Nettoie spécifiquement les verbornes :
    1. Déduplique (garde 1 copie par contenu)
    2. Force rotation=0°

    Args:
        msp: Le modelspace DXF.

    Returns:
        Le nombre de corrections appliquées.
    """
    n_fixed = 0

    # Collecter les verbornes par contenu
    vb_groups: Dict[str, list] = defaultdict(list)

    for ent in msp.query("TEXT MTEXT"):
        try:
            text = _get_entity_text(ent).strip()
            if not text:
                continue
            if not RE_VERBORNE.match(text):
                continue

            key = text.strip().lower()
            vb_groups[key].append(ent)
        except Exception:
            continue

    for key, entities in vb_groups.items():
        # Forcer rot=0 sur toutes
        for ent in entities:
            rot = getattr(ent.dxf, "rotation", 0.0)
            if abs(rot) > 1.0 and abs(rot - 360.0) > 1.0:
                ent.dxf.rotation = 0.0
                n_fixed += 1

        # Dédupliquer (garder la première)
        if len(entities) > 1:
            for dup in entities[1:]:
                try:
                    msp.delete_entity(dup)
                    n_fixed += 1
                except Exception:
                    continue

    if n_fixed > 0:
        logger.info("RULE verborne_cleanup: %d corrections", n_fixed)
    return n_fixed


# ── ORCHESTRATEUR : Application de toutes les règles ──────────────────

def apply_all_mined_rules(
    msp,
    parcel_poly: Optional[Polygon] = None,
    original_tc_positions: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, int]:
    """
    Applique toutes les règles extraites par mine_rules.py dans l'ordre
    optimal d'impact.

    Ordre :
    1. Déduplication (supprime les doublons avant tout traitement)
    2. Cleanup verbornes (dedup + rot=0)
    3. Force rotation=0 (coord, tc_nature, riverain_RT)
    4. Normalisation angles linéaires [-90°, +90°]
    5. Restauration position TC

    Args:
        msp: Le modelspace DXF.
        parcel_poly: Le polygone de la parcelle.
        original_tc_positions: Positions originales des TC (avant polylabel).

    Returns:
        Dict avec le nombre de corrections par règle.
    """
    results: Dict[str, int] = {}

    # 1. Déduplication générale
    results["dedup"] = apply_deduplication(msp, parcel_poly)

    # 2. Cleanup verbornes (dedup + rot=0)
    results["verborne_cleanup"] = apply_verborne_cleanup(msp)

    # 3. Force rotation=0
    results["force_rot_zero"] = apply_force_rotation_zero(msp, parcel_poly)

    # 4. Normalisation angles linéaires
    results["angle_normalize"] = apply_angle_normalization(msp)

    # 5. Restauration position TC
    results["tc_preserve"] = apply_tc_position_preserve(
        msp, parcel_poly, original_tc_positions
    )

    total = sum(results.values())
    if total > 0:
        logger.info(
            "MINED RULES APPLIED: total=%d (dedup=%d, vb=%d, rot0=%d, angle=%d, tc=%d)",
            total,
            results["dedup"],
            results["verborne_cleanup"],
            results["force_rot_zero"],
            results["angle_normalize"],
            results["tc_preserve"],
        )

    return results
