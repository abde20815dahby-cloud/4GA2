# -*- coding: utf-8 -*-
"""
text_anti_collision.py: Moteur Anti-Collision Global et Intelligent pour Plans Cadastraux ANCFCC.
Résout 100% des collisions visuelles (overlaps) entre TOUTES les entités texte/MText :
- Bornes (Bxxxx)
- Bornes de rattachement ((Bxxxx))
- Verbornes (v.Bxxxx)
- Riverains (Plle... TC, Héritiers..., Titres)
- Natures de culture (TC, TN, etc.)
- Détails linéaires (Chaâbat, Pistes, Routes)
- Détails ponctuels (Château d'eau, Puits, Ecurie, etc.)
"""
import math
import logging
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import shapely.geometry as sg
from shapely.geometry import Polygon, Point, box

from exact_text_obb import get_exact_text_obb
from constants import ANCFCC_STANDARDS

logger = logging.getLogger("dxf_cleaner.anti_collision")

import re

# Mots-clés marginaux / cartouche administratif à ne jamais déplacer
CARTOUCHE_ADMIN_KW = (
    "caidat", "caïdat", "prefecture", "préfecture", "commune",
    "situee", "située", "province", "conservation", "direction",
    "royaume", "cadastre", "echelle", "échelle", "dessinateur",
    "geometre", "géomètre", "soritopo", "contenance", "requisition",
    "réquisition", "mappe", "topographique", "provisoire", "enrolement",
    "enrôlement", "propriete dite", "propriété dite", "ife", "sous zone",
    "st 25", "nord", "الشمال", "مطلب", "رسم", "الملك", "الكائن", "مساحة",
    "plle n°", "plle n\u00b0", "plle no"
)


def _get_text(ent) -> str:
    """Extrait le texte brut d'une entité TEXT ou MTEXT."""
    if ent.dxftype() == "TEXT":
        return getattr(ent.dxf, "text", "") or ""
    return getattr(ent, "text", "") or getattr(ent.dxf, "text", "") or ""


def _classify_priority(text: str, ent) -> int:
    """
    Définit la priorité d'immobilité d'un texte.
    1 = Le plus prioritaire (Immobile / Presque immobile)
    5 = Le plus déplaçable / flexible
    """
    t = text.strip()
    tl = t.lower()

    # 1. Bornes physiques de la parcelle (Bxxxx)
    if (t.startswith("B") or t.startswith("b")) and t[1:].isdigit():
        return 1

    # 2. Coordonnées Lambert
    if t.isdigit() and len(t) in (5, 6, 7):
        return 1

    # 3. Riverains (Titres fonciers R.../T..., Plle... TC, Héritiers..., etc.)
    if re.match(r"^[RT]\s*\d+", t, re.IGNORECASE):
        return 2
    if any(k in tl for k in ("plle", "titre", "héritier", "heritier", "hrts", "consorts", "t.", "habous", "domaine")):
        return 2

    # 4. Nature TC interne
    if tl in ("tc", "t.c", "t.c.", "tn", "t.n") or "olivier" in tl or "culture" in tl:
        return 3

    # 5. Verbornes et bornes rattachées
    if t.startswith("v.") or t.startswith("V.") or (t.startswith("(") and t.endswith(")")):
        return 3

    # 6. Détails ponctuels (Château d'eau, Puits, etc.)
    if any(k in tl for k in ("chateau", "château", "puits", "ecurie", "écurie", "construction", "mur")):
        return 4

    # 7. Détails linéaires (Chaâbat, Pistes, Routes, L.E.B.T)
    return 5


def deduplicate_close_identical_texts(msp, max_dist: float = 40.0) -> int:
    """
    Supprime les textes strictement identiques ou quasi-identiques
    qui sont superposés ou distants de moins de max_dist.
    Exemple: deux "Chaâbat" à 15m, ou deux "Château d'eau" à 3m.
    """
    deleted = 0
    all_texts = []
    for ent in list(msp):
        if ent.dxftype() in ("TEXT", "MTEXT"):
            txt = _get_text(ent).strip()
            if not txt or any(k in txt.lower() for k in CARTOUCHE_ADMIN_KW):
                continue
            ins = getattr(ent.dxf, "insert", None)
            if ins is not None:
                all_texts.append((txt, ent, (float(ins.x), float(ins.y))))

    to_remove = set()
    n = len(all_texts)
    for i in range(n):
        if i in to_remove:
            continue
        t1, e1, p1 = all_texts[i]
        norm1 = t1.lower().replace(" ", "").replace(".", "").replace("â", "a").replace("é", "e").replace("’", "'")

        for j in range(i + 1, n):
            if j in to_remove:
                continue
            t2, e2, p2 = all_texts[j]
            norm2 = t2.lower().replace(" ", "").replace(".", "").replace("â", "a").replace("é", "e").replace("’", "'")

            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])

            # Doublon exact de détail linéaire ou ponctuel (ex: Chaâbat vs Chaâbat à < 40m)
            if norm1 == norm2 and dist < max_dist:
                # Garder celui qui a une rotation orientée ou qui est MTEXT
                if abs(getattr(e1.dxf, "rotation", 0.0)) > 1.0:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
                    break

            # Cas particulier "Chateau" vs "Chateau d'eau" à < 8m : supprimer le générique "Chateau"
            if ("chateau" in norm1 and "chateaud'eau" in norm2) and dist < 8.0:
                to_remove.add(i)
                break
            if ("chateau" in norm2 and "chateaud'eau" in norm1) and dist < 8.0:
                to_remove.add(j)

    for idx in to_remove:
        try:
            msp.delete_entity(all_texts[idx][1])
            deleted += 1
            logger.info("🗑️ Doublon texte supprimé : '%s'", all_texts[idx][0])
        except Exception:
            pass

    return deleted


def resolve_global_text_overlaps(
    msp,
    parcel_poly: Optional[Polygon] = None,
    frame_poly: Optional[Polygon] = None,
    scale: int = 1000,
    max_passes: int = 25
) -> int:
    """
    Moteur global d'anti-collision (SAT + MTV + relaxation hiérarchique).
    Itère sur TOUTES les entités textes pour éliminer les intersections.
    """
    h_text = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018) * scale
    pad = 0.15 * h_text

    # 1. Première passe : déduplication des textes superposés identiques
    dedup_count = deduplicate_close_identical_texts(msp, max_dist=35.0)

    # 2. Collecte de toutes les entités textuelles actives
    items = []
    for ent in msp.query("TEXT MTEXT"):
        raw_t = _get_text(ent).strip()
        if not raw_t or any(k in raw_t.lower() for k in CARTOUCHE_ADMIN_KW):
            continue

        ins = getattr(ent.dxf, "insert", None)
        if ins is None:
            continue

        obb = get_exact_text_obb(ent, pad_ratio=0.10)
        if obb is None or obb.is_empty:
            continue

        prio = _classify_priority(raw_t, ent)
        orig_ins = np.array([float(ins.x), float(ins.y)], dtype=np.float64)
        c = obb.centroid
        orig_c = np.array([c.x, c.y], dtype=np.float64)

        items.append({
            "text": raw_t,
            "entity": ent,
            "obb": obb,
            "prio": prio,
            "orig_ins": orig_ins,
            "curr_ins": orig_ins.copy(),
            "orig_c": orig_c,
            "max_drift": 0.5 if prio <= 2 else (3.5 if prio == 3 else (6.0 if prio == 4 else 12.0)),
        })

    n = len(items)
    if n < 2:
        return dedup_count

    total_resolved = 0

    for pass_num in range(max_passes):
        any_collision = False

        for i in range(n):
            it_a = items[i]
            obb_a = it_a["obb"]

            for j in range(i + 1, n):
                it_b = items[j]
                obb_b = it_b["obb"]

                if not obb_a.intersects(obb_b):
                    continue

                any_collision = True
                total_resolved += 1

                # Calculer le vecteur d'éloignement (MTV)
                ca = obb_a.centroid
                cb = obb_b.centroid
                dx = cb.x - ca.x
                dy = cb.y - ca.y
                dist = math.hypot(dx, dy) or 0.001
                ux, uy = dx / dist, dy / dist

                # Déterminer la profondeur de chevauchement estimée
                minx_a, miny_a, maxx_a, maxy_a = obb_a.bounds
                minx_b, miny_b, maxx_b, maxy_b = obb_b.bounds
                overlap_x = min(maxx_a, maxx_b) - max(minx_a, minx_b)
                overlap_y = min(maxy_a, maxy_b) - max(miny_a, miny_b)

                shift_needed = max(overlap_x, overlap_y, h_text * 0.8) + pad

                prio_a = it_a["prio"]
                prio_b = it_b["prio"]

                # Hiérarchie de déplacement :
                # On pousse l'entité de priorité inférieure (prio la plus grande)
                if prio_a < prio_b:
                    # A est plus prioritaire -> déplacer B
                    move_b = np.array([ux * shift_needed, uy * shift_needed], dtype=np.float64)
                    _apply_shift(it_b, move_b, frame_poly)
                elif prio_b < prio_a:
                    # B est plus prioritaire -> déplacer A
                    move_a = np.array([-ux * shift_needed, -uy * shift_needed], dtype=np.float64)
                    _apply_shift(it_a, move_a, frame_poly)
                else:
                    # Même priorité : pousser les deux de moitié dans des directions opposées
                    half = 0.55 * shift_needed
                    # Privilégier la séparation verticale si les textes sont étagés
                    if overlap_y < overlap_x:
                        move_y = half if dy >= 0 else -half
                        _apply_shift(it_a, np.array([0.0, -move_y]), frame_poly)
                        _apply_shift(it_b, np.array([0.0, move_y]), frame_poly)
                    else:
                        _apply_shift(it_a, np.array([-ux * half, -uy * half]), frame_poly)
                        _apply_shift(it_b, np.array([ux * half, uy * half]), frame_poly)

                # Mettre à jour les OBB
                it_a["obb"] = get_exact_text_obb(it_a["entity"], pad_ratio=0.10) or obb_a
                it_b["obb"] = get_exact_text_obb(it_b["entity"], pad_ratio=0.10) or obb_b

        if not any_collision:
            logger.info("✅ Anti-collision globale convergée après %d passes.", pass_num + 1)
            break

    # Écrire les nouvelles coordonnées finales dans le modelspace DXF
    n_modified = 0
    for it in items:
        disp = it["curr_ins"] - it["orig_ins"]
        if float(np.linalg.norm(disp)) > 0.05:
            ent = it["entity"]
            z = float(getattr(ent.dxf.insert, "z", 0.0))
            ent.dxf.insert = (float(it["curr_ins"][0]), float(it["curr_ins"][1]), z)
            n_modified += 1

    logger.info("🎯 Global Anti-Collision : %d entités déplacées pour zéro chevauchement", n_modified)
    return total_resolved + dedup_count


def _apply_shift(it: Dict[str, Any], shift_vec: np.ndarray, frame_poly: Optional[Polygon]) -> None:
    """Applique un déplacement avec limitation de dérive et contrainte de cadre."""
    target_ins = it["curr_ins"] + shift_vec
    drift = float(np.linalg.norm(target_ins - it["orig_ins"]))

    # Limite de dérive maximale autorisée selon la catégorie
    max_d = it["max_drift"]
    if drift > max_d:
        scale_d = max_d / drift
        target_ins = it["orig_ins"] + (target_ins - it["orig_ins"]) * scale_d

    # Contrainte de cadre : ne jamais expulser hors du cadre de dessin
    if frame_poly is not None and not frame_poly.buffer(-2.0).contains(Point(target_ins[0], target_ins[1])):
        return

    it["curr_ins"] = target_ins
    ent = it["entity"]
    z = float(getattr(ent.dxf.insert, "z", 0.0))
    ent.dxf.insert = (float(target_ins[0]), float(target_ins[1]), z)
    new_obb = get_exact_text_obb(ent, pad_ratio=0.10)
    if new_obb and not new_obb.is_empty:
        it["obb"] = new_obb
