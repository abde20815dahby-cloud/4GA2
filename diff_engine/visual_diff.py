# -*- coding: utf-8 -*-
"""
visual_diff.py: Moteur d'analyse différentielle entre un DXF produit (OURS) et la référence expert (DEF).
Compare entité par entité avec seuils de tolérance spatiale et classification sémantique.
"""
import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from shapely.geometry import Polygon, Point

from score_all import load_dxf_safe, extract_all_texts, DxfText, detect_frame_and_scale_simple, decode_dxf_unicode


@dataclass
class DiffItem:
    """Représente une entité textuelle comparée entre DEF et OURS."""
    status: str  # "MATCH", "SHIFTED", "MISSING", "PARASITE"
    category: str  # "borne", "paren_borne", "riverain_RT", "tc_nature", "linear_detail", etc.
    text_def: Optional[str] = None
    text_ours: Optional[str] = None
    pos_def: Optional[Tuple[float, float]] = None
    pos_ours: Optional[Tuple[float, float]] = None
    rot_def: Optional[float] = None
    rot_ours: Optional[float] = None
    dist: float = 0.0
    angle_diff: float = 0.0
    details: str = ""


@dataclass
class DiffResult:
    """Résultat complet de la comparaison différentielle."""
    parcel_id: str
    def_path: str
    ours_path: str
    category_filter: Optional[str]
    matches: List[DiffItem] = field(default_factory=list)      # Vert
    shifted: List[DiffItem] = field(default_factory=list)      # Jaune
    missing: List[DiffItem] = field(default_factory=list)      # Rouge
    parasites: List[DiffItem] = field(default_factory=list)    # Bleu
    parcel_poly: Optional[Polygon] = None
    all_def_texts: List[DxfText] = field(default_factory=list)
    all_ours_texts: List[DxfText] = field(default_factory=list)

    @property
    def total_def(self) -> int:
        return len(self.matches) + len(self.shifted) + len(self.missing)

    @property
    def total_ours(self) -> int:
        return len(self.matches) + len(self.shifted) + len(self.parasites)

    @property
    def is_perfect(self) -> bool:
        return len(self.missing) == 0 and len(self.parasites) == 0 and len(self.shifted) == 0


def get_category_tolerance(category: str, scale: int = 1000) -> Tuple[float, float]:
    """
    Renvoie les tolérances (dist_max_parfaite, angle_max_parfait) selon la catégorie et l'échelle.
    Tolérance cadastrale ANCFCC officielle: 2.0 mm sur plan papier.
    """
    scale_factor = scale / 1000.0
    if category == "borne":
        return (2.0 * scale_factor, 15.0)  # 2.0mm sur plan (4.0m à 1/2000, 2.0m à 1/1000)
    elif category == "paren_borne":
        return (2.5 * scale_factor, 30.0)
    elif category in ("riverain_RT", "riverain_name", "verborne"):
        return (5.0 * scale_factor, 10.0)
    elif category == "tc_nature":
        return (6.0 * scale_factor, 180.0)
    elif category == "linear_detail":
        return (12.0 * scale_factor, 15.0)
    return (3.0 * scale_factor, 15.0)


def normalize_text_content(text: str) -> str:
    """Normalise une chaîne de texte pour comparaison insensible à la casse et ponctuation."""
    if not text:
        return ""
    t = decode_dxf_unicode(text.strip().lower())
    t = t.replace("\n", " ").replace("\\p", " ").replace("’", "'").replace("`", "'")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\.,;:_]", "", t)
    # Remplacer les caractères arabes équivalents
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    return t.strip()


def extract_parcel_polygon_from_msp(msp) -> Optional[Polygon]:
    """Extrait le polygone principal de la parcelle depuis le modelspace."""
    for ent in msp.query("LWPOLYLINE POLYLINE"):
        layer = (getattr(ent.dxf, "layer", "") or "").lower()
        if layer in ("parcelle", "lmts", "limite", "parcel"):
            pts = []
            if ent.dxftype() == "LWPOLYLINE":
                pts = [(float(p[0]), float(p[1])) for p in ent.get_points(format="xy")]
            elif ent.dxftype() == "POLYLINE":
                pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in ent.vertices]
            if len(pts) >= 3:
                try:
                    poly = Polygon(pts)
                    if poly.is_valid and poly.area > 50.0:
                        return poly
                except Exception:
                    pass
    return None


def compare_dxf_files(
    def_path: str,
    ours_path: str,
    parcel_id: str = "",
    category_filter: Optional[str] = None
) -> DiffResult:
    """
    Compare un fichier DEF avec un fichier OURS.
    Si category_filter est spécifié (ex: 'borne', 'riverain_RT', etc.),
    on filtre uniquement sur cette catégorie.
    """
    def_doc = load_dxf_safe(def_path)
    ours_doc = load_dxf_safe(ours_path)

    def_texts = extract_all_texts(def_doc.modelspace())
    ours_texts = extract_all_texts(ours_doc.modelspace())

    parcel_poly = extract_parcel_polygon_from_msp(def_doc.modelspace())
    if parcel_poly is None:
        parcel_poly = extract_parcel_polygon_from_msp(ours_doc.modelspace())

    result = DiffResult(
        parcel_id=parcel_id or "PARCELLE",
        def_path=def_path,
        ours_path=ours_path,
        category_filter=category_filter,
        parcel_poly=parcel_poly,
        all_def_texts=def_texts,
        all_ours_texts=ours_texts
    )

    _, detected_scale = detect_frame_and_scale_simple(def_doc.modelspace())
    scale = detected_scale if detected_scale in (200, 500, 1000, 2000, 5000) else 1000

    # Filtrer par catégorie si demandé
    relevant_def = []
    for t in def_texts:
        if category_filter:
            if category_filter == "borne" and t.category != "borne":
                continue
            elif category_filter == "riverain" and t.category not in ("riverain_RT", "riverain_name", "verborne", "paren_borne", "tc_nature"):
                continue
            elif category_filter == "linear" and t.category != "linear_detail":
                continue
            elif category_filter not in ("borne", "riverain", "linear") and t.category != category_filter:
                continue
        relevant_def.append(t)

    relevant_ours = []
    for t in ours_texts:
        if category_filter:
            if category_filter == "borne" and t.category != "borne":
                continue
            elif category_filter == "riverain" and t.category not in ("riverain_RT", "riverain_name", "verborne", "paren_borne", "tc_nature"):
                continue
            elif category_filter == "linear" and t.category != "linear_detail":
                continue
            elif category_filter not in ("borne", "riverain", "linear") and t.category != category_filter:
                continue
        relevant_ours.append(t)

    # Déduplication des doublons dans DEF (artefacts DAO / double-clics / copies d'essai)
    unique_def = []
    seen_def_keys = set()
    # Pour les bornes, chaque identifiant (B1, B2...) est strictement unique sur la parcelle
    # S'il y a des doublons DAO dans DEF, on garde l'entité canonique la plus proche de OURS
    borne_def_map: Dict[str, List[DxfText]] = {}
    other_def: List[DxfText] = []

    for dt in relevant_def:
        norm = normalize_text_content(dt.text)
        if dt.category == "borne":
            borne_def_map.setdefault(norm, []).append(dt)
        else:
            key = (norm, round(dt.x, 1), round(dt.y, 1))
            if key not in seen_def_keys:
                seen_def_keys.add(key)
                other_def.append(dt)

    for norm, dt_list in borne_def_map.items():
        if len(dt_list) == 1:
            unique_def.append(dt_list[0])
        else:
            best_dt = dt_list[0]
            best_d = float("inf")
            ours_matching = [ot for ot in relevant_ours if normalize_text_content(ot.text) == norm]
            for dt in dt_list:
                if ours_matching:
                    for ot in ours_matching:
                        d = math.hypot(dt.x - ot.x, dt.y - ot.y)
                        if d < best_d:
                            best_d = d
                            best_dt = dt
                elif parcel_poly:
                    d = parcel_poly.exterior.distance(Point(dt.x, dt.y))
                    if d < best_d:
                        best_d = d
                        best_dt = dt
            unique_def.append(best_dt)

    unique_def.extend(other_def)
    relevant_def = unique_def

    # Appariement intelligent
    used_ours_indices: Set[int] = set()

    # Passe 1: Appariement par contenu normalisé exact le plus proche
    for dt in relevant_def:
        norm_dt = normalize_text_content(dt.text)
        if not norm_dt:
            continue

        best_idx = -1
        best_dist = float("inf")

        for oi, ot in enumerate(relevant_ours):
            if oi in used_ours_indices:
                continue
            norm_ot = normalize_text_content(ot.text)
            if norm_dt == norm_ot:
                d = math.hypot(dt.x - ot.x, dt.y - ot.y)
                if d < best_dist:
                    best_dist = d
                    best_idx = oi

        max_tol_dist, max_tol_ang = get_category_tolerance(dt.category, scale=scale)

        if best_idx >= 0:
            ot = relevant_ours[best_idx]
            used_ours_indices.add(best_idx)

            ang_diff = abs(float(dt.rotation) - float(ot.rotation)) % 360.0
            if ang_diff > 180.0:
                ang_diff = 360.0 - ang_diff

            if best_dist <= max_tol_dist:
                result.matches.append(DiffItem(
                    status="MATCH",
                    category=dt.category,
                    text_def=dt.text,
                    text_ours=ot.text,
                    pos_def=(dt.x, dt.y),
                    pos_ours=(ot.x, ot.y),
                    rot_def=dt.rotation,
                    rot_ours=ot.rotation,
                    dist=round(best_dist, 2),
                    angle_diff=round(ang_diff, 1),
                    details=f"Écart={best_dist:.2f}m (≤ {max_tol_dist}m)"
                ))
            else:
                result.shifted.append(DiffItem(
                    status="SHIFTED",
                    category=dt.category,
                    text_def=dt.text,
                    text_ours=ot.text,
                    pos_def=(dt.x, dt.y),
                    pos_ours=(ot.x, ot.y),
                    rot_def=dt.rotation,
                    rot_ours=ot.rotation,
                    dist=round(best_dist, 2),
                    angle_diff=round(ang_diff, 1),
                    details=f"Décalé de {best_dist:.2f}m (> seuil {max_tol_dist}m)"
                ))
        else:
            # Pas trouvé dans OURS -> Manquant (Rouge)
            result.missing.append(DiffItem(
                status="MISSING",
                category=dt.category,
                text_def=dt.text,
                pos_def=(dt.x, dt.y),
                rot_def=dt.rotation,
                details="Présent dans DEF mais absent dans OURS"
            ))

    # Passe 2: Identifier les parasites dans OURS (non utilisés)
    for oi, ot in enumerate(relevant_ours):
        if oi not in used_ours_indices:
            result.parasites.append(DiffItem(
                status="PARASITE",
                category=ot.category,
                text_ours=ot.text,
                pos_ours=(ot.x, ot.y),
                rot_ours=ot.rotation,
                details="Présent dans OURS mais absent dans DEF (Parasite)"
            ))

    return result
