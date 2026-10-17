# -*- coding: utf-8 -*-
r"""
score_all.py — Moteur de Scoring Multi-Dimensionnel DAHBY PRO SUITE
====================================================================
Compare automatiquement le résultat du pipeline (OURS) avec la référence
expert (DEF) sur 10 dimensions, pour TOUTES les parcelles disponibles.

Usage:
    python -m engines.dxf_cleaner.score_all
    python engines/dxf_cleaner/score_all.py --zone LEQSAQSA
    python engines/dxf_cleaner/score_all.py --parcel P216

Sortie : CSV (scores_baseline.csv) + rapport console.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import ezdxf
import ezdxf.recover

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Logging structuré ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_all")

# ── Constantes ─────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\Admin\Downloads\Sous_Zone 01")
ZONES = ["IGHANOUYINE", "LEQSAQSA"]
SCRATCH_DIR = Path(r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE\engines\dxf_cleaner\_score_scratch")
OUTPUT_CSV = Path(r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE\engines\dxf_cleaner\scores_baseline.csv")
SECTORS_DIR = r"C:\Users\Admin\Downloads\SS"

# Regex patterns pour classification des textes
RE_BORNE_LABEL = re.compile(r"^B\d+$", re.IGNORECASE)
RE_PAREN_BORNE = re.compile(r"^\(B\d+\)$", re.IGNORECASE)
RE_RIVERAIN_RT = re.compile(r"^[RT]\.?\s*\d+", re.IGNORECASE)
RE_VERBORNE = re.compile(r"^v\.?\s*\(?B\d+\)?", re.IGNORECASE)
RE_LINEAR_DETAIL = re.compile(
    r"(piste|chemin|route|oued|chaaba|cha[âa]bat|depression|L\.?E\.?[BMH]\.?T|sentier|canal|cloture|clôture)",
    re.IGNORECASE,
)
RE_COORD_NUMBER = re.compile(r"^\d{5,6}(\.\d+)?$")
RE_TC_NATURE = re.compile(
    r"(T\.?C\.?|terrain|culture|argan|amand|oliv|vigne|figuier|cactus|jachère|labour|irrig)",
    re.IGNORECASE,
)


# ── DXF Loading (copie de pipeline.py) ─────────────────────────────────
def _fix_malformed_group_codes(raw: bytes) -> bytes:
    """Répare les corruptions de codes de groupe DXF."""
    fixed = re.sub(rb"0[ \t]+0\r?\n", b"0\r\n", raw)
    pos_corrupt = -1
    for i in range(len(fixed) - 10):
        b = fixed[i]
        if b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19,
                 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            pos_corrupt = i
            break
    if pos_corrupt != -1:
        end_sec = fixed.find(b"ENDSEC", pos_corrupt)
        if end_sec != -1:
            fixed = fixed[:pos_corrupt] + b"\r\n  0\r\n" + fixed[end_sec:]
    return fixed


def load_dxf_safe(path: str) -> ezdxf.document.Drawing:
    """Charge un fichier DXF en réparant les corruptions."""
    with open(path, "rb") as fh:
        raw = fh.read()
    raw = _fix_malformed_group_codes(raw)
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
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


def decode_dxf_unicode(text: str) -> str:
    """Décode les séquences Unicode AutoCAD \\U+XXXX -> caractère réel."""
    if not text:
        return ""
    t = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), text)
    t = re.sub(r"\\u\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), t)
    return t


# ── Extraction de texte depuis un DXF ──────────────────────────────────
@dataclass
class DxfText:
    """Représente un texte extrait d'un fichier DXF."""
    text: str
    x: float
    y: float
    rotation: float
    height: float
    layer: str
    entity_type: str  # TEXT ou MTEXT

    @property
    def normalized_text(self) -> str:
        """Texte nettoyé pour comparaison — strip MTEXT codes et normalisation."""
        t = self.text.strip()
        # Décoder les séquences Unicode AutoCAD \U+XXXX -> caractère réel
        t = re.sub(r"\\U\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), t)
        t = re.sub(r"\\u\+([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), t)
        # Strip MTEXT formatting codes restants
        t = re.sub(r"\\f[^;]*;", "", t)
        t = re.sub(r"\\[HhWwTtQqAa][^;]*;", "", t)
        t = re.sub(r"\\[Cc]\d+;", "", t)
        t = re.sub(r"\\[LlOoKk]", "", t)
        t = re.sub(r"\\S([^;]*);", r"\1", t)
        t = t.replace("{", "").replace("}", "")
        t = t.replace("\\P", " ").replace("\\p", " ")
        t = t.replace("\\~", " ").replace("\\\\", "\\")
        t = t.replace("%%d", "°").replace("%%p", "±").replace("%%c", "Ø")
        t = t.replace("\n", " ").replace("\r", " ")
        # FIX 1.5: Coordonnées carroyage (supprimer .0 ou .00)
        t = re.sub(r"(\b\d{5,6})\.0+\b", r"\1", t)
        # Normaliser les verbornes v.(Bxxx) <-> v.Bxxx
        t = re.sub(r"\b[vV]\.?\s*\(([A-Za-z0-9]+)\)", r"v.\1", t)
        t = re.sub(r"\b[vV]\.?\s*([A-Za-z0-9]+)", r"v.\1", t)
        # Normaliser les espaces multiples
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    @property
    def category(self) -> str:
        """Catégorise le texte avec distinction stricte entre cartouche, riverains et détails."""
        nt = self.normalized_text
        raw_lower = self.text.strip().lower()
        # Décoder les séquences Unicode dans le texte brut pour le test cartouche
        raw_decoded = re.sub(r"\\u\+([0-9a-f]{4})", lambda m: chr(int(m.group(1), 16)), raw_lower)

        # 1. Textes administratifs et cartouche
        cartouche_keywords = (
            "echelle", "échelle", "plan levé", "plan leve", "plan établi", "plan etabli",
            "propriété dite", "propriete dite", "contenance", "st 2", "st2",
            "mappe de reperage", "mappe de repérage", "carte topographique",
            "géomètre", "geometre", "dessinateur", "operateur", "opérateur",
            "soritopo", "reconstitution", "bornage", "cadastre", "plan parcellaire",
            "plan provisoire", "nord", "شمال", "مطلب", "الملك", "المسمى", "رسم",
            "الكائن", "مساحة", "محافظة", "مملكة", "جماعة", "دائرة", "قيادة",
            "direction du cadastre", "prefecture", "préfecture", "caidat", "caïdat",
            "commune", "37-45-", "37-44-", "ife :", "limite de la propriété", "limite de la propriete",
            "située", "situee", "province", "conservation", "pce d'", "cle de", "cne.", "cne ",
            "douar", "oujda-angad", "mestferki"
        )
        if any(kw in raw_decoded for kw in cartouche_keywords) or any(kw in nt.lower() for kw in cartouche_keywords):
            return "cartouche"
        if re.search(r"plle\s*n[°o]?\s*:\s*\d+", raw_decoded):
            return "cartouche"
        if re.match(r"^titre\s*:\s*$", raw_decoded) or re.match(r"^réquisition\s*n[°o]?\s*:\s*$", raw_decoded):
            return "cartouche"

        # 2. Bornes
        if RE_BORNE_LABEL.match(nt):
            return "borne"
        if RE_PAREN_BORNE.match(nt):
            return "paren_borne"
        if RE_RIVERAIN_RT.match(nt):
            return "riverain_RT"
        if RE_VERBORNE.match(nt):
            return "verborne"

        # 3. Détails linéaires et coordonnées
        if RE_LINEAR_DETAIL.search(nt):
            return "linear_detail"
        if RE_COORD_NUMBER.match(nt):
            return "coord"

        # 4. Nature de culture (TC, terrain de culture...)
        if nt.lower() == "tc" or raw_decoded.upper() == "TC" or RE_TC_NATURE.search(nt):
            return "tc_nature"

        # 5. Riverains non-officiels (noms propres, Héritiers, Plle voisine, etc.)
        if any(kw in nt.lower() for kw in ("héritier", "heritier", "plle", "titre", "hér.")):
            return "riverain_name"
        if len(nt) > 3 and not nt.isdigit():
            return "riverain_name"
        return "other"



def extract_all_texts(msp) -> List[DxfText]:
    """Extrait tous les textes (TEXT + MTEXT) d'un modelspace DXF."""
    texts: List[DxfText] = []
    for ent in msp.query("TEXT"):
        try:
            raw_text = getattr(ent.dxf, "text", "")
            if not raw_text or not raw_text.strip():
                continue
            ins = ent.dxf.insert
            texts.append(DxfText(
                text=raw_text,
                x=ins.x if ins else 0.0,
                y=ins.y if ins else 0.0,
                rotation=getattr(ent.dxf, "rotation", 0.0),
                height=getattr(ent.dxf, "height", 2.0),
                layer=getattr(ent.dxf, "layer", "0"),
                entity_type="TEXT",
            ))
        except Exception:
            continue

    for ent in msp.query("MTEXT"):
        try:
            if hasattr(ent, "plain_text"):
                raw_text = ent.plain_text()
            else:
                raw_text = getattr(ent, "text", "")
            if not raw_text or not raw_text.strip():
                continue
            ins = ent.dxf.insert
            texts.append(DxfText(
                text=raw_text,
                x=ins.x if ins else 0.0,
                y=ins.y if ins else 0.0,
                rotation=getattr(ent.dxf, "rotation", 0.0),
                height=getattr(ent.dxf, "char_height", 2.0),
                layer=getattr(ent.dxf, "layer", "0"),
                entity_type="MTEXT",
            ))
        except Exception:
            continue

    return texts


# ── Matching de textes entre deux listes ───────────────────────────────
def _euclidean(t1: DxfText, t2: DxfText) -> float:
    """Distance euclidienne entre deux textes."""
    return math.sqrt((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2)


def _angle_diff(a1: float, a2: float) -> float:
    """Différence angulaire minimale en degrés [0, 180]."""
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


def _normalize_deep(text: str) -> str:
    """Normalisation profonde : accents, ponctuation, casse, abréviations usuelles."""
    t = text.lower().strip()
    t = re.sub(r"\\u\+([0-9a-f]{4})", lambda m: chr(int(m.group(1), 16)), t)
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = re.sub(r"\((?:ddp|dp|pu|bc(?:\s+du\s+[^\)]*)?)\)", "", t)
    t = re.sub(r"[.\s\-/\\_,;:()\[\]]+", "", t)
    t = re.sub(r"h[eé]r(itier)?s?\.?", "hrt", t)
    t = re.sub(r"^hrt", "", t)
    t = re.sub(r"^const(ruction)?", "const", t)
    t = re.sub(r"p(lle|arcelle)\s*", "p", t)
    t = re.sub(r"tc$", "", t)
    return t


def match_texts_by_content(
    def_texts: List[DxfText],
    ours_texts: List[DxfText],
    max_dist: float = 500.0,
) -> Tuple[List[Tuple[DxfText, DxfText]], List[DxfText], List[DxfText]]:
    """
    Match les textes DEF ↔ OURS par contenu normalisé.
    Passe 1 : correspondance exacte sur normalized_text.lower().
    Passe 2 : correspondance profonde via _normalize_deep() pour les reliquats.

    Returns:
        matched: list of (def_text, ours_text)
        def_only: textes dans DEF mais pas dans OURS (manquants)
        ours_only: textes dans OURS mais pas dans DEF (excédentaires)
    """
    # Grouper par contenu normalisé
    def_by_content: Dict[str, List[DxfText]] = {}
    for t in def_texts:
        key = t.normalized_text.lower()
        def_by_content.setdefault(key, []).append(t)

    ours_by_content: Dict[str, List[DxfText]] = {}
    for t in ours_texts:
        key = t.normalized_text.lower()
        ours_by_content.setdefault(key, []).append(t)

    matched: List[Tuple[DxfText, DxfText]] = []
    def_only: List[DxfText] = []
    ours_used: Set[int] = set()

    all_keys = set(def_by_content.keys()) | set(ours_by_content.keys())

    for key in all_keys:
        d_list = def_by_content.get(key, [])
        o_list = ours_by_content.get(key, [])

        # Match par proximité spatiale optimale (bipartite matching)
        if _HAS_SCIPY and (len(d_list) > 1 or len(o_list) > 1) and len(d_list) > 0 and len(o_list) > 0:
            cost = np.zeros((len(d_list), len(o_list)))
            for i, dt in enumerate(d_list):
                for j, ot in enumerate(o_list):
                    cost[i, j] = _euclidean(dt, ot)
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_rows = set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] <= max_dist:
                    matched.append((d_list[r], o_list[c]))
                    matched_rows.add(r)
                    for gi, gt in enumerate(ours_texts):
                        if gt is o_list[c] and gi not in ours_used:
                            ours_used.add(gi)
                            break
            for r in range(len(d_list)):
                if r not in matched_rows:
                    def_only.append(d_list[r])
        else:
            o_available = list(range(len(o_list)))
            for dt in d_list:
                best_idx = -1
                best_dist = max_dist + 1
                for oi in o_available:
                    ot = o_list[oi]
                    dist = _euclidean(dt, ot)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = oi
                if best_idx >= 0 and best_dist <= max_dist:
                    matched.append((dt, o_list[best_idx]))
                    o_available.remove(best_idx)
                    for gi, gt in enumerate(ours_texts):
                        if gt is o_list[best_idx] and gi not in ours_used:
                            ours_used.add(gi)
                            break
                else:
                    def_only.append(dt)

    # ── Passe 2 : Deep / Fuzzy matching pour les reliquats def_only ──
    remaining_def: List[DxfText] = []
    ours_available_indices = [gi for gi in range(len(ours_texts)) if gi not in ours_used]
    pass2_dist = 50.0  # Seuil de proximité pour les variantes typographiques

    for dt in def_only:
        dt_deep = _normalize_deep(dt.normalized_text)
        if not dt_deep:
            remaining_def.append(dt)
            continue

        best_gi = -1
        best_dist = pass2_dist + 1

        for gi in ours_available_indices:
            ot = ours_texts[gi]
            ot_deep = _normalize_deep(ot.normalized_text)
            if not ot_deep:
                continue

            # Correspondance exacte après normalisation profonde uniquement
            if dt_deep == ot_deep:
                dist = _euclidean(dt, ot)
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi

        if best_gi >= 0 and best_dist <= pass2_dist:
            matched.append((dt, ours_texts[best_gi]))
            ours_used.add(best_gi)
            ours_available_indices.remove(best_gi)
        else:
            remaining_def.append(dt)

    def_only = remaining_def

    # OURS non matchés
    ours_only: List[DxfText] = []
    for gi, gt in enumerate(ours_texts):
        if gi not in ours_used:
            key = gt.normalized_text.lower()
            if key not in def_by_content:
                ours_only.append(gt)

    return matched, def_only, ours_only


# ── Scoring par dimension ─────────────────────────────────────────────
@dataclass
class ParcelScore:
    """Score d'une parcelle sur les 10 dimensions."""
    parcel_id: str = ""
    zone: str = ""
    d1_presence: float = 0.0       # Présence des textes (F1-score)
    d2_bornes_pos: float = 0.0     # Position des bornes (Bxxx)
    d3_paren_pos: float = 0.0      # Position des (Bxx)
    d4_angles: float = 0.0         # Angles de rotation
    d5_riverains_pos: float = 0.0  # Position des riverains R.../T...
    d6_tc_pos: float = 0.0         # Position du TC
    d7_linear: float = 0.0         # Détails linéaires
    d8_duplicates: float = 0.0     # Doublons
    d9_escaped: float = 0.0        # Textes hors cadre
    d10_coords: float = 0.0        # Coordonnées carroyage
    total: float = 0.0
    n_def_texts: int = 0
    n_ours_texts: int = 0
    n_matched: int = 0
    n_def_only: int = 0
    n_ours_only: int = 0
    error: str = ""

    def compute_total(self) -> float:
        """Calcule la note globale pondérée."""
        weights = {
            "d1_presence": 2.0,
            "d2_bornes_pos": 1.5,
            "d3_paren_pos": 1.0,
            "d4_angles": 1.5,
            "d5_riverains_pos": 1.5,
            "d6_tc_pos": 1.0,
            "d7_linear": 1.0,
            "d8_duplicates": 1.5,
            "d9_escaped": 0.5,
            "d10_coords": 0.5,
        }
        total_w = sum(weights.values())
        weighted_sum = sum(
            getattr(self, dim) * w for dim, w in weights.items()
        )
        self.total = round(weighted_sum / total_w, 1)
        return self.total


def _score_position(distances: List[float], perfect_threshold: float = 2.0, fail_threshold: float = 50.0) -> float:
    """
    Score de position : 100 si distance=0, 0 si distance≥fail_threshold.

    Args:
        distances: Liste des distances euclidiennes en mètres.
        perfect_threshold: Distance en dessous de laquelle on considère parfait.
        fail_threshold: Distance au-dessus de laquelle on considère complètement raté.
    """
    if not distances:
        return 100.0

    scores = []
    for d in distances:
        if d <= perfect_threshold:
            scores.append(100.0)
        elif d >= fail_threshold:
            scores.append(0.0)
        else:
            # Interpolation linéaire
            scores.append(100.0 * (1.0 - (d - perfect_threshold) / (fail_threshold - perfect_threshold)))

    return round(sum(scores) / len(scores), 1)


def _score_angle(angle_diffs: List[float]) -> float:
    """
    Score d'angle : 100 si diff=0°, 0 si diff≥30°.
    """
    if not angle_diffs:
        return 100.0

    scores = []
    for d in angle_diffs:
        if d <= 1.0:
            scores.append(100.0)
        elif d >= 30.0:
            scores.append(0.0)
        else:
            scores.append(100.0 * (1.0 - (d - 1.0) / 29.0))

    return round(sum(scores) / len(scores), 1)


def _count_duplicates(texts: List[DxfText]) -> int:
    """Compte le nombre de textes dupliqués (même contenu normalisé)."""
    seen: Dict[str, int] = {}
    for t in texts:
        key = t.normalized_text.lower()
        seen[key] = seen.get(key, 0) + 1
    return sum(max(0, c - 1) for c in seen.values())


def score_parcel(
    def_texts: List[DxfText],
    ours_texts: List[DxfText],
    frame_bbox: Optional[Tuple[float, float, float, float]] = None,
    ours_frame_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> ParcelScore:
    """
    Calcule le score multi-dimensionnel d'une parcelle.

    Args:
        def_texts: Textes extraits du fichier DEF (référence expert)
        ours_texts: Textes extraits du fichier OURS (résultat pipeline)
        frame_bbox: Boîte englobante du cadre DEF (xmin, ymin, xmax, ymax)
        ours_frame_bbox: Boîte englobante du cadre OURS (pour D9)
    """
    score = ParcelScore()
    score.n_def_texts = len(def_texts)
    score.n_ours_texts = len(ours_texts)

    # ── D1: Présence des textes (F1-score) ────────────────────────────
    matched, def_only, ours_only = match_texts_by_content(def_texts, ours_texts)
    score.n_matched = len(matched)
    score.n_def_only = len(def_only)
    score.n_ours_only = len(ours_only)

    if len(def_texts) > 0:
        recall = len(matched) / len(def_texts)  # Proportion de DEF retrouvée
    else:
        recall = 1.0

    # Pour le precision, on considère aussi les doublons OURS
    total_ours_relevant = len(matched) + len(ours_only)
    if total_ours_relevant > 0:
        precision = len(matched) / total_ours_relevant
    else:
        precision = 1.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    score.d1_presence = round(f1 * 100, 1)

    # ── D2: Position des bornes (Bxxx) ────────────────────────────────
    borne_dists = [
        _euclidean(dt, ot)
        for dt, ot in matched
        if dt.category == "borne"
    ]
    score.d2_bornes_pos = _score_position(borne_dists, 1.0, 30.0)

    # ── D3: Position des bornes parenthésées (Bxx) ────────────────────
    paren_dists = [
        _euclidean(dt, ot)
        for dt, ot in matched
        if dt.category == "paren_borne"
    ]
    score.d3_paren_pos = _score_position(paren_dists, 2.0, 40.0)

    # ── D4: Angles de rotation ────────────────────────────────────────
    angle_diffs = [
        _angle_diff(dt.rotation, ot.rotation)
        for dt, ot in matched
        if dt.normalized_text.strip()  # Ignorer les textes vides
    ]
    score.d4_angles = _score_angle(angle_diffs)

    # ── D5: Position des riverains R.../T... ──────────────────────────
    riv_dists = [
        _euclidean(dt, ot)
        for dt, ot in matched
        if dt.category == "riverain_RT"
    ]
    score.d5_riverains_pos = _score_position(riv_dists, 3.0, 100.0)

    # ── D6: Position du TC ────────────────────────────────────────────
    tc_dists = [
        _euclidean(dt, ot)
        for dt, ot in matched
        if dt.category == "tc_nature"
    ]
    score.d6_tc_pos = _score_position(tc_dists, 5.0, 200.0)

    # ── D7: Détails linéaires (position + angle) ──────────────────────
    linear_pos_dists = []
    linear_ang_diffs = []
    for dt, ot in matched:
        if dt.category == "linear_detail":
            linear_pos_dists.append(_euclidean(dt, ot))
            linear_ang_diffs.append(_angle_diff(dt.rotation, ot.rotation))

    pos_score = _score_position(linear_pos_dists, 5.0, 100.0)
    ang_score = _score_angle(linear_ang_diffs)
    score.d7_linear = round((pos_score + ang_score) / 2, 1) if linear_pos_dists else 100.0

    # ── D8: Doublons ──────────────────────────────────────────────────
    n_dups_ours = _count_duplicates(ours_texts)
    n_dups_def = _count_duplicates(def_texts)
    extra_dups = max(0, n_dups_ours - n_dups_def)
    if len(ours_texts) > 0:
        dup_ratio = extra_dups / len(ours_texts)
        score.d8_duplicates = round(max(0.0, 100.0 - dup_ratio * 500), 1)
    else:
        score.d8_duplicates = 100.0

    # ── D9: Textes hors cadre ─────────────────────────────────────────
    eval_frame = ours_frame_bbox if ours_frame_bbox is not None else frame_bbox
    if eval_frame is not None:
        fx0, fy0, fx1, fy1 = eval_frame
        # Marge de tolérance (cartouche, marge annotations = 15% du cadre)
        margin_x = (fx1 - fx0) * 0.15
        margin_y = (fy1 - fy0) * 0.15
        expanded = (fx0 - margin_x, fy0 - margin_y, fx1 + margin_x, fy1 + margin_y)

        n_escaped = 0
        for t in ours_texts:
            if not (expanded[0] <= t.x <= expanded[2] and expanded[1] <= t.y <= expanded[3]):
                n_escaped += 1

        if len(ours_texts) > 0:
            escape_ratio = n_escaped / len(ours_texts)
            score.d9_escaped = round(max(0.0, 100.0 - escape_ratio * 500), 1)
        else:
            score.d9_escaped = 100.0
    else:
        score.d9_escaped = 100.0

    # ── D10: Coordonnées carroyage ────────────────────────────────────
    coord_dists = [
        _euclidean(dt, ot)
        for dt, ot in matched
        if dt.category == "coord"
    ]
    score.d10_coords = _score_position(coord_dists, 1.0, 20.0)

    # ── Total pondéré ─────────────────────────────────────────────────
    score.compute_total()
    return score


# ── Détection du cadre et de l'échelle (simplifié) ─────────────────────
def detect_frame_and_scale_simple(msp) -> Tuple[Tuple[float, float, float, float], int]:
    """Détecte le cadre et l'échelle d'un DXF (version simplifiée)."""
    scale = 2000
    for ent in msp.query("MTEXT TEXT"):
        txt = getattr(ent.dxf, "text", "")
        if hasattr(ent, "text") and not txt:
            txt = ent.text
        m = re.search(r"\b1\s*/\s*([1-9]\d{2,4})\b", txt)
        if m:
            scale = int(m.group(1))
            break

    parcel_pts = []
    for pl in msp.query("LWPOLYLINE"):
        if pl.dxf.layer.lower() == "parcelle":
            pts = [(p[0], p[1]) for p in pl.get_points()]
            if len(pts) >= 3:
                parcel_pts.extend(pts)

    if parcel_pts and scale >= 5000:
        p_xmin = min(p[0] for p in parcel_pts)
        p_xmax = max(p[0] for p in parcel_pts)
        p_ymin = min(p[1] for p in parcel_pts)
        p_ymax = max(p[1] for p in parcel_pts)
        max_dim = max(p_xmax - p_xmin, p_ymax - p_ymin)
        if max_dim <= 700.0:
            scale = 2000

    cadre_layers = {"0", "cadre", "cdre", "cadrep1", "habillage", "cadre_plan", "cadre_ext"}
    best_frame = None
    best_score = -1.0

    for pl in msp.query("LWPOLYLINE"):
        pts = [(p[0], p[1]) for p in pl.get_points()]
        if len(pts) in (4, 5):
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin

            if 0.08 * scale <= w <= 0.75 * scale and 0.08 * scale <= h <= 0.75 * scale:
                sc = 0.0
                if pl.dxf.layer.lower() in cadre_layers:
                    sc += 50.0
                if parcel_pts:
                    p_xmin = min(p[0] for p in parcel_pts)
                    p_xmax = max(p[0] for p in parcel_pts)
                    p_ymin = min(p[1] for p in parcel_pts)
                    p_ymax = max(p[1] for p in parcel_pts)
                    if (xmin <= p_xmin + 5.0 and xmax >= p_xmax - 5.0
                            and ymin <= p_ymin + 5.0 and ymax >= p_ymax - 5.0):
                        sc += 100.0
                if sc > best_score:
                    best_score = sc
                    best_frame = (xmin, ymin, xmax, ymax)

    if best_frame:
        return best_frame, scale

    if parcel_pts:
        p_xmin = min(p[0] for p in parcel_pts)
        p_xmax = max(p[0] for p in parcel_pts)
        p_ymin = min(p[1] for p in parcel_pts)
        p_ymax = max(p[1] for p in parcel_pts)
        pad_x = max((p_xmax - p_xmin) * 0.35, 20.0)
        pad_y = max((p_ymax - p_ymin) * 0.35, 20.0)
        return (p_xmin - pad_x, p_ymin - pad_y, p_xmax + pad_x, p_ymax + pad_y), scale

    return (0.0, 0.0, 0.26 * scale, 0.16 * scale), scale


# ── Découverte des paires DEF / OREGEN ─────────────────────────────────
def discover_parcel_pairs(
    zones: Optional[List[str]] = None,
    specific_parcel: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Découvre toutes les paires (DEF, OREGEN) disponibles.

    Returns:
        Liste de dicts avec clés : parcel_id, zone, def_path, oregen_path
    """
    if zones is None:
        zones = ZONES

    pairs: List[Dict[str, str]] = []

    for zone in zones:
        def_dir = BASE_DIR / zone / "DEF"
        oregen_dir = BASE_DIR / zone / "OREGEN"

        if not def_dir.exists() or not oregen_dir.exists():
            logger.warning("Zone %s : répertoire DEF ou OREGEN manquant", zone)
            continue

        for def_subdir in sorted(def_dir.iterdir()):
            if not def_subdir.is_dir():
                continue

            pid = def_subdir.name  # ex: P14, P216

            # Chercher le fichier DEF (gère les noms variables)
            def_path = None
            for candidate in def_subdir.glob("*-PLAN.dxf"):
                # Éviter les sous-dossiers dupliqués (P216/P216/P216-PLAN.dxf)
                if candidate.parent == def_subdir:
                    def_path = str(candidate)
                    break

            if def_path is None:
                # Chercher aussi les noms avec "DEFINITIVE"
                for candidate in def_subdir.glob("*DEFINITIVE*-PLAN.dxf"):
                    if candidate.parent == def_subdir:
                        def_path = str(candidate)
                        break

            if def_path is None:
                # Chercher n'importe quel .dxf
                for candidate in def_subdir.glob("*.dxf"):
                    if candidate.parent == def_subdir:
                        def_path = str(candidate)
                        break

            if def_path is None:
                logger.debug("Pas de DEF trouvé pour %s/%s", zone, pid)
                continue

            # Chercher le fichier OREGEN correspondant
            # Structure 1 (LEQSAQSA) : OREGEN/P{N}/P{N}-PLAN.dxf
            # Structure 2 (IGHANOUYINE) : OREGEN/PLAN-IND-DXF_.../P{N}/P{N}-PLAN.dxf
            oregen_path = None

            # Essai direct
            oregen_subdir = oregen_dir / pid
            if oregen_subdir.exists() and oregen_subdir.is_dir():
                for candidate in oregen_subdir.glob("*-PLAN.dxf"):
                    oregen_path = str(candidate)
                    break
                if oregen_path is None:
                    for candidate in oregen_subdir.glob("*.dxf"):
                        oregen_path = str(candidate)
                        break

            # Essai récursif (structure IGHANOUYINE avec sous-dossier intermédiaire)
            if oregen_path is None:
                for sub in oregen_dir.iterdir():
                    if sub.is_dir() and sub.name != pid:
                        nested = sub / pid
                        if nested.exists() and nested.is_dir():
                            for candidate in nested.glob("*-PLAN.dxf"):
                                oregen_path = str(candidate)
                                break
                            if oregen_path is None:
                                for candidate in nested.glob("*.dxf"):
                                    oregen_path = str(candidate)
                                    break
                    if oregen_path is not None:
                        break

            if oregen_path is None:
                logger.debug("Pas d'OREGEN trouvé pour %s/%s", zone, pid)
                continue

            if specific_parcel and pid != specific_parcel:
                continue

            pairs.append({
                "parcel_id": pid,
                "zone": zone,
                "def_path": def_path,
                "oregen_path": oregen_path,
            })

    logger.info("Paires découvertes : %d", len(pairs))
    return pairs


# ── Pipeline léger pour générer OURS ───────────────────────────────────
def run_pipeline_on_oregen(oregen_path: str, output_path: str) -> bool:
    """
    Exécute le pipeline de nettoyage sur un fichier OREGEN.

    Returns:
        True si succès, False sinon.
    """
    try:
        # Import du pipeline
        sys.path.insert(0, str(Path(r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")))
        from engines.dxf_cleaner.pipeline import clean_dxf_file
        clean_dxf_file(oregen_path, output_path)
        return True
    except Exception as ex:
        logger.error("Pipeline échoué pour %s : %s", oregen_path, ex)
        return False


# ── Scoring direct DEF vs OREGEN (sans pipeline) ──────────────────────
def score_def_vs_raw(
    def_path: str,
    oregen_path: str,
) -> ParcelScore:
    """
    Score le fichier OREGEN brut contre le DEF (avant pipeline).
    Utile pour mesurer l'apport du pipeline.
    """
    try:
        def_doc = load_dxf_safe(def_path)
        def_msp = def_doc.modelspace()
        def_texts = extract_all_texts(def_msp)
        def_frame, _ = detect_frame_and_scale_simple(def_msp)

        ore_doc = load_dxf_safe(oregen_path)
        ore_msp = ore_doc.modelspace()
        ore_texts = extract_all_texts(ore_msp)
        ore_frame, _ = detect_frame_and_scale_simple(ore_msp)

        return score_parcel(def_texts, ore_texts, def_frame, ours_frame_bbox=ore_frame)
    except Exception as ex:
        score = ParcelScore()
        score.error = str(ex)
        return score


# ── Scoring DEF vs OURS (après pipeline) ──────────────────────────────
def score_def_vs_ours(
    def_path: str,
    ours_path: str,
) -> ParcelScore:
    """
    Score le résultat du pipeline (OURS) contre le DEF expert.
    """
    try:
        def_doc = load_dxf_safe(def_path)
        def_msp = def_doc.modelspace()
        def_texts = extract_all_texts(def_msp)
        def_frame, _ = detect_frame_and_scale_simple(def_msp)

        ours_doc = load_dxf_safe(ours_path)
        ours_msp = ours_doc.modelspace()
        ours_texts = extract_all_texts(ours_msp)
        ours_frame, _ = detect_frame_and_scale_simple(ours_msp)

        return score_parcel(def_texts, ours_texts, def_frame, ours_frame_bbox=ours_frame)
    except Exception as ex:
        score = ParcelScore()
        score.error = str(ex)
        return score


# ── Main : Scoring de toutes les parcelles ─────────────────────────────
def run_full_scoring(
    zones: Optional[List[str]] = None,
    specific_parcel: Optional[str] = None,
    run_pipeline: bool = True,
    compare_raw: bool = False,
) -> List[ParcelScore]:
    """
    Exécute le scoring sur toutes les parcelles disponibles.

    Args:
        zones: Liste des zones à traiter (défaut: toutes)
        specific_parcel: Si spécifié, ne traite que cette parcelle
        run_pipeline: Si True, exécute le pipeline sur OREGEN pour créer OURS
        compare_raw: Si True, compare aussi OREGEN brut vs DEF
    """
    pairs = discover_parcel_pairs(zones, specific_parcel)
    if not pairs:
        logger.error("Aucune paire DEF/OREGEN trouvée !")
        return []

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    scores: List[ParcelScore] = []

    for i, pair in enumerate(pairs, 1):
        pid = pair["parcel_id"]
        zone = pair["zone"]
        def_path = pair["def_path"]
        oregen_path = pair["oregen_path"]

        logger.info("━━━ [%d/%d] %s/%s ━━━", i, len(pairs), zone, pid)

        # Étape 1 : Générer OURS via pipeline
        ours_path = str(SCRATCH_DIR / f"{pid}_OURS.dxf")

        if run_pipeline:
            logger.info("  → Pipeline sur %s...", os.path.basename(oregen_path))
            success = run_pipeline_on_oregen(oregen_path, ours_path)
            if not success:
                score = ParcelScore(parcel_id=pid, zone=zone, error="Pipeline failed")
                scores.append(score)
                continue
        elif not os.path.exists(ours_path):
            logger.warning("  → OURS non trouvé et pipeline désactivé : %s", ours_path)
            score = ParcelScore(parcel_id=pid, zone=zone, error="No OURS file")
            scores.append(score)
            continue

        # Étape 2 : Scorer DEF vs OURS
        logger.info("  → Scoring DEF vs OURS...")
        score = score_def_vs_ours(def_path, ours_path)
        score.parcel_id = pid
        score.zone = zone

        if score.error:
            logger.error("  ✗ Erreur : %s", score.error)
        else:
            logger.info(
                "  ✓ TOTAL=%.1f%% | D1=%.0f D2=%.0f D3=%.0f D4=%.0f D5=%.0f D6=%.0f D7=%.0f D8=%.0f D9=%.0f D10=%.0f",
                score.total, score.d1_presence, score.d2_bornes_pos, score.d3_paren_pos,
                score.d4_angles, score.d5_riverains_pos, score.d6_tc_pos,
                score.d7_linear, score.d8_duplicates, score.d9_escaped, score.d10_coords,
            )

        scores.append(score)

        # Optionnel : Score OREGEN brut
        if compare_raw:
            raw_score = score_def_vs_raw(def_path, oregen_path)
            logger.info("  📊 RAW OREGEN total=%.1f%%", raw_score.total)

    return scores


def write_csv(scores: List[ParcelScore], output_path: Optional[str] = None) -> str:
    """Écrit les scores dans un fichier CSV."""
    if output_path is None:
        output_path = str(OUTPUT_CSV)

    fieldnames = [
        "parcel_id", "zone",
        "d1_presence", "d2_bornes_pos", "d3_paren_pos", "d4_angles",
        "d5_riverains_pos", "d6_tc_pos", "d7_linear", "d8_duplicates",
        "d9_escaped", "d10_coords", "total",
        "n_def_texts", "n_ours_texts", "n_matched", "n_def_only", "n_ours_only",
        "error",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            row = asdict(score)
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    logger.info("CSV écrit : %s", output_path)
    return output_path


def print_summary(scores: List[ParcelScore]) -> None:
    """Affiche un résumé global des scores."""
    valid_scores = [s for s in scores if not s.error]
    if not valid_scores:
        logger.warning("Aucun score valide à afficher !")
        return

    n = len(valid_scores)
    dims = [
        "d1_presence", "d2_bornes_pos", "d3_paren_pos", "d4_angles",
        "d5_riverains_pos", "d6_tc_pos", "d7_linear", "d8_duplicates",
        "d9_escaped", "d10_coords", "total",
    ]
    dim_labels = {
        "d1_presence": "D1 Présence",
        "d2_bornes_pos": "D2 Bornes Pos",
        "d3_paren_pos": "D3 (Bxx) Pos",
        "d4_angles": "D4 Angles",
        "d5_riverains_pos": "D5 Riverains",
        "d6_tc_pos": "D6 TC Pos",
        "d7_linear": "D7 Linéaire",
        "d8_duplicates": "D8 Doublons",
        "d9_escaped": "D9 Hors Cadre",
        "d10_coords": "D10 Coords",
        "total": "TOTAL",
    }

    print("\n" + "=" * 80)
    print("RAPPORT DE SCORING — DAHBY PRO SUITE")
    print(f"Parcelles évaluées : {n} / {len(scores)} (erreurs : {len(scores) - n})")
    print("=" * 80)
    print(f"{'Dimension':<20} {'Moyenne':>8} {'Min':>8} {'Max':>8} {'<80%':>6}")
    print("-" * 52)

    for dim in dims:
        vals = [getattr(s, dim) for s in valid_scores]
        avg = sum(vals) / n
        mn = min(vals)
        mx = max(vals)
        below_80 = sum(1 for v in vals if v < 80)
        marker = " <<WEAK>>" if avg < 80 else ""
        print(f"{dim_labels[dim]:<20} {avg:>7.1f}% {mn:>7.1f}% {mx:>7.1f}% {below_80:>5}{marker}")

    print("=" * 80)

    # Top 5 pires parcelles
    worst = sorted(valid_scores, key=lambda s: s.total)[:5]
    print("\n[WORST] TOP 5 PIRES PARCELLES :")
    for s in worst:
        print(f"  {s.zone}/{s.parcel_id}: {s.total}% -- D1={s.d1_presence} D4={s.d4_angles} D8={s.d8_duplicates}")

    # Top 5 meilleures parcelles
    best = sorted(valid_scores, key=lambda s: s.total, reverse=True)[:5]
    print("\n[BEST] TOP 5 MEILLEURES PARCELLES :")
    for s in best:
        print(f"  {s.zone}/{s.parcel_id}: {s.total}%")

    print()


# ── Point d'entrée ────────────────────────────────────────────────────
def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Score multi-dimensionnel DAHBY PRO SUITE")
    parser.add_argument("--zone", type=str, default=None, help="Zone spécifique (IGHANOUYINE ou LEQSAQSA)")
    parser.add_argument("--parcel", type=str, default=None, help="Parcelle spécifique (ex: P216)")
    parser.add_argument("--no-pipeline", action="store_true", help="Ne pas exécuter le pipeline (utiliser OURS existants)")
    parser.add_argument("--compare-raw", action="store_true", help="Comparer aussi OREGEN brut vs DEF")
    parser.add_argument("--output", type=str, default=None, help="Chemin CSV de sortie")
    parser.add_argument("--discover-only", action="store_true", help="Lister les paires sans scorer")

    args = parser.parse_args()

    zones = [args.zone] if args.zone else None

    if args.discover_only:
        pairs = discover_parcel_pairs(zones, args.parcel)
        for p in pairs:
            print(f"  {p['zone']}/{p['parcel_id']} : DEF={p['def_path']}")
        print(f"\nTotal : {len(pairs)} paires")
        return

    scores = run_full_scoring(
        zones=zones,
        specific_parcel=args.parcel,
        run_pipeline=not args.no_pipeline,
        compare_raw=args.compare_raw,
    )

    csv_path = write_csv(scores, args.output)
    print_summary(scores)
    print(f"📊 CSV sauvegardé : {csv_path}")


if __name__ == "__main__":
    main()
