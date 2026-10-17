# -*- coding: utf-8 -*-
r"""
mine_rules.py — Extraction Automatique de Règles depuis les Paires DEF/OREGEN
==============================================================================
Analyse toutes les paires (OREGEN → DEF) et en extrait des règles
déterministes décrivant les transformations que l'expert applique.

Usage:
    python engines/dxf_cleaner/mine_rules.py
    python engines/dxf_cleaner/mine_rules.py --zone LEQSAQSA
    python engines/dxf_cleaner/mine_rules.py --parcel P216 --verbose

Sortie : mined_rules.json + rapport console.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

import ezdxf
import ezdxf.recover

# ── Logging structuré ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mine_rules")

# ── Constantes ─────────────────────────────────────────────────────────
BASE_DIR = Path(r"C:\Users\Admin\Downloads\Sous_Zone 01")
ZONES = ["IGHANOUYINE", "LEQSAQSA"]
OUTPUT_JSON = Path(r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE\engines\dxf_cleaner\mined_rules.json")

# Regex patterns pour classification des textes
RE_BORNE_LABEL = re.compile(r"^B\d+$", re.IGNORECASE)
RE_PAREN_BORNE = re.compile(r"^\(B\d+\)$", re.IGNORECASE)
RE_RIVERAIN_RT = re.compile(r"^[RT]\d+", re.IGNORECASE)
RE_VERBORNE = re.compile(r"^v\.?\s*B\d+", re.IGNORECASE)
RE_LINEAR_DETAIL = re.compile(
    r"(piste|chemin|route|oued|chaaba|cha[âa]bat|depression|L\.?E\.?B\.?T|sentier|canal)",
    re.IGNORECASE,
)
RE_COORD_NUMBER = re.compile(r"^\d{5,6}(\.\d+)?$")
RE_TC_NATURE = re.compile(
    r"(T\.?C\.?|terrain|culture|argan|amand|oliv|vigne|figuier|cactus|jachère|labour|irrig)",
    re.IGNORECASE,
)


# ── DXF Loading ────────────────────────────────────────────────────────
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


# ── Extraction de texte ────────────────────────────────────────────────
@dataclass
class DxfText:
    """Représente un texte extrait d'un fichier DXF."""
    text: str
    x: float
    y: float
    rotation: float
    height: float
    layer: str
    entity_type: str

    @property
    def normalized_text(self) -> str:
        """Texte nettoyé pour comparaison."""
        t = self.text.strip()
        t = re.sub(r"\s+", " ", t)
        return t

    @property
    def category(self) -> str:
        """Catégorise le texte."""
        nt = self.normalized_text
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
            raw_text = ent.text
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


# ── Matching de textes ─────────────────────────────────────────────────
def _euclidean(t1: DxfText, t2: DxfText) -> float:
    """Distance euclidienne entre deux textes."""
    return math.sqrt((t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2)


def _angle_diff(a1: float, a2: float) -> float:
    """Différence angulaire minimale en degrés [0, 180]."""
    d = abs(a1 - a2) % 360
    return min(d, 360 - d)


@dataclass
class TextTransform:
    """Transformation observée entre un texte OREGEN et son équivalent DEF."""
    text: str
    category: str
    parcel_id: str
    zone: str
    action: str  # "kept", "deleted", "added", "modified"
    ore_x: float = 0.0
    ore_y: float = 0.0
    ore_rot: float = 0.0
    def_x: float = 0.0
    def_y: float = 0.0
    def_rot: float = 0.0
    delta_x: float = 0.0
    delta_y: float = 0.0
    delta_rot: float = 0.0
    distance: float = 0.0


def analyze_pair(
    def_texts: List[DxfText],
    ore_texts: List[DxfText],
    parcel_id: str,
    zone: str,
    max_dist: float = 500.0,
) -> List[TextTransform]:
    """
    Analyse une paire (DEF, OREGEN) et enregistre toutes les transformations.

    Returns:
        Liste de TextTransform décrivant ce qui a changé.
    """
    transforms: List[TextTransform] = []

    # Grouper par contenu normalisé
    def_by_content: Dict[str, List[DxfText]] = defaultdict(list)
    for t in def_texts:
        def_by_content[t.normalized_text.lower()].append(t)

    ore_by_content: Dict[str, List[DxfText]] = defaultdict(list)
    for t in ore_texts:
        ore_by_content[t.normalized_text.lower()].append(t)

    all_keys = set(def_by_content.keys()) | set(ore_by_content.keys())

    for key in all_keys:
        d_list = def_by_content.get(key, [])
        o_list = ore_by_content.get(key, [])

        if not d_list and o_list:
            # Texte dans OREGEN mais pas dans DEF → DELETED by expert
            for ot in o_list:
                transforms.append(TextTransform(
                    text=ot.normalized_text,
                    category=ot.category,
                    parcel_id=parcel_id,
                    zone=zone,
                    action="deleted",
                    ore_x=ot.x, ore_y=ot.y, ore_rot=ot.rotation,
                ))
            continue

        if d_list and not o_list:
            # Texte dans DEF mais pas dans OREGEN → ADDED by expert
            for dt in d_list:
                transforms.append(TextTransform(
                    text=dt.normalized_text,
                    category=dt.category,
                    parcel_id=parcel_id,
                    zone=zone,
                    action="added",
                    def_x=dt.x, def_y=dt.y, def_rot=dt.rotation,
                ))
            continue

        # Les deux existent — match par proximité
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
                ot = o_list[best_idx]
                o_available.remove(best_idx)
                dx = dt.x - ot.x
                dy = dt.y - ot.y
                drot = _angle_diff(dt.rotation, ot.rotation)

                transforms.append(TextTransform(
                    text=dt.normalized_text,
                    category=dt.category,
                    parcel_id=parcel_id,
                    zone=zone,
                    action="modified" if (abs(dx) > 0.1 or abs(dy) > 0.1 or drot > 0.1) else "kept",
                    ore_x=ot.x, ore_y=ot.y, ore_rot=ot.rotation,
                    def_x=dt.x, def_y=dt.y, def_rot=dt.rotation,
                    delta_x=dx, delta_y=dy, delta_rot=drot,
                    distance=best_dist,
                ))
            else:
                # DEF text sans match OREGEN
                transforms.append(TextTransform(
                    text=dt.normalized_text,
                    category=dt.category,
                    parcel_id=parcel_id,
                    zone=zone,
                    action="added",
                    def_x=dt.x, def_y=dt.y, def_rot=dt.rotation,
                ))

        # Surplus OREGEN non matchés → DELETED or DEDUPLICATED
        for oi in o_available:
            ot = o_list[oi]
            # Si le même contenu existe dans d_list, c'est un doublon OREGEN supprimé
            if d_list:
                action = "deduplicated"
            else:
                action = "deleted"
            transforms.append(TextTransform(
                text=ot.normalized_text,
                category=ot.category,
                parcel_id=parcel_id,
                zone=zone,
                action=action,
                ore_x=ot.x, ore_y=ot.y, ore_rot=ot.rotation,
            ))

    return transforms


# ── Découverte des paires ──────────────────────────────────────────────
def discover_parcel_pairs(
    zones: Optional[List[str]] = None,
    specific_parcel: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Découvre toutes les paires (DEF, OREGEN) disponibles."""
    if zones is None:
        zones = ZONES

    pairs: List[Dict[str, str]] = []

    for zone in zones:
        def_dir = BASE_DIR / zone / "DEF"
        oregen_dir = BASE_DIR / zone / "OREGEN"

        if not def_dir.exists() or not oregen_dir.exists():
            continue

        for def_subdir in sorted(def_dir.iterdir()):
            if not def_subdir.is_dir():
                continue

            pid = def_subdir.name

            # Chercher DEF
            def_path = None
            for pattern in ["*-PLAN.dxf", "*DEFINITIVE*-PLAN.dxf", "*.dxf"]:
                for candidate in def_subdir.glob(pattern):
                    if candidate.parent == def_subdir:
                        def_path = str(candidate)
                        break
                if def_path:
                    break

            if def_path is None:
                continue

            # Chercher OREGEN (direct ou récursif)
            oregen_path = None
            oregen_subdir = oregen_dir / pid
            if oregen_subdir.exists() and oregen_subdir.is_dir():
                for candidate in oregen_subdir.glob("*-PLAN.dxf"):
                    oregen_path = str(candidate)
                    break
                if not oregen_path:
                    for candidate in oregen_subdir.glob("*.dxf"):
                        oregen_path = str(candidate)
                        break

            if oregen_path is None:
                for sub in oregen_dir.iterdir():
                    if sub.is_dir() and sub.name != pid:
                        nested = sub / pid
                        if nested.exists() and nested.is_dir():
                            for candidate in nested.glob("*-PLAN.dxf"):
                                oregen_path = str(candidate)
                                break
                            if not oregen_path:
                                for candidate in nested.glob("*.dxf"):
                                    oregen_path = str(candidate)
                                    break
                    if oregen_path:
                        break

            if oregen_path is None:
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


# ── Rule Mining Engine ─────────────────────────────────────────────────
@dataclass
class MinedRule:
    """Une règle extraite automatiquement."""
    rule_id: str
    category: str
    rule_type: str  # force_rotation, delete, deduplicate, position_preserve, angle_normalize
    description: str
    confidence: float  # 0.0 - 1.0
    evidence_count: int
    total_count: int
    impact_parcels: int  # nombre de parcelles affectées
    details: Dict[str, Any] = field(default_factory=dict)


def mine_rules_from_transforms(
    all_transforms: List[TextTransform],
    total_parcels: int,
) -> List[MinedRule]:
    """
    Extrait des règles déterministes à partir des transformations observées.

    Args:
        all_transforms: Toutes les transformations de toutes les paires.
        total_parcels: Nombre total de paires analysées.
    """
    rules: List[MinedRule] = []
    rule_counter = 0

    # Grouper par catégorie
    by_category: Dict[str, List[TextTransform]] = defaultdict(list)
    for t in all_transforms:
        by_category[t.category].append(t)

    for category, transforms in by_category.items():
        # ── R1: Force rotation = 0° ──────────────────────────────────
        # Pour chaque catégorie, vérifier si DEF a toujours rot=0°
        kept_or_modified = [t for t in transforms if t.action in ("kept", "modified")]
        if kept_or_modified:
            rot_zero_count = sum(1 for t in kept_or_modified if abs(t.def_rot) < 1.0)
            total = len(kept_or_modified)
            confidence = rot_zero_count / total if total > 0 else 0
            parcels_affected = len(set(t.parcel_id for t in kept_or_modified if abs(t.ore_rot) > 1.0 and abs(t.def_rot) < 1.0))

            if confidence >= 0.85 and total >= 5:
                rule_counter += 1
                rules.append(MinedRule(
                    rule_id=f"RULE_{rule_counter:03d}",
                    category=category,
                    rule_type="force_rotation_zero",
                    description=f"{category} texts always have rotation=0.0 in expert DEF",
                    confidence=round(confidence, 3),
                    evidence_count=rot_zero_count,
                    total_count=total,
                    impact_parcels=parcels_affected,
                    details={
                        "mean_ore_rot": round(sum(abs(t.ore_rot) for t in kept_or_modified) / total, 1),
                        "mean_def_rot": round(sum(abs(t.def_rot) for t in kept_or_modified) / total, 1),
                    },
                ))

        # ── R2: Angle normalisé [-90, +90] ──────────────────────────
        if kept_or_modified:
            norm_90_count = sum(1 for t in kept_or_modified if -91.0 <= t.def_rot <= 91.0)
            total = len(kept_or_modified)
            confidence = norm_90_count / total if total > 0 else 0
            parcels_with_bad_angle = len(set(
                t.parcel_id for t in kept_or_modified
                if not (-91.0 <= t.ore_rot <= 91.0) and -91.0 <= t.def_rot <= 91.0
            ))

            if confidence >= 0.90 and total >= 5 and parcels_with_bad_angle >= 2:
                rule_counter += 1
                rules.append(MinedRule(
                    rule_id=f"RULE_{rule_counter:03d}",
                    category=category,
                    rule_type="angle_normalize_90",
                    description=f"{category} angles always in [-90°, +90°] range in DEF",
                    confidence=round(confidence, 3),
                    evidence_count=norm_90_count,
                    total_count=total,
                    impact_parcels=parcels_with_bad_angle,
                ))

        # ── R3: Suppression (deleted) ────────────────────────────────
        deleted = [t for t in transforms if t.action == "deleted"]
        if deleted:
            # Compter les parcelles avec des suppressions
            parcels_with_deletion = len(set(t.parcel_id for t in deleted))
            total_in_category = len(transforms)

            if parcels_with_deletion >= 3 and len(deleted) >= 5:
                rule_counter += 1
                rules.append(MinedRule(
                    rule_id=f"RULE_{rule_counter:03d}",
                    category=category,
                    rule_type="delete",
                    description=f"{category} texts are deleted by expert in {parcels_with_deletion} parcels",
                    confidence=round(len(deleted) / total_in_category, 3),
                    evidence_count=len(deleted),
                    total_count=total_in_category,
                    impact_parcels=parcels_with_deletion,
                    details={
                        "sample_texts": list(set(t.text for t in deleted))[:10],
                    },
                ))

        # ── R4: Déduplication ────────────────────────────────────────
        deduplicated = [t for t in transforms if t.action == "deduplicated"]
        if deduplicated:
            parcels_with_dedup = len(set(t.parcel_id for t in deduplicated))
            n_dedup = len(deduplicated)

            if parcels_with_dedup >= 3 and n_dedup >= 5:
                rule_counter += 1
                rules.append(MinedRule(
                    rule_id=f"RULE_{rule_counter:03d}",
                    category=category,
                    rule_type="deduplicate",
                    description=f"{category} texts: expert keeps only 1 copy (deduplication)",
                    confidence=round(n_dedup / max(1, len(transforms)), 3),
                    evidence_count=n_dedup,
                    total_count=len(transforms),
                    impact_parcels=parcels_with_dedup,
                    details={
                        "avg_copies_removed_per_parcel": round(n_dedup / parcels_with_dedup, 1),
                    },
                ))

        # ── R5: Position préservée (faible déplacement) ──────────────
        if kept_or_modified:
            small_move = [t for t in kept_or_modified if t.distance < 5.0]
            large_move = [t for t in kept_or_modified if t.distance >= 20.0]

            if len(small_move) >= 5 and len(small_move) / len(kept_or_modified) >= 0.7:
                rule_counter += 1
                dists = [t.distance for t in kept_or_modified]
                rules.append(MinedRule(
                    rule_id=f"RULE_{rule_counter:03d}",
                    category=category,
                    rule_type="position_preserve",
                    description=f"{category} texts keep their original position (mean shift < 5m)",
                    confidence=round(len(small_move) / len(kept_or_modified), 3),
                    evidence_count=len(small_move),
                    total_count=len(kept_or_modified),
                    impact_parcels=len(set(t.parcel_id for t in large_move)),
                    details={
                        "mean_distance": round(sum(dists) / len(dists), 2),
                        "max_distance": round(max(dists), 2),
                        "std_distance": round(
                            (sum((d - sum(dists)/len(dists))**2 for d in dists) / len(dists)) ** 0.5, 2
                        ),
                    },
                ))

    # ── Cross-category rules ──────────────────────────────────────────

    # R6: Informal names deleted on R.../T... façades
    rt_parcels = set(t.parcel_id for t in all_transforms if t.category == "riverain_RT" and t.action in ("kept", "modified"))
    deleted_names = [t for t in all_transforms if t.category == "riverain_name" and t.action == "deleted"]
    deleted_on_rt_facade = [t for t in deleted_names if t.parcel_id in rt_parcels]
    if len(deleted_on_rt_facade) >= 3:
        rule_counter += 1
        rules.append(MinedRule(
            rule_id=f"RULE_{rule_counter:03d}",
            category="riverain_name",
            rule_type="delete_informal_on_RT_facade",
            description="Informal names deleted when façade has R.../T... title",
            confidence=round(len(deleted_on_rt_facade) / max(1, len(deleted_names)), 3),
            evidence_count=len(deleted_on_rt_facade),
            total_count=len(deleted_names),
            impact_parcels=len(set(t.parcel_id for t in deleted_on_rt_facade)),
            details={
                "sample_deleted": list(set(t.text for t in deleted_on_rt_facade))[:10],
            },
        ))

    # Sort by impact (confidence × evidence_count)
    rules.sort(key=lambda r: r.confidence * r.evidence_count, reverse=True)

    return rules


# ── Global statistics ──────────────────────────────────────────────────
def compute_statistics(
    all_transforms: List[TextTransform],
    total_parcels: int,
) -> Dict[str, Any]:
    """Calcule des statistiques globales."""
    stats: Dict[str, Any] = {
        "total_pairs_analyzed": total_parcels,
        "total_transforms": len(all_transforms),
    }

    # Par action
    action_counts = Counter(t.action for t in all_transforms)
    stats["actions"] = dict(action_counts)

    # Par catégorie
    cat_counts = Counter(t.category for t in all_transforms)
    stats["categories"] = dict(cat_counts)

    # Par catégorie × action
    cat_action: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in all_transforms:
        cat_action[t.category][t.action] += 1
    stats["category_actions"] = {k: dict(v) for k, v in cat_action.items()}

    # Rotation statistics par catégorie
    rot_stats: Dict[str, Dict[str, float]] = {}
    for cat in cat_counts:
        relevant = [t for t in all_transforms if t.category == cat and t.action in ("kept", "modified")]
        if relevant:
            ore_rots = [abs(t.ore_rot) for t in relevant]
            def_rots = [abs(t.def_rot) for t in relevant]
            rot_stats[cat] = {
                "mean_ore_rot": round(sum(ore_rots) / len(ore_rots), 2),
                "mean_def_rot": round(sum(def_rots) / len(def_rots), 2),
                "n_zero_ore": sum(1 for r in ore_rots if r < 1.0),
                "n_zero_def": sum(1 for r in def_rots if r < 1.0),
                "total": len(relevant),
            }
    stats["rotation_by_category"] = rot_stats

    return stats


# ── Main ───────────────────────────────────────────────────────────────
def run_mining(
    zones: Optional[List[str]] = None,
    specific_parcel: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[List[MinedRule], Dict[str, Any]]:
    """
    Exécute le mining de règles sur toutes les paires disponibles.

    Returns:
        (rules, statistics)
    """
    pairs = discover_parcel_pairs(zones, specific_parcel)
    if not pairs:
        logger.error("Aucune paire trouvée !")
        return [], {}

    all_transforms: List[TextTransform] = []
    errors = 0

    for i, pair in enumerate(pairs, 1):
        pid = pair["parcel_id"]
        zone = pair["zone"]

        if i % 20 == 0 or verbose:
            logger.info("[%d/%d] %s/%s", i, len(pairs), zone, pid)

        try:
            def_doc = load_dxf_safe(pair["def_path"])
            def_texts = extract_all_texts(def_doc.modelspace())

            ore_doc = load_dxf_safe(pair["oregen_path"])
            ore_texts = extract_all_texts(ore_doc.modelspace())

            transforms = analyze_pair(def_texts, ore_texts, pid, zone)
            all_transforms.extend(transforms)

            if verbose:
                logger.info(
                    "  → %d transforms (kept=%d, modified=%d, deleted=%d, added=%d, dedup=%d)",
                    len(transforms),
                    sum(1 for t in transforms if t.action == "kept"),
                    sum(1 for t in transforms if t.action == "modified"),
                    sum(1 for t in transforms if t.action == "deleted"),
                    sum(1 for t in transforms if t.action == "added"),
                    sum(1 for t in transforms if t.action == "deduplicated"),
                )
        except Exception as ex:
            logger.error("Erreur sur %s/%s : %s", zone, pid, ex)
            errors += 1

    logger.info("Analyse terminée : %d paires, %d transforms, %d erreurs",
                len(pairs), len(all_transforms), errors)

    # Mine rules
    rules = mine_rules_from_transforms(all_transforms, len(pairs))
    stats = compute_statistics(all_transforms, len(pairs))

    return rules, stats


def save_results(
    rules: List[MinedRule],
    stats: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Sauvegarde les résultats en JSON."""
    if output_path is None:
        output_path = str(OUTPUT_JSON)

    result = {
        "rules": [asdict(r) for r in rules],
        "statistics": stats,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Résultats sauvegardés : %s", output_path)
    return output_path


def print_rules_summary(rules: List[MinedRule], stats: Dict[str, Any]) -> None:
    """Affiche un résumé des règles extraites."""
    print("\n" + "=" * 80)
    print("RÈGLES EXTRAITES — MINE RULES ENGINE")
    print(f"Total : {len(rules)} règles")
    print("=" * 80)

    for rule in rules:
        impact = rule.confidence * rule.evidence_count
        print(f"\n{'─' * 70}")
        print(f"  {rule.rule_id}: {rule.description}")
        print(f"  Type    : {rule.rule_type}")
        print(f"  Conf.   : {rule.confidence * 100:.1f}% ({rule.evidence_count}/{rule.total_count})")
        print(f"  Impact  : {rule.impact_parcels} parcelles")
        if rule.details:
            for k, v in rule.details.items():
                print(f"  {k}: {v}")

    print(f"\n{'═' * 80}")
    print("STATISTIQUES GLOBALES")
    print(f"{'═' * 80}")
    if "actions" in stats:
        print(f"Actions : {stats['actions']}")
    if "rotation_by_category" in stats:
        print(f"\nRotation par catégorie :")
        for cat, rs in stats["rotation_by_category"].items():
            print(f"  {cat}: mean_ore={rs['mean_ore_rot']}°, mean_def={rs['mean_def_rot']}°, "
                  f"zero_ore={rs['n_zero_ore']}/{rs['total']}, zero_def={rs['n_zero_def']}/{rs['total']}")
    print()


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Extraction de règles DAHBY PRO SUITE")
    parser.add_argument("--zone", type=str, default=None, help="Zone spécifique")
    parser.add_argument("--parcel", type=str, default=None, help="Parcelle spécifique (ex: P216)")
    parser.add_argument("--output", type=str, default=None, help="Chemin JSON de sortie")
    parser.add_argument("--verbose", action="store_true", help="Sortie verbeuse")

    args = parser.parse_args()
    zones = [args.zone] if args.zone else None

    rules, stats = run_mining(zones, args.parcel, args.verbose)
    output_path = save_results(rules, stats, args.output)
    print_rules_summary(rules, stats)
    print(f"📊 JSON sauvegardé : {output_path}")


if __name__ == "__main__":
    main()
