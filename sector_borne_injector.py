# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/sector_borne_injector.py — Injection des bornes manquantes depuis les plans de secteur
======================================================================================================
Permet d'injecter dans un plan parcellaire individuel (ex: OREGEN) les étiquettes et blocs de bornes (Bxx)
qui figurent dans les plans d'ensemble / plans de secteur (SS) mais sont absentes de l'extrait individuel.

Caractéristiques techniques :
  - Extraction tolérante des fichiers DXF de secteur corrompus via ezdxf.recover.readfile()
  - Réparation préventive des codes de groupe malformés ('0  0' et octets binaires parasites)
  - Indexation spatiale avec mise en cache binaire (.sector_borne_cache.pkl)
  - Filtrage géométrique par bounding box parcellaire + marge (défaut : 50m)
  - Déduplication stricte par rapport aux étiquettes déjà présentes dans le modèle
  - Injection conforme aux standards ANCFCC (calque 'bornes', ratio de hauteur de texte cadastral)
  - Injection des blocs de bornes (INSERT) associés à proximité immédiate
"""

import io
import logging
import math
import os
import pickle
import re
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Any, NamedTuple

import ezdxf
from ezdxf.recover import readfile as recover_read
from shapely.geometry import Polygon, Point, box

from constants import ANCFCC_STANDARDS

logger = logging.getLogger("dxf_cleaner.sector_borne_injector")

SECTOR_CACHE_FILENAME = ".sector_borne_cache.pkl"
DEFAULT_SECTORS_DIR = r"C:\Users\Admin\Downloads\SS"
B_LABEL_REGEX = re.compile(r"^B\d+$", re.IGNORECASE)
BORNE_BLOCK_NAMES = {"borne_plantee", "borne_marquee", "borne_marquee_roche"}


class SectorBorne(NamedTuple):
    """
    Entrée représentant une borne extraite d'un plan de secteur.
    Comporte les coordonnées du texte, le fichier source, et les coordonnées
    éventuelles du bloc de borne associé.
    """
    x: float
    y: float
    sector_file: str
    block_name: Optional[str] = None
    block_x: Optional[float] = None
    block_y: Optional[float] = None


# Alias rétrocompatible
SectorBorneEntry = SectorBorne


def fix_malformed_group_codes(raw: bytes) -> bytes:
    """Répare les corruptions de codes de groupe DXF (ex: '0  0' ou octets binaires parasites)."""
    fixed = re.sub(rb"0[ \t]+0\r?\n", b"0\r\n", raw)
    pos_corrupt = -1
    for i in range(len(fixed) - 10):
        b = fixed[i]
        if b in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            pos_corrupt = i
            break
    if pos_corrupt != -1:
        end_sec = fixed.find(b"ENDSEC", pos_corrupt)
        if end_sec != -1:
            fixed = fixed[:pos_corrupt] + b"\r\n  0\r\n" + fixed[end_sec:]
    return fixed


def _read_sector_dxf_safe(filepath: str):
    """Charge un fichier DXF de secteur avec assainissement binaire et mode recover."""
    with open(filepath, "rb") as fh:
        raw = fh.read()
    raw = fix_malformed_group_codes(raw)
    with tempfile.NamedTemporaryFile(suffix=".dxf", delete=False) as tf:
        tf.write(raw)
        tmp_path = tf.name
    try:
        try:
            result = recover_read(tmp_path)
            doc = result[0] if isinstance(result, tuple) else result
        except Exception:
            doc = ezdxf.readfile(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return doc


def build_sector_borne_index(
    sectors_dir: str = DEFAULT_SECTORS_DIR,
    cache_path: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, List[SectorBorne]]:
    """
    Scanne tous les fichiers DXF de secteur et extrait toutes les bornes Bxx avec leurs coordonnées.
    Gère la mise en cache sur disque (.sector_borne_cache.pkl).
    
    Retourne :
        dict {label_text: [SectorBorne(x, y, sector_file, ...), ...]}
    """
    if not os.path.isdir(sectors_dir):
        logger.warning("Répertoire des secteurs introuvable : %s", sectors_dir)
        return {}

    if cache_path is None:
        cache_path = os.path.join(sectors_dir, SECTOR_CACHE_FILENAME)

    if not force_refresh and os.path.isfile(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cached_data = pickle.load(fh)
            if isinstance(cached_data, dict) and cached_data:
                logger.info("Index des bornes de secteur chargé depuis le cache (%d entrées) : %s", len(cached_data), cache_path)
                return cached_data
        except Exception as exc:
            logger.warning("Échec de lecture du cache des bornes (%s), reconstruction de l'index : %s", cache_path, exc)

    logger.info("Construction de l'index des bornes de secteur depuis : %s", sectors_dir)
    index: Dict[str, List[SectorBorne]] = {}
    sector_files = sorted([f for f in os.listdir(sectors_dir) if f.upper().endswith(".DXF")])

    for sf in sector_files:
        fp = os.path.join(sectors_dir, sf)
        logger.info("Indexation du fichier secteur : %s", sf)
        try:
            doc = _read_sector_dxf_safe(fp)
            msp = doc.modelspace()
        except Exception as exc:
            logger.error("Erreur lors de la lecture du fichier secteur %s : %s", sf, exc)
            continue

        # 1. Extraction des blocs de bornes (INSERT)
        borne_inserts: List[Tuple[float, float, str]] = []
        for ent in msp:
            if ent.dxftype() == "INSERT":
                bname = getattr(ent.dxf, "name", "").strip().lower()
                if "borne" in bname or "brn" in bname:
                    ip = getattr(ent.dxf, "insert", None)
                    if ip is not None:
                        real_name = getattr(ent.dxf, "name", "").strip()
                        borne_inserts.append((round(float(ip.x), 3), round(float(ip.y), 3), real_name))

        # 2. Extraction des étiquettes TEXT et MTEXT de bornes Bxx
        count_file = 0
        for ent in msp:
            dt = ent.dxftype()
            if dt in ("TEXT", "MTEXT"):
                try:
                    ip = getattr(ent.dxf, "insert", None)
                    if ip is None and dt == "TEXT":
                        ip = getattr(ent.dxf, "align_point", None)
                    if ip is None:
                        continue
                    raw_text = ent.dxf.text if dt == "TEXT" else ent.text
                    clean_text = re.sub(r"\s+", "", str(raw_text).strip().upper())
                    if B_LABEL_REGEX.match(clean_text):
                        tx = round(float(ip.x), 3)
                        ty = round(float(ip.y), 3)

                        # Recherche du bloc de borne le plus proche dans un rayon de 10m
                        best_block_name = None
                        best_bx, best_by = None, None
                        min_dist_sq = 100.0  # rayon max de 10m
                        for bx_c, by_c, b_name in borne_inserts:
                            d2 = (tx - bx_c) ** 2 + (ty - by_c) ** 2
                            if d2 < min_dist_sq:
                                min_dist_sq = d2
                                best_block_name = b_name
                                best_bx, best_by = bx_c, by_c

                        entry = SectorBorne(
                            x=tx,
                            y=ty,
                            sector_file=sf,
                            block_name=best_block_name,
                            block_x=best_bx,
                            block_y=best_by,
                        )
                        if clean_text not in index:
                            index[clean_text] = []
                        index[clean_text].append(entry)
                        count_file += 1
                except Exception:
                    continue

        logger.info("  %s : %d étiquettes Bxx répertoriées", sf, count_file)

    # Sauvegarde dans le cache
    try:
        with open(cache_path, "wb") as fh:
            pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cache des bornes de secteur sauvegardé avec succès (%d clés uniques) : %s", len(index), cache_path)
    except Exception as exc:
        logger.warning("Impossible d'enregistrer le cache des bornes dans %s : %s", cache_path, exc)

    return index


_GLOBAL_SECTOR_INDEX: Optional[Dict[str, List[SectorBorne]]] = None


def get_sector_borne_index(sectors_dir: str = DEFAULT_SECTORS_DIR) -> Dict[str, List[SectorBorne]]:
    """Retourne l'index en mémoire globale (singleton) pour éviter les relectures multiples."""
    global _GLOBAL_SECTOR_INDEX
    if _GLOBAL_SECTOR_INDEX is None:
        _GLOBAL_SECTOR_INDEX = build_sector_borne_index(sectors_dir=sectors_dir)
    return _GLOBAL_SECTOR_INDEX


def extract_existing_borne_labels(msp) -> Set[str]:
    """Extrait l'ensemble de tous les labels de bornes existants dans l'espace objet."""
    existing = set()
    for ent in msp:
        if ent.dxftype() in ("TEXT", "MTEXT"):
            try:
                raw_text = ent.dxf.text if ent.dxftype() == "TEXT" else ent.text
                clean_text = re.sub(r"\s+", "", str(raw_text).strip().upper())
                m = B_LABEL_REGEX.match(clean_text)
                if m or re.match(r"^\(B\d+\)$", clean_text):
                    existing.add(clean_text)
            except Exception:
                continue
    return existing


def extract_existing_borne_positions(msp) -> List[Tuple[float, float]]:
    """Extrait les coordonnées de toutes les étiquettes de bornes existantes."""
    positions = []
    for ent in msp:
        if ent.dxftype() in ("TEXT", "MTEXT"):
            try:
                raw_text = ent.dxf.text if ent.dxftype() == "TEXT" else ent.text
                clean_text = re.sub(r"\s+", "", str(raw_text).strip().upper())
                if B_LABEL_REGEX.match(clean_text) or re.match(r"^\(B\d+\)$", clean_text):
                    ip = getattr(ent.dxf, "insert", None)
                    if ip is None and ent.dxftype() == "TEXT":
                        ip = getattr(ent.dxf, "align_point", None)
                    if ip is not None:
                        positions.append((float(ip.x), float(ip.y)))
            except Exception:
                continue
    return positions


def inject_missing_bornes(
    msp,
    parcel_poly: Optional[Polygon],
    sector_index: Optional[Dict[str, List[SectorBorne]]] = None,
    scale: int = 2000,
    existing_labels: Optional[Set[str]] = None,
    margin_m: float = 20.0,
    max_boundary_dist: float = 2.0,
) -> int:
    """
    Injecte les étiquettes et blocs de bornes manquants dans l'espace objet (msp).
    CHANTIER 1 : Seules les bornes strictement sur la limite parcellaire (<= max_boundary_dist)
    sont injectées, éliminant toute pollution de bornes parasites dans les parcelles voisines.
    
    Paramètres :
        msp : Modelspace ezdxf cible
        parcel_poly : Polygone Shapely de l'emprise parcellaire
        sector_index : Index des bornes de secteur (chargé automatiquement si None)
        scale : Échelle du plan (défaut : 2000)
        existing_labels : Ensemble des labels déjà présents (extrait de msp si None)
        margin_m : Marge préliminaire de recherche (défaut : 20.0m)
        max_boundary_dist : Distance max stricte au contour de la parcelle (défaut : 2.0m)
        
    Retourne :
        Nombre de bornes injectées
    """
    if parcel_poly is None or parcel_poly.is_empty:
        logger.warning("Injection de bornes impossible : polygone de parcelle absent ou vide.")
        return 0

    if sector_index is None:
        sector_index = get_sector_borne_index()

    if not sector_index:
        logger.warning("Index des bornes de secteur vide ou non disponible.")
        return 0

    if existing_labels is None:
        existing_labels = extract_existing_borne_labels(msp)

    existing_positions = extract_existing_borne_positions(msp)

    # Délimitation de la boîte englobante avec marge
    minx, miny, maxx, maxy = parcel_poly.bounds
    search_bbox = box(minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)

    # Calcul de la hauteur de texte selon les standards ANCFCC
    height_ratio = float(ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018))
    text_height = max(1.5, height_ratio * scale)

    # S'assurer que le calque 'bornes' existe dans le document
    doc = msp.doc
    if "bornes" not in doc.layers:
        try:
            doc.layers.add("bornes", color=3)  # Vert ANCFCC standard
        except Exception:
            pass

    # Vérifier les blocs existants dans le document pour duplication/insertion sécurisée
    doc_blocks = set(b.name for b in doc.blocks)

    # Coordonnées des blocs INSERT de bornes existants
    existing_insert_positions = []
    for ent in msp.query("INSERT"):
        bname = getattr(ent.dxf, "name", "").lower()
        if "borne" in bname or "brn" in bname:
            ip = getattr(ent.dxf, "insert", None)
            if ip is not None:
                existing_insert_positions.append((float(ip.x), float(ip.y)))

    injected_count = 0
    injected_labels: Set[str] = set()

    existing_nums = [
        int(m.group())
        for l in existing_labels
        for m in [re.search(r"\d+", l)]
        if m
    ]
    min_num = min(existing_nums) if existing_nums else 0
    max_num = max(existing_nums) if existing_nums else 999999

    for label_name, entries in sector_index.items():
        if label_name in existing_labels or label_name in injected_labels:
            continue

        cand_match = re.search(r"\d+", label_name)
        if cand_match and existing_nums:
            cand_num = int(cand_match.group())
            # Filtrer les bornes hors-série étrangères (ex: B13071 dans une parcelle 1..50)
            is_compatible = (
                any(abs(cand_num - en) <= 5 for en in existing_nums)
                or (min_num - 10 <= cand_num <= max_num + 10)
            )
            if not is_compatible:
                continue

        # Trouver une entrée STRICTEMENT sur la limite parcellaire
        valid_entry: Optional[SectorBorne] = None
        min_dist_to_parcel = float("inf")

        for entry in entries:
            pt_txt = Point(entry.x, entry.y)
            pt_blk = (
                Point(entry.block_x, entry.block_y)
                if (entry.block_x is not None and entry.block_y is not None)
                else None
            )

            d_blk = parcel_poly.exterior.distance(pt_blk) if pt_blk is not None else 999.0
            d_txt = parcel_poly.exterior.distance(pt_txt)
            d = min(d_blk, d_txt)

            # CHANTIER 1 : Condition stricte d'appartenance au contour
            if d <= max_boundary_dist and d < min_dist_to_parcel:
                # Vérifier qu'aucune étiquette existante n'est déjà présente à moins de 3.0m
                is_near_existing = any(
                    math.hypot(entry.x - ex_x, entry.y - ex_y) < 3.0
                    for ex_x, ex_y in existing_positions
                )
                if not is_near_existing:
                    min_dist_to_parcel = d
                    valid_entry = entry

        if valid_entry is not None:
            # 1. Insertion de l'étiquette TEXT
            try:
                msp.add_text(
                    label_name,
                    dxfattribs={
                        "layer": "bornes",
                        "height": text_height,
                        "insert": (valid_entry.x, valid_entry.y, 0.0),
                    },
                )
                existing_labels.add(label_name)
                injected_labels.add(label_name)
                existing_positions.append((valid_entry.x, valid_entry.y))
                injected_count += 1
                logger.debug(
                    "Borne injectée : %s à (%.2f, %.2f) depuis secteur %s (dist=%.1fm)",
                    label_name,
                    valid_entry.x,
                    valid_entry.y,
                    valid_entry.sector_file,
                    min_dist_to_parcel,
                )
            except Exception as exc:
                logger.error("Erreur lors de l'insertion de l'étiquette %s : %s", label_name, exc)
                continue

            # 2. Insertion du bloc de borne (INSERT) si présent et non doublon
            if valid_entry.block_name and valid_entry.block_x is not None and valid_entry.block_y is not None:
                bname = valid_entry.block_name
                bx, by = valid_entry.block_x, valid_entry.block_y
                is_near_insert = any(
                    math.hypot(bx - ex_x, by - ex_y) < 1.0
                    for ex_x, ex_y in existing_insert_positions
                )
                if not is_near_insert:
                    # Si le bloc est défini dans le document
                    if bname in doc_blocks:
                        try:
                            msp.add_blockref(
                                bname,
                                (bx, by, 0.0),
                                dxfattribs={"layer": "bornes"},
                            )
                            existing_insert_positions.append((bx, by))
                        except Exception as exc:
                            logger.debug("Bloc %s non inséré : %s", bname, exc)
                    elif "borne_plantee" in doc_blocks:
                        # Fallback sur borne_plantee
                        try:
                            msp.add_blockref(
                                "borne_plantee",
                                (bx, by, 0.0),
                                dxfattribs={"layer": "bornes"},
                            )
                            existing_insert_positions.append((bx, by))
                        except Exception:
                            pass

    logger.info("Total de bornes injectées depuis le secteur : %d", injected_count)
    return injected_count
