# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/constants.py — Constantes et expressions régulières cadastrales ANCFCC
========================================================================================
Centralise les constantes de traitement parcellaire, les calques normalisés,
les expressions régulières de reconnaissance topographique et les paramètres par défaut.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, Set

DEFAULT_MEMORY: Dict[str, Any] = {
    "acadver": "AC1018",
    "codepage": "ANSI_1256",
    "aunits": 2,
    "auprec": 4,
    "lunits": 2,
    "luprec": 4,
    "angbase": 90.0,
    "angdir": 1,
    "ltscale": 2.0,
    "plinegen": 1,
    "measurement": 1,
    "base_scale_default": 2000,
    "frame_paper_width_m": 0.16,
    "frame_paper_height_m": 0.26,
    "grid_step_ratio": 0.10,
    "min_grid_to_frame_ratio": 0.0075,
    "min_grid_to_borne_ratio": 0.00125,
    "borne_text_height_ratio": 0.0018,
    "isolated_borne_offset_mult": 2.68,
    "cluster_distance_threshold_mult": 2.25,
    "min_borne_label_gap_mult": 0.35,
    "riverain_lateral_offset_ratio": 0.0082,
    "contenance_bottom_offset_ratio": 0.0200,
}


def load_brain_memory() -> Dict[str, Any]:
    mem = dict(DEFAULT_MEMORY)
    candidates = [
        Path(__file__).resolve().parents[2] / "02_CADASTRE_MASTER_REFERENCE" / "CADASTRE_LEARNED_RULES.json",
        Path("02_CADASTRE_MASTER_REFERENCE/CADASTRE_LEARNED_RULES.json"),
        Path(r"C:\Users\Admin\Desktop\ARGON_4\02_CADASTRE_MASTER_REFERENCE\CADASTRE_LEARNED_RULES.json"),
        Path(r"C:\Users\Admin\Desktop\app dahby\CADASTRE_MASTER_REFERENCE\CADASTRE_LEARNED_RULES.json"),
        Path(r"C:\Users\Admin\Music\DAHBY_PRO_SUITE\02_CADASTRE_MASTER_REFERENCE\CADASTRE_LEARNED_RULES.json"),
        Path(r"C:\Users\Admin\.gemini\config\plugins\wa7ch-apex-core\cadastre_reference\CADASTRE_LEARNED_RULES.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                mem["raw_json"] = raw
                # Sections 1 à 5 (Format consolidé)
                for sec_key in (
                    "1_qawanin_al_7adf_wa_tarbi3",
                    "2_qawanin_al_bornes_wa_taza7om",
                    "3_qawanin_at_tafasil_al_khattiya",
                    "4_qawanin_al_mojawirin_wa_markaz_tc",
                    "5_qawanin_al_idafat_wa_ancfcc",
                ):
                    sec = raw.get(sec_key, {})
                    mem.update(sec)
                # Format legacy rétro-compatibilité
                mem.update(raw.get("header_standards", {}))
                mem.update(raw.get("agent_eraser_grid", {}))
                mem.update(raw.get("agent_borne_mover", {}))
                mem.update(raw.get("agent_detail_mover", {}))
                mem.update(raw.get("agent_riverain_tc", {}))
                # Clés canoniques garanties
                if "grid_croisillons_step_ratio_S" in mem:
                    mem["grid_step_ratio"] = mem["grid_croisillons_step_ratio_S"]
                if "min_grid_to_frame_ratio_S" in mem:
                    mem["min_grid_to_frame_ratio"] = mem["min_grid_to_frame_ratio_S"]
                if "min_grid_to_borne_ratio_S" in mem:
                    mem["min_grid_to_borne_ratio"] = mem["min_grid_to_borne_ratio_S"]
                if "borne_text_height_ratio_S" in mem:
                    mem["borne_text_height_ratio"] = mem["borne_text_height_ratio_S"]
                if "isolated_borne_offset_mult_H" in mem:
                    mem["isolated_borne_offset_mult"] = mem["isolated_borne_offset_mult_H"]
                if "cluster_distance_threshold_mult_H" in mem:
                    mem["cluster_distance_threshold_mult"] = mem["cluster_distance_threshold_mult_H"]
                if "min_borne_label_gap_mult_H" in mem:
                    mem["min_borne_label_gap_mult"] = mem["min_borne_label_gap_mult_H"]
                if "riverain_lateral_offset_ratio_S" in mem:
                    mem["riverain_lateral_offset_ratio"] = mem["riverain_lateral_offset_ratio_S"]
                if "contenance_bottom_offset_ratio_S" in mem:
                    mem["contenance_bottom_offset_ratio"] = mem["contenance_bottom_offset_ratio_S"]
                break
            except Exception:
                pass
    return mem


ANCFCC_STANDARDS: Dict[str, Any] = load_brain_memory()

# Séquence officielle des 17 étapes de traitement cadastral (IFE Tools v4.0.7 & DAHBY PRO)
ETAPES_PARCELLAIRE: Tuple[str, ...] = (
    "Chargement du DXF",
    "Suppression entités quasi-nulles",
    "Détection géométrie de parcelle",
    "Détection cadre du plan",
    "Suppression textes en doublon",
    "Suppression textes d'annotation",
    "Déplacement noms de bornes",
    "Contrainte stricte des labels Bxx",
    "Positionnement textes verborne",
    "Réinitialisation rotation textes internes",
    "Suppression symboles croix",
    "Suppression des riverains",
    "Découpage au cadre",
    "Centrage textes sur lignes",
    "Gestion des chevauchements",
    "Forçage retour à la ligne",
    "Enregistrement",
)

# Calques de lignes doubles linéaires (voies, cours d'eau, limites mitoyennes)
DBL_LAYERS: Set[str] = {
    "SAGUIA", "SENTIER", "piste", "route", "MASREF", "limit", "CHAABA",
    "LE_BT", "masref", "seguia", "PISTE", "lemt", "saguia", "le_bt",
    "LEHT", "chaaba", "LIMITE", "SEGUIA", "LE_BT", "limite", "LE",
    "LIMIT", "leht", "sentier", "ROUTE"
}

# Regex pour identifier les textes associés aux doubles lignes
DBL_TEXT_RE: re.Pattern = re.compile(
    r"^(oued|seguia|saguia|sentier|piste|route|chaaba|masref|canal|berge|berm|haie|talus|lkhzaz|ounza|saguia\s+b[eé]ton|route\s+goudronn)",
    re.IGNORECASE
)

# Calques d'éléments linéaires pour calcul de dérive et accrochage
LINEAR_LAYERS: Set[str] = {
    "MASREF", "LEHT", "LE", "limite", "le_bt", "leht", "LEMT", "LIMIT",
    "ROUTE", "SENTIER", "route", "LIMITE", "lemt", "PISTE", "chaaba",
    "piste", "seguia", "sentier", "CHAABA", "limit", "masref"
}

# Regex pour identifier les désignations linéaires le long des limites
LINEAR_RE: re.Pattern = re.compile(
    r"^(oued|seguia|saguia|sentier|piste|route|chaaba|chaaба|masref|limit|saguia|canal|berge|berm|haie|talus)\b",
    re.IGNORECASE
)

# Regex pour identifier les textes de mise en page / cartouche hors emprise parcellaire
LAYOUT_RE: re.Pattern = re.compile(
    r"^(Propri|Situ[eé]|Titre|R[eé]quisition|Direction|ST\b|Echel|Plan|Limit|NORD|الشمال|Plle\s*[nN]|IFE|Mappe\s+de|Carte\s+topo|G[eé]om[eè]tre|Dessinateur|Contenance|superficie|Surface|Plan\s+[eé]tabli|Plan\s+Provisoire)",
    re.IGNORECASE
)

# Paramètres par défaut de l'algorithme de nettoyage
DEFAULT_OPTIONS: Dict[str, Any] = {
    "del_near_zero": True,
    "del_duplicates": True,
    "del_annotation": True,
    "move_bornes": True,
    "fix_bornes_strict": True,
    "allow_scale_down": True,
    "pos_verborne": True,
    "reset_rotation": True,
    "del_crosses": True,
    "clean_dtls": True,
    "clip_frame": True,
    "centre_texts": True,
    "add_lv_chemin": False,
    "resolve_overlaps": True,
    "force_wrap": True,
    "wrap_char_count": 15,
    "verborne_spacing": 5,
    "center_texts": [],
    "populate_trees": True,
    "use_sector_riverains": True,
    "sectors_dir": r"C:\Users\Admin\Downloads\SS",
    "sz_reference_path": r"C:\Users\Admin\Downloads\SS",
    "selected_sz": "AUTO",
    "sz_force_refresh": False,
    "use_rust": True,
    "reference_path": None,
    "use_reference": True,
}
