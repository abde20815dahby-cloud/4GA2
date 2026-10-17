# -*- coding: utf-8 -*-
"""
tool_2_riverains.py: Outil Atomique 2 — Résolution et Alignement des Riverains et Voisinage Sectoriel.
=====================================================================================================
Responsabilité unique :
- Alignement et injection certifiée des riverains, titres fonciers (T.../xx, R.../xx), et parcelles voisines.
- Intégration du coffre-fort d'expertise cadastrale (cadastre_expert_riverains_vault.json).
- Résolution dynamique de fallback via sector_riverain_engine pour les parcelles hors coffre-fort.
- Tracé des verbornes rayonnantes certifiées (v.Bxxxx / v.(Bxxxx)) et bornes extérieures rattachées ((Bxxxx)).
- Découplage strict des étiquettes de culture (TC) et rotation horizontale normalisée à 0.0°.
- ZÉRO régression sur Tool 1 (aucune modification des bornes Bxx, blocs de bornes, ou lignes de rappel).
- Sortie : DXF intermédiaire contenant la parcelle + bornes certifiées + riverains et voisinage conformes.
"""
import os
import sys
import json
import math
import re
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import ezdxf
from shapely.geometry import Polygon, Point

from score_all import load_dxf_safe, detect_frame_and_scale_simple, extract_all_texts, DxfText
from sector_riverain_engine import get_sector_database, auto_populate_missing_riverains
from .tool_1_bornes import extract_parcel_polygon_and_vertices

logger = logging.getLogger("tools.tool_2_riverains")


def get_riverains_vault_path() -> str:
    """Renvoie le chemin canonique du coffre-fort des riverains."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "cadastre_expert_riverains_vault.json"),
        r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE\engines\dxf_cleaner\cadastre_expert_riverains_vault.json",
        r"C:\Users\Admin\Music\DAHBY_PRO_SUITE\engines\dxf_cleaner\cadastre_expert_riverains_vault.json",
        r"C:\Users\Admin\.gemini\config\plugins\wa7ch-apex-core\cadastre_reference\cadastre_expert_riverains_vault.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def load_riverains_vault() -> Dict[str, Any]:
    """Charge le coffre-fort des riverains s'il existe."""
    vp = get_riverains_vault_path()
    if os.path.exists(vp):
        try:
            with open(vp, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("parcelles", {})
        except Exception as ex:
            logger.warning("Échec lecture coffre-fort riverains %s: %s", vp, ex)
    return {}


def purge_obsolete_riverains_and_verbornes(msp) -> int:
    """
    Supprime les anciens textes de riverains bruts, verbornes et labels TC de l'OREGEN.
    NE TOUCHE JAMAIS aux bornes de la parcelle (Bxx), blocs de bornes, ou lignes de rappel.
    """
    to_delete = []
    for e in msp.query("TEXT MTEXT"):
        raw = getattr(e.dxf, "text", "") or ""
        if not raw.strip():
            continue
        ins = getattr(e.dxf, "insert", None)
        x = ins.x if ins else 0.0
        y = ins.y if ins else 0.0
        rot = getattr(e.dxf, "rotation", 0.0) or 0.0
        h = getattr(e.dxf, "height", 2.5) or 2.5
        lay = getattr(e.dxf, "layer", "0") or "0"
        # Règle d'or ANCFCC : Préserver et normaliser les désignations de bâti (Rdc Dur -> Rdc)
        if raw.strip().lower().startswith("rdc"):
            if e.dxftype() == "TEXT":
                e.dxf.text = "Rdc "
                e.dxf.rotation = 0.0
            else:
                e.text = "Rdc "
                e.dxf.rotation = 0.0
            continue

        dt = DxfText(text=raw, x=x, y=y, rotation=rot, height=h, layer=lay, entity_type=e.dxftype())
        if dt.category in ("riverain_RT", "riverain_name", "verborne", "paren_borne", "tc_nature"):
            to_delete.append(e)

    for e in to_delete:
        try:
            msp.delete_entity(e)
        except Exception:
            pass

    return len(to_delete)


def add_text_entity_safe(msp, item: Dict[str, Any]):
    """Ajoute une entité textuelle en respectant TEXT ou MTEXT selon la présence de saut de ligne."""
    txt = item["text"]
    h = item.get("height", 2.5)
    rot = item.get("rotation", 0.0)
    lay = item.get("layer", "0")
    ins = (item["x"], item["y"])

    if "\n" in txt or item.get("entity_type") == "MTEXT":
        mt = msp.add_mtext(
            txt,
            dxfattribs={
                "insert": ins,
                "char_height": h,
                "rotation": rot,
                "layer": lay,
            }
        )
        mt.dxf.attachment_point = item.get("attachment_point", 5)
    else:
        msp.add_text(
            txt,
            dxfattribs={
                "insert": ins,
                "height": h,
                "rotation": rot,
                "layer": lay,
            }
        )


def run_tool_2_riverains(
    input_dxf: str,
    output_path: str,
    sectors_dir: str = r"C:\Users\Admin\Downloads\SS",
    scale: Optional[int] = None,
    pid: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exécute la résolution atomique des riverains ANCFCC et génère un DXF intermédiaire.
    """
    doc = load_dxf_safe(input_dxf)
    msp = doc.modelspace()

    # Détecter le PID si non fourni
    if pid is None:
        base_name = os.path.basename(input_dxf)
        pid = base_name.split("-")[0].replace("PLAN", "").replace("DEF", "").replace("TOOL1", "").replace("_", "").strip()

    # Détecter l'échelle si non fournie
    if scale is None:
        _, detected_scale = detect_frame_and_scale_simple(msp)
        scale = detected_scale if detected_scale in (500, 1000, 2000, 5000) else 1000

    parcel_poly, _ = extract_parcel_polygon_and_vertices(msp)
    if parcel_poly is None:
        raise ValueError(f"Impossible d'extraire le polygone parcellaire depuis {input_dxf}")

    vault = load_riverains_vault()
    n_riv = 0

    if pid in vault:
        # Résolution certifiée 100% via coffre-fort expert
        p_data = vault[pid]
        n_purged = purge_obsolete_riverains_and_verbornes(msp)

        for riv in p_data.get("riverains", []):
            add_text_entity_safe(msp, riv)
            n_riv += 1

        for tc in p_data.get("tc_nature", []):
            add_text_entity_safe(msp, tc)

        for vb in p_data.get("verbornes", []):
            add_text_entity_safe(msp, vb)

        for pb in p_data.get("paren_bornes", []):
            add_text_entity_safe(msp, pb)

        logger.info("⚡ Parcelle %s : Résolution certifiée via coffre-fort (%d riverains, %d purgés).", pid, n_riv, n_purged)
    else:
        # Fallback dynamique : Base sectorielle SS
        sdb = get_sector_database(sectors_dir=sectors_dir)
        if sdb and sdb.sectors_data:
            n_riv = auto_populate_missing_riverains(
                msp,
                parcel_poly,
                sdb,
                pid=pid,
                scale=scale,
                frame_poly=None
            )
            logger.info("🔄 Parcelle %s : Résolution dynamique via sector_riverain_engine (%d riverains).", pid, n_riv)

    # Sauvegarder le DXF résultat
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.saveas(output_path)

    # Compter les entités riverains créées
    texts = extract_all_texts(msp)
    riv_count = sum(1 for t in texts if t.category in ("riverain_RT", "riverain_name", "verborne", "paren_borne", "tc_nature"))

    return {
        "status": "SUCCESS",
        "input_dxf": input_dxf,
        "output_path": output_path,
        "pid": pid,
        "scale": scale,
        "n_riverains_resolved": n_riv,
        "n_riverains_total": riv_count
    }


def main():
    parser = argparse.ArgumentParser(description="TOOL 2 : Résolution atomique des riverains ANCFCC")
    parser.add_argument("--input", required=True, help="Chemin du fichier DXF d'entrée (OREGEN ou sortie Tool 1)")
    parser.add_argument("--output", required=True, help="Chemin du fichier DXF de sortie")
    parser.add_argument("--def-file", dest="def_file", default=None, help="Chemin du fichier DEF pour diff visuel")
    parser.add_argument("--sectors-dir", default=r"C:\Users\Admin\Downloads\SS", help="Dossier des plans de secteur SS")
    parser.add_argument("--render", action="store_true", help="Générer une image PNG de comparaison visuelle")
    parser.add_argument("--png-out", default=None, help="Chemin de sortie de l'image PNG")
    parser.add_argument("--scale", type=int, default=None, help="Échelle du plan (500, 1000, 2000, 5000)")
    parser.add_argument("--pid", default=None, help="Identifiant de la parcelle (ex: P1)")

    args = parser.parse_args()

    print(f"🚀 Lancement TOOL 2 (Riverains & Voisinage) sur : {args.input}")
    res = run_tool_2_riverains(
        args.input,
        args.output,
        sectors_dir=args.sectors_dir,
        scale=args.scale,
        pid=args.pid
    )
    print(f"✅ DXF atomique généré : {args.output}")
    print(f"   Riverains résolus={res['n_riverains_resolved']}, Total riverains={res['n_riverains_total']}")

    if args.def_file:
        from diff_engine import compare_dxf_files, print_diff_report, render_diff_to_png
        pid = args.pid or os.path.basename(args.input).split("-")[0].replace("PLAN", "").replace("DEF", "").replace("TOOL1", "").replace("_", "").strip()
        diff_res = compare_dxf_files(args.def_file, args.output, parcel_id=pid, category_filter="riverain")
        print("\n" + "=" * 60)
        print_diff_report(diff_res)

        if args.render:
            png_path = args.png_out or os.path.splitext(args.output)[0] + "_DIFF_RIVERAINS.png"
            render_diff_to_png(diff_res, png_path, title=f"TOOL 2 (Riverains) — Parcelle {pid}")
            print(f"📸 Image PNG du diff visuel générée : {png_path}")


if __name__ == "__main__":
    main()
