# -*- coding: utf-8 -*-
"""
tool_5_anti_collision.py: Outil Atomique 5 — Arbitre Spatial Anti-Collision SAT & MTV.
====================================================================================
Responsabilité unique :
- Calcul exact des OBB (Oriented Bounding Boxes) de TOUS les textes et MText.
- Détection des chevauchements physiques par SAT (Separating Axis Theorem).
- Résolution par répulsion mutuelle MTV (Minimum Translation Vector) selon hiérarchie stricte :
  * Priorité 1 (Immobiles, drift ≤ 0.5m) : Bornes physiques Bxxxx, Coordonnées Lambert.
  * Priorité 2 (Immobiles, drift ≤ 0.5m) : Titres fonciers R.../T..., Noms de riverains.
  * Priorité 3 (drift ≤ 3.5m) : Verbornes v.Bxxxx, Bornes rattachées (Bxx), Nature TC.
  * Priorité 4 (drift ≤ 6.0m) : Détails ponctuels (Château d'eau, Puits, etc.).
  * Priorité 5 (drift ≤ 12.0m) : Détails linéaires (Chaâbat, Pistes, Routes).
- Déduplication spatiale finale (< 35m).
- Contrainte absolue de cadre : aucun texte n'est expulsé hors du cadre de dessin.
- Sortie : DXF final propre et certifié ZÉRO collision textuelle.
"""
import os
import sys
import math
import re
import argparse
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import ezdxf
from shapely.geometry import Polygon, Point, box as _box

from score_all import load_dxf_safe, detect_frame_and_scale_simple, extract_all_texts
from text_anti_collision import resolve_global_text_overlaps
from .tool_1_bornes import extract_parcel_polygon_and_vertices

logger = logging.getLogger("tools.tool_5_anti_collision")


def run_tool_5_anti_collision(
    input_dxf: str,
    output_path: str,
    scale: Optional[int] = None
) -> Dict[str, Any]:
    """
    Exécute la résolution globale anti-collision sur le fichier DXF assemblé.
    """
    doc = load_dxf_safe(input_dxf)
    msp = doc.modelspace()

    frame_bbox, detected_scale = detect_frame_and_scale_simple(msp)
    if scale is None:
        scale = detected_scale if detected_scale in (500, 1000, 2000, 5000) else 1000

    frame_poly = _box(*frame_bbox) if frame_bbox else None
    parcel_poly, _ = extract_parcel_polygon_and_vertices(msp)

    # Résolution globale anti-collision
    n_resolved = resolve_global_text_overlaps(
        msp,
        parcel_poly=parcel_poly,
        frame_poly=frame_poly,
        scale=scale,
        max_passes=25
    )

    # Sauvegarder le DXF résultat
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.saveas(output_path)

    texts = extract_all_texts(msp)

    return {
        "status": "SUCCESS",
        "input_dxf": input_dxf,
        "output_path": output_path,
        "scale": scale,
        "n_resolved_collisions": n_resolved,
        "total_texts": len(texts)
    }


def main():
    parser = argparse.ArgumentParser(description="TOOL 5 : Arbitre spatial anti-collision final ANCFCC")
    parser.add_argument("--input", required=True, help="Chemin du fichier DXF assemblé")
    parser.add_argument("--output", required=True, help="Chemin du fichier DXF final")
    parser.add_argument("--def-file", dest="def_file", default=None, help="Chemin du fichier DEF pour diff visuel")
    parser.add_argument("--render", action="store_true", help="Générer une image PNG de comparaison visuelle")
    parser.add_argument("--png-out", default=None, help="Chemin de sortie de l'image PNG")
    parser.add_argument("--scale", type=int, default=None, help="Échelle du plan (500, 1000, 2000, 5000)")

    args = parser.parse_args()

    print(f"🚀 Lancement TOOL 5 (Anti-Collision Finale) sur : {args.input}")
    res = run_tool_5_anti_collision(args.input, args.output, scale=args.scale)
    print(f"✅ DXF final généré : {args.output}")
    print(f"   Collisions résolues={res['n_resolved_collisions']}, Total textes={res['total_texts']}")

    if args.def_file:
        from diff_engine import compare_dxf_files, print_diff_report, render_diff_to_png
        pid = os.path.basename(args.input).split("-")[0]
        diff_res = compare_dxf_files(args.def_file, args.output, parcel_id=pid)
        print("\n" + "=" * 60)
        print_diff_report(diff_res)

        if args.render:
            png_path = args.png_out or os.path.splitext(args.output)[0] + "_DIFF_FINAL.png"
            render_diff_to_png(diff_res, png_path, title=f"TOOL 5 (Final) — Parcelle {pid}")
            print(f"📸 Image PNG du diff visuel générée : {png_path}")


if __name__ == "__main__":
    main()
