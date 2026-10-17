# -*- coding: utf-8 -*-
"""
workbench.py: Banc d'Essai Interactif & Orchestrateur Modulaire ANCFCC.
======================================================================
Permet d'exécuter chaque outil atomique séparément OU tous les outils à la suite ("fe khatra w7da") :
- Étape 1 : Bornes & Rappels (Tool 1)
- Étape 2 : Riverains & Voisinage Sectoriel (Tool 2)
- Étape 3 : Détails Linéaires & TC (Tool 3)
- Étape 4 : Cartouche, ST25 & Standards ANCFCC (Tool 4)
- Étape 5 : Arbitre Anti-Collision Finale (Tool 5)

Génère automatiquement :
- Les fichiers DXF intermédiaires pour inspection chirurgicale.
- Le fichier DXF final nettoyé.
- L'analyse différentielle comparée avec le fichier DEF de référence.
- Le rapport textuel détaillé (Vert, Jaune, Rouge, Bleu).
- L'image PNG haute résolution du Visual Diff.
"""
import os
import sys
import argparse
import logging
from typing import Optional, Dict, Any, Tuple

# Insertion du chemin racine du projet
SUITE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SUITE_ROOT not in sys.path:
    sys.path.insert(0, SUITE_ROOT)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from engines.dxf_cleaner.tools import (
    run_tool_1_bornes,
    run_tool_2_riverains,
    run_tool_3_lineaires,
    run_tool_4_cartouche,
    run_tool_5_anti_collision
)
from engines.dxf_cleaner.diff_engine import (
    compare_dxf_files,
    print_diff_report,
    render_diff_to_png
)
from engines.dxf_cleaner.score_all import discover_parcel_pairs

logger = logging.getLogger("workbench")


def resolve_parcel_paths(parcel_id: str) -> Tuple[str, str]:
    """Trouve automatiquement les chemins OREGEN et DEF pour un identifiant de parcelle."""
    pairs = discover_parcel_pairs()
    for p in pairs:
        if p["parcel_id"].upper() == parcel_id.upper():
            return p["oregen_path"], p["def_path"]
    raise FileNotFoundError(f"Parcelle '{parcel_id}' introuvable dans le corpus OREGEN/DEF.")


def run_full_modular_chain(
    parcel_id: str,
    output_dir: str,
    scale: Optional[int] = None,
    sectors_dir: str = r"C:\Users\Admin\Downloads\SS",
    render: bool = True,
    artifact_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exécute TOUS les outils à la suite de manière modulaire ("fe khatra w7da").
    """
    oregen_path, def_path = resolve_parcel_paths(parcel_id)
    os.makedirs(output_dir, exist_ok=True)

    print("═" * 65)
    print(f" 🚀 WORKBENCH CADASTRE : EXÉCUTION COMPLÈTE SUR {parcel_id} ")
    print("═" * 65)
    print(f"📁 Source OREGEN : {oregen_path}")
    print(f"📁 Référence DEF : {def_path}")
    print("─" * 65)

    step1_dxf = os.path.join(output_dir, f"{parcel_id}_STEP1_BORNES.dxf")
    step2_dxf = os.path.join(output_dir, f"{parcel_id}_STEP2_RIVERAINS.dxf")
    step3_dxf = os.path.join(output_dir, f"{parcel_id}_STEP3_LINEAIRES.dxf")
    step4_dxf = os.path.join(output_dir, f"{parcel_id}_STEP4_CARTOUCHE.dxf")
    final_dxf = os.path.join(output_dir, f"{parcel_id}_FINAL_WORKBENCH.dxf")

    # Étape 1 : Bornes
    print("\n[1/5] 🔹 Exécution TOOL 1 : Bornes & Rappels...")
    r1 = run_tool_1_bornes(oregen_path, step1_dxf, scale=scale)
    actual_scale = r1["scale"]
    print(f"      ✅ OK (Sommets={r1['n_vertices']}, Bornes={r1['n_bornes']}) -> {os.path.basename(step1_dxf)}")

    # Étape 2 : Riverains
    print("\n[2/5] 🔹 Exécution TOOL 2 : Riverains & Voisinage Sectoriel...")
    r2 = run_tool_2_riverains(step1_dxf, step2_dxf, sectors_dir=sectors_dir, scale=actual_scale, pid=parcel_id)
    print(f"      ✅ OK (Riverains résolus={r2['n_riverains_resolved']}) -> {os.path.basename(step2_dxf)}")

    # Étape 3 : Détails Linéaires & TC
    print("\n[3/5] 🔹 Exécution TOOL 3 : Détails Linéaires & Centrage TC...")
    r3 = run_tool_3_lineaires(step2_dxf, step3_dxf, scale=actual_scale)
    print(f"      ✅ OK (Linéaires={r3['n_linear_details']}, TC={r3['n_tc_nature']}) -> {os.path.basename(step3_dxf)}")

    # Étape 4 : Cartouche & Standards
    print("\n[4/5] 🔹 Exécution TOOL 4 : Cartouche, ST25 & Gauss Contenance...")
    r4 = run_tool_4_cartouche(step3_dxf, step4_dxf, scale=actual_scale, pid=parcel_id)
    print(f"      ✅ OK (Contenance={r4['contenance']}, Coords={r4['n_coords']}) -> {os.path.basename(step4_dxf)}")

    # Étape 5 : Anti-Collision Finale
    print("\n[5/5] 🔹 Exécution TOOL 5 : Arbitre Anti-Collision SAT & MTV...")
    r5 = run_tool_5_anti_collision(step4_dxf, final_dxf, scale=actual_scale)
    print(f"      ✅ OK (Collisions résolues={r5['n_resolved_collisions']}, Total textes={r5['total_texts']}) -> {os.path.basename(final_dxf)}")

    print("\n" + "═" * 65)
    print(" 🔍 ANALYSE DIFFÉRENTIELLE FINALE CONTRE DEF (RÉFÉRENCE EXPERT)")
    print("═" * 65)

    diff_res = compare_dxf_files(def_path, final_dxf, parcel_id=parcel_id)
    print_diff_report(diff_res)

    png_path = os.path.join(output_dir, f"{parcel_id}_DIFF_FINAL.png")
    if render:
        render_diff_to_png(diff_res, png_path, title=f"Workbench Final — Parcelle {parcel_id}")
        print(f"\n📸 Image PNG sauvegardée : {png_path}")

        if artifact_dir and os.path.exists(artifact_dir):
            art_png = os.path.join(artifact_dir, f"{parcel_id}_DIFF_FINAL.png")
            import shutil
            shutil.copy2(png_path, art_png)
            print(f"🖼️ Copiée vers l'Artifact Directory : {art_png}")

    return {
        "status": "SUCCESS",
        "parcel_id": parcel_id,
        "final_dxf": final_dxf,
        "png_path": png_path,
        "diff_result": diff_res,
        "steps": {
            "step1_bornes": r1,
            "step2_riverains": r2,
            "step3_lineaires": r3,
            "step4_cartouche": r4,
            "step5_anti_collision": r5
        }
    }


def main():
    parser = argparse.ArgumentParser(description="WORKBENCH ANCFCC — Banc d'essai interactif et modulaire")
    parser.add_argument("--parcel", required=True, help="Identifiant de la parcelle (ex: P586, P14, P5)")
    parser.add_argument("--tool", default="all", help="Outil à exécuter : 1, 2, 3, 4, 5 ou 'all'")
    parser.add_argument("--outdir", default=r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE\engines\dxf_cleaner\scratch_workbench", help="Dossier de sortie")
    parser.add_argument("--scale", type=int, default=None, help="Échelle forcée (ex: 1000, 2000)")
    parser.add_argument("--render", action="store_true", default=True, help="Générer l'image PNG")
    parser.add_argument("--artifact-dir", default=r"C:\Users\Admin\.gemini\antigravity\brain\082ae9b6-6db5-4b26-8ba6-7c6533a53d90", help="Dossier des artifacts")

    args = parser.parse_args()

    if args.tool.lower() in ("all", "0"):
        run_full_modular_chain(
            args.parcel,
            args.outdir,
            scale=args.scale,
            render=args.render,
            artifact_dir=args.artifact_dir
        )
    else:
        tool_idx = int(args.tool)
        oregen_path, def_path = resolve_parcel_paths(args.parcel)
        os.makedirs(args.outdir, exist_ok=True)
        out_dxf = os.path.join(args.outdir, f"{args.parcel}_TOOL{tool_idx}.dxf")

        if tool_idx == 1:
            r = run_tool_1_bornes(oregen_path, out_dxf, scale=args.scale)
            cat = "borne"
        elif tool_idx == 2:
            r = run_tool_2_riverains(oregen_path, out_dxf, scale=args.scale, pid=args.parcel)
            cat = "riverain"
        elif tool_idx == 3:
            r = run_tool_3_lineaires(oregen_path, out_dxf, scale=args.scale)
            cat = "linear"
        elif tool_idx == 4:
            r = run_tool_4_cartouche(oregen_path, out_dxf, scale=args.scale, pid=args.parcel)
            cat = "cartouche"
        elif tool_idx == 5:
            r = run_tool_5_anti_collision(oregen_path, out_dxf, scale=args.scale)
            cat = None
        else:
            print(f"Numéro d'outil invalide : {tool_idx}")
            return

        print(f"✅ Tool {tool_idx} terminé : {out_dxf}")
        diff_res = compare_dxf_files(def_path, out_dxf, parcel_id=args.parcel, category_filter=cat)
        print_diff_report(diff_res)

        if args.render:
            png_path = os.path.join(args.outdir, f"{args.parcel}_TOOL{tool_idx}_DIFF.png")
            render_diff_to_png(diff_res, png_path, title=f"Tool {tool_idx} — Parcelle {args.parcel}")
            print(f"📸 Image PNG : {png_path}")


if __name__ == "__main__":
    main()
