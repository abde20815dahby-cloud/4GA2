# -*- coding: utf-8 -*-
"""
batch_scan_tool1.py: Scanner Parallèle Massif Tool 1 (Bornes & Rappels Master)
==============================================================================
Exécute Tool 1 sur l'intégralité du corpus (137 parcelles) en parallèle multi-processus :
- Génère les images PNG haute résolution du Visual Diff pour chaque parcelle.
- Ne sauvegarde aucun DXF lourd sur le disque (mode éphémère / temporaire).
- Calcule les métriques globales et par parcelle (Vert, Jaune, Rouge, Bleu).
- Exporte un CSV détaillé et un rapport Markdown consolidé.
"""
import os
import sys
import time
import math
import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

# Assurer l'encodage UTF-8 sous Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SUITE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SUITE_ROOT not in sys.path:
    sys.path.insert(0, SUITE_ROOT)

import matplotlib
matplotlib.use("Agg")

from engines.dxf_cleaner.score_all import discover_parcel_pairs
from engines.dxf_cleaner.tools.tool_1_bornes import run_tool_1_bornes
from engines.dxf_cleaner.diff_engine import compare_dxf_files, render_diff_to_png

logger = logging.getLogger("batch_scan_tool1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

OUTPUT_PNG_DIR = os.path.join(os.path.dirname(__file__), "visual_diff_tool1_pngs")
TEMP_DXF_DIR = os.path.join(os.path.dirname(__file__), "temp_batch_tool1")


def worker_process_parcel(pair: Dict[str, str], png_dir: str, temp_dir: str) -> Dict[str, Any]:
    """Traite une parcelle de manière isolée pour exécution parallèle."""
    pid = pair["parcel_id"]
    oregen_path = pair["oregen_path"]
    def_path = pair["def_path"]

    tmp_dxf = os.path.join(temp_dir, f"tmp_{pid}_tool1.dxf")
    png_path = os.path.join(png_dir, f"{pid}_DIFF_BORNES.png")

    result = {
        "parcel_id": pid,
        "status": "ERROR",
        "scale": 1000,
        "total_def": 0,
        "total_ours": 0,
        "matches": 0,
        "shifted": 0,
        "missing": 0,
        "parasites": 0,
        "pct_green": 0.0,
        "png_path": png_path,
        "error_msg": ""
    }

    try:
        # 1. Exécution de Tool 1 dans un fichier temporaire
        t1_res = run_tool_1_bornes(oregen_path, tmp_dxf)
        result["scale"] = t1_res.get("scale", 1000)

        # 2. Comparaison différentielle stricte sur la catégorie 'borne'
        diff_res = compare_dxf_files(def_path, tmp_dxf, parcel_id=pid, category_filter="borne")

        # 3. Rendu PNG haute résolution
        render_diff_to_png(diff_res, png_path, title=f"TOOL 1 (Bornes) — Parcelle {pid}")

        # 4. Extraction des métriques
        n_def = diff_res.total_def
        n_ours = diff_res.total_ours
        n_matches = len(diff_res.matches)
        n_shifted = len(diff_res.shifted)
        n_missing = len(diff_res.missing)
        n_parasites = len(diff_res.parasites)

        pct = (n_matches / n_def * 100.0) if n_def > 0 else 100.0

        result["status"] = "SUCCESS"
        result["total_def"] = n_def
        result["total_ours"] = n_ours
        result["matches"] = n_matches
        result["shifted"] = n_shifted
        result["missing"] = n_missing
        result["parasites"] = n_parasites
        result["pct_green"] = round(pct, 2)

    except Exception as exc:
        result["error_msg"] = str(exc)

    finally:
        # Nettoyage immédiat du DXF temporaire (aucune empreinte DXF)
        if os.path.exists(tmp_dxf):
            try:
                os.remove(tmp_dxf)
            except Exception:
                pass

    return result


def main():
    os.makedirs(OUTPUT_PNG_DIR, exist_ok=True)
    os.makedirs(TEMP_DXF_DIR, exist_ok=True)

    print("🔍 Découverte des parcelles OREGEN/DEF...")
    pairs = discover_parcel_pairs()
    total_parcels = len(pairs)
    print(f"📦 Total découvert : {total_parcels} parcelles.")

    max_workers = min(os.cpu_count() or 4, 10)
    print(f"🚀 Lancement du scan parallèle sur {max_workers} cœurs CPU simultanément...")

    start_time = time.time()
    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {
            executor.submit(worker_process_parcel, p, OUTPUT_PNG_DIR, TEMP_DXF_DIR): p["parcel_id"]
            for p in pairs
        }

        completed_count = 0
        for future in as_completed(future_to_pid):
            completed_count += 1
            res = future.result()
            results.append(res)
            pid = res["parcel_id"]
            status = res["status"]
            if status == "SUCCESS":
                pct = res["pct_green"]
                print(f"[{completed_count:03d}/{total_parcels:03d}] Parcelle {pid:8s} | 🟩 {res['matches']:2d} | 🟨 {res['shifted']:2d} | 🟥 {res['missing']:2d} | 🔵 {res['parasites']:2d} | {pct:5.1f}%")
            else:
                print(f"[{completed_count:03d}/{total_parcels:03d}] Parcelle {pid:8s} | ❌ ERREUR: {res['error_msg'][:60]}")

    elapsed = time.time() - start_time

    # Nettoyer le dossier temporaire
    try:
        os.rmdir(TEMP_DXF_DIR)
    except Exception:
        pass

    # Trier les résultats par identifiant de parcelle naturel
    def sort_key(r):
        num_m = os.path.basename(r["parcel_id"])
        import re
        nums = re.findall(r"\d+", num_m)
        return int(nums[0]) if nums else 0

    results.sort(key=sort_key)

    # Calcul des métriques globales
    total_def_bornes = sum(r["total_def"] for r in results if r["status"] == "SUCCESS")
    total_ours_bornes = sum(r["total_ours"] for r in results if r["status"] == "SUCCESS")
    total_matches = sum(r["matches"] for r in results if r["status"] == "SUCCESS")
    total_shifted = sum(r["shifted"] for r in results if r["status"] == "SUCCESS")
    total_missing = sum(r["missing"] for r in results if r["status"] == "SUCCESS")
    total_parasites = sum(r["parasites"] for r in results if r["status"] == "SUCCESS")

    global_pct = (total_matches / total_def_bornes * 100.0) if total_def_bornes > 0 else 0.0
    parcels_100_pct = sum(1 for r in results if r["status"] == "SUCCESS" and r["pct_green"] == 100.0)

    print("\n" + "=" * 70)
    print(" 🏆 BILAN MASSIF DU SCAN TOOL 1 (BORNES & RAPPELS MASTER)")
    print("=" * 70)
    print(f"⏱️ Temps total d'exécution : {elapsed:.2f} secondes ({elapsed/total_parcels:.2f}s / parcelle)")
    print(f"📦 Parcelles analysées     : {len(results)} / {total_parcels}")
    print(f"🟢 Parcelles à 100% Vert  : {parcels_100_pct} ({parcels_100_pct / len(results) * 100.0:.1f}%)")
    print(f"📊 Total Bornes DEF        : {total_def_bornes}")
    print(f"🟩 MATCHES PARFAITS (Vert) : {total_matches} ({global_pct:.2f}%)")
    print(f"🟨 DÉCALÉS (Jaune)         : {total_shifted}")
    print(f"🟥 MANQUANTS (Rouge)       : {total_missing}")
    print(f"🔵 PARASITES (Bleu)        : {total_parasites}")
    print("=" * 70)

    # Export CSV
    csv_path = r"C:\Users\Admin\.gemini\antigravity\brain\082ae9b6-6db5-4b26-8ba6-7c6533a53d90\scores_tool1_137_parcels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "parcel_id", "status", "scale", "total_def", "total_ours",
            "matches", "shifted", "missing", "parasites", "pct_green", "error_msg"
        ])
        writer.writeheader()
        for r in results:
            row = dict(r)
            row.pop("png_path", None)
            writer.writerow(row)
    print(f"📄 Fichier CSV exporté : {csv_path}")

    # Export Markdown
    md_path = r"C:\Users\Admin\.gemini\antigravity\brain\082ae9b6-6db5-4b26-8ba6-7c6533a53d90\RAPPORT_GLOBAL_TOOL1_137_PARCELLES.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🏆 RAPPORT MASSIF TOOL 1 : SCAN GLOBAL DES 137 PARCELLES\n\n")
        f.write(f"**Date d'exécution** : 17 Octobre 2026  \n")
        f.write(f"**Mode d'exécution** : Multi-processus parallèle ({max_workers} workers)  \n")
        f.write(f"**Temps d'exécution** : {elapsed:.2f} secondes ({elapsed/total_parcels:.2f}s / parcelle)  \n\n")
        f.write("## 1. 📊 Métriques Globales Consolidées\n\n")
        f.write("| Métrique | Valeur Consolidée | Taux Global |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Parcelles 100% Conformes** | **{parcels_100_pct} / {len(results)}** | **{parcels_100_pct / len(results) * 100.0:.1f}%** |\n")
        f.write(f"| **Total Bornes dans DEF** | **{total_def_bornes}** | 100.0% |\n")
        f.write(f"| 🟩 **Matches Parfaits (Vert)** | **{total_matches}** | **{global_pct:.2f}%** |\n")
        f.write(f"| 🟨 **Décalés Hors Seuil (Jaune)** | **{total_shifted}** | **{total_shifted/max(total_def_bornes,1)*100.0:.2f}%** |\n")
        f.write(f"| 🟥 **Manquants (Rouge)** | **{total_missing}** | **{total_missing/max(total_def_bornes,1)*100.0:.2f}%** |\n")
        f.write(f"| 🔵 **Parasites (Bleu)** | **{total_parasites}** | **{total_parasites/max(total_def_bornes,1)*100.0:.2f}%** |\n\n")
        f.write("## 2. 📋 Détail par Parcelle\n\n")
        f.write("| Parcelle | Échelle | Total DEF | 🟩 Vert | 🟨 Jaune | 🟥 Rouge | 🔵 Bleu | Taux Vert | Statut |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            if r["status"] == "SUCCESS":
                st = "🟢 100%" if r["pct_green"] == 100.0 else ("🟡 Partiel" if r["pct_green"] >= 80.0 else "🔴 Attention")
                f.write(f"| {r['parcel_id']} | 1/{r['scale']} | {r['total_def']} | {r['matches']} | {r['shifted']} | {r['missing']} | {r['parasites']} | {r['pct_green']:.1f}% | {st} |\n")
            else:
                f.write(f"| {r['parcel_id']} | — | — | — | — | — | — | 0.0% | ❌ Erreur |\n")
    print(f"📑 Rapport Markdown exporté : {md_path}")


if __name__ == "__main__":
    main()
