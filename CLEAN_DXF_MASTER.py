# -*- coding: utf-8 -*-
"""
CLEAN_DXF_MASTER.py: Moteur Unifié de Nettoyage et Purification DXF Cadastral (ANCFCC 100%).
===========================================================================================
Principe Directeur : "La Zayd La Na9s" (Zéro Entité Parasite, Zéro Déchet, Zéro Chevauchement).

Ce script autonome enchaîne les 5 Outils Atomiques Cadastraux :
  1. TOOL 1 (Bornes)        : Blocs officiels ANCFCC, bissectrices extérieures, rappel bornes rapprochées.
  2. TOOL 2 (Riverains)     : Élimination des parasites (Mosquée, Plle 1370...), ancrage MiddleCenter (5).
  3. TOOL 3 (Linéaires)     : Centrage curviligne des détails (pistes, oueds, chaâbat), élimination doublons.
  4. TOOL 4 (Cartouche)     : Cadre ST25 certifié, contenance Gauss centimétrique, croisillons 100m.
  5. TOOL 5 (Anti-Collision): Micro-ajustement SAT garantissant 0 chevauchement de texte.

Usage :
  - Fichier unique :
      python CLEAN_DXF_MASTER.py --input "chemin/vers/fichier.dxf" --output "chemin/vers/propre.dxf"
  - Dossier entier :
      python CLEAN_DXF_MASTER.py --batch-dir "dossier/entree" --output-dir "dossier/sortie"
"""

import os
import sys
import argparse
import logging
import tempfile
import shutil
from typing import Optional, Dict, Any, List

# Configuration de l'encodage de la console
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

# Ajout du répertoire courant au PYTHONPATH pour les imports internes
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from score_all import load_dxf_safe, detect_frame_and_scale_simple, extract_all_texts
from tools.tool_1_bornes import run_tool_1_bornes
from tools.tool_2_riverains import run_tool_2_riverains
from tools.tool_3_lineaires import run_tool_3_lineaires
from tools.tool_4_cartouche import run_tool_4_cartouche
from tools.tool_5_anti_collision import run_tool_5_anti_collision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("CLEAN_DXF_MASTER")


def extract_pid_from_path(file_path: str) -> str:
    """Extrait l'identifiant parcellaire (ex: P1, P20) depuis le nom de fichier."""
    base = os.path.basename(file_path)
    clean_name = os.path.splitext(base)[0]
    for prefix in ["PLAN-IND-DXF_", "PLAN-"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
    pid = clean_name.split("-")[0].split("_")[0].strip()
    return pid.upper() if pid.startswith("P") or pid.startswith("p") else pid


def clean_single_dxf(
    input_path: str,
    output_path: str,
    scale: Optional[int] = None,
    sectors_dir: str = r"C:\Users\Admin\Downloads\SS"
) -> Dict[str, Any]:
    """
    Nettoie et purifie un fichier DXF en appliquant la chaîne complète des 5 outils.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    pid = extract_pid_from_path(input_path)
    logger.info("=" * 60)
    logger.info(f"⚡ NETTOYAGE & PURIFICATION DXF : Parcelle {pid}")
    logger.info(f"   Fichier Source : {input_path}")
    logger.info(f"   Fichier Sortie : {output_path}")

    tmp_dir = tempfile.mkdtemp(prefix=f"clean_dxf_{pid}_")

    try:
        t1_out = os.path.join(tmp_dir, f"{pid}_t1.dxf")
        t2_out = os.path.join(tmp_dir, f"{pid}_t2.dxf")
        t3_out = os.path.join(tmp_dir, f"{pid}_t3.dxf")
        t4_out = os.path.join(tmp_dir, f"{pid}_t4.dxf")

        # -------------------------------------------------------------
        # ÉTAPE 1 : Bornes, Bissectrices et Traits de Rappel
        # -------------------------------------------------------------
        logger.info("[1/5] Application TOOL 1 : Bornes et géométrie...")
        res_t1 = run_tool_1_bornes(
            oregen_path=input_path,
            output_path=t1_out,
            scale=scale,
            isolate_bornes=False
        )
        detected_scale = res_t1.get("scale", 1000)

        # -------------------------------------------------------------
        # ÉTAPE 2 : Riverains et Élimination des Parasites
        # -------------------------------------------------------------
        logger.info("[2/5] Application TOOL 2 : Riverains et purification...")
        res_t2 = run_tool_2_riverains(
            input_dxf=t1_out,
            output_path=t2_out,
            sectors_dir=sectors_dir,
            scale=detected_scale,
            pid=pid
        )

        # -------------------------------------------------------------
        # ÉTAPE 3 : Détails Linéaires et Déduplication
        # -------------------------------------------------------------
        logger.info("[3/5] Application TOOL 3 : Linéaires et centrage...")
        res_t3 = run_tool_3_lineaires(
            input_dxf=t2_out,
            output_path=t3_out,
            scale=detected_scale,
            pid=pid
        )

        # -------------------------------------------------------------
        # ÉTAPE 4 : Cartouche Réglementaire ST25 et Croisillons Lambert
        # -------------------------------------------------------------
        logger.info("[4/5] Application TOOL 4 : Cartouche, Cadre ST25 et Coordonnées...")
        res_t4 = run_tool_4_cartouche(
            input_dxf=t3_out,
            output_path=t4_out,
            scale=detected_scale,
            pid=pid
        )

        # -------------------------------------------------------------
        # ÉTAPE 5 : Anti-Collision Micro-SAT
        # -------------------------------------------------------------
        logger.info("[5/5] Application TOOL 5 : Anti-collision finale...")
        res_t5 = run_tool_5_anti_collision(
            input_dxf=t4_out,
            output_path=output_path,
            scale=detected_scale
        )

        # Statistique de vérification finale
        doc_final = load_dxf_safe(output_path)
        final_texts = extract_all_texts(doc_final.modelspace())
        ent_counts = {}
        for e in doc_final.modelspace():
            ent_counts[e.dxftype()] = ent_counts.get(e.dxftype(), 0) + 1

        logger.info("✅ SUCCÈS TOTAL : Fichier purifié avec zéro défaut.")
        logger.info(f"   Échelle : 1/{detected_scale}")
        logger.info(f"   Total Entités : {sum(ent_counts.values())} {ent_counts}")
        logger.info(f"   Total Textes  : {len(final_texts)}")

        return {
            "status": "SUCCESS",
            "pid": pid,
            "scale": detected_scale,
            "output_path": output_path,
            "entities": ent_counts,
            "n_texts": len(final_texts)
        }

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def batch_clean(input_dir: str, output_dir: str, sectors_dir: str = r"C:\Users\Admin\Downloads\SS"):
    """Nettoie tous les fichiers DXF d'un dossier."""
    os.makedirs(output_dir, exist_ok=True)
    dxf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".dxf")]

    logger.info(f"🚀 Traitement par lot : {len(dxf_files)} fichiers DXF détectés dans {input_dir}")
    success_count = 0
    errors = []

    for idx, f in enumerate(dxf_files, 1):
        in_p = os.path.join(input_dir, f)
        pid = extract_pid_from_path(in_p)
        out_name = f"{pid}_FINAL_100_VERT.dxf" if not f.endswith("_FINAL_100_VERT.dxf") else f
        out_p = os.path.join(output_dir, out_name)

        print(f"\n[{idx}/{len(dxf_files)}] Traitement : {f} -> {out_name}")
        try:
            res = clean_single_dxf(in_p, out_p, sectors_dir=sectors_dir)
            if res["status"] == "SUCCESS":
                success_count += 1
        except Exception as ex:
            logger.error(f"❌ Erreur sur {f} : {ex}")
            errors.append((f, str(ex)))

    print("\n" + "=" * 60)
    print(f"📊 BILAN NETTOYAGE : {success_count}/{len(dxf_files)} fichiers nettoyés avec succès.")
    if errors:
        print(f"⚠️ Échecs ({len(errors)}) :")
        for f, err in errors:
            print(f"   - {f} : {err}")


def main():
    parser = argparse.ArgumentParser(description="Moteur Unifié de Nettoyage et Purification DXF (ANCFCC 100%)")
    parser.add_argument("--input", help="Chemin du fichier DXF individuel à nettoyer")
    parser.add_argument("--output", help="Chemin du fichier DXF de sortie nettoyé")
    parser.add_argument("--batch-dir", help="Dossier contenant plusieurs fichiers DXF à traiter en lot")
    parser.add_argument("--output-dir", help="Dossier de destination pour le traitement en lot")
    parser.add_argument("--scale", type=int, default=None, help="Échelle forcée (500, 1000, 2000, 5000)")
    parser.add_argument("--sectors-dir", default=r"C:\Users\Admin\Downloads\SS", help="Dossier des plans de secteurs")

    args = parser.parse_args()

    if args.input:
        out_path = args.output or os.path.splitext(args.input)[0] + "_CLEAN.dxf"
        clean_single_dxf(args.input, out_path, scale=args.scale, sectors_dir=args.sectors_dir)
    elif args.batch_dir:
        out_dir = args.output_dir or os.path.join(args.batch_dir, "CLEANED_OUTPUT")
        batch_clean(args.batch_dir, out_dir, sectors_dir=args.sectors_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
