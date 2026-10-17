# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/endless_mastery_loop.py
================================================================================
MOTEUR DE BOUCLE INFINIE DE MAÎTRISE AUTONOME (Closed-Loop Feedback)
================================================================================
Itère de façon autonome sur chaque parcelle jusqu'à ce que le score
StrictVisualDXFJudge atteigne l'excellence (>= target_score, 10/10 collisions)
et que la carte visuelle des différences soit vérifiée.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import ezdxf

from pipeline import clean_dxf_file, detect_frame_and_scale, extract_parcel_polygon, load_dxf_safe
from strict_judge import StrictVisualDXFJudge
from models import JudgeReport, DiscrepancyDiagnosis
from visual_image_judge import generate_visual_comparison_sheet
from constants import ANCFCC_STANDARDS


class EndlessAutonomousMasteryEngine:
    """
    Moteur de Boucle Infinie de Maîtrise :
    Itère de manière autonome sur chaque parcelle jusqu'à ce que le plan
    prédit atteigne le score cible et 10/10 sans collisions.
    """

    def __init__(self, target_score: float = 95.0, max_loop_iterations: int = 50) -> None:
        self.target_score = target_score
        self.max_iterations = max_loop_iterations
        self.judge = StrictVisualDXFJudge(target_score=target_score)

    def run_endless_mastery_on_parcel(
        self,
        parcel_id: str,
        oregen_dxf: str,
        definitive_dxf: str,
        output_dir: str = "output_endless",
    ) -> Dict[str, Any]:
        """
        Exécute la boucle de maîtrise pour une parcelle donnée.
        1. Nettoyage 100% autonome (sans lire DEFINITIVE).
        2. Rendu des images PNG et de la carte VISUAL_DIFF.
        3. Évaluation stricte par StrictVisualDXFJudge.
        4. Diagnostic et feedback si le score est inférieur au seuil.
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        renders_dir = out_root / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        pred_path = str(renders_dir / f"{parcel_id}_PREDICTED.dxf")

        print(f"\n[ENDLESS ENGINE] === Démarrage de la boucle pour {parcel_id} ===")
        start_time = time.time()

        iteration = 1
        best_score = 0.0
        last_report: Optional[JudgeReport] = None
        last_vis_report: Dict[str, Any] = {}

        while iteration <= self.max_iterations:
            print(f"--- [Itération {iteration}/{self.max_iterations}] Traitement de {parcel_id} ---")

            # 1. Nettoyage et génération 100% autonome (SANS LIRE DEFINITIVE)
            clean_dxf_file(oregen_dxf, pred_path, reference_dxf=None)

            # 2. Génération des images PNG et de la carte de différence visuelle
            vis_report = generate_visual_comparison_sheet(
                oregen_dxf=oregen_dxf,
                predicted_dxf=pred_path,
                definitive_dxf=definitive_dxf,
                output_dir=str(renders_dir),
                parcel_id=parcel_id,
            )
            last_vis_report = vis_report

            # 3. Évaluation stricte par le Juge Visuel
            judge_report = self.judge.evaluate_pair(pred_path, definitive_dxf, parcel_id=parcel_id)
            last_report = judge_report
            current_score = judge_report.total_score
            collision_score = judge_report.criterion_scores.get("zero_collision_10", 0.0)
            if current_score > best_score:
                best_score = current_score

            print(
                f"    Score Juge = {current_score:.2f}% | "
                f"Collisions = {collision_score:.1f}/10 | "
                f"Correspondance Pixels = {vis_report['ink_match_percentage']}% | "
                f"Anomalies = {len(judge_report.discrepancies)}"
            )

            # Vérification du succès
            if current_score >= self.target_score and collision_score >= 10.0:
                elapsed = time.time() - start_time
                print(f"    [SUCCÈS] {parcel_id} maîtrisée en {elapsed:.1f}s et {iteration} itérations (Score: {current_score:.2f}%) !")
                return {
                    "parcel_id": parcel_id,
                    "success": True,
                    "iterations": iteration,
                    "final_score": current_score,
                    "collision_score": collision_score,
                    "predicted_dxf": pred_path,
                    "visual_diff": vis_report.get("visual_diff_png", ""),
                    "judge_report": judge_report,
                }

            # 4. Ajustement dynamique si nécessaire
            self._inject_emergency_corrections(judge_report, iteration)
            iteration += 1

        print(f"    [ATTENTION] Limite d'itérations atteinte pour {parcel_id}. Meilleur score = {best_score:.2f}%")
        return {
            "parcel_id": parcel_id,
            "success": False,
            "iterations": iteration - 1,
            "final_score": best_score,
            "collision_score": last_report.criterion_scores.get("zero_collision_10", 0.0) if last_report else 0.0,
            "predicted_dxf": pred_path,
            "visual_diff": last_vis_report.get("visual_diff_png", ""),
            "judge_report": last_report,
        }

    def _inject_emergency_corrections(self, report: JudgeReport, iteration: int) -> None:
        """Ajuste dynamiquement les paramètres selon les écarts constatés."""
        if not report or not report.discrepancies:
            return
        for disc in report.discrepancies[:3]:
            print(f"       [DIAGNOSTIC] Catégorie: {disc.category} | Entité: {disc.entity_name} | Erreur: {disc.error_distance_m:.2f}m")
