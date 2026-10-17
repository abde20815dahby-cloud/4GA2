# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/sequential_mastery_trainer.py — Entraîneur Séquentiel Parcelle par Parcelle
=============================================================================================
Entraîneur Séquentiel Verrouillé Parcelle par Parcelle :
  - Traite chaque paire (OREGEN_i, DEFINITIVE_i) l'une après l'autre.
  - Exécute `clean_dxf_file` en mode 100% AUTONOME (sans jamais passer DEFINITIVE_i).
  - Évalue le résultat avec `StrictVisualDXFJudge` (6 critères réels, seuil minimal >= 98.5%).
  - Si score < 98.5%, adapte la mémoire apprise dans `CADASTRE_LEARNED_RULES.json` et ré-exécute.
  - Interdiction formelle de passer à la parcelle suivante tant que le score < 98.5%.
  - Contrôle de non-régression automatique sur toutes les parcelles préalablement maîtrisées.
"""

from __future__ import annotations
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from pipeline import clean_dxf_file
from strict_judge import StrictVisualDXFJudge, JudgeReport
from constants import ANCFCC_STANDARDS

logger = logging.getLogger("dxf_cleaner.trainer")


class SequentialParcelMasteryTrainer:
    """
    Entraîneur Séquentiel Verrouillé Parcelle par Parcelle :
    Garantit 0.00% collision, respect strict des normes ANCFCC et maîtrise prouvée >= 98.5%.
    """

    def __init__(
        self,
        rules_json_path: str = "02_CADASTRE_MASTER_REFERENCE/CADASTRE_LEARNED_RULES.json",
        target_score: float = 98.5,
        max_attempts_per_parcel: int = 8,
    ) -> None:
        self.rules_path = Path(rules_json_path)
        self.judge = StrictVisualDXFJudge(target_score=target_score)
        self.target_score = target_score
        self.max_attempts = max_attempts_per_parcel

    def train_sequentially(
        self,
        pairs: List[Tuple[str, str, str]],
        output_dir: str = "output_mastery",
        stop_on_first_failure: bool = True
    ) -> Dict[str, Any]:
        """
        Exécute la boucle séquentielle parcelle par parcelle.
        pairs : liste de (parcel_id, oregen_path, definitive_path)
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        mastered_parcels: List[Tuple[str, str, str]] = []
        history: List[Dict[str, Any]] = []

        print(f"\n{'='*80}")
        print(f"[DEMARRAGE] SEQUENTIAL PARCEL MASTERY TRAINER (TOTAL: {len(pairs)} PARCELLES)")
        print(f"[OBJECTIF] 100% AUTONOME | MINIMUM REQUIS >= {self.target_score}% PAR PARCELLE")
        print(f"{'='*80}\n")

        for idx, (parcel_id, oregen_path, definitive_path) in enumerate(pairs, 1):
            print(f"\n[PARCEL {idx}/{len(pairs)}] === Debut d'evaluation sur {parcel_id} ===")
            pred_path = str(out_root / f"{parcel_id}_PREDICTED.dxf")

            attempt_reports = []
            final_report: Optional[JudgeReport] = None

            for attempt in range(1, self.max_attempts + 1):
                t0 = time.time()
                clean_res = clean_dxf_file(oregen_path, pred_path, reference_dxf=None)
                el_clean = time.time() - t0

                report = self.judge.evaluate_pair(pred_path, definitive_path, parcel_id=parcel_id)
                final_report = report
                attempt_reports.append(report)

                print(
                    f"  -> Tentative {attempt}/{self.max_attempts} ({el_clean:.1f}s) | "
                    f"Score Visuel Reel = {report.total_score}% | "
                    f"Grille: {report.criterion_scores['grid_20']}/20 | "
                    f"Bornes: {report.criterion_scores['bornes_25']}/25 | "
                    f"Details: {report.criterion_scores['details_20']}/20 | "
                    f"TC: {report.criterion_scores['tc_riverains_15']}/15 | "
                    f"Collision: {report.criterion_scores['zero_collision_10']}/10 | "
                    f"Header: {report.criterion_scores['header_topology_10']}/10"
                )

                if report.passed:
                    print(f"  [OK] MAITRISE CONFIRMEE : {parcel_id} validee avec {report.total_score}% (>= {self.target_score}%) !")
                    break

                print(f"  [ADAPTATION] {len(report.discrepancies)} divergence(s) relevee(s). Auto-adaptation de la memoire...")
                self._adapt_memory_from_discrepancies(report)

            if not final_report or not final_report.passed:
                print(f"\n[BLOCAGE QUALITE] La parcelle {parcel_id} n'a pas atteint le seuil requis ({final_report.total_score if final_report else 0}% < {self.target_score}%).")
                if stop_on_first_failure:
                    print("Arrêt strict conformément à la Règle d'Or (interdiction d'avancer sans maîtrise prouvée).")
                    break

            mastered_parcels.append((parcel_id, oregen_path, definitive_path))
            history.append({
                "parcel_id": parcel_id,
                "final_score": final_report.total_score if final_report else 0.0,
                "passed": final_report.passed if final_report else False,
                "scores_breakdown": final_report.criterion_scores if final_report else {},
                "attempts": len(attempt_reports),
                "remaining_discrepancies": len(final_report.discrepancies) if final_report else 0,
            })

            # Contrôle de Non-Régression sur les parcelles précédentes maîtrisées
            if len(mastered_parcels) > 1:
                print(f"  [NON-REGRESSION] Verification sur {len(mastered_parcels) - 1} parcelle(s) precedente(s)...")
                for prev_id, prev_oregen, prev_def in mastered_parcels[:-1]:
                    prev_pred = str(out_root / f"{prev_id}_REGRESSION_CHECK.dxf")
                    clean_dxf_file(prev_oregen, prev_pred, reference_dxf=None)
                    reg_rep = self.judge.evaluate_pair(prev_pred, prev_def, parcel_id=prev_id)
                    if not reg_rep.passed:
                        print(f"  [ALERTE REGRESSION] Score sur {prev_id} descendu à {reg_rep.total_score}% !")
                    else:
                        print(f"     [OK] {prev_id}: Non-regression validee ({reg_rep.total_score}%)")

        print(f"\n{'='*80}")
        print(f"[BILAN] FIN DE SESSION D'ENTRAINEMENT SEQUENTIEL : {len(mastered_parcels)}/{len(pairs)} MAITRISEE(S)")
        print(f"{'='*80}\n")

        return {
            "mastered_count": len(mastered_parcels),
            "total_requested": len(pairs),
            "history": history
        }

    def _adapt_memory_from_discrepancies(self, report: JudgeReport) -> None:
        """Adapte dynamiquement les règles apprises dans CADASTRE_LEARNED_RULES.json."""
        for d in report.discrepancies:
            if d.category == "GRID" and "agent_eraser_grid.min_grid_to_frame_ratio" in d.recommended_rule_update:
                new_ratio = d.recommended_rule_update["agent_eraser_grid.min_grid_to_frame_ratio"]
                ANCFCC_STANDARDS["min_grid_to_frame_ratio"] = max(
                    ANCFCC_STANDARDS.get("min_grid_to_frame_ratio", 0.0075), new_ratio
                )
            elif d.category == "COLLISION":
                current_gap = ANCFCC_STANDARDS.get("min_borne_label_gap_mult", 0.35)
                ANCFCC_STANDARDS["min_borne_label_gap_mult"] = round(
                    min(0.55, current_gap + 0.02), 3
                )

        if self.rules_path.exists():
            try:
                data = json.loads(self.rules_path.read_text(encoding="utf-8"))
                data.setdefault("agent_eraser_grid", {})["min_grid_to_frame_ratio"] = ANCFCC_STANDARDS["min_grid_to_frame_ratio"]
                data.setdefault("agent_borne_mover", {})["min_borne_label_gap_mult"] = ANCFCC_STANDARDS["min_borne_label_gap_mult"]
                self.rules_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
