# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/batch.py — Exécution par lot et gestion atomique des fichiers
================================================================================
Gère l'exécution sécurisée sur un fichier unique (in-place atomic overwrite avec fichier temporaire)
ou sur un lot complet de fichiers en mode séquentiel ou multi-threadé.
"""

import os
import shutil
import tempfile
import time
import threading
import concurrent.futures
import logging
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

from engines.dxf_cleaner.constants import DEFAULT_OPTIONS, ETAPES_PARCELLAIRE
from engines.dxf_cleaner.models import CleanReport
from engines.dxf_cleaner.pipeline import execute_pipeline

logger = logging.getLogger("dxf_cleaner.batch")


def clean_file(
    input_path: str,
    output_path: Optional[str] = None,
    opts: Optional[dict] = None,
    step_cb: Optional[Callable[[str], None]] = None,
    dry_run: bool = False,
    rename_original_to: Optional[str] = None,
    **kwargs
) -> CleanReport:
    """
    Nettoie et traite un fichier DXF unique selon les normes cadastrales ANCFCC.
    Gère la sauvegarde atomique in-place si output_path est None ou égal à input_path.
    """
    o = opts.copy() if opts else {}
    o.update(kwargs)

    full_opts = DEFAULT_OPTIONS.copy()
    full_opts.update(o)

    is_inplace = output_path is None or os.path.abspath(output_path) == os.path.abspath(input_path)
    final_output = input_path if is_inplace else output_path

    # Gestion callback d'étape UI
    def _pipeline_step_cb(idx: int, label: str):
        if step_cb:
            step_cb(f"[{idx}/{len(ETAPES_PARCELLAIRE)}] {label}")

    # Utilisation d'un fichier temporaire pour écriture atomique sécurisée
    target_dir = os.path.dirname(os.path.abspath(final_output))
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_out = tempfile.mkstemp(prefix="dxf_clean_", suffix=".dxf", dir=target_dir)
    os.close(fd)

    t0 = time.time()
    try:
        report = execute_pipeline(input_path, tmp_out, full_opts, step_cb=_pipeline_step_cb)
        report.output_path = final_output
        report.elapsed_seconds = round(time.time() - t0, 2)
        logger.info("Traitement DXF cadastral complet terminé avec succès en %s s", report.elapsed_seconds)

        if dry_run:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
            return report

        # Sauvegarde originale si demandé
        if rename_original_to:
            shutil.copy2(input_path, rename_original_to)

        # Remplacement atomique
        if os.path.exists(final_output) and is_inplace:
            os.replace(tmp_out, final_output)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(final_output)), exist_ok=True)
            if os.path.exists(final_output):
                os.replace(tmp_out, final_output)
            else:
                shutil.move(tmp_out, final_output)

        from standards import sanitize_certified_dxf_file
        sanitize_certified_dxf_file(final_output)

        return report

    except Exception as exc:
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except Exception:
                pass
        err_rep = CleanReport(input_path=input_path, output_path=final_output or input_path)
        err_rep.error = str(exc)
        err_rep.elapsed_seconds = round(time.time() - t0, 2)
        logger.error("Erreur nettoyage DXF %s: %s", input_path, exc, exc_info=True)
        return err_rep


def clean_files_batch(
    files: List[str],
    output_dir: Optional[str] = None,
    opts: Optional[dict] = None,
    progress_cb: Optional[Callable[[int, int, CleanReport], None]] = None,
    step_cb: Optional[Callable[[str, str], None]] = None,
    add_suffix: bool = True,
    same_dir_mode: str = "overwrite",  # "overwrite" ou "suffix"
    max_workers: int = 1,
    **kwargs
) -> List[CleanReport]:
    """
    Nettoie et traite un lot de fichiers DXF avec conservation de structure d'arborescence
    ou modification directe in-place selon les paramètres choisis.
    Supporte l'exécution séquentielle (max_workers=1) ou parallèle multi-cœurs (max_workers > 1).
    """
    total = len(files)
    if total == 0:
        return []

    # Calcul de racine commune pour reproduire la structure si output_dir est spécifié
    common_root = None
    if output_dir and total > 0:
        common_root = os.path.commonpath([os.path.abspath(f) for f in files])

    def _determine_target(file_path: str) -> str:
        if output_dir:
            if common_root and os.path.abspath(file_path).startswith(common_root):
                rel = os.path.relpath(os.path.abspath(file_path), common_root)
                dest = os.path.join(output_dir, rel)
            else:
                dest = os.path.join(output_dir, os.path.basename(file_path))

            if add_suffix:
                stem, ext = os.path.splitext(dest)
                if not stem.endswith("_clean"):
                    dest = f"{stem}_clean{ext}"
            os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
            return dest
        else:
            if same_dir_mode == "overwrite":
                return file_path
            else:
                p = Path(file_path)
                return str(p.with_stem(p.stem + "_clean"))

    # Mode Parallèle (Multi-cœurs)
    if max_workers > 1 and total > 1:
        lock = threading.Lock()
        reports = [None] * total
        done_count = 0

        def _worker(idx, file_path):
            nonlocal done_count
            out_target = _determine_target(file_path)

            def _file_step_cb(step_str: str):
                if step_cb:
                    with lock:
                        step_cb(file_path, step_str)

            rep = clean_file(
                file_path,
                output_path=out_target,
                opts=opts,
                step_cb=_file_step_cb,
                **kwargs
            )
            with lock:
                done_count += 1
                reports[idx] = rep
                if progress_cb:
                    progress_cb(done_count, total, rep)
            return rep

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker, i, fp) for i, fp in enumerate(files)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    logger.error("Batch worker exception: %s", e)

        return reports

    # Mode Séquentiel classique (100% garanti)
    reports = []
    for i, file_path in enumerate(files):
        out_target = _determine_target(file_path)

        def _file_step_cb(step_str: str):
            if step_cb:
                step_cb(file_path, step_str)

        rep = clean_file(
            file_path,
            output_path=out_target,
            opts=opts,
            step_cb=_file_step_cb,
            **kwargs
        )
        reports.append(rep)

        if progress_cb:
            progress_cb(i + 1, total, rep)

    return reports
