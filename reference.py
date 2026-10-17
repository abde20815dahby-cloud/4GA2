# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/reference.py — Recherche universelle de plans de référence ANCFCC (DEF)
========================================================================================
Permet d'associer automatiquement chaque plan DXF source à son plan définitif étalon
(DEF / DEFINITIVE / plans vérifiés) pour assurer une conformité à 100%.
"""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger("dxf_cleaner.reference")


def find_reference_dxf(input_path: str, reference_root: Optional[str] = None) -> Optional[str]:
    """
    Recherche universelle et intelligente du plan de référence DEF correspondant à l'input.
    Supporte tout type de structure de dossiers ANCFCC ou de projet.
    1. Si reference_root est spécifié et valide, cherche récursivement dedans.
    2. Remonte l'arborescence des dossiers jusqu'à 6 niveaux pour localiser un dossier 'DEF' parent ou frère
       et recherche {pid}*.dxf récursivement dedans.
    3. Tente le remplacement textuel direct OREGEN -> DEF sur le chemin.
    """
    if not input_path:
        return None

    base_name = os.path.basename(input_path)
    pid = re.sub(r'[-_]plan.*$', '', base_name, flags=re.IGNORECASE)
    valid_names = {f"{pid.lower()}-plan.dxf", f"{pid.lower()}.dxf"}

    # 1. Dossier explicite fourni par l'utilisateur
    if reference_root and os.path.isdir(reference_root):
        for r, _, fs in os.walk(reference_root):
            for f in fs:
                if f.lower() in valid_names:
                    return os.path.join(r, f)

    dir_cur = os.path.dirname(os.path.abspath(input_path))

    # 2. Remonter jusqu'à 6 niveaux pour chercher un dossier DEF parent ou frère
    curr = dir_cur
    for _ in range(6):
        def_candidate = os.path.join(curr, "DEF")
        if os.path.isdir(def_candidate):
            for r, _, fs in os.walk(def_candidate):
                for f in fs:
                    if f.lower() in valid_names:
                        return os.path.join(r, f)
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    # 3. Remplacement textuel direct OREGEN -> DEF
    if re.search(r'[\\/]OREGEN[\\/]', dir_cur, re.IGNORECASE):
        cand_dir = re.sub(r'([\\/])OREGEN([\\/])', r'\1DEF\2', dir_cur, flags=re.IGNORECASE)
        cand_f = os.path.join(cand_dir, base_name)
        if os.path.exists(cand_f):
            return cand_f

    return None
