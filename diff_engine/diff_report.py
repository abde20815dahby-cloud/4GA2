# -*- coding: utf-8 -*-
"""
diff_report.py: Générateur de rapports textuels et formatés pour l'analyse différentielle OURS vs DEF.
Fournit une synthèse claire et détaillée de chaque catégorie avec alertes visuelles.
"""
from typing import Optional
from visual_diff import DiffResult, DiffItem


def format_diff_report(result: DiffResult, detailed: bool = True) -> str:
    """
    Génère un rapport textuel complet et formaté prêt à l'affichage terminal ou fichier.
    """
    lines = []
    w = 60
    lines.append("═" * w)
    lines.append(f" 📊 RAPPORT DIFFÉRENTIEL : PARCELLE {result.parcel_id} ")
    if result.category_filter:
        lines.append(f" 🎯 FILTRE ACTIF : {result.category_filter.upper()}")
    lines.append("═" * w)
    lines.append(f"📁 Fichier DEF  : {result.def_path}")
    lines.append(f"📁 Fichier OURS : {result.ours_path}")
    lines.append("─" * w)

    n_match = len(result.matches)
    n_shift = len(result.shifted)
    n_miss = len(result.missing)
    n_para = len(result.parasites)
    total_def = result.total_def
    total_ours = result.total_ours

    lines.append(f"📋 RÉSUMÉ DES ENTITÉS :")
    lines.append(f"   • Total dans DEF   : {total_def}")
    lines.append(f"   • Total dans OURS  : {total_ours}")
    lines.append(f"   • 🟩 MATCHES PARFAITS (Vert)  : {n_match}")
    lines.append(f"   • 🟨 DÉCALÉS HORS SEUIL (Jaune): {n_shift}")
    lines.append(f"   • 🟥 MANQUANTS DANS OURS (Rouge): {n_miss}")
    lines.append(f"   • 🔵 PARASITES EN TROP (Bleu)  : {n_para}")
    lines.append("─" * w)

    if detailed:
        if result.shifted:
            lines.append("🟨 DÉTAIL DES ENTITÉS DÉCALÉES :")
            for it in result.shifted:
                lines.append(f"   ⚠️ '{it.text_def}' : {it.details}")
                if it.pos_def and it.pos_ours:
                    lines.append(f"      DEF=({it.pos_def[0]:.2f}, {it.pos_def[1]:.2f}) -> OURS=({it.pos_ours[0]:.2f}, {it.pos_ours[1]:.2f})")
            lines.append("─" * w)

        if result.missing:
            lines.append("🟥 DÉTAIL DES ENTITÉS MANQUANTES (Dans DEF mais pas OURS) :")
            for it in result.missing:
                pos_str = f"({it.pos_def[0]:.2f}, {it.pos_def[1]:.2f})" if it.pos_def else "inconnu"
                lines.append(f"   ❌ '{it.text_def}' à {pos_str} [Cat={it.category}]")
            lines.append("─" * w)

        if result.parasites:
            lines.append("🔵 DÉTAIL DES PARASITES (Dans OURS mais pas DEF) :")
            for it in result.parasites:
                pos_str = f"({it.pos_ours[0]:.2f}, {it.pos_ours[1]:.2f})" if it.pos_ours else "inconnu"
                lines.append(f"   👾 '{it.text_ours}' à {pos_str} [Cat={it.category}]")
            lines.append("─" * w)

    # Verdict
    lines.append("🏁 VERDICT :")
    if result.is_perfect:
        lines.append("   ✅ CONFORME À 100% (Aucun manquant, aucun parasite, aucun décalage)")
    elif n_miss == 0 and n_para == 0 and n_shift <= 2:
        lines.append("   ⚠️ ACCEPTABLE AVEC LÉGERS DÉCALAGES (0 manquant, 0 parasite)")
    else:
        lines.append(f"   ❌ NON CONFORME : {n_miss} manquant(s), {n_para} parasite(s), {n_shift} décalé(s)")
    lines.append("═" * w)

    return "\n".join(lines)


def print_diff_report(result: DiffResult, detailed: bool = True) -> None:
    """Affiche le rapport dans la console standard."""
    print(format_diff_report(result, detailed=detailed))
