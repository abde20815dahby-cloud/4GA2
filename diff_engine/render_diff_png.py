# -*- coding: utf-8 -*-
"""
render_diff_png.py: Générateur d'images PNG comparatives haute résolution (Visual Diff).
Affiche la parcelle et les entités textuelles avec code couleur :
- 🟩 Vert : Match parfait
- 🟨 Jaune : Décalé par rapport au DEF (avec flèche de direction)
- 🟥 Rouge : Manquant (présent dans DEF uniquement)
- 🔵 Bleu : Parasite (présent dans OURS uniquement)
"""
import os
from typing import Optional
import matplotlib
matplotlib.use("Agg")  # Mode sans affichage GUI pour génération de fichiers image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from visual_diff import DiffResult, DiffItem


def render_diff_to_png(
    result: DiffResult,
    output_png_path: str,
    title: Optional[str] = None,
    dpi: int = 200
) -> str:
    """
    Génère un rendu graphique 2D haute résolution de la comparaison différentielle.
    Sauvegarde l'image sous output_png_path et renvoie le chemin absolu.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_png_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12), dpi=dpi)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#ffffff")

    # 1. Dessiner le polygone parcellaire s'il existe
    if result.parcel_poly and not result.parcel_poly.is_empty:
        poly = result.parcel_poly
        x, y = poly.exterior.xy
        ax.plot(x, y, color="#212529", linewidth=2.2, label="Limite Parcellaire (LMTS)", zorder=2)
        ax.fill(x, y, color="#e9ecef", alpha=0.35, zorder=1)

        # Note: Ne PAS dessiner de points génériques sur les sommets du polygone,
        # car les points de limite naturelle (piste, chaâba, oued) ne sont pas des bornes.

    # 2. Dessiner les entités par catégorie de statut

    # 🟩 VERT : Matches parfaits
    for it in result.matches:
        if it.pos_ours:
            ax.scatter(it.pos_ours[0], it.pos_ours[1], color="#2b8a3e", s=45, marker="o", zorder=5)
            lbl = it.text_ours or it.text_def or ""
            ax.annotate(
                lbl,
                xy=it.pos_ours,
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#2b8a3e",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ebfbee", edgecolor="#b2f2bb", alpha=0.85)
            )

    # 🟨 JAUNE : Décalés (avec flèche vers la position DEF)
    for it in result.shifted:
        if it.pos_ours:
            ax.scatter(it.pos_ours[0], it.pos_ours[1], color="#f59f00", s=60, marker="s", zorder=5)
            lbl = f"{it.text_ours} (Δ={it.dist:.1f}m)"
            ax.annotate(
                lbl,
                xy=it.pos_ours,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#d9480f",
                zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff9db", edgecolor="#ffe066", alpha=0.9)
            )
            # Flèche vers la position DEF si connue
            if it.pos_def:
                ax.annotate(
                    "",
                    xy=it.pos_def,
                    xytext=it.pos_ours,
                    arrowprops=dict(arrowstyle="->", color="#f59f00", linestyle="--", lw=1.5),
                    zorder=4
                )
                ax.scatter(it.pos_def[0], it.pos_def[1], color="#fab005", s=30, marker="x", zorder=4)

    # 🟥 ROUGE : Manquants (Présents dans DEF uniquement)
    for it in result.missing:
        if it.pos_def:
            ax.scatter(it.pos_def[0], it.pos_def[1], color="#e03131", s=70, marker="X", zorder=7)
            lbl = f"MANQUANT: {it.text_def}"
            ax.annotate(
                lbl,
                xy=it.pos_def,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#c92a2a",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe3e3", edgecolor="#ffa8a8", alpha=0.95)
            )

    # 🔵 BLEU : Parasites (Présents dans OURS uniquement)
    for it in result.parasites:
        if it.pos_ours:
            ax.scatter(it.pos_ours[0], it.pos_ours[1], color="#1971c2", s=70, marker="*", zorder=7)
            lbl = f"PARASITE: {it.text_ours}"
            ax.annotate(
                lbl,
                xy=it.pos_ours,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                weight="bold",
                color="#1864ab",
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e7f5ff", edgecolor="#a5d8ff", alpha=0.95)
            )

    # 3. Personnalisation des axes et titre
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linestyle=":", alpha=0.6, color="#ced4da")

    n_m = len(result.matches)
    n_s = len(result.shifted)
    n_r = len(result.missing)
    n_b = len(result.parasites)

    main_title = title or f"Visual Diff ANCFCC — Parcelle {result.parcel_id}"
    subtitle = f"Match [Vert]: {n_m}  |  Décalé [Jaune]: {n_s}  |  Manquant [Rouge]: {n_r}  |  Parasite [Bleu]: {n_b}"

    ax.set_title(f"{main_title}\n{subtitle}", fontsize=13, weight="bold", pad=12, color="#212529")
    ax.set_xlabel("Coordonnée X Lambert (m)", fontsize=9, color="#495057")
    ax.set_ylabel("Coordonnée Y Lambert (m)", fontsize=9, color="#495057")

    # Légende explicite
    legend_elements = [
        patches.Patch(color="#2b8a3e", label=f"Match Conforme ({n_m})"),
        patches.Patch(color="#f59f00", label=f"Décalé > Seuil ({n_s})"),
        patches.Patch(color="#e03131", label=f"Manquant / Erreur DEF ({n_r})"),
        patches.Patch(color="#1971c2", label=f"Parasite / En trop ({n_b})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_png_path)
