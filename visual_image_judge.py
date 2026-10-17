# -*- coding: utf-8 -*-
"""
01_PRODUCTION_CLEANER_PIPELINE/dxf_cleaner/visual_image_judge.py
================================================================================
MOTEUR DE RENDU IMAGE HAUTE DÉFINITION & DIFF VISUEL PIXEL-PAR-PIXEL
================================================================================
Convertit les plans DXF bruts, prédits et de référence en images PNG 250+ DPI,
et génère la carte visuelle des différences (Vert = Superposé, Rouge = Erreur/Déplacé,
Bleu = Manquant/Oublié) avec un calage géométrique rigoureux sur le cadre du plan.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from shapely.geometry import Polygon

from pipeline import detect_frame_and_scale, load_dxf_safe


def get_cadastral_view_box(doc, padding_ratio: float = 0.02) -> Tuple[float, float, float, float]:
    """
    Extrait l'emprise exacte du cadre de plan cadastral pour verrouiller
    un système de coordonnées d'affichage strictement identique entre deux DXF.
    """
    msp = doc.modelspace()
    frame_bbox, scale = detect_frame_and_scale(msp)
    xmin, ymin, xmax, ymax = frame_bbox

    # Si le cadre est valide (largeur > 50m et coordonnées réelles terrain)
    if (xmax - xmin) > 50.0 and (ymax - ymin) > 50.0 and xmin > 10000.0:
        pad_x = (xmax - xmin) * padding_ratio
        pad_y = (ymax - ymin) * padding_ratio
        return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)

    # Fallback : boîte englobante de toutes les entités du modelspace
    all_x, all_y = [], []
    for e in msp.query("LINE LWPOLYLINE INSERT TEXT MTEXT"):
        if hasattr(e.dxf, "insert"):
            all_x.append(e.dxf.insert.x)
            all_y.append(e.dxf.insert.y)
        elif hasattr(e.dxf, "start"):
            all_x.extend([e.dxf.start.x, e.dxf.end.x])
            all_y.extend([e.dxf.start.y, e.dxf.end.y])
        elif e.dxftype() == "LWPOLYLINE":
            for p in e.get_points():
                all_x.append(p[0])
                all_y.append(p[1])

    if all_x and all_y:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        span_x = max(max_x - min_x, 100.0)
        span_y = max(max_y - min_y, 100.0)
        return (min_x - span_x * 0.05, min_y - span_y * 0.05, max_x + span_x * 0.05, max_y + span_y * 0.05)

    return (0.0, 0.0, 1000.0, 1000.0)


def render_dxf_to_png(
    dxf_path: str,
    png_path: str,
    dpi: int = 220,
    view_box: Optional[Tuple[float, float, float, float]] = None,
    dark_mode: bool = True,
) -> Path:
    """
    Convertit un fichier DXF en une image PNG haute résolution fidèle à AutoCAD.
    Verrouille la fenêtre de vue (view_box) si fournie pour garantir un alignement pixel parfait.
    Par défaut en dark_mode=True (fond sombre AutoCAD #1e2227) pour un rendu éclatant
    des entités blanches (Couleur 7), rouges, vertes, et cyans.
    """
    doc = load_dxf_safe(dxf_path)
    msp = doc.modelspace()

    if view_box is None:
        view_box = get_cadastral_view_box(doc)

    xmin, ymin, xmax, ymax = view_box
    w_m = max(xmax - xmin, 1.0)
    h_m = max(ymax - ymin, 1.0)
    aspect = h_m / w_m

    fig_w = 12.0
    fig_h = max(8.0, min(24.0, fig_w * aspect))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    bg_color = "#1e2227" if dark_mode else "white"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ctx = RenderContext(doc)
    out = MatplotlibBackend(ax)
    Frontend(ctx, out).draw_layout(msp, finalize=False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()

    out_file = Path(png_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), dpi=dpi, facecolor=bg_color)
    plt.close(fig)
    return out_file


def generate_visual_comparison_sheet(
    oregen_dxf: str,
    predicted_dxf: str,
    definitive_dxf: str,
    output_dir: str,
    parcel_id: str = "P14",
    dpi: int = 200,
    dark_mode: bool = True,
) -> Dict[str, Any]:
    """
    Génère la planche d'inspection visuelle complète (PNG) permettant à l'Agent IA
    et à l'utilisateur d'inspecter de leurs propres yeux chaque détail entre
    PREDICTED (100% autonome depuis OREGEN seul) et DEFINITIVE (Golden Master).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Récupérer l'emprise commune à partir du plan définitif pour verrouiller l'alignement
    doc_def = load_dxf_safe(definitive_dxf)
    shared_view_box = get_cadastral_view_box(doc_def)

    p_oregen = render_dxf_to_png(
        oregen_dxf, str(out_dir / f"{parcel_id}_01_OREGEN.png"), dpi=dpi, view_box=shared_view_box, dark_mode=dark_mode
    )
    p_pred = render_dxf_to_png(
        predicted_dxf, str(out_dir / f"{parcel_id}_02_PREDICTED.png"), dpi=dpi, view_box=shared_view_box, dark_mode=dark_mode
    )
    p_def = render_dxf_to_png(
        definitive_dxf, str(out_dir / f"{parcel_id}_03_DEFINITIVE.png"), dpi=dpi, view_box=shared_view_box, dark_mode=dark_mode
    )

    # Lecture des images pour calcul pixel-diff
    img_pred = plt.imread(str(p_pred))
    img_def = plt.imread(str(p_def))

    h = min(img_pred.shape[0], img_def.shape[0])
    w = min(img_pred.shape[1], img_def.shape[1])
    arr_p = img_pred[:h, :w, :3]
    arr_d = img_def[:h, :w, :3]

    bg_rgb = np.array([0.118, 0.133, 0.153], dtype=np.float32) if dark_mode else np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Détection des pixels d'encre (distance par rapport au fond)
    dist_p = np.max(np.abs(arr_p - bg_rgb), axis=2)
    dist_d = np.max(np.abs(arr_d - bg_rgb), axis=2)
    ink_p = dist_p > 0.08
    ink_d = dist_d > 0.08

    both_ok = ink_p & ink_d
    only_pred_err = ink_p & (~ink_d)
    only_def_missed = (~ink_p) & ink_d

    diff_canvas = np.full((h, w, 3), bg_rgb, dtype=np.float32)
    # Vert vif = Parfaitement superposé | Rouge vif = Erreur / Déplacé | Cyan vif = Manquant
    diff_canvas[both_ok] = [0.15, 0.90, 0.25]
    diff_canvas[only_pred_err] = [0.95, 0.20, 0.20]
    diff_canvas[only_def_missed] = [0.20, 0.75, 1.00]

    diff_path = out_dir / f"{parcel_id}_04_VISUAL_DIFF.png"
    fig, axes = plt.subplots(1, 3, figsize=(24, 12), dpi=dpi)

    panel_bg = "#1e2227" if dark_mode else "white"
    title_color = "white" if dark_mode else "#0f172a"
    fig.patch.set_facecolor(panel_bg)

    axes[0].imshow(arr_p)
    axes[0].set_title(f"{parcel_id} - PREDICTED (100% Autonome)", fontsize=14, fontweight="bold", color="#4ade80" if dark_mode else "darkgreen")
    axes[0].axis("off")

    axes[1].imshow(arr_d)
    axes[1].set_title(f"{parcel_id} - DEFINITIVE (Référence Expert)", fontsize=14, fontweight="bold", color="#38bdf8" if dark_mode else "navy")
    axes[1].axis("off")

    axes[2].imshow(diff_canvas)
    axes[2].set_title("DIFF VISUEL PIXEL (Vert=Match | Rouge=Erreur Prédite | Cyan=Oublié)", fontsize=14, fontweight="bold", color=title_color)
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(str(diff_path), dpi=dpi, bbox_inches="tight", facecolor=panel_bg)
    plt.close(fig)

    total_def_ink = max(int(np.sum(ink_d)), 1)
    pixel_overlap_ratio = float(np.sum(both_ok)) / float(total_def_ink)

    return {
        "oregen_png": str(p_oregen),
        "predicted_png": str(p_pred),
        "definitive_png": str(p_def),
        "visual_diff_png": str(diff_path),
        "ink_match_percentage": round(pixel_overlap_ratio * 100.0, 2),
    }
