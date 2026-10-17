# -*- coding: utf-8 -*-
"""
engines/dxf_cleaner/models.py — Modèles de données géométriques, SAT et rapports d'exécution
===========================================================================================
Fournit les classes de boîtes englobantes orientées (OrientedBox2D), le solveur SAT d'intersection,
le calcul exact des largeurs de glyphes par style AutoCAD et le rapport d'exécution CleanReport.
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List
import numpy as np
from shapely.geometry import Polygon

STYLE_WIDTH_FACTORS: Dict[str, Tuple[float, float]] = {
    "CQ_ARIAL": (0.99, 0.0),
    "CQ_COORD": (0.85, 0.0),
    "CQ_VRBN": (0.75, 30.0),
    "CQ_ROMANS": (0.65, 30.0),
    "CQ_MAPPE": (1.125, 30.0),
    "CQ_PDITV": (0.45, 0.0),
    "CQ_PRID": (1.00, 0.0),
    "CQ_ROMANT": (1.00, 0.0),
    "CQ_ROMAND": (1.00, 0.0),
    "arabicfont": (1.00, 0.0),
}


def strip_dxf_formatting(raw_text: str) -> str:
    """Supprime les balises de formattage MTEXT complexes d'AutoCAD."""
    txt = re.sub(r"\\A\d;", "", raw_text)
    txt = re.sub(r"\{\\f[^;]+;([^}]*)\}", r"\1", txt)
    txt = re.sub(r"\{\\H[^;]+;\\S([^;]+);\}", r"\1", txt)
    return txt.strip()


def estimate_exact_text_dimensions(
    raw_text: str, height: float, style_name: str = "CQ_ARIAL"
) -> Tuple[float, float]:
    """Estime avec précision la largeur et la hauteur de la boîte englobante."""
    clean = strip_dxf_formatting(raw_text)
    lines = clean.split(r"\P") if r"\P" in clean else [clean]
    wf, obliq_deg = STYLE_WIDTH_FACTORS.get(style_name, (0.95, 0.0))
    max_chars = max((len(line) for line in lines), default=1)
    width = max_chars * height * 0.72 * wf + height * math.tan(math.radians(obliq_deg))
    total_height = height + (len(lines) - 1) * (5.0 / 3.0) * height
    return max(width, 1.2 * height), total_height


@dataclass
class OrientedBox2D:
    """Boîte englobante 2D orientée (OBB) pour le test de séparation SAT."""
    center: np.ndarray
    half_extents: np.ndarray
    angle_rad: float = 0.0
    corners: np.ndarray = field(init=False)
    axes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.update_pose(self.center, self.angle_rad)

    def update_pose(self, center: np.ndarray, angle_rad: float = 0.0) -> None:
        self.center = np.asarray(center, dtype=np.float64)
        self.angle_rad = float(angle_rad)
        c, s = math.cos(self.angle_rad), math.sin(self.angle_rad)
        self.axes = np.array([[c, s], [-s, c]], dtype=np.float64)
        wx = self.axes[0] * self.half_extents[0]
        wy = self.axes[1] * self.half_extents[1]
        self.corners = np.array([
            self.center - wx - wy,
            self.center + wx - wy,
            self.center + wx + wy,
            self.center - wx + wy,
        ], dtype=np.float64)

    def to_polygon(self) -> Polygon:
        return Polygon(self.corners)


def sat_overlap_mtv(box_a: OrientedBox2D, box_b: OrientedBox2D) -> Tuple[bool, np.ndarray, float]:
    """
    Separating Axis Theorem (SAT) :
    Détermine si deux OBB se chevauchent et calcule le Minimum Translation Vector (MTV).
    """
    axes = np.vstack((box_a.axes, box_b.axes))
    min_overlap = float("inf")
    mtv_axis = np.zeros(2, dtype=np.float64)
    diff = box_b.center - box_a.center

    for axis in axes:
        pa = np.dot(box_a.corners, axis)
        pb = np.dot(box_b.corners, axis)
        overlap = min(np.max(pa), np.max(pb)) - max(np.min(pa), np.min(pb))
        if overlap <= 0.0:
            return False, np.zeros(2, dtype=np.float64), 0.0
        if overlap < min_overlap:
            min_overlap = float(overlap)
            mtv_axis = axis if np.dot(diff, axis) >= 0.0 else -axis

    return True, mtv_axis, min_overlap


@dataclass
class CleanReport:
    """Rapport d'exécution d'un fichier DXF cadastral."""
    input_path: str
    output_path: str = ""
    n_lines_repaired: int = 0
    near_zero_deleted: int = 0
    duplicates_deleted: int = 0
    annotation_deleted: int = 0
    bornes_moved: int = 0
    bornes_strict: Tuple[int, int, int] = (0, 0, 0)
    verborne_positioned: int = 0
    rotations_reset: int = 0
    crosses_deleted: int = 0
    dtls_filtered: int = 0
    frame_clipped: int = 0
    texts_centered: int = 0
    overlaps_resolved: int = 0
    texts_wrapped: int = 0
    trees_populated: int = 0
    riverains_from_sector: int = 0
    bornes_injected_from_sector: int = 0
    scale_advice: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    elapsed_seconds: float = 0.0
    parcel_area: float = 0.0
    drawing_scale: str = ""

    @property
    def ok(self) -> bool:
        """Indique si le fichier a été traité avec succès sans erreur."""
        return not self.error

    def summary_text(self) -> str:
        """ملخص إحصائي دقيق وتفصيلي بالعمليات المنجزة في البلان."""
        fname = os.path.basename(self.input_path)
        if not self.ok:
            return f"❌ خطأ في معالجة {fname}: {self.error}"

        parts = [f"✅ تم تنظيف وتصحيح: {fname}"]
        if self.parcel_area > 0:
            scale_str = f" (السلم: {self.drawing_scale})" if self.drawing_scale else ""
            parts.append(f"• مساحة البقعة: {self.parcel_area:,.2f} m²{scale_str}")
        if self.near_zero_deleted:
            parts.append(f"• الكيانات الشبحية المحذوفة: {self.near_zero_deleted}")
        if self.duplicates_deleted:
            parts.append(f"• النصوص المكررة المحذوفة: {self.duplicates_deleted}")
        if self.annotation_deleted:
            parts.append(f"• نصوص الملاحظات المحذوفة: {self.annotation_deleted}")
        if self.bornes_moved:
            parts.append(f"• أسماء الأنصاب المحركة: {self.bornes_moved}")
        b_moved, b_scaled, b_match = self.bornes_strict
        if b_moved or b_scaled or b_match:
            parts.append(f"• أنصاب Bxx: {b_moved} منقولة، {b_scaled} مصغرة، {b_match} متطابقة")
        if self.verborne_positioned:
            parts.append(f"• نصوص VERBORNE الموضوعة: {self.verborne_positioned}")
        if self.crosses_deleted:
            parts.append(f"• علامات التقاطع المحذوفة: {self.crosses_deleted}")
        if self.texts_centered:
            parts.append(f"• نصوص التفاصيل المتمركزة: {self.texts_centered}")
        if self.overlaps_resolved:
            parts.append(f"• التداخلات المعالجة: {self.overlaps_resolved}")
        if self.texts_wrapped:
            parts.append(f"• نصوص مقسمة على أسطر: {self.texts_wrapped}")
        if self.riverains_from_sector:
            parts.append(f"• الجيران المستخرجون من البلان العام: {self.riverains_from_sector} جار")
        if self.scale_advice and self.scale_advice.get("suggested_frame"):
            parts.append(f"• ⚡ تنبيه الإطار: {self.scale_advice.get('msg', '')}")
        if self.elapsed_seconds > 0:
            parts.append(f"• مدة المعالجة: {self.elapsed_seconds:.2f}s")

        return "\n".join(parts)


@dataclass
class DiscrepancyDiagnosis:
    """Diagnostic précis d'un écart visuel relevé par StrictVisualDXFJudge."""
    category: str
    entity_name: str
    pred_pos: Tuple[float, float]
    def_pos: Tuple[float, float]
    error_distance_m: float
    context_info: Dict[str, Any] = field(default_factory=dict)
    recommended_rule_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeReport:
    """Rapport d'évaluation visuelle stricte calculé par StrictVisualDXFJudge."""
    parcel_id: str
    total_score: float
    criterion_scores: Dict[str, float]
    passed: bool
    discrepancies: List[DiscrepancyDiagnosis] = field(default_factory=list)


# -----------------------------------------------------------------------------
# TABLE OFFICIELLE DES 30 BLOCS ANCFCC (MASTER_ANCFCC_TEMPLATE.dxf)
# Dimensions réelles locales (W, H, BBox [xmin, ymin, xmax, ymax], Catégorie)
# -----------------------------------------------------------------------------
ANCFCC_MASTER_BLOCKS_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "borne_plantee": {"width": 0.040, "height": 0.040, "bbox": [-0.020, -0.020, 0.020, 0.020], "radius": 0.020, "category": "BORNE", "desc": "Borne plantée officielle"},
    "borne_marquee": {"width": 1.392, "height": 1.392, "bbox": [-0.696, -0.696, 0.696, 0.696], "radius": 0.696, "category": "BORNE", "desc": "Borne marquée au sol"},
    "borne_marquee_roche": {"width": 1.392, "height": 1.392, "bbox": [-0.696, -0.696, 0.696, 0.696], "radius": 0.696, "category": "BORNE", "desc": "Borne marquée sur roche"},
    "BLKR": {"width": 0.040, "height": 0.040, "bbox": [-0.020, -0.020, 0.020, 0.020], "radius": 0.020, "category": "BORNE", "desc": "Borne de rattachement"},
    "CROISI": {"width": 0.200, "height": 0.200, "bbox": [-0.100, -0.100, 0.100, 0.100], "radius": 0.100, "category": "GRID", "desc": "Croisillon de carroyage Lambert 100m"},
    "CARRE": {"width": 0.100, "height": 0.100, "bbox": [-0.050, -0.050, 0.050, 0.050], "radius": 0.050, "category": "GRID", "desc": "Symbole carré de repère"},
    "v113": {"width": 0.350, "height": 0.500, "bbox": [-0.175, 0.000, 0.175, 0.500], "radius": 0.350, "category": "NORD", "desc": "Flèche du Nord cadastral ANCFCC"},
    "puit": {"width": 3.451, "height": 3.794, "bbox": [-1.916, -2.873, 1.535, 0.921], "radius": 1.90, "category": "DETAIL", "desc": "Puits d'eau"},
    "metfia": {"width": 4.627, "height": 5.005, "bbox": [-3.112, -3.518, 1.515, 1.486], "radius": 2.50, "category": "DETAIL", "desc": "Matfiya / Citerne d'eau"},
    "metfia1": {"width": 4.627, "height": 5.005, "bbox": [-3.112, -3.518, 1.515, 1.486], "radius": 2.50, "category": "DETAIL", "desc": "Matfiya variante"},
    "PEBT": {"width": 3.802, "height": 5.342, "bbox": [-1.901, -0.350, 1.901, 4.992], "radius": 2.70, "category": "ELECTRICITE", "desc": "Poteau Électrique Basse Tension"},
    "PEMT": {"width": 3.070, "height": 5.604, "bbox": [-1.535, -0.350, 1.535, 5.254], "radius": 2.80, "category": "ELECTRICITE", "desc": "Poteau Moyenne Tension"},
    "PEHT": {"width": 3.070, "height": 6.058, "bbox": [-1.535, -0.350, 1.535, 5.708], "radius": 3.00, "category": "ELECTRICITE", "desc": "Poteau Haute Tension"},
    "LE": {"width": 1.484, "height": 1.503, "bbox": [-0.742, -0.752, 0.742, 0.752], "radius": 0.75, "category": "ELECTRICITE", "desc": "Ligne Électrique"},
    "marque": {"width": 1.969, "height": 1.969, "bbox": [-1.007, -0.952, 0.962, 1.017], "radius": 1.00, "category": "BORNE", "desc": "Marque cadastrale"},
    "olivier": {"width": 2.640, "height": 1.657, "bbox": [-1.333, 0.000, 1.307, 1.657], "radius": 1.35, "category": "VEGETATION", "desc": "Olivier"},
    "arbre": {"width": 0.366, "height": 1.782, "bbox": [1.102, 0.000, 1.468, 1.782], "radius": 1.00, "category": "VEGETATION", "desc": "Arbre générique"},
    "arganier": {"width": 1.327, "height": 4.067, "bbox": [-0.663, 0.000, 0.664, 4.067], "radius": 2.00, "category": "VEGETATION", "desc": "Arganier"},
    "amendier": {"width": 1.327, "height": 4.067, "bbox": [-0.663, 0.000, 0.664, 4.067], "radius": 2.00, "category": "VEGETATION", "desc": "Amandier"},
    "figuier": {"width": 1.327, "height": 4.067, "bbox": [-0.663, 0.000, 0.664, 4.067], "radius": 2.00, "category": "VEGETATION", "desc": "Figuier"},
    "arbre_fruitier": {"width": 0.366, "height": 1.629, "bbox": [0.000, 0.000, 0.366, 1.629], "radius": 0.85, "category": "VEGETATION", "desc": "Arbre fruitier"},
    "prunier": {"width": 3.148, "height": 4.126, "bbox": [-1.425, 0.000, 1.723, 4.126], "radius": 2.10, "category": "VEGETATION", "desc": "Prunier"},
    "caroubier": {"width": 1.794, "height": 5.088, "bbox": [-0.850, 0.000, 0.944, 5.088], "radius": 2.50, "category": "VEGETATION", "desc": "Caroubier"},
    "palmier": {"width": 4.726, "height": 4.374, "bbox": [-2.245, 0.000, 2.482, 4.374], "radius": 2.40, "category": "VEGETATION", "desc": "Palmier"},
    "grendadier": {"width": 4.195, "height": 4.247, "bbox": [-2.185, 0.002, 2.010, 4.249], "radius": 2.10, "category": "VEGETATION", "desc": "Grenadier"},
    "cactus": {"width": 4.819, "height": 3.072, "bbox": [-2.410, -0.876, 2.409, 2.196], "radius": 2.40, "category": "VEGETATION", "desc": "Cactus / Haie d'aloe"},
    "cimetiere": {"width": 2.780, "height": 5.249, "bbox": [-1.428, -0.083, 1.352, 5.166], "radius": 2.60, "category": "DETAIL", "desc": "Cimetière"},
    "regard": {"width": 2.635, "height": 2.635, "bbox": [-1.317, -1.317, 1.317, 1.317], "radius": 1.32, "category": "DETAIL", "desc": "Regard technique"},
    "fontaine": {"width": 2.914, "height": 2.551, "bbox": [-1.457, -1.275, 1.457, 1.275], "radius": 1.45, "category": "DETAIL", "desc": "Fontaine publique"},
    "CMAP": {"width": 1.000, "height": 1.000, "bbox": [-0.500, -0.500, 0.500, 0.500], "radius": 0.50, "category": "TEXT", "desc": "Symbole de mappage"}
}


# -----------------------------------------------------------------------------
# TABLE OFFICIELLE DES 63 CALQUES ANCFCC (MASTER_ANCFCC_TEMPLATE.dxf)
# -----------------------------------------------------------------------------
ANCFCC_MASTER_LAYERS: Dict[str, Dict[str, Any]] = {
    "0": {"color": 7, "category": "STANDARD", "desc": "Calque générique par défaut"},
    "parcelle": {"color": 1, "category": "LIMITES", "desc": "Limite de la parcelle principale"},
    "LIMIT": {"color": 1, "category": "LIMITES", "desc": "Limites de parcelles adjacentes"},
    "MITOYEN": {"color": 1, "category": "LIMITES", "desc": "Limites mitoyennes"},
    "bornes": {"color": 3, "category": "BORNES", "desc": "Symboles de bornes du périmètre"},
    "BRNS": {"color": 3, "category": "BORNES", "desc": "Noms de bornes et traits de rappel"},
    "points": {"color": 3, "category": "BORNES", "desc": "Points topographiques levés"},
    "P_B_T": {"color": 3, "category": "BORNES", "desc": "Points de bornage terrain"},
    "P_M_T": {"color": 3, "category": "BORNES", "desc": "Points de marquage terrain"},
    "riverain": {"color": 2, "category": "ANNOTATION", "desc": "Textes des propriétaires et parcelles riveraines"},
    "titres": {"color": 2, "category": "ANNOTATION", "desc": "Numéros de titres fonciers adjacents"},
    "plle": {"color": 2, "category": "ANNOTATION", "desc": "Numéros des parcelles voisines"},
    "etiquette": {"color": 7, "category": "ANNOTATION", "desc": "Étiquettes de contenance et numéros"},
    "annotation": {"color": 7, "category": "ANNOTATION", "desc": "Annotations diverses"},
    "routes": {"color": 4, "category": "VOIRIE", "desc": "Routes goudronnées et voies aménagées"},
    "piste": {"color": 4, "category": "VOIRIE", "desc": "Pistes carrossables et voies tertiaires"},
    "passages": {"color": 4, "category": "VOIRIE", "desc": "Passages piétons et servitudes de passage"},
    "chemin": {"color": 4, "category": "VOIRIE", "desc": "Chemins ruraux"},
    "sentier": {"color": 4, "category": "VOIRIE", "desc": "Sentiers muletiers"},
    "Chaaba": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Chaâba (ravin d'écoulement d'eau)"},
    "oued": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Oued / Cours d'eau principal"},
    "seguia": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Séguia d'irrigation"},
    "source": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Source d'eau naturelle"},
    "puits": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Puits d'eau"},
    "bassin": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Bassin de rétention d'eau"},
    "merja": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Merja / Zone marécageuse"},
    "canalisation": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Canalisations d'eau"},
    "masref": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Masref / Fossé de drainage"},
    "mtfia": {"color": 5, "category": "HYDROGRAPHIE", "desc": "Matfiya / Citerne souterraine"},
    "le_bt": {"color": 6, "category": "ELECTRICITE", "desc": "Ligne électrique Basse Tension"},
    "lemt": {"color": 6, "category": "ELECTRICITE", "desc": "Ligne électrique Moyenne Tension"},
    "leht": {"color": 6, "category": "ELECTRICITE", "desc": "Ligne électrique Haute Tension"},
    "LE": {"color": 6, "category": "ELECTRICITE", "desc": "Lignes électriques génériques"},
    "L.E": {"color": 6, "category": "ELECTRICITE", "desc": "Textes et polylignes L.E.B.T"},
    "P_TELEPHONIQUE": {"color": 6, "category": "RESEAUX", "desc": "Poteaux téléphoniques"},
    "mur": {"color": 8, "category": "CONSTRUCTIONS", "desc": "Murs en maçonnerie ou clôture bâtie"},
    "construction": {"color": 8, "category": "CONSTRUCTIONS", "desc": "Bâtiments, maisons, hangars"},
    "tazouta": {"color": 8, "category": "CONSTRUCTIONS", "desc": "Tazouta (abri traditionnel en pierres sèches)"},
    "metmoura": {"color": 8, "category": "CONSTRUCTIONS", "desc": "Matmoura / Silo souterrain"},
    "arbres": {"color": 3, "category": "VEGETATION", "desc": "Arbres isolés"},
    "vegetation": {"color": 3, "category": "VEGETATION", "desc": "Massifs végétaux et cultures"},
    "Plantation": {"color": 3, "category": "VEGETATION", "desc": "Plantations arborées organisées"},
    "CACTUS": {"color": 3, "category": "VEGETATION", "desc": "Haies de cactus / figuiers de barbarie"},
    "cimetiere": {"color": 9, "category": "DIVERS", "desc": "Cimetières et sépultures"},
    "dépression": {"color": 9, "category": "DIVERS", "desc": "Dépressions topographiques"},
    "REGARD": {"color": 9, "category": "RESEAUX", "desc": "Regards de visite techniques"},
    "FONTAINE": {"color": 9, "category": "DIVERS", "desc": "Fontaines publiques"},
    "CADRE": {"color": 7, "category": "CADRE", "desc": "Cadre extérieur du plan"},
    "CDRE": {"color": 7, "category": "CADRE", "desc": "Cadre intérieur de découpage"},
    "CADREP1": {"color": 7, "category": "CADRE", "desc": "Cadre de planche format P1"},
    "HABILLAGE": {"color": 7, "category": "CARTOUCHE", "desc": "Cartouche, titres administratifs et échelle"},
    "legende": {"color": 7, "category": "CARTOUCHE", "desc": "Légende des signes conventionnels"},
    "T-LEGENDE": {"color": 7, "category": "CARTOUCHE", "desc": "Textes descriptifs de la légende"},
    "BLKS": {"color": 7, "category": "BLOCS", "desc": "Entités blocs insérées"},
    "PRID": {"color": 7, "category": "DIVERS", "desc": "Périmètres d'irrigation et d'aménagement"},
    "DTLS": {"color": 7, "category": "DETAILS", "desc": "Détails topographiques généraux"},
    "TopoAlt": {"color": 7, "category": "TOPO", "desc": "Altimétrie / Cotes de nivellement"},
    "TopoMat": {"color": 7, "category": "TOPO", "desc": "Matérialisation des stations polygonales"},
    "CUL": {"color": 7, "category": "AGRICOLE", "desc": "Cultures annuelles"},
    "CCAL": {"color": 7, "category": "DIVERS", "desc": "Chantiers et carrières"},
    "pe": {"color": 6, "category": "ELECTRICITE", "desc": "Poteaux électriques"},
    "opposition": {"color": 1, "category": "JURIDIQUE", "desc": "Zones frappées d'opposition juridique"},
    "Defpoints": {"color": 7, "category": "SYSTEME", "desc": "Calque système de cotation non imprimable"}
}


def get_block_exact_obb(insert_entity, scale_factor: float = 1.0) -> Optional[OrientedBox2D]:
    """
    Calcule l'OBB (Oriented Bounding Box) exacte d'une instance de bloc AutoCAD (INSERT)
    à partir de ses dimensions d'origine dans MASTER_ANCFCC_TEMPLATE.dxf.
    """
    bname = getattr(insert_entity.dxf, "name", "").strip()
    if bname not in ANCFCC_MASTER_BLOCKS_DIMENSIONS:
        return None

    info = ANCFCC_MASTER_BLOCKS_DIMENSIONS[bname]
    ins_x = float(insert_entity.dxf.insert.x)
    ins_y = float(insert_entity.dxf.insert.y)
    rot_deg = float(getattr(insert_entity.dxf, "rotation", 0.0))
    rot_rad = math.radians(rot_deg)

    xs = float(getattr(insert_entity.dxf, "xscale", 1.0)) * scale_factor
    ys = float(getattr(insert_entity.dxf, "yscale", 1.0)) * scale_factor

    xmin, ymin, xmax, ymax = info["bbox"]
    w = (xmax - xmin) * xs
    h = (ymax - ymin) * ys
    cx_loc = (xmin + xmax) * 0.5 * xs
    cy_loc = (ymin + ymax) * 0.5 * ys

    # Rotation du centre local
    cos_r, sin_r = math.cos(rot_rad), math.sin(rot_rad)
    cx_world = ins_x + (cx_loc * cos_r - cy_loc * sin_r)
    cy_world = ins_y + (cx_loc * sin_r + cy_loc * cos_r)

    return OrientedBox2D(
        center=np.array([cx_world, cy_world], dtype=np.float64),
        half_extents=np.array([w * 0.5, h * 0.5], dtype=np.float64),
        angle_rad=rot_rad
    )

