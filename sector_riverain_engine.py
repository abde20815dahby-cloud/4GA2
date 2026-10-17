# -*- coding: utf-8 -*-
"""
engines/sector_riverain_engine.py — Moteur Cadastral de Résolution des Riverains par Secteurs (SS)
==================================================================================================
Utilise les plans parcellaires d'ensemble par secteur (PLAN-TOT-SECTEUR dans C:\\Users\\Admin\\Downloads\\SS)
comme base de référence spatiale maîtresse (Marji3) pour :
1. Découvrir et positionner automatiquement les riverains réels (JIRAN) le long de chaque limite :
   - Parcelles voisines (Plle xxxx TC) et leurs limites rayonnantes sur calque 'parcelle'
   - Titres Fonciers et Réquisitions (T.xxxx/xx TC ou R.xxxx/xx TC)
   - Châaba (Chaâbat sur calque 0)
   - Pistes et Voies publiques (piste de 10 m / Route goudronnée de 15m)
   - Propriétaires nommés du calque riverain (Nom Prénom sur calque 0)
   - Bornes de rattachement extérieures (Bxxxx) et verbornes (v.Bxxxx)
2. Corriger et garantir le numéro de parcelle interne conforme (Plle {id} TC) et éliminer les faux labels (ex: Plle 255 TC).
"""
import os
import sys
import math
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
import shapely.geometry as sg
from shapely.ops import nearest_points, linemerge
import numpy as np
from scipy.spatial import cKDTree

try:
    import parcel_core as core
except ImportError:
    from engines import parcel_core as core

try:
    from constants import ANCFCC_STANDARDS
except ImportError:
    from engines.dxf_cleaner.constants import ANCFCC_STANDARDS

logger = logging.getLogger("sector_riverain_engine")


class SectorDatabase:
    """
    Base spatiale maîtresse chargeant et indexant les plans d'ensemble des Sous-Zones (SZ1 à SZ5).
    Supporte :
    1. Un dossier complet contenant les DXF des sous-zones (ex: C:\\Users\\Admin\\Downloads\\SS)
    2. Un fichier DXF unique d'une Sous-Zone spécifique (ex: PLAN-TOT-SECTEUR_...dxf)
    3. Rechargement dynamique (force_refresh) lors de modifications récentes du plan maître
    4. Filtrage par Sous-Zone (SZ1, SZ2, SZ3, SZ4, SZ5 ou Auto)
    """
    def __init__(
        self,
        sectors_dir: str = r"C:\Users\Admin\Downloads\SS",
        force_refresh: bool = False,
        selected_sz: Optional[str] = None
    ):
        self.sectors_dir = sectors_dir
        self.sectors_path = sectors_dir
        self.force_refresh = force_refresh
        self.selected_sz = selected_sz
        self.sectors_data: List[Dict[str, Any]] = []
        self._load_all_sectors()

    def _load_all_sectors(self):
        if not os.path.exists(self.sectors_dir):
            logger.warning("Chemin du référentiel Sous-Zone introuvable: %s", self.sectors_dir)
            return

        is_file = os.path.isfile(self.sectors_dir)
        base_dir = os.path.dirname(os.path.abspath(self.sectors_dir)) if is_file else self.sectors_dir

        cache_path = os.path.join(base_dir, ".sector_cache.pkl") if not is_file else None

        # Vérification du cache disque si non forcé
        if not self.force_refresh and cache_path and os.path.exists(cache_path):
            try:
                cache_mtime = os.path.getmtime(cache_path)
                dxf_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.lower().endswith(".dxf")]
                any_newer = any(os.path.getmtime(df) > cache_mtime for df in dxf_files)
                if not any_newer:
                    import pickle
                    with open(cache_path, "rb") as f:
                        all_data = pickle.load(f)
                    self.sectors_data = self._filter_by_sz(all_data, self.selected_sz)
                    logger.info("⚡ [CACHE HIT] Secteurs cadastraux chargés depuis le cache disque (%d secteur(s)).", len(self.sectors_data))
                    return
                else:
                    logger.info("🔄 [PLAN MODIFIÉ] Fichiers DXF du secteur plus récents que le cache. Re-synchronisation...")
            except Exception as ex:
                logger.warning("Cache des secteurs illisible, ré-indexation: %s", ex)

        # Indexation des fichiers
        if is_file:
            files_to_load = [self.sectors_dir]
        else:
            files_to_load = [os.path.join(base_dir, f) for f in sorted(os.listdir(base_dir)) if f.lower().endswith(".dxf")]

        logger.info("Chargement et indexation de %d plan(s) maître(s) Sous-Zone...", len(files_to_load))

        indexed = []
        for idx, fpath in enumerate(files_to_load, start=1):
            fn = os.path.basename(fpath)
            sz_tag = f"SZ{idx}"
            m_sz = re.search(r'SZ[_\s-]*(\d+)|Sous[_\s-]*Zone[_\s-]*(\d+)', fn, re.IGNORECASE)
            if m_sz:
                num = m_sz.group(1) or m_sz.group(2)
                sz_tag = f"SZ{int(num)}"

            try:
                doc = core.load_dxf(fpath)
                msp = doc.modelspace()
                s_data = self._index_single_sector(msp, fn)
                s_data["sz_tag"] = sz_tag
                s_data["filepath"] = fpath
                indexed.append(s_data)
                logger.info(
                    "[%s] Plan maître %s indexé : %d polygones parcelles, %d titres, %d chaabas, %d riverains.",
                    sz_tag, fn, len(s_data["parcel_polygons"]), len(s_data["titres"]),
                    len(s_data["chaabas"]), len(s_data["riverains"])
                )
            except Exception as ex:
                logger.error("Erreur lors du chargement du secteur %s: %s", fn, ex)

        if not is_file and indexed and cache_path:
            try:
                import pickle
                with open(cache_path, "wb") as f:
                    pickle.dump(indexed, f)
                logger.info("💾 Cache des secteurs cadastraux sauvegardé avec succès.")
            except Exception as ex:
                logger.warning("Impossible de sauvegarder le cache des secteurs: %s", ex)

        self.sectors_data = self._filter_by_sz(indexed, self.selected_sz)

    def _filter_by_sz(self, sectors: List[Dict[str, Any]], selected_sz: Optional[str]) -> List[Dict[str, Any]]:
        if not selected_sz or selected_sz.upper() in ("AUTO", "TOUT", "ALL", ""):
            return sectors
        clean_target = selected_sz.upper().replace(" ", "").replace("_", "").replace("-", "")
        m = re.search(r'(\d+)', clean_target)
        if m:
            target_num = m.group(1)
            filtered = [s for s in sectors if target_num in s.get("sz_tag", "") or f"SZ{target_num}" in s.get("sz_tag", "")]
            if filtered:
                return filtered
        return sectors

    def _index_single_sector(self, msp, filename: str) -> Dict[str, Any]:
        raw_polys = []
        raw_labels = []
        titres = []
        chaabas = []
        oueds = []
        pistes = []
        routes = []
        riverains = []
        bornes = []
        borne_blocks = {}
        limit_lines = []
        minx, miny, maxx, maxy = 1e9, 1e9, -1e9, -1e9

        for e in msp:
            t = e.dxftype()
            lay = e.dxf.layer.lower()

            if t in ("TEXT", "MTEXT"):
                try:
                    txt = core.get_text_content(e).strip()
                    if not txt:
                        continue
                    tx, ty = core.get_insert(e)
                    pt = sg.Point(tx, ty)

                    minx = min(minx, tx)
                    miny = min(miny, ty)
                    maxx = max(maxx, tx)
                    maxy = max(maxy, ty)

                    if "titre" in lay:
                        titres.append((pt, txt))
                    elif "chaaba" in lay:
                        chaabas.append((pt, txt))
                    elif "oued" in lay:
                        oueds.append((pt, txt))
                    elif "riverain" in lay:
                        riverains.append((pt, txt))
                    elif "piste" in lay:
                        pistes.append((pt, txt))
                    elif "route" in lay:
                        routes.append((pt, txt))
                    elif "plle" in lay or "parcelle" in lay:
                        raw_labels.append((pt, txt))
                    else:
                        clean_b = txt.replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                        if "borne" in lay or (clean_b.startswith("B") and len(clean_b) > 1 and clean_b[1:].isdigit()):
                            bornes.append((pt, clean_b))
                except Exception:
                    pass

            elif t == "INSERT":
                try:
                    bname = getattr(e.dxf, "name", "")
                    if "borne" in bname.lower():
                        ins = getattr(e.dxf, "insert", None)
                        if ins:
                            borne_blocks[(round(ins.x, 2), round(ins.y, 2))] = bname
                except Exception:
                    pass

            elif t in ("LINE", "LWPOLYLINE"):
                pts = []
                is_closed = False
                if t == "LWPOLYLINE":
                    pts = list(e.get_points())
                    is_closed = getattr(e, "is_closed", False)
                else:
                    pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]

                if len(pts) >= 2:
                    coords = [(p[0], p[1]) for p in pts]
                    ls = sg.LineString(coords)
                    limit_lines.append((ls, lay, coords))
                    bx = ls.bounds
                    minx = min(minx, bx[0])
                    miny = min(miny, bx[1])
                    maxx = max(maxx, bx[2])
                    maxy = max(maxy, bx[3])

                    if "chaaba" in lay:
                        chaabas.append((ls, "Chaâbat"))
                    elif "oued" in lay:
                        oueds.append((ls, "Oued"))
                    elif "piste" in lay:
                        pistes.append((ls, "piste de 8 m"))
                    elif "route" in lay:
                        routes.append((ls, "Route goudronnée de 15m"))

                    if is_closed and len(pts) >= 3 and lay in ("plle", "parcelle"):
                        try:
                            poly = sg.Polygon(coords)
                            if poly.is_valid and poly.area > 5.0:
                                raw_polys.append(poly)
                        except Exception:
                            pass

        # Apparier les polygones parcellaires avec leurs labels de numéro
        parcel_polygons: List[Tuple[sg.Polygon, str]] = []
        for poly in raw_polys:
            matched_lbl = None
            min_dist = 1e9
            for lpt, ltxt in raw_labels:
                if poly.contains(lpt):
                    matched_lbl = ltxt
                    break
                d = poly.distance(lpt)
                if d < min_dist and d < 45.0:
                    min_dist = d
                    matched_lbl = ltxt
            if matched_lbl:
                clean_pid = re.sub(r'^[Pp]lle|^[Pp]', '', matched_lbl.strip()).strip()
                parcel_polygons.append((poly, clean_pid))

        bbox = sg.box(minx, miny, maxx, maxy) if minx < maxx else sg.box(0, 0, 0, 0)
        return {
            "filename": filename,
            "bbox": bbox,
            "parcel_polygons": parcel_polygons,
            "titres": titres,
            "chaabas": chaabas,
            "oueds": oueds,
            "pistes": pistes,
            "routes": routes,
            "riverains": riverains,
            "bornes": bornes,
            "borne_blocks": borne_blocks,
            "limit_lines": limit_lines,
        }

    def get_sector_for_parcel(self, parcel_poly: sg.Polygon, pid: str = "") -> Optional[Dict[str, Any]]:
        """Trouve le secteur contenant la parcelle."""
        if not parcel_poly or parcel_poly.is_empty:
            return None
        c = parcel_poly.centroid

        candidates = [s for s in self.sectors_data if s["bbox"].contains(c)]
        if not candidates:
            candidates = self.sectors_data
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        if pid:
            base_p = re.sub(r'[-_]plan.*$', '', os.path.basename(pid), flags=re.IGNORECASE)
            clean_target = re.sub(r'^[Pp]lle|^[Pp]', '', base_p).strip()
            for s in candidates:
                for poly, spid in s["parcel_polygons"]:
                    if spid == clean_target and poly.distance(c) < 80.0:
                        return s

        best_sec = candidates[0]
        max_nearby = -1
        buf = parcel_poly.buffer(60.0)
        for s in candidates:
            cnt = sum(1 for poly, _ in s["parcel_polygons"] if buf.intersects(poly))
            if cnt > max_nearby:
                max_nearby = cnt
                best_sec = s

        return best_sec

class SectorTopologyLinker:
    r"""
    AGENT 7 — SectorTopologyLinker :
    Ne traite jamais une parcelle en aveugle si les données du secteur (polygones voisins,
    liste des riverains, bornes de rattachement) sont disponibles.
    - Par intersection spatiale (STRtree), détecte automatiquement les parcelles adjacentes
      (ex: Plle 15\PTC) et les noms des propriétaires riverains (ex: Héritiers ...) et génère
      leurs MTEXT parallèles à chaque limite.
    - Détecte les segments de limites externes qui partent des bornes du périmètre vers des bornes
      voisines hors-cadre et génère automatiquement les amorces directionnelles v.B...
      (style CQ_VRBN, hauteur 3.6m à 1/2000) ainsi que les bornes rattachées entre parenthèses (B...).
    """
    def __init__(self, sector_db: Optional[SectorDatabase] = None):
        self.sector_db = sector_db

    def link_parcel_to_sector(
        self,
        msp,
        parcel_poly: sg.Polygon,
        pid: str = "",
        scale: float = 1000.0,
        frame_poly: Optional[sg.Polygon] = None,
        sector_db: Optional[SectorDatabase] = None,
    ) -> int:
        db = sector_db or self.sector_db
        if db is None:
            try:
                db = get_sector_database()
            except Exception:
                db = None
        if db is None:
            return 0
        return auto_populate_missing_riverains(
            msp=msp,
            parcel_poly=parcel_poly,
            sector_db=db,
            pid=pid,
            scale=scale,
            frame_poly=frame_poly,
        )


def get_facing_segment_tangent(parcel_poly: sg.Polygon, pt: sg.Point) -> Tuple[float, float]:
    """
    Calcule le vecteur directeur unitaire (ux, uy) du segment de limite de la parcelle
    le plus proche du point pt.
    Normalisé pour être lisible de gauche à droite / de bas en haut (Bottom/Right convention ANCFCC).
    """
    if not parcel_poly or parcel_poly.is_empty:
        return (1.0, 0.0)
    coords = list(parcel_poly.exterior.coords)
    best_seg = None
    min_d = 1e9
    for i in range(len(coords) - 1):
        seg = sg.LineString([coords[i], coords[i+1]])
        d = seg.distance(pt)
        if d < min_d:
            min_d = d
            best_seg = seg
    if not best_seg:
        return (1.0, 0.0)
    p0, p1 = best_seg.coords[0], best_seg.coords[1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    # Normalisation pour lisibilité (bottom/right): ux >= 0 ou si vertical uy > 0
    if ux < 0 or (abs(ux) < 1e-5 and uy < 0):
        ux, uy = -ux, -uy
    return (ux, uy)


def auto_populate_missing_riverains(
    msp,
    parcel_poly: sg.Polygon,
    sector_db: SectorDatabase,
    pid: str = "",
    scale: float = 1000.0,
    frame_poly: Optional[sg.Polygon] = None
) -> int:
    r"""
    Résolution complète et automatique des riverains cadastraux conforme ANCFCC (100% DEF) :
    1. Corrige le label interne de la parcelle (Plle {id} TC) et purge tout faux numéro.
    2. Identifie toutes les parcelles voisines dans le secteur dans un rayon de 25m :
       - Insère le texte riverain Plle {nbr} TC sur calque 0 orienté vers l'extérieur.
       - Insère les limites de parcelles et de voirie/chaaba sur calque 'DTLS'.
    3. Résout les propriétaires riverains nommés (calque 'riverain') avec expansion Héritiers et \PTC.
    4. Résout les titres fonciers (calque 'titres').
    5. Insère les éléments linéaires uniques (Chaâbat, piste de 8 m, Route goudronnée de 15m).
    6. Insère les bornes de rattachement extérieures (Bxxxx / (Bxxxx)) avec leurs blocs ANCFCC réels.
    7. Calcule et insère les verbornes (v.Bxxxx) aux extrémités des lignes rayonnantes.
    """
    if not parcel_poly or parcel_poly.is_empty:
        return 0

    if frame_poly is None:
        frame_poly = core.get_frame_polygon(msp)

    base_p = re.sub(r'[-_]plan.*$', '', os.path.basename(pid), flags=re.IGNORECASE)
    clean_pid = re.sub(r'^[Pp]lle|^[Pp]', '', base_p).strip()
    sec = sector_db.get_sector_for_parcel(parcel_poly, pid=pid)
    if not sec:
        return 0

    added_count = 0
    text_height = 2.0 * (scale / 1000.0)
    borne_text_height = 1.75 * (scale / 1000.0)

    # Identifier les vrais voisins du secteur (< 25m)
    # Règle Cadastrale ANCFCC: Si une voie publique (piste, route) sépare deux parcelles,
    # la voie publique est la limite riveraine. La parcelle d'en face n'est pas un voisin direct.
    road_geoms = [ls for ls, lay, _ in sec.get("limit_lines", []) if lay.lower() in ("piste", "route", "oued")]
    valid_neighbor_pids = set()
    nearby_neighbors = []
    for p_nbr, nbr_pid in sec.get("parcel_polygons", []):
        if nbr_pid != clean_pid:
            d = parcel_poly.distance(p_nbr)
            if d <= 25.0:
                pt_poly, pt_nbr = nearest_points(parcel_poly, p_nbr)
                conn_seg = sg.LineString([pt_poly, pt_nbr])
                # Si la séparation est supérieure à 3.0m et qu'une voirie traverse le segment de liaison:
                is_separated_by_road = False
                if d >= 3.0:
                    for rg in road_geoms:
                        if conn_seg.intersects(rg.buffer(0.5)):
                            is_separated_by_road = True
                            break
                if not is_separated_by_road:
                    valid_neighbor_pids.add(nbr_pid)
                    nearby_neighbors.append((p_nbr, nbr_pid, d))

    # 1. Nettoyer les faux labels internes uniquement (ne jamais détruire les riverains réels)
    for e in list(msp):
        if e.dxftype() in ("TEXT", "MTEXT"):
            try:
                t = core.get_text_content(e).strip()
                tx, ty = core.get_insert(e)
                pt = sg.Point(tx, ty)
                if re.search(r"\b\d+\b", t) and ("plle" in t.lower() or "p" in t.lower()):
                    found_nums = re.findall(r"\b\d+\b", t)
                    if parcel_poly.buffer(3.0).contains(pt):
                        if clean_pid and clean_pid not in found_nums:
                            e.destroy()
            except Exception:
                pass

    # Vérifier si TC ou Plle {clean_pid} TC existe à l'intérieur
    has_internal_pid = False
    for e in msp:
        if e.dxftype() in ("TEXT", "MTEXT"):
            try:
                t = core.get_text_content(e).strip()
                tx, ty = core.get_insert(e)
                if parcel_poly.contains(sg.Point(tx, ty)):
                    if "tc" in t.lower() or (clean_pid and clean_pid in t):
                        has_internal_pid = True
                        break
            except Exception:
                pass

    if not has_internal_pid and clean_pid:
        c = parcel_poly.centroid
        msp.add_mtext(
            f"Plle {clean_pid}\\PTC",
            dxfattribs={"layer": "0", "char_height": text_height, "insert": (c.x, c.y, 0.0)}
        )
        added_count += 1

    # 2. Relever les textes et bornes existants
    existing_texts = []
    existing_borne_names: Set[str] = set()
    for e in list(msp):
        if e.dxftype() in ("TEXT", "MTEXT"):
            try:
                raw_t = getattr(e, "text", getattr(e.dxf, "text", ""))
                if re.search(r"^(Jbel|Oued|Chaaba|Chaâba|Châaba|Forêt|Foret)", raw_t, re.IGNORECASE):
                    t = re.sub(r"(\\+P|\n|\\)?\s*T\.?C\.?$", "", raw_t, flags=re.IGNORECASE).strip()
                    if hasattr(e, "text"):
                        e.text = t
                    if hasattr(e.dxf, "text"):
                        e.dxf.text = t
                else:
                    t = core.get_text_content(e).strip()
                existing_texts.append(t)
                bt = t.replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                if bt.startswith("B"):
                    existing_borne_names.add(bt)
            except Exception:
                pass

    # 3. Résolution des parcelles voisines (Jiran) dans le secteur (< 25m)
    neighbor_frontage_endpoints = []
    for p_nbr, nbr_pid, d in nearby_neighbors:
        lbl = f"Plle {nbr_pid}\\PTC"
        # Calculer la façade faisant face au voisin pour un placement centré et rigoureux
        facing_frontage = parcel_poly.boundary.intersection(p_nbr.buffer(max(12.0, d + 4.0)))
        if not facing_frontage.is_empty:
            f_mid = facing_frontage.centroid
            pt_poly, pt_nbr = nearest_points(f_mid, p_nbr)
        else:
            pt_poly, pt_nbr = nearest_points(parcel_poly, p_nbr)

        dx = pt_nbr.x - pt_poly.x
        dy = pt_nbr.y - pt_poly.y
        L = math.hypot(dx, dy)
        if L > 0.01:
            ux, uy = dx / L, dy / L
        else:
            c_nbr = p_nbr.centroid
            c_p = parcel_poly.centroid
            dx2, dy2 = c_nbr.x - c_p.x, c_nbr.y - c_p.y
            L2 = math.hypot(dx2, dy2) or 1.0
            ux, uy = dx2 / L2, dy2 / L2

        # Profondeur dans la parcelle voisine (au moins 28m pour dégager la zone de berge/voirie)
        offset = max(28.0, min(45.0, d + 26.0))
        lx = pt_poly.x + ux * offset
        ly = pt_poly.y + uy * offset
        if frame_poly and not frame_poly.buffer(-4.0).contains(sg.Point(lx, ly)):
            np_in, _ = nearest_points(frame_poly.buffer(-4.0), sg.Point(lx, ly))
            lx, ly = np_in.x, np_in.y

        tan_ux, tan_uy = get_facing_segment_tangent(parcel_poly, sg.Point(lx, ly))

        # Vérifier si un label pour ce voisin existe déjà
        matched_ent = None
        for e in list(msp):
            if e.dxftype() in ("TEXT", "MTEXT"):
                t_chk = core.get_text_content(e).strip()
                if (f"plle {nbr_pid}" in t_chk.lower() or f"p{nbr_pid}" in t_chk.lower()) and nbr_pid != clean_pid:
                    matched_ent = e
                    break

        if matched_ent:
            # Repositionner et orienter parallèlement à la limite cadastrale avec garde-fou
            ex, ey = core.get_insert(matched_ent)
            disp = math.hypot(ex - lx, ey - ly)
            current_facade_dist = parcel_poly.exterior.distance(sg.Point(ex, ey)) if parcel_poly else 0.0
            is_near_borne = any(math.hypot(ex - bx, ey - by) < 8.0 for bx, by in sec.get("borne_blocks", {}).keys())
            if (current_facade_dist > 40.0 and disp < 80.0) or (is_near_borne and disp < 80.0):
                matched_ent.dxf.insert = (lx, ly, 0.0)
            if hasattr(matched_ent.dxf, "text_direction"):
                matched_ent.dxf.text_direction = (tan_ux, tan_uy, 0.0)
            if hasattr(matched_ent.dxf, "rotation"):
                matched_ent.dxf.rotation = 0.0
            logger.info("🔄 Voisin repositionné et orienté sur façade : Plle %s TC à (%.1f, %.1f) dir=(%.3f, %.3f)", nbr_pid, lx, ly, tan_ux, tan_uy)
        else:
            msp.add_mtext(
                lbl,
                dxfattribs={
                    "layer": "0",
                    "char_height": text_height,
                    "insert": (lx, ly, 0.0),
                    "text_direction": (tan_ux, tan_uy, 0.0),
                }
            )
            existing_texts.append(lbl)
            added_count += 1
            logger.info("➕ Voisin ajouté orienté : Plle %s TC à (%.1f, %.1f) dir=(%.3f, %.3f)", nbr_pid, lx, ly, tan_ux, tan_uy)

    # 3b. Calage et repositionnement des riverains nominatifs réels (Héritiers..., Jbel...) depuis le secteur
    for r_pt, r_txt in sec.get("riverains", []):
        d_r = parcel_poly.distance(r_pt)
        if 0.5 < d_r <= 45.0 and not parcel_poly.contains(r_pt):
            norm_r = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\u0600-\u06FF]", " ", r_txt)).strip().lower()
            if len(norm_r) < 3:
                continue
            matched_r_ent = None
            for e in list(msp):
                if e.dxftype() in ("TEXT", "MTEXT"):
                    e_t = core.get_text_content(e).strip()
                    norm_e = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\u0600-\u06FF]", " ", e_t)).strip().lower()
                    if norm_r in norm_e or norm_e in norm_r or (len(norm_r) > 10 and len(norm_e) > 10 and (norm_r[:15] == norm_e[:15])):
                        matched_r_ent = e
                        break
            if matched_r_ent:
                ex, ey = core.get_insert(matched_r_ent)
                is_near_borne = any(math.hypot(ex - bx, ey - by) < 8.0 for bx, by in sec.get("borne_blocks", {}).keys())
                d_lim = parcel_poly.exterior.distance(sg.Point(ex, ey))
                target_x, target_y = float(r_pt.x), float(r_pt.y)
                pt_sec = sg.Point(r_pt.x, r_pt.y)
                is_outside_frame = frame_poly and not frame_poly.buffer(-5.0).contains(pt_sec)
                if is_outside_frame or is_near_borne or d_lim < 3.0 or math.hypot(ex - r_pt.x, ey - r_pt.y) > 5.0:
                    if is_outside_frame or is_near_borne:
                        coords = list(parcel_poly.exterior.coords)
                        best_seg = None
                        min_sd = 999.0
                        for si in range(len(coords) - 1):
                            s_line = sg.LineString([coords[si], coords[si+1]])
                            sd = s_line.distance(pt_sec)
                            if sd < min_sd:
                                min_sd = sd
                                best_seg = (coords[si], coords[si+1], s_line)
                        if best_seg and best_seg[2].length >= 10.0:
                            p1, p2, s_line = best_seg
                            h_t = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018) * scale
                            offset_d = max(8.0, 2.5 * h_t)
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            L_s = math.hypot(dx, dy) or 1.0
                            nx, ny = -dy / L_s, dx / L_s
                            for frac in [0.5, 0.6, 0.4, 0.7, 0.3]:
                                pt_s = s_line.interpolate(frac, normalized=True)
                                c1 = sg.Point(pt_s.x + nx * offset_d, pt_s.y + ny * offset_d)
                                c2 = sg.Point(pt_s.x - nx * offset_d, pt_s.y - ny * offset_d)
                                cand = c1 if not parcel_poly.contains(c1) else c2
                                if not frame_poly or frame_poly.buffer(-5.0).contains(cand):
                                    target_x, target_y = cand.x, cand.y
                                    break
                    disp_r = math.hypot(target_x - ex, target_y - ey)
                    if disp_r <= 40.0:
                        matched_r_ent.dxf.insert = (target_x, target_y, 0.0)
                        tan_ux, tan_uy = get_facing_segment_tangent(parcel_poly, sg.Point(target_x, target_y))
                        if hasattr(matched_r_ent.dxf, "text_direction"):
                            matched_r_ent.dxf.text_direction = (tan_ux, tan_uy, 0.0)
                        if hasattr(matched_r_ent.dxf, "rotation"):
                            matched_r_ent.dxf.rotation = 0.0
                        logger.info("🔄 Riverain nominatif recalé sur façade : '%s' à (%.2f, %.2f) dir=(%.3f, %.3f)", r_txt, target_x, target_y, tan_ux, tan_uy)
                continue

    # 3c. Déconfliction des riverains et titres trop proches des bornes
    all_borne_pts = [
        (float(bx), float(by)) for bx, by in sec.get("borne_blocks", {}).keys()
    ]
    for ins in msp.query("INSERT"):
        all_borne_pts.append((ins.dxf.insert.x, ins.dxf.insert.y))
    for t in msp.query("TEXT MTEXT"):
        txt = getattr(t.dxf, "text", "").strip()
        if txt.startswith("B") or txt.startswith("(B"):
            all_borne_pts.append((t.dxf.insert.x, t.dxf.insert.y))
    b_kdtree = cKDTree(all_borne_pts) if all_borne_pts else None

    if b_kdtree is not None and parcel_poly and parcel_poly.is_valid:
        h_t = ANCFCC_STANDARDS.get("borne_text_height_ratio", 0.0018) * scale
        for e in list(msp):
            if e.dxftype() in ("TEXT", "MTEXT"):
                txt = core.get_text_content(e).strip()
                tl = txt.lower()
                if any(k in tl for k in ("titre", "t.", "t1", "t2", "t3", "r.", "req", "plle", "p.", "hrts", "héritiers", "heritiers", "jbel", "oued", "foret", "forêt")):
                    ex, ey = core.get_insert(e)
                    if not parcel_poly.contains(sg.Point(ex, ey)):
                        db, _ = b_kdtree.query([ex, ey])
                        if db < 4.5 * h_t:
                            coords = list(parcel_poly.exterior.coords)
                            best_cand = None
                            best_db = db
                            for i in range(len(coords) - 1):
                                seg = sg.LineString([coords[i], coords[i+1]])
                                if seg.distance(sg.Point(ex, ey)) < 35.0:
                                    for frac in [0.2, 0.35, 0.5, 0.65, 0.8]:
                                        pt_seg = seg.interpolate(frac, normalized=True)
                                        dx = coords[i+1][0] - coords[i][0]
                                        dy = coords[i+1][1] - coords[i][1]
                                        L_s = math.hypot(dx, dy) or 1.0
                                        nx, ny = -dy / L_s, dx / L_s
                                        cand_x = pt_seg.x + nx * (4.0 * h_t)
                                        cand_y = pt_seg.y + ny * (4.0 * h_t)
                                        if parcel_poly.contains(sg.Point(cand_x, cand_y)):
                                            cand_x = pt_seg.x - nx * (4.0 * h_t)
                                            cand_y = pt_seg.y - ny * (4.0 * h_t)
                                        db_cand, _ = b_kdtree.query([cand_x, cand_y])
                                        if db_cand > best_db:
                                            best_db = db_cand
                                            best_cand = (cand_x, cand_y)
                            if best_cand and best_db >= 4.0 * h_t:
                                disp_c = math.hypot(best_cand[0] - ex, best_cand[1] - ey)
                                if disp_c <= 25.0:
                                    e.dxf.insert = (float(best_cand[0]), float(best_cand[1]), 0.0)
                                    if hasattr(e.dxf, "rotation"):
                                        e.dxf.rotation = 0.0
                                    logger.info("🔄 Texte riverain déconflicté des bornes : '%s' déplacé à (%.2f, %.2f) d_borne=%.1fm", txt, best_cand[0], best_cand[1], best_db)

    # 4. Import des polylignes limites du secteur (chaaba, oued, piste, route, l.e) sur DTLS/L.E
    # Règle d'or ANCFCC: Ne JAMAIS importer de polygone fermé de voisin (ring) sur DTLS
    # Directive Cadre : Les lignes ne doivent JAMAIS toucher ou dépasser le cadre (marge stricte >= 8m).
    inner_frame = None
    if frame_poly:
        frame_margin = max(8.0, 0.004 * scale)
        inner_frame = frame_poly.buffer(-frame_margin)
        if inner_frame.geom_type == "MultiPolygon":
            inner_frame = max(inner_frame.geoms, key=lambda g: g.area)

        for ls, lay, coords in sec.get("limit_lines", []):
            lay_l = lay.lower()
            if lay_l in ("chaaba", "oued", "piste", "route", "l.e", "riverain"):
                if ls.is_ring:
                    continue
                # Remplacement de toute entité L.E locale tronquée par la ligne complète du secteur
                if lay_l == "l.e":
                    for old_le in list(msp):
                        if old_le.dxftype() in ("LWPOLYLINE", "POLYLINE", "LINE") and getattr(old_le.dxf, "layer", "").upper() == "L.E":
                            msp.delete_entity(old_le)
                else:
                    # Si le plan local possède déjà ce détail linéaire, ne pas importer de doublon depuis le secteur
                    has_local = any(e.dxftype() in ("LWPOLYLINE", "POLYLINE", "LINE") and getattr(e.dxf, "layer", "").lower() == lay_l for e in msp)
                    if has_local:
                        continue
                # Ignorer les lignes quasi-identiques à la limite de la parcelle
                coords_on_bnd = sum(1 for p in coords if parcel_poly.boundary.distance(sg.Point(p)) < 0.2)
                if len(coords) > 2 and (coords_on_bnd / len(coords)) > 0.85:
                    continue
                d = parcel_poly.distance(ls)
                if d <= 25.0:
                    clipped_ls = ls.intersection(inner_frame)
                    if not clipped_ls.is_empty:
                        geoms = [clipped_ls] if isinstance(clipped_ls, sg.LineString) else getattr(clipped_ls, "geoms", [])
                        for g in geoms:
                            if isinstance(g, sg.LineString) and g.length >= 2.0:
                                try:
                                    target_lay = "L.E" if lay_l == "l.e" else "DTLS"
                                    g_coords = list(g.coords)
                                    msp.add_lwpolyline(g_coords, dxfattribs={"layer": target_lay})
                                    neighbor_frontage_endpoints.append((g_coords[0], g_coords[-1]))
                                    added_count += 1
                                except Exception:
                                    pass

    # 4b. Synthèse de la berge opposée de chaâba (7ODOD CH3BA DYAL JAR sur DTLS)
    # Règle ANCFCC: Ne synthétiser que si le plan contient déjà des détails DTLS du géomètre
    has_local_dtls = any(e.dxftype() in ("LWPOLYLINE", "POLYLINE", "LINE") and getattr(e.dxf, "layer", "").upper() == "DTLS" for e in msp)
    for ls_ch, ch_name in (sec.get("chaabas", []) if has_local_dtls else []):
        if ls_ch.length >= 20.0 and ls_ch.distance(parcel_poly) < 2.5:
            coords_parcel = list(parcel_poly.exterior.coords)[:-1]
            chain = [p for p in coords_parcel if ls_ch.distance(sg.Point(p)) < 2.5]
            if len(chain) >= 2:
                # Vérifier si une berge opposée existe déjà sur DTLS dans ce couloir (rayon 8m)
                has_opposite_bank = False
                for e_dtls in msp:
                    lay = getattr(e_dtls.dxf, "layer", "").upper()
                    if e_dtls.dxftype() in ("LWPOLYLINE", "POLYLINE") and (lay in ("DTLS", "CHAABA", "0", "DETAIL") or "CHA" in lay):
                        d_dtls = e_dtls.get_points(format="xy") if e_dtls.dxftype() == "LWPOLYLINE" else []
                        if len(d_dtls) >= 2:
                            ls_dtls = sg.LineString(d_dtls)
                            # Une vraie berge opposée doit longer la chaîne sur au moins 40% de ses sommets à une distance de 2 à 15m
                            matching_pts = sum(1 for p in chain if 2.0 <= ls_dtls.distance(sg.Point(p)) <= 15.0)
                            if matching_pts >= max(2, int(len(chain) * 0.40)):
                                has_opposite_bank = True
                                break
                if not has_opposite_bank:
                    sub_indices = [coords_parcel.index(p) for p in chain]
                    sub_indices.sort()
                    ordered_chain = [coords_parcel[i] for i in sub_indices]
                    
                    # Décalage normal extérieur AutoCAD 6.0m
                    n_c = len(ordered_chain)
                    segs = []
                    for idx in range(n_c - 1):
                        p0, p1 = ordered_chain[idx], ordered_chain[idx+1]
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        L_seg = math.hypot(dx, dy) or 1.0
                        nx, ny = -dy / L_seg, dx / L_seg
                        mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
                        if parcel_poly.contains(sg.Point(mx + nx * 0.5, my + ny * 0.5)):
                            nx, ny = -nx, -ny
                        s0 = (p0[0] + nx * 6.0, p0[1] + ny * 6.0)
                        s1 = (p1[0] + nx * 6.0, p1[1] + ny * 6.0)
                        segs.append((s0, s1))
                    out_bank = [segs[0][0]]
                    for idx in range(len(segs) - 1):
                        (s0, s1), (t0, t1) = segs[idx], segs[idx+1]
                        x1, y1 = s0; x2, y2 = s1
                        x3, y3 = t0; x4, y4 = t1
                        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                        if abs(denom) > 1e-6:
                            ix = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
                            iy = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
                            out_bank.append((ix, iy))
                        else:
                            out_bank.append(s1)
                    out_bank.append(segs[-1][1])
                    
                    if len(out_bank) >= 2:
                        ls_bank = sg.LineString(out_bank)
                        if inner_frame:
                            ls_bank = ls_bank.intersection(inner_frame)
                        if not ls_bank.is_empty:
                            b_geoms = [ls_bank] if isinstance(ls_bank, sg.LineString) else getattr(ls_bank, "geoms", [])
                            for bg in b_geoms:
                                if isinstance(bg, sg.LineString) and bg.length >= 5.0:
                                    msp.add_lwpolyline(list(bg.coords), dxfattribs={"layer": "DTLS"})
                                    added_count += 1
                                    logger.info("🌊 [BERGE OPPOSÉE CHAABA] Limite de chaâba voisin créée sur DTLS (Npts=%d, L=%.1fm)", len(bg.coords), bg.length)

    # 5. Segments de limite reliant directement un sommet de parcelle à une borne extérieure ou limite naturelle
    connecting_lines = []
    seen_conn_segs = set()
    ext_borne_pts = [sg.Point(float(bx), float(by)) for bx, by in sec.get("borne_blocks", {}).keys()]
    chaaba_geoms = [geom for geom, _ in sec.get("chaabas", [])]

    for ls, lay, coords in sec.get("limit_lines", []):
        if any(k in lay.lower() for k in ("construction", "mur", "batiment", "hangar")):
            continue
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i+1]
            pt0 = sg.Point(float(p0[0]), float(p0[1]))
            pt1 = sg.Point(float(p1[0]), float(p1[1]))
            d0 = parcel_poly.distance(pt0)
            d1 = parcel_poly.distance(pt1)
            touches = (d0 < 0.2 and 0.5 < d1 <= 20.0) or (d1 < 0.2 and 0.5 < d0 <= 20.0)
            if touches:
                out_pt = pt1 if d0 < 0.2 else pt0
                is_to_ext_borne = any(out_pt.distance(bpt) < 1.0 for bpt in ext_borne_pts)
                if is_to_ext_borne:
                    key_seg = (min((p0[0], p0[1]), (p1[0], p1[1])), max((p0[0], p0[1]), (p1[0], p1[1])))
                    key_seg_round = ((round(key_seg[0][0], 1), round(key_seg[0][1], 1)), (round(key_seg[1][0], 1), round(key_seg[1][1], 1)))
                    if key_seg_round not in seen_conn_segs:
                        seen_conn_segs.add(key_seg_round)
                        target_lay = "DTLS"
                        seg = sg.LineString([p0, p1])
                        try:
                            msp.add_line((p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0), dxfattribs={"layer": target_lay})
                            connecting_lines.append(seg)
                            added_count += 1
                        except Exception:
                            pass

    # 5b. Deuxième saut de limite reliant une borne externe rattachée à une autre borne frontière (ex: B500 -> B106)
    conn_endpoints = []
    for seg in connecting_lines:
        conn_endpoints.append(sg.Point(float(seg.coords[0][0]), float(seg.coords[0][1])))
        conn_endpoints.append(sg.Point(float(seg.coords[1][0]), float(seg.coords[1][1])))

    for ls, lay, coords in sec.get("limit_lines", []):
        for i in range(len(coords) - 1):
            p0, p1 = coords[i], coords[i+1]
            pt0 = sg.Point(float(p0[0]), float(p0[1]))
            pt1 = sg.Point(float(p1[0]), float(p1[1]))
            for cpt in conn_endpoints:
                d0 = cpt.distance(pt0)
                d1 = cpt.distance(pt1)
                touches_conn = (d0 < 0.5 and 0.5 < d1 <= 80.0) or (d1 < 0.5 and 0.5 < d0 <= 80.0)
                if touches_conn:
                    out_pt = pt1 if d0 < 0.5 else pt0
                    if inner_frame and not inner_frame.contains(out_pt):
                        continue
                    is_borne = any(out_pt.distance(bpt) < 4.0 for bpt, _ in sec.get("bornes", []))
                    if is_borne:
                        key_seg = (min((p0[0], p0[1]), (p1[0], p1[1])), max((p0[0], p0[1]), (p1[0], p1[1])))
                        key_seg_round = ((round(key_seg[0][0], 1), round(key_seg[0][1], 1)), (round(key_seg[1][0], 1), round(key_seg[1][1], 1)))
                        if key_seg_round not in seen_conn_segs:
                            seen_conn_segs.add(key_seg_round)
                            msp.add_line((p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0), dxfattribs={"layer": "DTLS"})
                            connecting_lines.append(sg.LineString([p0, p1]))
                            neighbor_frontage_endpoints.append((p0, p1))
                            added_count += 1

    # 6. Bornes extérieures de rattachement (Bxxxx / (Bxxxx)) avec leurs blocs réels
    borne_blocks_dict = sec.get("borne_blocks", {})
    for (bx, by), blk_name in borne_blocks_dict.items():
        bpt = sg.Point(float(bx), float(by))
        d = parcel_poly.distance(bpt)
        max_borne_d = min(25.0, max(15.0, 15.0 * (scale / 1000.0)))
        if 0.5 < d <= max_borne_d and not parcel_poly.contains(bpt):
            # Règle ANCFCC anti-collision : Si une borne externe non connectée est à moins de 10.0m
            # d'une borne déjà présente sur la limite de parcelle, ne pas l'importer (borne doublon/vis-à-vis)
            if any(math.hypot(bx - p[0], by - p[1]) < 10.0 for p in parcel_poly.exterior.coords):
                continue
            matched_bname = None
            for b_t_pt, b_t_name in sec.get("bornes", []):
                if math.hypot(b_t_pt.x - bx, b_t_pt.y - by) < 4.0:
                    matched_bname = b_t_name
                    break
            if not matched_bname:
                continue
            clean_b = matched_bname.replace("(", "").replace(")", "").replace("v.", "").strip().upper()

            # Est-ce que cette borne est connectée par une ligne à la parcelle ?
            is_connected = any(seg.distance(bpt) < 0.5 for seg in connecting_lines)
            if not is_connected:
                continue
            lbl = f"({clean_b})"

            # Vérifier si un bloc existe déjà pour cette borne
            has_blk = any(
                be.dxftype() == "INSERT" and math.hypot(be.dxf.insert.x - bx, be.dxf.insert.y - by) < 1.0
                for be in msp
            )
            if not has_blk:
                try:
                    msp.add_blockref(blk_name, insert=(bx, by, 0.0), dxfattribs={"layer": "bornes"})
                except Exception:
                    pass

            # Calcul de la position optimale de l'étiquette
            ins_x, ins_y = None, None
            c_frame = frame_poly.centroid if frame_poly else parcel_poly.centroid
            v_dir = (c_frame.x - bx, c_frame.y - by)
            L_v = math.hypot(v_dir[0], v_dir[1]) or 1.0
            u_dir = (v_dir[0] / L_v, v_dir[1] / L_v)

            # Détection de la bissectrice géométrique si la borne est sur une polyligne
            bisector_dir = None
            for e_pl in msp:
                if e_pl.dxftype() == "LWPOLYLINE":
                    pts = [(p[0], p[1]) for p in e_pl.get_points()]
                    for idx, pt in enumerate(pts):
                        if math.hypot(pt[0] - bx, pt[1] - by) < 1.0:
                            p_c = np.array([bx, by])
                            p_p = np.array(pts[idx - 1]) if idx > 0 else (np.array(pts[-1]) if getattr(e_pl, 'is_closed', False) else None)
                            p_n = np.array(pts[idx + 1]) if idx < len(pts) - 1 else (np.array(pts[0]) if getattr(e_pl, 'is_closed', False) else None)
                            if p_p is not None and p_n is not None:
                                v1 = p_p - p_c
                                v2 = p_n - p_c
                                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                                if n1 > 1e-3 and n2 > 1e-3:
                                    u1, u2 = v1 / n1, v2 / n2
                                    bis = u1 + u2
                                    n_b = float(np.linalg.norm(bis))
                                    if n_b > 1e-3:
                                        b_cand = bis / n_b
                                        if parcel_poly and parcel_poly.contains(sg.Point(float(p_c[0] + b_cand[0] * 3.0), float(p_c[1] + b_cand[1] * 3.0))):
                                            b_cand = -b_cand
                                        bisector_dir = b_cand
                            break
                    if bisector_dir is not None:
                        break

            existing_boxes = []
            for te in msp:
                if te.dxftype() in ("TEXT", "MTEXT"):
                    try:
                        from engines.dxf_cleaner.exact_text_obb import get_exact_text_obb
                        existing_boxes.append(get_exact_text_obb(te, pad_ratio=0.0))
                    except Exception:
                        t_content = core.get_text_content(te).strip()
                        if t_content:
                            w_t = len(t_content) * 0.55 * borne_text_height
                            tx, ty = te.dxf.insert.x, te.dxf.insert.y
                            existing_boxes.append(sg.box(tx, ty - borne_text_height, tx + w_t, ty))

            cand_w = len(lbl) * 0.60 * borne_text_height
            cand_h = borne_text_height * 1.0

            candidate_angles = [225, 240, 245, 250, 255, 270, 290, 315, 330, 0, 45, 90, 135, 180, 200, 215]
            if bisector_dir is not None:
                b_ang_deg = math.degrees(math.atan2(bisector_dir[1], bisector_dir[0])) % 360
                candidate_angles = [b_ang_deg, (b_ang_deg - 10) % 360, (b_ang_deg + 10) % 360, (b_ang_deg - 20) % 360, (b_ang_deg + 20) % 360] + candidate_angles

            best_score = -1e9
            r_candidates = [1.2, 1.8, 2.5, 3.5, 4.5] if bisector_dir is not None else [0.8, 1.2, 1.8, 2.2, 2.5]
            for r_mult in r_candidates:
                r_dist = r_mult * borne_text_height
                for ang in candidate_angles:
                    rad = math.radians(ang)
                    cx = bx + r_dist * math.cos(rad)
                    cy = by + r_dist * math.sin(rad)
                    cpt = sg.Point(cx, cy)
                    if parcel_poly.buffer(0.5).contains(cpt):
                        continue
                    if frame_poly and not frame_poly.buffer(-4.0).contains(cpt):
                        continue
                    cand_box = sg.box(cx, cy - cand_h, cx + cand_w, cy)
                    if any(cand_box.intersects(eb) for eb in existing_boxes):
                        continue
                    d_p = parcel_poly.distance(cpt)
                    min_b_dist = min([d_p] + [math.hypot(cx - obx, cy - oby) for obx, oby in sec.get("borne_blocks", {}).keys() if (obx, oby) != (bx, by)])
                    if min_b_dist < 1.0 * borne_text_height:
                        continue
                    d_frame = frame_poly.exterior.distance(cpt) if frame_poly else 20.0
                    align_center = math.cos(rad) * u_dir[0] + math.sin(rad) * u_dir[1]
                    if bisector_dir is not None:
                        align_bis = math.cos(rad) * bisector_dir[0] + math.sin(rad) * bisector_dir[1]
                        score = align_bis * 15.0 + align_center * 8.0 + min(d_frame, 20.0) * 1.5 + min(min_b_dist, 6.0)
                    else:
                        score = align_center * 8.0 + min(d_frame, 20.0) * 1.5 + min(min_b_dist, 6.0) - abs(r_mult - 2.0) * 1.5
                    if score > best_score:
                        best_score = score
                        ins_x, ins_y = cx, cy

            if ins_x is None:
                def_dir = bisector_dir if bisector_dir is not None else u_dir
                ins_x, ins_y = bx + def_dir[0] * 2.0 * borne_text_height, by + def_dir[1] * 2.0 * borne_text_height

            # Vérifier si l'étiquette ou sa verborne existe déjà dans le dessin
            existing_ent = None
            is_verborne = False
            for e in list(msp):
                if e.dxftype() in ("TEXT", "MTEXT"):
                    raw_val = core.get_text_content(e).strip().upper()
                    t_val = raw_val.replace("(", "").replace(")", "").strip()
                    if t_val == clean_b:
                        ins = core.get_insert(e)
                        if ins and math.hypot(ins[0] - ins_x, ins[1] - ins_y) < 25.0:
                            existing_ent = e
                            break
                    if t_val == f"V.{clean_b}" or (raw_val.startswith("V.") and clean_b in raw_val):
                        ins = core.get_insert(e)
                        if ins and math.hypot(ins[0] - ins_x, ins[1] - ins_y) < 25.0:
                            existing_ent = e
                            is_verborne = True
                            break

            if is_verborne:
                continue

            if existing_ent:
                # Mettre à jour le texte (enlever parenthèses si connecté) et repositionner
                if hasattr(existing_ent, "text"):
                    existing_ent.text = lbl
                else:
                    existing_ent.dxf.text = lbl
                existing_ent.dxf.insert = (ins_x, ins_y, 0.0)
                if hasattr(existing_ent.dxf, "text_direction"):
                    existing_ent.dxf.text_direction = (1.0, 0.0, 0.0)
                if hasattr(existing_ent.dxf, "rotation"):
                    existing_ent.dxf.rotation = 0.0
                logger.info("🔄 Borne extérieure mise à jour : '%s' à (%.1f, %.1f)", lbl, ins_x, ins_y)
            else:
                msp.add_mtext(
                    lbl,
                    dxfattribs={
                        "layer": "0",
                        "char_height": borne_text_height,
                        "insert": (ins_x, ins_y, 0.0),
                        "text_direction": (1.0, 0.0, 0.0),
                    }
                )
                existing_borne_names.add(clean_b)
                added_count += 1
                logger.info("➕ Borne extérieure ajoutée : '%s' (%s) à (%.1f, %.1f)", lbl, blk_name, ins_x, ins_y)

    # 7. Verbornes (v.Bxxxx) : étiquetage précis aux extrémités de façade des parcelles voisines
    existing_vb_inserts = []
    for e in list(msp):
        if e.dxftype() in ("TEXT", "MTEXT") and core.get_text_content(e).strip().lower().startswith("v."):
            existing_vb_inserts.append(core.get_insert(e)[:2])

    seen_vb_endpoints = set()
    for ep_start, ep_end in neighbor_frontage_endpoints:
        for pt_end in (ep_start, ep_end):
            # Ne pas ajouter si déjà couvert par une verborne existante dans un rayon de 25m
            if any(math.hypot(pt_end[0] - vx, pt_end[1] - vy) < 25.0 for vx, vy in existing_vb_inserts):
                continue
            # Règle ANCFCC: Ne JAMAIS placer de verborne (v.Bxxxx) à proximité d'une borne déjà tracée sur ce plan
            if any(math.hypot(pt_end[0] - be.dxf.insert.x, pt_end[1] - be.dxf.insert.y) < 10.0 for be in msp if be.dxftype() == "INSERT" and "borne" in getattr(be.dxf, "name", "").lower()):
                continue
            pt_obj = sg.Point(float(pt_end[0]), float(pt_end[1]))
            if parcel_poly.distance(pt_obj) > 2.0:
                # Règle ANCFCC: Ne placer une verborne que si une ligne physique existe dans le dessin msp (< 3.0m)
                has_line_near = False
                for e_line in msp.query("LINE LWPOLYLINE"):
                    if e_line.dxftype() == "LINE":
                        ls_m = sg.LineString([(e_line.dxf.start.x, e_line.dxf.start.y), (e_line.dxf.end.x, e_line.dxf.end.y)])
                    else:
                        pts_m = list(e_line.get_points(format="xy"))
                        if len(pts_m) < 2:
                            continue
                        ls_m = sg.LineString(pts_m)
                    if ls_m.distance(pt_obj) < 3.0:
                        has_line_near = True
                        break
                if not has_line_near:
                    continue

                key_end = (round(pt_end[0], 0), round(pt_end[1], 0))
                if key_end in seen_vb_endpoints:
                    continue
                seen_vb_endpoints.add(key_end)

                best_vb = None
                # Cas 0 : pt_end coïncide directement avec une borne du secteur (ex: B106 à l'extrémité de la ligne rattachée)
                if sec.get("bornes"):
                    closest_b = min(sec.get("bornes"), key=lambda b: math.hypot(b[0].x - pt_end[0], b[0].y - pt_end[1]))
                    if math.hypot(closest_b[0].x - pt_end[0], closest_b[0].y - pt_end[1]) < 4.0:
                        clean_vb = closest_b[1].replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                        if clean_vb not in existing_borne_names and parcel_poly.distance(closest_b[0]) > 8.0:
                            best_vb = clean_vb

                # Recherche cognitive de la verborne cible le long de la ligne unclipped du secteur
                matched_sec_lines = [
                    (ls, lay, coords) for ls, lay, coords in sec.get("limit_lines", [])
                    if lay.lower() in ("piste", "route", "plle", "chaaba", "oued", "l.e", "riverain")
                    and ls.distance(pt_obj) < 2.5
                ]

                # Cas A : Ligne rayonnante ou voirie vers l'extérieur (ex: piste de 8m vers B102)
                for ls, lay, coords in matched_sec_lines:
                    d_start = parcel_poly.distance(sg.Point(coords[0]))
                    d_end = parcel_poly.distance(sg.Point(coords[-1]))
                    far_end = coords[-1] if d_end > d_start else coords[0]

                    d_term0 = math.hypot(pt_end[0] - coords[0][0], pt_end[1] - coords[0][1])
                    d_term1 = math.hypot(pt_end[0] - coords[-1][0], pt_end[1] - coords[-1][1])
                    d_term = min(d_term0, d_term1)

                    if d_term >= 8.0 and max(d_start, d_end) > 25.0:
                        # La ligne a été tronquée par le cadre : la destination est far_end
                        for bpt, bname in sec.get("bornes", []):
                            if bpt.distance(sg.Point(far_end)) < 18.0:
                                clean_vb = bname.replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                                if clean_vb not in existing_borne_names and parcel_poly.distance(bpt) > 8.0:
                                    best_vb = clean_vb
                                    break
                    elif d_term < 8.0:
                        # La ligne originale se termine naturellement ici
                        term_pt = coords[0] if d_term0 < 8.0 else coords[-1]
                        for bpt, bname in sec.get("bornes", []):
                            if bpt.distance(sg.Point(term_pt)) < 10.0:
                                clean_vb = bname.replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                                if clean_vb not in existing_borne_names and parcel_poly.distance(bpt) > 8.0:
                                    best_vb = clean_vb
                                    break
                    if best_vb:
                        break

                # Cas B : Continuité le long des sommets du secteur
                if not best_vb:
                    for ls, lay, coords in matched_sec_lines:
                        closest_idx = min(range(len(coords)), key=lambda i: math.hypot(coords[i][0]-pt_end[0], coords[i][1]-pt_end[1]))
                        for step in (1, -1):
                            curr = closest_idx + step
                            while 0 <= curr < len(coords):
                                c_pt = coords[curr]
                                for b_t_pt, b_t_name in sec.get("bornes", []):
                                    if math.hypot(b_t_pt.x - c_pt[0], b_t_pt.y - c_pt[1]) < 6.0:
                                        clean_vb = b_t_name.replace("(", "").replace(")", "").replace("v.", "").strip().upper()
                                        if clean_vb not in existing_borne_names and parcel_poly.distance(b_t_pt) > 8.0:
                                            best_vb = clean_vb
                                            break
                                if best_vb:
                                    break
                                curr += step
                            if best_vb:
                                break
                        if best_vb:
                            break

                if best_vb:
                    seen_vb_endpoints.add(key_end)
                    vb_lbl = f"v.{best_vb}"
                    # Trouver la ligne directrice la plus proche pour orienter la verborne
                    vb_tan_ux, vb_tan_uy = 1.0, 0.0
                    min_l_d = 1e9
                    for e_l in msp:
                        if e_l.dxftype() in ("LINE", "LWPOLYLINE") and getattr(e_l.dxf, "layer", "") in ("DTLS", "L.E", "Chaaba", "0"):
                            if e_l.dxftype() == "LINE":
                                l_seg = sg.LineString([(e_l.dxf.start.x, e_l.dxf.start.y), (e_l.dxf.end.x, e_l.dxf.end.y)])
                            else:
                                l_pts = [(p[0], p[1]) for p in e_l.get_points()]
                                if len(l_pts) >= 2:
                                    l_seg = min((sg.LineString([l_pts[i], l_pts[i+1]]) for i in range(len(l_pts)-1)), key=lambda s: s.distance(pt_obj))
                                else:
                                    continue
                            d_l = l_seg.distance(pt_obj)
                            if d_l < min_l_d:
                                min_l_d = d_l
                                p0, p1 = l_seg.coords[0], l_seg.coords[1]
                                ldx, ldy = p1[0] - p0[0], p1[1] - p0[1]
                                lL = math.hypot(ldx, ldy) or 1.0
                                nux, nuy = ldx / lL, ldy / lL
                                if nux < 0 or (abs(nux) < 1e-5 and nuy < 0):
                                    nux, nuy = -nux, -nuy
                                vb_tan_ux, vb_tan_uy = nux, nuy

                    # Vérifier si cette verborne existe déjà
                    existing_vb_ent = None
                    for e in list(msp):
                        if e.dxftype() in ("TEXT", "MTEXT"):
                            t_chk = core.get_text_content(e).strip()
                            if t_chk == vb_lbl or t_chk == f"v.{best_vb}":
                                existing_vb_ent = e
                                break

                    c_frame = frame_poly.centroid if frame_poly else parcel_poly.centroid
                    to_c_x = c_frame.x - pt_end[0]
                    to_c_y = c_frame.y - pt_end[1]
                    L_to_c = math.hypot(to_c_x, to_c_y) or 1.0
                    ins_x = pt_end[0] + (to_c_x / L_to_c) * 4.0
                    ins_y = pt_end[1] + (to_c_y / L_to_c) * 4.0

                    if existing_vb_ent:
                        existing_vb_ent.dxf.insert = (ins_x, ins_y, 0.0)
                        if hasattr(existing_vb_ent.dxf, "text_direction"):
                            existing_vb_ent.dxf.text_direction = (vb_tan_ux, vb_tan_uy, 0.0)
                        if hasattr(existing_vb_ent.dxf, "rotation"):
                            existing_vb_ent.dxf.rotation = 0.0
                    else:
                        doc = getattr(msp, "doc", None)
                        if doc and hasattr(doc, "styles") and "CQ_VRBN" not in doc.styles:
                            try:
                                doc.styles.new("CQ_VRBN", dxfattribs={"font": "romans.shx"})
                            except Exception:
                                pass
                        msp.add_mtext(
                            vb_lbl,
                            dxfattribs={
                                "layer": "0",
                                "char_height": max(borne_text_height, 3.6 * (scale / 2000.0)),
                                "insert": (ins_x, ins_y, 0.0),
                                "text_direction": (vb_tan_ux, vb_tan_uy, 0.0),
                                "style": "CQ_VRBN",
                            }
                        )
                        existing_texts.append(vb_lbl)
                        existing_borne_names.add(best_vb)
                        added_count += 1
                        logger.info("➕ Verborne rayonnante ajoutée : '%s' à (%.1f, %.1f) dir=(%.3f, %.3f)", vb_lbl, ins_x, ins_y, vb_tan_ux, vb_tan_uy)

                    # Troncature de la polyligne de voirie/limite à la verborne pour ne jamais dépasser vers le cadre
                    for e_l in list(msp):
                        if e_l.dxftype() == "LWPOLYLINE" and getattr(e_l.dxf, "layer", "").upper() in ("DTLS", "L.E", "PISTE", "ROUTE"):
                            l_pts = list(e_l.get_points(format="xy"))
                            if len(l_pts) >= 3:
                                dists = [math.hypot(p[0] - pt_end[0], p[1] - pt_end[1]) for p in l_pts]
                                min_d = min(dists)
                                if min_d < 10.0:
                                    k = dists.index(min_d)
                                    d_start = parcel_poly.distance(sg.Point(l_pts[0]))
                                    d_end = parcel_poly.distance(sg.Point(l_pts[-1]))
                                    if d_start < d_end and k < len(l_pts) - 1:
                                        new_pts = l_pts[:k+1]
                                        if len(new_pts) >= 2:
                                            e_l.set_points([(p[0], p[1], 0, 0, 0) for p in new_pts])
                                            logger.info("✂️ Ligne de voirie tronquée à la verborne %s (Npts %d -> %d)", vb_lbl, len(l_pts), len(new_pts))
                                    elif d_end < d_start and k > 0:
                                        new_pts = l_pts[k:]
                                        if len(new_pts) >= 2:
                                            e_l.set_points([(p[0], p[1], 0, 0, 0) for p in new_pts])
                                            logger.info("✂️ Ligne de voirie tronquée à la verborne %s (Npts %d -> %d)", vb_lbl, len(l_pts), len(new_pts))

    # 8. Propriétaires nommés (calque 'riverain') avec expansion Héritiers et \PTC
    def _norm_cmp(s: str) -> str:
        s = re.sub(r'\\ptc|\bptc\b|\\p|\btc\b|\bt\.c\b|\n|\r', ' ', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(heritiers|hritiers|hrts|hert|heritier|hritier)\b', 'heritiers', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(ouled|ould|oulad)\b', 'ouled', s, flags=re.IGNORECASE)
        s = re.sub(r'\b(ben|ebn|bin)\b', 'ben', s, flags=re.IGNORECASE)
        s_chars = re.sub(r'[oeuy\u00E9\u00E8\u00EA\u00EB]', '', s.lower())
        s_chars = re.sub(r'\btc\b', ' ', s_chars)
        return re.sub(r'[^a-z0-9\u0600-\u06FF]', '', s_chars)

    # Règle ANCFCC: Ne chercher que les propriétaires immédiatement adjacents (<= 40m max)
    max_search_d = min(40.0, max(25.0, 25.0 * (scale / 1000.0)))
    for pt, rtxt in sec.get("riverains", []):
        if not parcel_poly.contains(pt):
            d = parcel_poly.distance(pt)
            if 0.5 < d <= max_search_d:
                clean_r = rtxt.strip()
                clean_r = re.sub(r'^\bHert\b|\bHrts\b', 'Héritiers', clean_r, flags=re.IGNORECASE)
                is_natural = bool(re.match(r"^(Jbel|Oued|Chaaba|Chaâba|Châaba|Forêt|Foret)$", clean_r, re.IGNORECASE))
                # Ne pas forcer \PTC sur les propriétaires nommés : dans le DEF expert, ils sont écrits sans suffixe TC
                clean_r_norm = _norm_cmp(clean_r)
                already_has_r = any(clean_r_norm == _norm_cmp(t) for t in existing_texts if len(_norm_cmp(t)) >= 3)

                ins_pt = pt
                if frame_poly and not frame_poly.buffer(-2.0).contains(pt):
                    np_f, _ = nearest_points(frame_poly.buffer(-5.0), pt)
                    ins_pt = np_f
                # Préserver la position réelle du secteur sans déplacement intempestif si d <= 65m
                if parcel_poly.distance(ins_pt) > 65.0:
                    np_p, _ = nearest_points(parcel_poly, ins_pt)
                    v_dx, v_dy = ins_pt.x - np_p.x, ins_pt.y - np_p.y
                    v_l = math.hypot(v_dx, v_dy) or 1.0
                    ins_pt = sg.Point(np_p.x + (v_dx / v_l) * 40.0, np_p.y + (v_dy / v_l) * 40.0)

                tan_ux, tan_uy = get_facing_segment_tangent(parcel_poly, ins_pt)
                # Les propriétaires nommés sont toujours horizontaux (rot=0), les éléments naturels suivent la limite
                r_dir = (tan_ux, tan_uy, 0.0) if is_natural else (1.0, 0.0, 0.0)

                if not already_has_r:
                    msp.add_mtext(
                        clean_r,
                        dxfattribs={
                            "layer": "0",
                            "char_height": text_height,
                            "insert": (ins_pt.x, ins_pt.y, 0.0),
                            "text_direction": r_dir,
                        }
                    )
                    existing_texts.append(clean_r)
                    added_count += 1
                    logger.info("➕ Riverain nommé ajouté : '%s' à (%.1f, %.1f) dir=(%.3f, %.3f)", clean_r, ins_pt.x, ins_pt.y, r_dir[0], r_dir[1])
                else:
                    # Mettre à jour l'orientation du riverain existant
                    for e in list(msp):
                        if e.dxftype() in ("TEXT", "MTEXT"):
                            t_chk = core.get_text_content(e)
                            if _norm_cmp(t_chk) == clean_r_norm:
                                e_ins = sg.Point(core.get_insert(e)[:2])
                                t_ux, t_uy = get_facing_segment_tangent(parcel_poly, e_ins)
                                if hasattr(e.dxf, "text_direction"):
                                    e.dxf.text_direction = (t_ux, t_uy, 0.0)
                                if hasattr(e.dxf, "rotation"):
                                    e.dxf.rotation = 0.0

    # 9. Titres fonciers
    for pt, ttxt in sec.get("titres", []):
        if not parcel_poly.contains(pt):
            d = parcel_poly.distance(pt)
            if 0.5 < d <= max_search_d:
                clean_t = ttxt.split("(")[0].strip()
                lbl_t = f"{clean_t}\\PTC" if not clean_t.upper().endswith("TC") else clean_t
                lbl_t_norm = _norm_cmp(clean_t)
                already_has_t = any(lbl_t_norm == _norm_cmp(t.split("(")[0]) for t in existing_texts if len(_norm_cmp(t.split("(")[0])) >= 3)
                if not already_has_t:
                    ins_pt = pt
                    if frame_poly and not frame_poly.buffer(-2.0).contains(pt):
                        np_f, _ = nearest_points(frame_poly.buffer(-5.0), pt)
                        ins_pt = np_f
                    msp.add_mtext(lbl_t, dxfattribs={"layer": "0", "char_height": text_height, "insert": (ins_pt.x, ins_pt.y, 0.0)})
                    existing_texts.append(lbl_t)
                    added_count += 1
                    logger.info("➕ Titre ajouté : '%s' à (%.1f, %.1f)", clean_t, ins_pt.x, ins_pt.y)

    # 10. Châaba : fermeture de berge (calque Chaaba) et toponyme au milieu
    seen_chaaba_pts = []
    # 10. Châaba : toponyme au milieu si manquant

    nb_chaabat_labels = sum(1 for t in existing_texts if "chaaba" in t.lower() or "chaâba" in t.lower())
    for geom, _ in sec.get("chaabas", []):
        d = parcel_poly.distance(geom)
        if d < 25.0:
            clipped = geom.intersection(frame_poly) if frame_poly else geom
            if clipped.is_empty:
                continue
            geoms = [clipped] if isinstance(clipped, sg.LineString) else getattr(clipped, "geoms", [])
            for g in geoms:
                if isinstance(g, sg.LineString) and g.length >= 25.0:
                    if nb_chaabat_labels >= 1:
                        break
                    # Point d'insertion centré dans le chenal de la chaâba
                    frac = 0.5 if nb_chaabat_labels == 0 else 0.25
                    mid = g.interpolate(frac, normalized=True)
                    if any(math.hypot(mid.x - px, mid.y - py) < 45.0 for px, py in seen_chaaba_pts):
                        continue
                    if any(math.hypot(mid.x - bx, mid.y - by) < 10.0 for bx, by in sec.get("borne_blocks", {}).keys()):
                        continue
                    seen_chaaba_pts.append((mid.x, mid.y))
                    p_before = g.interpolate(max(0.01, frac - 0.02), normalized=True)
                    p_after = g.interpolate(min(0.99, frac + 0.02), normalized=True)
                    dx = p_after.x - p_before.x
                    dy = p_after.y - p_before.y
                    rot = math.degrees(math.atan2(dy, dx)) % 360.0
                    if 90.0 < rot <= 270.0:
                        rot = (rot + 180.0) % 360.0
                    msp.add_text("Chaâbat", dxfattribs={"layer": "0", "height": text_height, "rotation": rot, "insert": (mid.x, mid.y, 0.0)})
                    existing_texts.append("Chaâbat")
                    nb_chaabat_labels += 1
                    added_count += 1
                    logger.info("➕ Chaâbat ajoutée au milieu du cours d'eau à (%.1f, %.1f) rot=%.1f", mid.x, mid.y, rot)

    # 11. Piste : positionnement au milieu de l'emprise de voirie avec rotation tangente
    if not any("piste" in t.lower() for t in existing_texts):
        for geom, plbl in sec.get("pistes", []):
            d = parcel_poly.distance(geom)
            if d < 25.0:
                clipped = geom.intersection(frame_poly) if frame_poly else geom
                if clipped.is_empty:
                    continue
                geoms = [clipped] if isinstance(clipped, sg.LineString) else getattr(clipped, "geoms", [])
                for g in geoms:
                    if isinstance(g, sg.LineString) and g.length >= 20.0:
                        mid = g.interpolate(0.5, normalized=True)
                        if any(math.hypot(mid.x - bx, mid.y - by) < 10.0 for bx, by in sec.get("borne_blocks", {}).keys()):
                            continue
                        p_before = g.interpolate(0.48, normalized=True)
                        p_after = g.interpolate(0.52, normalized=True)
                        dx = p_after.x - p_before.x
                        dy = p_after.y - p_before.y
                        rot = math.degrees(math.atan2(dy, dx)) % 360.0
                        if 90.0 < rot <= 270.0:
                            rot = (rot + 180.0) % 360.0
                        pst_txt = plbl if plbl else "piste de 8 m"
                        msp.add_text(pst_txt, dxfattribs={"layer": "0", "height": text_height, "rotation": rot, "insert": (mid.x, mid.y, 0.0)})
                        existing_texts.append(pst_txt)
                        added_count += 1
                        logger.info("➕ Piste ajoutée au milieu du segment à (%.1f, %.1f) rot=%.1f", mid.x, mid.y, rot)
                        break
                if any("piste" in t.lower() for t in existing_texts):
                    break

    # 12. Route goudronnée : positionnement au milieu du tronçon riverain avec rotation tangente
    if not any("route" in t.lower() for t in existing_texts):
        min_rt_d = 1e9
        best_rt_pt = None
        best_rt_rot = 0.0
        best_rt_lbl = "Route goudronnée de 15 m"
        for geom, rlbl in sec.get("routes", []):
            d = parcel_poly.distance(geom)
            if d < min_rt_d and d < 25.0:
                min_rt_d = d
                if isinstance(geom, sg.LineString) and geom.length >= 10.0:
                    proj = geom.project(parcel_poly.centroid)
                    p_mid_rt = geom.interpolate(max(0.1, min(0.9, proj / geom.length)), normalized=True)
                    delta_d = min(1.0, geom.length * 0.05)
                    p_b = geom.interpolate(max(0.0, proj - delta_d))
                    p_a = geom.interpolate(min(geom.length, proj + delta_d))
                    dx_rt = p_a.x - p_b.x
                    dy_rt = p_a.y - p_b.y
                    rot_rt = math.degrees(math.atan2(dy_rt, dx_rt)) % 360.0
                    if 90.0 < rot_rt <= 270.0:
                        rot_rt = (rot_rt + 180.0) % 360.0
                    best_rt_pt = p_mid_rt
                    best_rt_rot = rot_rt
                    best_rt_lbl = rlbl if rlbl else "Route goudronnée de 15 m"
                else:
                    p_poly, p_geo = nearest_points(parcel_poly, geom)
                    best_rt_pt = p_geo
                    best_rt_rot = 0.0
                    best_rt_lbl = rlbl if rlbl else "Route goudronnée de 15 m"
        if best_rt_pt:
            msp.add_text(best_rt_lbl, dxfattribs={"layer": "0", "height": text_height, "rotation": best_rt_rot, "insert": (best_rt_pt.x, best_rt_pt.y, 0.0)})
            existing_texts.append(best_rt_lbl)
            added_count += 1

    return added_count


_GLOBAL_SECTOR_DB: Optional[SectorDatabase] = None

def get_sector_database(
    sectors_dir: str = r"C:\Users\Admin\Downloads\SS",
    force_refresh: bool = False,
    selected_sz: Optional[str] = None
) -> Optional[SectorDatabase]:
    """
    Charge et maintient en mémoire cache la base des secteurs / Sous-Zones.
    Supporte à la fois un dossier complet ou un fichier DXF unique de Sous-Zone.
    """
    global _GLOBAL_SECTOR_DB
    need_reload = (
        _GLOBAL_SECTOR_DB is None
        or _GLOBAL_SECTOR_DB.sectors_dir != sectors_dir
        or _GLOBAL_SECTOR_DB.selected_sz != selected_sz
        or force_refresh
    )
    if need_reload:
        if os.path.exists(sectors_dir):
            _GLOBAL_SECTOR_DB = SectorDatabase(
                sectors_dir=sectors_dir,
                force_refresh=force_refresh,
                selected_sz=selected_sz
            )
    return _GLOBAL_SECTOR_DB


def inspect_sz_reference(sectors_path: str, selected_sz: Optional[str] = None) -> Dict[str, Any]:
    """
    Inspecte et retourne les statistiques du référentiel Sous-Zone pour l'interface UI.
    Permet à l'utilisateur de vérifier instantanément que le plan maître a été chargé
    et combien de parcelles et riverains y ont été trouvés.
    """
    if not os.path.exists(sectors_path):
        return {"status": "error", "message": f"المسار غير موجود: {sectors_path}"}

    try:
        db = SectorDatabase(sectors_dir=sectors_path, force_refresh=True, selected_sz=selected_sz)
        if not db.sectors_data:
            return {"status": "error", "message": "لم يتم العثور على أي مخططات DXF صالحة في المسار."}

        tot_parcels = sum(len(s.get("parcel_polygons", [])) for s in db.sectors_data)
        tot_titres = sum(len(s.get("titres", [])) for s in db.sectors_data)
        tot_riv = sum(len(s.get("riverains", [])) for s in db.sectors_data)
        tot_chaaba = sum(len(s.get("chaabas", [])) for s in db.sectors_data)

        details = []
        for s in db.sectors_data:
            details.append({
                "tag": s.get("sz_tag", "SZ"),
                "file": s.get("filename", ""),
                "parcels": len(s.get("parcel_polygons", [])),
                "titres": len(s.get("titres", [])),
                "riverains": len(s.get("riverains", []))
            })

        return {
            "status": "ok",
            "sectors_count": len(db.sectors_data),
            "total_parcels": tot_parcels,
            "total_titres": tot_titres,
            "total_riverains": tot_riv,
            "total_chaabas": tot_chaaba,
            "details": details
        }
    except Exception as exc:
        return {"status": "error", "message": f"خطأ أثناء فحص المخطط المرجعي: {exc}"}

