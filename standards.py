# -*- coding: utf-8 -*-
r"""
engines/dxf_cleaner/standards.py — Standards ANCFCC, En-têtes, XDATA COVADIS et Encodage Unicode
================================================================================================
AGENT 5 : Injection des variables de configuration système AutoCAD AC1018, ANSI_1256,
          application d'échelle COVADIS_ECHELLE, et encodage Unicode officiel \U+XXXX.
VERROU REJCAD : RejCadPrecisionLock pour arrondi centimétrique strict et contenance Gauss Ha a ca.
"""

import os
import logging
import math
import re
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from ezdxf.document import Drawing
from ezdxf.lldxf.const import DXFValueError

from constants import ANCFCC_STANDARDS

BORNE_NAME_RE = re.compile(r"^(?:v\.)?\(?(?:B\d+[A-Za-z0-9_/-]*|BP\d+[A-Za-z0-9_/-]*|BBP\d+[A-Za-z0-9_/-]*|R\d+|P\d+)\)?$", re.IGNORECASE)

logger = logging.getLogger("dxf_cleaner.standards")


def encode_ancfcc_unicode(text: str) -> str:
    r"""
    Encode les lettres accentuées françaises au format officiel AutoCAD \U+XXXX,
    tout en préservant le texte arabe en UTF-8 natif (exactement comme dans les plans DEF ANCFCC).
    """
    if not text:
        return ""
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+3].upper() == "\\U+" and i + 7 <= n:
            out.append(text[i:i+7])
            i += 7
            continue
        ch = text[i]
        code = ord(ch)
        # Seules les lettres accentuées françaises et symboles sont encodés en \U+00XX
        # L'arabe (0x0600..0x06FF) reste en UTF-8 natif propre sans étirement !
        if code in (0xE9, 0xE8, 0xEA, 0xEB, 0xE0, 0xE2, 0xF4, 0xFB, 0xEE, 0xEF, 0xE7, 0xB0):
            out.append(f"\\U+{code:04X}")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


class RejCadPrecisionLock:
    r"""
    VERROU DE PRÉCISION REJCAD (ANCFCC 100.00% CONFORMITÉ) :
    1. Force un arrondi centimétrique strict round(val, 2) sur tous les sommets
       de la polyligne 'parcelle' et sur les points d'insertion des blocs de bornes
       ('borne_plantee', 'borne_marquee', 'bornes').
       Élimine toute dérive flottante (ex: 793098.4599999998 -> 793098.46).
    2. Recalcule la contenance exacte de Gauss (Shoelace) directement à partir
       de ces coordonnées arrondies au centimètre : Ha a ca.
    3. Met à jour la formule de Contenance officielle ANCFCC :
       \A1;Contenance :   XX{\H0.7x;\SHa^ ;} YY{\H0.7x;\SA^ ;} ZZ{\H0.7x;\SCa^ ;} : \U+0645\U+0633\U+0627\U+062D\U+0629
    """

    @staticmethod
    def enforce_centimetric_precision(msp) -> Tuple[int, float, Dict[str, Any]]:
        """
        Applique le verrou centimétrique et recalcule la contenance exacte de Gauss.
        Retourne (nb_entites_arrondies, surface_m2, dict_contenance).
        """
        rounded_count = 0
        parcel_pts: List[Tuple[float, float]] = []

        # 1. Sommets de la polyligne de parcelle
        for pl in msp.query('LWPOLYLINE[layer=="parcelle"]'):
            pts = list(pl.get_points())
            if len(pts) >= 3:
                new_pts = []
                for p in pts:
                    rx, ry = round(float(p[0]), 2), round(float(p[1]), 2)
                    rest = p[2:]
                    new_pts.append((rx, ry, *rest))
                    parcel_pts.append((rx, ry))
                pl.set_points(new_pts)
                rounded_count += len(new_pts)

        # 2. Points d'insertion des blocs de bornes
        for ent in msp.query("INSERT"):
            lay = ent.dxf.layer.lower()
            bname = getattr(ent.dxf, "name", "").lower()
            if "borne" in lay or "borne" in bname:
                ins = ent.dxf.insert
                rx = round(float(ins.x), 2)
                ry = round(float(ins.y), 2)
                rz = round(float(ins.z), 2)
                ent.dxf.insert = (rx, ry, rz)
                rounded_count += 1

        # 3. Calcul de la contenance exacte de Gauss (Shoelace)
        contenance_info: Dict[str, Any] = {"ha": 0, "a": 0, "ca": 0, "m2": 0.0}
        if len(parcel_pts) >= 3:
            pts_arr = np.array(parcel_pts, dtype=np.float64)
            if float(np.linalg.norm(pts_arr[0] - pts_arr[-1])) < 0.01:
                pts_arr = pts_arr[:-1]
            x, y = pts_arr[:, 0], pts_arr[:, 1]
            area_m2 = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

            ha = int(area_m2 // 10000)
            rem = area_m2 - (ha * 10000)
            a = int(rem // 100)
            ca = int(round(rem - (a * 100)))

            if ca >= 100:
                ca = 0
                a += 1
            if a >= 100:
                a = 0
                ha += 1

            contenance_info = {"ha": ha, "a": a, "ca": ca, "m2": round(area_m2, 2)}
            logger.info("📐 [REJCAD LOCK] Contenance Gauss calculée : %02d Ha %02d a %02d ca (%.2f m²)", ha, a, ca, area_m2)

            # 4. Mise à jour ou injection du texte Contenance
            existing_cont = None
            for ent in msp.query("MTEXT TEXT"):
                raw_t = ent.text if ent.dxftype() == "MTEXT" else getattr(ent.dxf, "text", "")
                if "contenance" in raw_t.lower():
                    existing_cont = ent
                    break

            if ha > 0:
                contenance_text = (
                    f"\\A1;Contenance :   {ha:02d}{{\\H0.7x;\\SHa^ ;}} "
                    f"{a:02d}{{\\H0.7x;\\SA^ ;}} {ca:02d}{{\\H0.7x;\\SCa^ ;}} : "
                    f"\\U+0645\\U+0633\\U+0627\\U+062D\\U+0629"
                )
            else:
                contenance_text = (
                    f"\\A1;Contenance :   {a:d}{{\\H0.7x;\\SA^ ;}} "
                    f"{ca:02d}{{\\H0.7x;\\SCa^ ;}} : "
                    f"\\U+0645\\U+0633\\U+0627\\U+062D\\U+0629"
                )

            if existing_cont is not None:
                # Si le texte existant contient déjà le format officiel avec Ha/A/Ca, préserver le texte officiel du PV
                raw_ex = existing_cont.text if existing_cont.dxftype() == "MTEXT" else getattr(existing_cont.dxf, "text", "")
                if not ("{\\h" in raw_ex.lower() or "ca^" in raw_ex.lower() or "a^" in raw_ex.lower()):
                    if existing_cont.dxftype() == "MTEXT":
                        existing_cont.text = contenance_text
                    existing_cont.dxf.text = contenance_text
            elif len(parcel_pts) >= 3:
                # Injection automatique dans la zone cartouche basse si manquant
                cx = float(np.mean([p[0] for p in parcel_pts]))
                min_y = min(p[1] for p in parcel_pts)
                cartouche_y = min_y - 40.0
                msp.add_mtext(
                    contenance_text,
                    dxfattribs={
                        "layer": "0",
                        "insert": (cx, cartouche_y, 0.0),
                        "char_height": 2.5,
                        "attachment_point": 5,  # Middle Center
                    }
                )

        return rounded_count, contenance_info["m2"], contenance_info


def normalize_cartouche_and_standards(doc: Drawing, scale: float = 2000.0, frame_poly=None, drafter: str = "el-bekay") -> None:
    """
    Normalisation rigoureuse du cartouche, des orientations de texte et des calques :
    1. Cartouche administratif :
       - 'Dessinateur : Automatique' -> 'Dessinateur : el-bekay'
       - Nettoyage des suffixes parasites ('Jbel\\PTC' -> 'Jbel')
       - Préservation stricte de l'orientation horizontale et des positions du cartouche.
    2. Marges verticales extérieures :
       - Seuls 'ST 25' et 'Réquisition N° :' sur le cadre supérieur peuvent être verticaux.
       - Les textes arabes du cartouche restent HORIZONTAUX et dans leur encodage natif.
    """
    msp = doc.modelspace()

    # 1. Normalisation des calques des entités géométriques
    for ent in list(msp):
        lay = getattr(ent.dxf, "layer", "")
        if lay == "LIMIT":
            ent.dxf.layer = "DTLS"
        elif lay.lower() == "chaaba" and lay != "Chaaba":
            ent.dxf.layer = "Chaaba"

    # 2. Normalisation des textes et cartouches
    for ent in list(msp):
        if ent.dxftype() not in ("TEXT", "MTEXT"):
            continue

        raw_txt = ent.text if ent.dxftype() == "MTEXT" else getattr(ent.dxf, "text", "")
        if not raw_txt:
            continue
        txt = raw_txt

        # Drafter / Dessinateur
        if "Dessinateur : Automatique" in txt and drafter:
            txt = txt.replace("Dessinateur : Automatique", f"Dessinateur : {drafter}")

        # Normalisation administrative cartouche (sans déplacer l'insertion Y)
        if any(k in txt.lower() for k in ("pce d'oujda", "cle de", "prefecture d'oujda", "préfecture d'oujda", "caidat oued isly", "caïdat oued isly", "cne. mestferki")):
            txt = "Située à : Préfecture d'Oujda-Angad Caïdat Oued Isly Commune Mestferki"

        if "اقليم وجدة-أنكاد" in txt or "\\U+0627\\U+0642\\U+0644\\U+064A\\U+0645" in txt or "دائرة  جماعة مستفركي" in txt:
            txt = "{\\fTimes New Roman|b1|i0|c178|p18;الكائن ب :عمالة وجدة انكاد قيادة واد اسلي جماعة مستفركي}"

        # Jbel\\PTC -> Jbel
        if "jbel" in txt.lower():
            txt = re.sub(r"(\\+P|\n|\\)?\s*T\.?C\.?$", "", txt, flags=re.IGNORECASE).strip()

        # Dépression : préserver sur calque 0
        if txt.strip().lower() == "depression":
            ent.dxf.layer = "0"

        # Normalisation de l'échelle du cartouche
        if "echelle" in txt.lower() and scale:
            txt = re.sub(r"Echelle\s*:\s*1\s*/\s*\d+", f"Echelle : 1/{int(scale)}", txt, flags=re.IGNORECASE)

        # Normalisation réglementaire des hauteurs de texte selon l'échelle ANCFCC
        h_borne = ANCFCC_STANDARDS.get("borne_text_height_ratio_S", 0.0018) * scale
        h_road = 0.0020 * scale
        h_linear = 0.0018 * scale
        h_anno = 0.0012 * scale
        lay_l = getattr(ent.dxf, "layer", "").lower()
        tl = txt.strip().lower()

        is_road_text = any(k in tl for k in ("route", "piste", "chemin", "voie", "canal"))
        is_linear_text = any(k in tl for k in ("chaaba", "chaâba", "oued", "seguia", "l.e", "lebt", "lemt", "leht"))

        if BORNE_NAME_RE.match(txt.strip()):
            if ent.dxftype() == "TEXT":
                ent.dxf.height = h_borne
            elif hasattr(ent.dxf, "char_height"):
                ent.dxf.char_height = h_borne
        elif is_road_text:
            if ent.dxftype() == "TEXT":
                ent.dxf.height = h_road
            elif hasattr(ent.dxf, "char_height"):
                ent.dxf.char_height = h_road
        elif is_linear_text:
            if ent.dxftype() == "TEXT":
                ent.dxf.height = h_linear
            elif hasattr(ent.dxf, "char_height"):
                ent.dxf.char_height = h_linear
        elif lay_l == "annotation" or any(k == tl or tl.startswith(k + " ") or tl.endswith(" " + k) for k in (
            "construction", "autre", "bassin", "puits", "puit", "mur", "garage", "hangar", "depot",
            "habitation", "maison", "villa", "ruine", "ecurie", "rdc", "chateau", "magasin", "etage",
            "cave", "arbre", "figuier", "olivier", "cactus", "onep", "metmoura", "tazouta"
        )):
            if ent.dxftype() == "TEXT":
                ent.dxf.height = h_anno
            elif hasattr(ent.dxf, "char_height"):
                ent.dxf.char_height = h_anno
        elif txt.strip().isdigit() and len(txt.strip()) == 6:
            if ent.dxftype() == "TEXT":
                ent.dxf.height = h_borne
            elif hasattr(ent.dxf, "char_height"):
                ent.dxf.char_height = h_borne

        # Seuls les codes administratifs en marge supérieure ('ST 25', 'Réquisition N° :') sont verticaux
        is_vertical_margin = False
        ins_y = getattr(ent.dxf.insert, "y", 0.0) if hasattr(ent.dxf, "insert") else 0.0
        if frame_poly and ins_y > (frame_poly.bounds[3] - 25.0):
            if any(k in txt for k in ("ST 25", "Réquisition", "R\\U+00E9quisition")):
                is_vertical_margin = True

        if is_vertical_margin and ent.dxftype() == "MTEXT":
            ent.dxf.rotation = 0.0
            ent.dxf.text_direction = (1e-16, -1.0, 0.0)
            ent.dxf.attachment_point = 3  # TopRight

        # Coordonnées Lambert au bas du cadre (ex: 793000) : direction (0, 1, 0)
        s_txt = txt.strip()
        if s_txt.isdigit() and len(s_txt) == 6:
            ins = getattr(ent.dxf, "insert", None)
            if ins and frame_poly:
                minx, miny, maxx, maxy = frame_poly.bounds
                if abs(ins[1] - miny) < 15.0:  # En bas du cadre
                    ent.dxf.rotation = 0.0
                    if ent.dxftype() == "MTEXT":
                        ent.dxf.text_direction = (1e-16, 1.0, 0.0)
                        ent.dxf.attachment_point = 7  # BottomLeft

        # Application de l'encodage Unicode officiel
        enc_txt = encode_ancfcc_unicode(txt)
        if ent.dxftype() == "MTEXT":
            ent.text = enc_txt
        else:
            ent.dxf.text = enc_txt


def apply_ancfcc_header_and_xdata(doc: Drawing, scale: int = 2000) -> None:
    """
    AGENT 5 : Injection des normes ANCFCC (HEADER + XDATA COVADIS + Unicode).
    """
    try:
        doc.dxfversion = ANCFCC_STANDARDS["acadver"]
    except Exception:
        pass

    hdr = doc.header
    hdr["$ACADVER"] = ANCFCC_STANDARDS["acadver"]
    hdr["$DWGCODEPAGE"] = ANCFCC_STANDARDS["codepage"]
    hdr["$AUNITS"] = ANCFCC_STANDARDS["aunits"]
    hdr["$AUPREC"] = ANCFCC_STANDARDS["auprec"]
    hdr["$LUNITS"] = ANCFCC_STANDARDS["lunits"]
    hdr["$LUPREC"] = ANCFCC_STANDARDS["luprec"]
    hdr["$ANGBASE"] = ANCFCC_STANDARDS["angbase"]
    hdr["$ANGDIR"] = ANCFCC_STANDARDS["angdir"]
    hdr["$LTSCALE"] = ANCFCC_STANDARDS["ltscale"]
    hdr["$PLINEGEN"] = ANCFCC_STANDARDS["plinegen"]
    hdr["$MEASUREMENT"] = ANCFCC_STANDARDS["measurement"]
    hdr["$USERI1"] = int(scale // 10)
    hdr["$HANDSEED"] = "FFFFFF"

    # Fix AutoCAD Error 50: In AC1018, $XCLIPFRAME has group code 290 (strict boolean 0 or 1).
    # Values >= 2 (default in newer AutoCAD) cause AutoCAD 2018/2025 to crash with Error 50 on load!
    if "$XCLIPFRAME" in hdr:
        if hdr["$XCLIPFRAME"] not in (0, 1):
            hdr["$XCLIPFRAME"] = 1
    else:
        hdr["$XCLIPFRAME"] = 1

    for app_name in ("COVADIS_ECHELLE", "EZDXF"):
        if app_name not in doc.appids:
            doc.appids.add(app_name)

    ms_block = doc.block_records.get("*Model_Space")
    if ms_block is not None:
        try:
            ms_block.discard_xdata("COVADIS_ECHELLE")
        except DXFValueError:
            pass
        ms_block.set_xdata("COVADIS_ECHELLE", [(1040, float(scale / 10.0))])


def sanitize_certified_dxf_file(dxf_path: str) -> None:
    """
    AutoCAD Binary / Header Sanitizer (Garantie Zéro Erreur 50):
    1. Corrige l'Erreur 50 AutoCAD : Remplace la valeur invalide 2 du code de groupe 290 ($XCLIPFRAME) par 1.
    2. Normalise les fins de ligne en CRLF (Windows) pour compatibilité 100% avec AutoCAD 2004 à 2026.
    """
    if not os.path.exists(dxf_path):
        return
    try:
        with open(dxf_path, "rb") as f:
            raw = f.read()

        # Remplacement direct sur les octets bruts pour $XCLIPFRAME 290 2 -> 290 1
        raw_fixed = re.sub(rb"(\$XCLIPFRAME\r?\n\s*290\r?\n)\s*2", rb"\g<1>1", raw)

        # Normalisation CRLF
        if b"\r\n" not in raw_fixed and b"\n" in raw_fixed:
            raw_fixed = raw_fixed.replace(b"\n", b"\r\n")

        with open(dxf_path, "wb") as f:
            f.write(raw_fixed)
    except Exception as _ex_san:
        logger.warning("Échec de la sanitisation du fichier DXF %s : %s", dxf_path, _ex_san)


def apply_ancfcc_cadastral_standards(doc, scale: float = 2000.0, frame_poly=None, drafter: str = "el-bekay") -> None:
    """
    Applique l'ensemble des standards ANCFCC sur le document DXF.
    """
    apply_ancfcc_header_and_xdata(doc, int(scale))
    normalize_cartouche_and_standards(doc, scale=scale, frame_poly=frame_poly, drafter=drafter)
    msp = doc.modelspace()
    RejCadPrecisionLock.enforce_centimetric_precision(msp)


# Rétrocompatibilité interne avec les versions précédentes
_encode_autocad_unicode = encode_ancfcc_unicode
_apply_definitive_standards = apply_ancfcc_cadastral_standards
lock_rejcad_centimetric_precision = RejCadPrecisionLock.enforce_centimetric_precision
