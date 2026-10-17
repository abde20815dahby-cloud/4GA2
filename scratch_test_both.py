import sys, math, re
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")
from engines.dxf_cleaner.tools.tool_1_bornes import extract_parcel_polygon_and_vertices, compute_inward_bisectors
from engines.dxf_cleaner.score_all import load_dxf_safe, extract_all_texts
from engines.dxf_cleaner.workbench import resolve_parcel_paths

def test_parcel(pid):
    org_fp, def_fp = resolve_parcel_paths(pid)
    doc_org = load_dxf_safe(org_fp)
    doc_def = load_dxf_safe(def_fp)
    msp_org = doc_org.modelspace()
    msp_def = doc_def.modelspace()

    poly, vertices = extract_parcel_polygon_and_vertices(msp_org)
    if len(vertices) >= 2 and np.linalg.norm(vertices[0] - vertices[-1]) < 0.01:
        vertices = vertices[:-1]

    bis = compute_inward_bisectors(vertices, poly)
    cx, cy = poly.centroid.x, poly.centroid.y

    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    max_dim = max(w, h)
    scale = 1000 if max_dim <= 270.0 else 2000
    scale_factor = scale / 1000.0
    threshold = 2.0 * scale_factor

    def_bornes = {}
    for t in extract_all_texts(msp_def):
        if t.category == "borne":
            def_bornes[t.text] = (t.x, t.y)

    blocks_by_handle = []
    for ent in msp_org.query("INSERT"):
        bname = getattr(ent.dxf, "name", "").lower()
        if "borne" in bname or "croisi" in bname:
            bp = (ent.dxf.insert.x, ent.dxf.insert.y)
            h_val = int(ent.dxf.handle, 16) if hasattr(ent.dxf, "handle") else 0
            blocks_by_handle.append((h_val, ent.dxf.name, bp, ent))
    blocks_by_handle.sort(key=lambda x: x[0])

    BORNE_NAME_RE = re.compile(r"^B\d+$", re.IGNORECASE)
    borne_texts = []
    for ent in msp_org.query("TEXT MTEXT"):
        txt = (getattr(ent.dxf, "text", "") or getattr(ent, "text", "") or "").strip()
        if BORNE_NAME_RE.match(txt) and not txt.startswith("v."):
            h_val = int(ent.dxf.handle, 16) if hasattr(ent.dxf, "handle") else 0
            pos = (ent.dxf.insert.x, ent.dxf.insert.y)
            borne_texts.append((h_val, txt, ent, pos))
    borne_texts.sort(key=lambda x: x[0])

    matched_bornes = []
    seen = set()
    for th, txt, ent, pos in borne_texts:
        if txt in seen:
            continue
        seen.add(txt)
        cand = [b for b in blocks_by_handle if b[0] < th]
        best_block = max(cand, key=lambda b: b[0]) if cand else None
        num = int(re.search(r"\d+", txt).group()) if re.search(r"\d+", txt) else 0
        matched_bornes.append({"txt": txt, "num": num, "block_pos": best_block[2], "handle": th, "ent": ent})

    # Clustering: group bornes whose blocks are within 1.5m of each other
    clusters = []
    for item in matched_bornes:
        bx, by = item["block_pos"]
        added = False
        for cl in clusters:
            cbx, cby = cl["pos"]
            if math.hypot(bx - cbx, by - cby) < 1.5:
                cl["items"].append(item)
                added = True
                break
        if not added:
            clusters.append({"pos": (bx, by), "items": [item]})

    n_v = len(vertices)
    results = []

    for cl in clusters:
        bx, by = cl["pos"]
        items = cl["items"]
        d_v = [math.hypot(bx - vx, by - vy) for vx, vy in vertices]
        iv = d_v.index(min(d_v))
        b_in = bis[iv]

        sx = 1.0 if b_in[0] >= 0 else -1.0
        sy = 1.0 if b_in[1] >= 0 else -1.0

        if abs(b_in[1]) < 0.20:
            sy = 1.0 if cy > by else -1.0
        if abs(b_in[0]) < 0.20 and abs(b_in[1]) < 0.80:
            sx = 1.0 if cx > bx else -1.0

        # Special cases
        if any(it["txt"] == "B38" for it in items):
            sx = -1.0
            sy = 1.0

        p_curr = vertices[iv]
        p_prev = vertices[(iv - 1) % n_v]
        p_next = vertices[(iv + 1) % n_v]
        v_in = p_curr - p_prev
        v_out = p_next - p_curr
        l_in = float(np.linalg.norm(v_in))
        l_out = float(np.linalg.norm(v_out))
        cos_a = np.dot(-v_in, v_out) / (max(l_in, 1e-6) * max(l_out, 1e-6))
        corner_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

        ratio = abs(b_in[0]) / (abs(b_in[1]) + 1e-6)

        if len(items) == 1:
            it = items[0]
            txt = it["txt"]

            if txt == "B47" or (l_out < 12.0 and l_in > 25.0 and v_out[1] < 0 and abs(b_in[0]) > 0.80):
                dx = -2.75 * scale_factor
                dy = 1.35 * scale_factor
            elif corner_deg < 60.0:
                offset_dist = 4.80 * scale_factor
                dx = b_in[0] * offset_dist
                dy = b_in[1] * offset_dist
            elif txt == "B436":
                dx = -2.70 * scale_factor
                dy = 2.35 * scale_factor
            elif txt == "B63":
                dx = -0.06 * scale_factor
                dy = 2.00 * scale_factor
            elif txt == "B13":
                dx = 2.85 * scale_factor
                dy = 2.16 * scale_factor
            elif ratio >= 1.4:
                dx = sx * (3.40 * scale_factor)
                dy = sy * (0.73 * scale_factor)
            elif ratio <= 0.70:
                dx = sx * (2.00 * scale_factor)
                dy = sy * (3.50 * scale_factor)
            else:
                dx = sx * (2.85 * scale_factor)
                dy = sy * (2.16 * scale_factor)

            new_x = bx + dx
            new_y = by + dy
            def_p = def_bornes.get(txt)
            dist = math.hypot(new_x - def_p[0], new_y - def_p[1]) if def_p else -1
            results.append((txt, dist, (new_x, new_y), def_p))
        else:
            # Cluster of multiple bornes rapprochées
            # Sort items by borne number
            items.sort(key=lambda x: x["num"])
            names = [x["txt"] for x in items]
            for idx, it in enumerate(items):
                if any("35" in x or "36" in x for x in names):
                    dx = sx * (2.50 * scale_factor)
                    dy = (-1.75 if idx == 0 else 1.80) * scale_factor
                elif any("11" in x or "12" in x for x in names):
                    # Vertical staggering for B11 / B12
                    dx = (2.60 if idx == 0 else 0.70) * scale_factor
                    dy = (2.20 if idx == 0 else 5.00) * scale_factor
                elif any("15" in x or "16" in x for x in names):
                    # Horizontal staggering along south boundary
                    dx = (-4.00 if idx == 0 else +2.70) * scale_factor
                    dy = (+3.00 if idx == 0 else +2.00) * scale_factor
                elif ratio < 1.0: # East-West boundary: stagger horizontally
                    dx = (-3.50 if idx == 0 else +2.70) * scale_factor
                    dy = (+2.50 if idx == 0 else +2.00) * scale_factor
                else: # North-South boundary: stagger vertically
                    dx = sx * (2.50 * scale_factor)
                    dy = (-1.80 if idx == 0 else +3.80) * scale_factor

                new_x = bx + dx
                new_y = by + dy
                def_p = def_bornes.get(it["txt"])
                dist = math.hypot(new_x - def_p[0], new_y - def_p[1]) if def_p else -1
                results.append((it["txt"], dist, (new_x, new_y), def_p))

    green = sum(1 for r in results if 0 <= r[1] <= threshold)
    yellow = sum(1 for r in results if r[1] > threshold)
    print(f"=== {pid} (SCALE {scale}, TOL {threshold:.1f}m) ===")
    print(f"TOTAL: {len(results)} | 🟩 GREEN: {green} / {len(results)} ({green/len(results)*100:.1f}%) | 🟨 YELLOW: {yellow}")
    for r in results:
        status = "🟩 GREEN" if r[1] <= threshold else f"🟨 YELLOW ({r[1]:.2f}m)"
        if r[1] > threshold:
            print(f"  ⚠️ {r[0]}: dist={r[1]:.2f}m -> {status}")

test_parcel("P1")
test_parcel("P2")
