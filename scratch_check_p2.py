import sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")
from engines.dxf_cleaner.score_all import load_dxf_safe
from engines.dxf_cleaner.workbench import resolve_parcel_paths

org_fp, def_fp = resolve_parcel_paths("P2")
doc_org = load_dxf_safe(org_fp)
doc_def = load_dxf_safe(def_fp)

h_target = int("35318", 16)
for e in doc_org.modelspace():
    if hasattr(e.dxf, "handle"):
        h = int(e.dxf.handle, 16)
        if abs(h - h_target) <= 15:
            ins = getattr(e.dxf, "insert", None)
            pos_str = f"({ins.x:.2f}, {ins.y:.2f})" if ins else "None"
            txt = getattr(e, "text", getattr(e.dxf, "text", ""))
            name = getattr(e.dxf, "name", "")
            print(f"OREGEN Handle {hex(h)}: {e.dxftype()} name='{name}' text='{txt}' at {pos_str}")

print("\n--- DEF around B19 ---")
for e in doc_def.modelspace():
    if hasattr(e.dxf, "handle"):
        ins = getattr(e.dxf, "insert", None)
        if ins and abs(ins.x - 793361.27) < 5 and abs(ins.y - 440450.86) < 5:
            txt = getattr(e, "text", getattr(e.dxf, "text", ""))
            name = getattr(e.dxf, "name", "")
            print(f"DEF Handle {e.dxf.handle}: {e.dxftype()} name='{name}' text='{txt}' at ({ins.x:.2f}, {ins.y:.2f})")
