import sys, os, glob
sys.path.insert(0, r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")
from engines.dxf_cleaner.score_all import load_dxf_safe, extract_all_texts

target = (793361.27, 440450.86)
print("Searching for B19 near (793361.27, 440450.86)...")

for root, dirs, files in os.walk(r"C:\Users\Admin\Downloads"):
    for f in files:
        if f.lower().endswith(".dxf") and "p2" not in f.lower():
            fp = os.path.join(root, f)
            try:
                with open(fp, "rb") as fh:
                    raw = fh.read()
                if b"B19" in raw:
                    doc = load_dxf_safe(fp)
                    for t in extract_all_texts(doc.modelspace()):
                        if t.text == "B19":
                            d = ((t.x-target[0])**2 + (t.y-target[1])**2)**0.5
                            if d < 15:
                                print(f"FOUND B19 in {fp} at ({t.x:.2f}, {t.y:.2f}) dist={d:.2f}")
            except Exception:
                pass
