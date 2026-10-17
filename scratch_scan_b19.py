import sys, glob, os
sys.path.insert(0, r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")
from engines.dxf_cleaner.score_all import load_dxf_safe, extract_all_texts

oregen_files = glob.glob(r"C:\Users\Admin\Downloads\Sous_Zone 01\IGHANOUYINE\OREGEN\**\*.dxf", recursive=True)
target = (793361.27, 440450.86)
print(f"Scanning {len(oregen_files)} OREGEN files...")
for of in oregen_files:
    try:
        doc = load_dxf_safe(of)
        for t in extract_all_texts(doc.modelspace()):
            if t.text == "B19":
                d = ((t.x-target[0])**2 + (t.y-target[1])**2)**0.5
                if d < 50:
                    print(f"Found B19 in OREGEN: {os.path.basename(os.path.dirname(of))} at ({t.x:.2f}, {t.y:.2f}) dist={d:.2f}")
    except Exception:
        pass
