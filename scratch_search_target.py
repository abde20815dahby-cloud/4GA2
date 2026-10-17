import glob
from ezdxf.recover import readfile as recover_read

files = glob.glob(r"C:\Users\Admin\Downloads\SS\*.dxf")
target = (793361.27, 440450.86)
print(f"Scanning {len(files)} sector files for text near target {target}...")

for f in files:
    try:
        doc, _ = recover_read(f)
        for e in doc.modelspace().query("TEXT MTEXT"):
            t = (getattr(e.dxf, "text", "") or getattr(e, "text", "") or "").strip()
            if hasattr(e.dxf, "insert"):
                p = (e.dxf.insert.x, e.dxf.insert.y)
                d = ((p[0]-target[0])**2 + (p[1]-target[1])**2)**0.5
                if d < 100:
                    print(f"Found '{t}' in {f} at {p} dist={d:.2f}")
    except Exception as exc:
        print("Error in", f, exc)
