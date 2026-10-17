import sys
sys.path.insert(0, r"C:\Users\Admin\Desktop\app dahby\DAHBY_PRO_SUITE")
from engines.dxf_cleaner.workbench import resolve_parcel_paths

org_fp, _ = resolve_parcel_paths("P2")
with open(org_fp, "rb") as f:
    content = f.read()

idx = 0
while True:
    pos = content.find(b"B19", idx)
    if pos == -1:
        break
    start = max(0, pos - 100)
    end = min(len(content), pos + 100)
    snippet = content[start:end].decode("latin1", errors="replace")
    print(f"Match at {pos}:\n{snippet}\n---")
    idx = pos + 3
