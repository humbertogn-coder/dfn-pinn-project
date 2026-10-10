"""Render the equations of the student guide to PNG (matplotlib mathtext) and write a manifest with sizes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "eq"
OUT.mkdir(exist_ok=True)
DPI = 220
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": 15})

eqs = json.loads((HERE / "equations.json").read_text())
manifest = {}
for key, tex in eqs.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${tex}$", fontsize=15, color="#0b0b0b")
    path = OUT / f"{key}.png"
    try:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    except Exception as exc:  # report and continue
        print("FAILED", key, exc)
        plt.close(fig)
        continue
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    manifest[key] = {"file": str(path), "w": w, "h": h, "dpi": DPI}
    print(f"{key}: {w}x{h}")
(HERE / "eq_manifest.json").write_text(json.dumps(manifest, indent=1))
