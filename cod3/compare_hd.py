"""Check our decoded COD3 textures against a community HD set (World at War mod images converted with
iwi_to_png.py, upscaled from correctly decoded originals; partly renamed, plus recoloured variants).

  python -I cod3/compare_hd.py <models root> <hd png folder> <report.json> [sheet.png]
Each HD colour image ("~..." normal/specular maps, "mp_" copies and recolours skipped) is compared with
every character and weapon colour texture of ours (texture_index.json: c_*, of_c_*, w_*), shrunk to 64x64
grey (correlation). The best match tells which of our textures it is: a strong one (>= 0.85) confirms
ours, a middling one (0.5..0.85) means ours is that texture decoded from a wrong spot or with damage, a
weak one means we have no good copy of it. A name match is used when the names agree. Writes the
pairs, worst first, and optionally a sheet of the doubtful ones (ours | HD).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SKIP = ("_afrika", "_winter", "_wn_", "_wn", "_alpha", "_medic", "_v2", "wet", "dirt")


def vec(path, size=64):
    a = np.asarray(Image.open(path).convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    a = a - a.mean()
    n = float(np.linalg.norm(a))
    return a / n if n else a


if __name__ == "__main__":
    root, hd, report = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    index = json.loads((root / "texture_index.json").read_text())
    ours = sorted(n for n in index if n.startswith(("c_", "of_c_", "w_")) and n.endswith(("_c", "_diff", "_d")))
    names = []
    vecs = []
    for n in ours:
        try:
            vecs.append(vec(root / index[n]))
            names.append(n)
        except Exception:
            pass
    M = np.stack(vecs)
    rows = []
    for f in sorted(hd.glob("*.png")):
        h = f.stem.lower()
        if h.startswith(("~", "mp_", "char_rus")) or any(s in h for s in SKIP):
            continue
        v = vec(f)
        scores = M @ v
        k = int(np.argmax(scores))
        same = h if h in index else None
        rows.append({"hd": h, "best": names[k], "corr": round(float(scores[k]), 3),
                     "name_match": same, "name_corr": round(float(M[names.index(same)] @ v), 3) if same in names else None,
                     "ours": index[names[k]], "hd_file": str(f)})
    rows.sort(key=lambda r: r["corr"])
    report.write_text(json.dumps(rows, indent=1))
    good = [r for r in rows if r["corr"] >= 0.85]
    mid = [r for r in rows if 0.5 <= r["corr"] < 0.85]
    weak = [r for r in rows if r["corr"] < 0.5]
    print(f"{len(rows)} HD images: {len(good)} match ours well, {len(mid)} partly (ours shifted/damaged?), "
          f"{len(weak)} have no good copy of ours")
    for r in mid + weak:
        print(f"  {r['corr']:6.3f} {r['hd']:34} ~ {r['best']}")
    if len(sys.argv) > 4:
        doubt = (weak + mid)[:40]
        t = 150
        sheet = Image.new("RGB", (4 * 2 * t, max(1, (len(doubt) + 3) // 4) * (t + 14)), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(doubt):
            x, y = (k % 4) * 2 * t, (k // 4) * (t + 14)
            sheet.paste(Image.open(root / r["ours"]).convert("RGB").resize((t, t)), (x, y + 14))
            sheet.paste(Image.open(r["hd_file"]).convert("RGB").resize((t, t)), (x + t, y + 14))
            d.text((x + 2, y + 1), f"{r['hd']} {r['corr']}"[:44], fill=(255, 220, 90))
        sheet.save(sys.argv[4])
