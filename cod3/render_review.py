"""Review sheets for checking every catalog model by eye: 12 per sheet, numbered, two views each.

  python -I cod3/render_review.py <catalog folder> <category> <out folder> [first sheet] [last sheet]
Writes <out>/<category>_NNN.png (sheet NNN = models 12*(NNN-1) .. 12*NNN-1, catalog order by name) and
<out>/<category>_index.json (sheet -> model names), so a model can be named by "sheet 7 #3".
Each model gets a three-quarter view and a view from the opposite side.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gallery as RG  # noqa: E402

PER = 12
COLS = 4


def two_views(obj):
    a = RG.render(obj)
    # the opposite side: mirror the model in X and Z (turns it 180 degrees about the vertical)
    text = obj.read_text().splitlines()
    flipped = []
    for line in text:
        if line.startswith("v "):
            x, y, z = map(float, line.split()[1:4])
            line = f"v {-x} {y} {-z}"
        flipped.append(line)
    tmp = obj.with_name(obj.stem + "__back.obj")
    tmp.write_text("\n".join(flipped) + "\n")
    try:
        b = RG.render(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return a, b


def main():
    catalog, cat, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    last = int(sys.argv[5]) if len(sys.argv) > 5 else 10 ** 6
    out.mkdir(parents=True, exist_ok=True)
    models = sorted((c for c in json.loads((catalog / "catalog.json").read_text()) if c["category"] == cat),
                    key=lambda c: c["name"])
    index = {}
    t = RG.TILE
    for s in range(0, len(models), PER):
        n = s // PER + 1
        chunk = models[s:s + PER]
        index[n] = [c["name"] for c in chunk]
        if not first <= n <= last:
            continue
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * 2 * t, rows * (t + 20)), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        for k, c in enumerate(chunk):
            x, y = (k % COLS) * 2 * t, (k // COLS) * (t + 20)
            try:
                a, b = two_views(catalog / c["file"])
            except Exception:
                a = b = None
            for j, img in enumerate((a, b)):
                if img:
                    sheet.paste(img, (x + j * t, y + 20))
            d.text((x + 4, y + 3), f"#{k + 1} {c['name'].rsplit('.', 1)[0]} ({c['tris']} tris)"[:70], fill=(255, 220, 90))
        sheet.save(out / f"{cat}_{n:03d}.png")
        print(out / f"{cat}_{n:03d}.png", flush=True)
    (out / f"{cat}_index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
