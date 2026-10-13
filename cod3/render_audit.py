"""Contact sheets of the models texture_audit.py flagged, worst first, to check them by eye.

  python -I cod3/render_audit.py <catalog folder> <out prefix> [max models] [min score]
Skips models with no texture at all (nothing to check). Labels show the score and the worst reason.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gallery as RG  # noqa: E402


def main():
    catalog, prefix = Path(sys.argv[1]), sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
    res = json.loads((catalog / "texture_audit.json").read_text())
    picks = [r for r in res if r["score"] >= floor and any(s["texture"] for s in r["surfaces"])
             and r["category"] != "scenery"][:limit]
    per = RG.COLS * RG.ROWS
    for s in range(0, len(picks), per):
        sheet = Image.new("RGB", (RG.COLS * RG.TILE, RG.ROWS * (RG.TILE + 30)), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(picks[s:s + per]):
            try:
                img = RG.render(catalog / r["file"])
            except Exception:
                img = None
            x, y = (k % RG.COLS) * RG.TILE, (k // RG.COLS) * (RG.TILE + 30)
            if img:
                sheet.paste(img, (x, y + 30))
            worst = max(r["surfaces"], key=lambda q: q["bad"])
            d.text((x + 4, y + 2), f"{r['score']:.2f} {r['name']}"[:52], fill=(255, 220, 90))
            d.text((x + 4, y + 15), f"{worst['texture']} {worst['why']}"[:52], fill=(180, 180, 180))
        out = f"{prefix}_{s // per + 1:02d}.png"
        sheet.save(out)
        print(out, flush=True)


if __name__ == "__main__":
    main()
