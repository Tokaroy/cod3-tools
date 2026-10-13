"""Sheet of an archive's exported models whose names match a pattern (quick visual check).

  python -I cod3/render_models.py <models root> <archive> <name regex> <out.png>
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gallery as RG  # noqa: E402

root, arch, pat, out = Path(sys.argv[1]), sys.argv[2], re.compile(sys.argv[3], re.I), sys.argv[4]
M = root / arch / "models"
ms = [m for m in json.loads((M / "models.json").read_text()) if pat.search(m["name"])]
cols = min(4, max(1, len(ms)))
s = Image.new("RGB", (RG.TILE * cols, (RG.TILE + 16) * ((len(ms) + cols - 1) // cols)), (20, 22, 26))
d = ImageDraw.Draw(s)
for i, m in enumerate(ms):
    x, y = (i % cols) * RG.TILE, (i // cols) * (RG.TILE + 16)
    t = RG.render(M / m["file"])
    if t:
        s.paste(t, (x, y + 16))
    d.text((x + 4, y + 2), m["name"], fill=(255, 220, 90))
s.save(out)
print(out, [m["name"] for m in ms])
