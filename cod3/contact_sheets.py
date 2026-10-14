"""Picture sheets of the COD3 catalog: every model, one three-quarter view each, named, 30 per sheet.

  python -I cod3/contact_sheets.py <catalog folder> <out folder> [category ...]
Categories: characters, soldiers (the assembled ones), weapons, vehicles, props, scenery, effects (default:
all). Writes <out>/<category>/<category>_NN.png. Renders in parallel (render_gallery.render).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_gallery as RG  # noqa: E402

TILE, COLS, ROWS = 300, 6, 5
ALL = ("characters", "soldiers", "weapons", "vehicles", "props", "scenery", "effects")


def one(path):
    RG.TILE = TILE
    try:
        return RG.render(Path(path))
    except Exception:
        return None


def sheets(files, out, title, pool):
    imgs = list(pool.map(one, files, chunksize=4))
    per = COLS * ROWS
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{out.name}_*.png"):
        old.unlink()
    for k in range(0, len(files), per):
        chunk = list(zip(files[k:k + per], imgs[k:k + per]))
        rows = (len(chunk) + COLS - 1) // COLS
        s = Image.new("RGB", (COLS * TILE, rows * (TILE + 18) + 30), (24, 24, 24))
        d = ImageDraw.Draw(s)
        d.text((8, 8), f"{title}  ({k + 1}-{k + len(chunk)} of {len(files)})", fill=(255, 255, 255))
        for i, (f, img) in enumerate(chunk):
            x, y = (i % COLS) * TILE, (i // COLS) * (TILE + 18) + 30
            if img is not None:
                s.paste(img if img.size == (TILE, TILE) else img.resize((TILE, TILE)), (x, y + 18))
            d.text((x + 4, y + 3), Path(f).stem[:44], fill=(255, 220, 90))
        name = out / f"{out.name}_{k // per + 1:02d}.png"
        s.save(name)
        print(name, flush=True)


def main():
    cat, out = Path(sys.argv[1]), Path(sys.argv[2])
    wanted = sys.argv[3:] or ALL
    models = json.loads((cat / "catalog.json").read_text())
    with ProcessPoolExecutor(6) as pool:
        for c in wanted:
            if c == "soldiers":
                files = sorted(str(p) for p in (cat / "characters_assembled").glob("*.obj"))
            else:
                files = [str(cat / m["file"]) for m in sorted(models, key=lambda m: m["name"].lower()) if m["category"] == c]
            if files:
                sheets(files, out / c, f"COD3 {c}", pool)


if __name__ == "__main__":  # the process pool's workers re-import this file
    main()
