"""Which catalog models still lack textures: per category, models with untextured surfaces (collision
hulls excluded), worst first.

  python -I cod3/texture_report.py <catalog folder> [category] [models root] [--top N]
The models root (extract_models output, for links.json) defaults to <catalog>/../models.
"""
import json
import sys
from pathlib import Path


def main():
    argv = sys.argv[1:]
    top = 30
    if "--top" in argv:
        i = argv.index("--top")
        top = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    cat = json.loads((Path(args[0]) / "catalog.json").read_text())
    if len(args) > 1:
        cat = [c for c in cat if c["category"] == args[1]]
    root = Path(args[2]) if len(args) > 2 else Path(args[0]).parent / "models"
    links = {}
    rows, per = [], {}
    for c in cat:
        if c["archive"] not in links:
            lf = root / c["archive"] / "links.json"
            links[c["archive"]] = json.loads(lf.read_text()) if lf.exists() else {}
        how = next((v for k, v in links[c["archive"]].items() if k.lower().split("\\")[-1] == c["name"]), [])
        texs = [t for k, t in enumerate(c.get("textures") or [])
                if not (k < len(how) and how[k].get("how") == "collision")]  # collision hulls: no texture
        missing = sum(1 for t in texs if not t)
        p = per.setdefault(c["category"], [0, 0])
        p[0] += 1
        p[1] += bool(missing)
        if missing:
            rows.append((missing / max(len(texs), 1), c["tris"], c["name"], c["category"], missing, len(texs)))
    for k, (n, m) in sorted(per.items()):
        print(f"{k}: {m} of {n} models have untextured surfaces")
    for frac, tris, name, category, missing, total in sorted(rows, key=lambda r: (-r[0], -r[1]))[:top]:
        print(f"  {category:10} {name:45} {missing}/{total} surfaces untextured ({tris} tris)")


if __name__ == "__main__":
    main()
