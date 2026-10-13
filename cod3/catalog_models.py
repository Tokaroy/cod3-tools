"""Gather every exported Call of Duty 3 model (extract_models.py output) into one folder per category.

  python -I cod3/catalog_models.py <extracted root> <catalog folder>
Each unique model name is kept once, the copy with the most triangles (archives repeat shared models).
Writes <catalog>/<category>/<name>.obj + .mtl, every colour texture they use once in
<catalog>/textures/, and <catalog>/catalog.json (name, category, source archive, counts, textures).
The folder is self-contained (relative paths), so it can be moved or zipped. Categories come from the
game's own name prefixes. Run link_textures.py first so the models have their textures.
"""
import json
import re
import shutil
import sys
from pathlib import Path

LEVELS = ("mp_", "mayenne", "forest", "hostage", "crossroads", "island", "fuelplant", "falaise", "nightdrop",
          "stlambert", "chambois", "blackbaron", "saint", "laisonriver", "mace", "credits", "sky", "shadow")
WEAPONS = ("w_", "vg_", "bazooka", "lee_enfield", "1903springfield", "g43", "kar98", "m1garand", "thompson",
           "mp40", "stg44", "bar_", "springfield", "mg42", "flak88_shell")
VEHICLES = ("v_", "p_husk_", "stuka", "pantherv", "bicycle", "tiger", "sherman", "halftrack", "jeep")


def category(name: str) -> str:
    n = name.lower().split("\\")[-1]
    if n.endswith(".xecell") or n.startswith(("sbmodel", "bmodel")) or n.startswith(LEVELS) or re.match(r"^\d", n):
        return "scenery"
    if n.startswith(("c_", "_uspriv", "corpse")):
        return "characters"
    if n.startswith(WEAPONS):
        return "weapons"
    if n.startswith(VEHICLES):
        return "vehicles"
    if n.startswith("pfx_"):
        return "effects"
    return "props"


def drop_hulls(lines):
    """Leave out simple hull surfaces (shadow/occlusion shapes): under 64 vertices yet spanning at least
    90% of the whole model on every axis. Drawn opaque they hide the model."""
    import numpy as np
    verts = np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith("v ")] or [[0, 0, 0]])
    full = np.ptp(verts, 0) + 1e-6
    out, block, used = [], [], set()

    def flush():
        if not block:
            return
        idx = {int(t.split("/")[0]) - 1 for l in block if l.startswith("f ") for t in l.split()[1:4]}
        if idx and len(idx) < 64 and (np.ptp(verts[sorted(idx)], 0) >= 0.9 * full).all():
            out.extend(l for l in block if not l.startswith("f "))  # keep vertices (indices stay valid)
        else:
            out.extend(block)

    for l in lines:
        if l.startswith("g "):
            flush()
            block = [l]
        elif block:
            block.append(l)
        else:
            out.append(l)
    flush()
    return out


def main():
    root, dest = Path(sys.argv[1]), Path(sys.argv[2])
    best = {}
    for listing in root.rglob("models/models.json"):
        archive = listing.parent.parent
        for m in json.loads(listing.read_text()):
            key = m["name"].lower().split("\\")[-1]
            if key not in best or m["tris"] > best[key][1]["tris"]:
                best[key] = (archive, m)
    index = json.loads((root / "texture_index.json").read_text()) if (root / "texture_index.json").exists() else {}
    if (root / "texture_index_bound.json").exists():  # bound-only copies (link_textures.py)
        index.update(json.loads((root / "texture_index_bound.json").read_text()))
    links_cache, copied, catalog = {}, set(), []
    (dest / "textures").mkdir(parents=True, exist_ok=True)
    for key, (archive, m) in sorted(best.items()):
        cat = category(key)
        out = dest / cat / (re.sub(r"[^A-Za-z0-9_.-]", "_", key).rsplit(".", 1)[0] + ".obj")
        out.parent.mkdir(parents=True, exist_ok=True)
        if archive not in links_cache:
            lf = archive / "links.json"
            links_cache[archive] = json.loads(lf.read_text()) if lf.exists() else {}
        texs = [x["texture"] for x in links_cache[archive].get(m["name"], [])]
        files = [x.get("file") for x in links_cache[archive].get(m["name"], [])]  # a per-model copy choice
        obj = (archive / "models" / m["file"]).read_text().splitlines()
        obj = [f"mtllib {out.stem}.mtl" if l.startswith("mtllib") else l for l in obj]
        obj = drop_hulls(obj)
        out.write_text("\n".join(obj) + "\n")
        mtl = []
        for k in range(m["surfaces"]):
            mtl += [f"newmtl s{k}", "Kd 1 1 1"]
            t = texs[k] if k < len(texs) else None
            if t and t in index:
                src = files[k] if k < len(files) and files[k] else index[t]
                fn = re.sub(r"[^A-Za-z0-9_.$-]", "_", t) + ("" if src == index[t] else
                     "__" + re.sub(r"[^A-Za-z0-9_]", "_", src.split("/textures_named/")[0])) + ".png"
                if fn not in copied:
                    shutil.copyfile(root / src, dest / "textures" / fn)
                    copied.add(fn)
                mtl.append(f"map_Kd ../textures/{fn}")
        out.with_suffix(".mtl").write_text("\n".join(mtl) + "\n")
        catalog.append({"name": key, "category": cat, "file": str(out.relative_to(dest)).replace("\\", "/"),
                        "archive": str(archive.relative_to(root)).replace("\\", "/"), "textures": texs,
                        "skinned": m["skinned"], "surfaces": m["surfaces"], "verts": m["verts"], "tris": m["tris"]})
    (dest / "catalog.json").write_text(json.dumps(catalog, indent=1))
    counts = {}
    for c in catalog:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"{len(catalog)} unique models: {counts}; {len(copied)} textures")


if __name__ == "__main__":
    main()
