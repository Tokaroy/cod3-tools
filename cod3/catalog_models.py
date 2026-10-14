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

import numpy as np

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


def drop_hulls(lines, collision=()):
    """Leave out surfaces that aren't drawn: collision hulls (surface numbers in `collision`, the linker's
    "collision" surfaces) and simple hull surfaces (shadow/occlusion shapes): under 64 vertices yet spanning
    at least 90% of the whole model on every axis. Drawn opaque they hide the model. A model is never left
    without surfaces: the size test only drops a surface while another one remains."""
    import numpy as np
    verts = np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith("v ")] or [[0, 0, 0]])
    full = np.ptp(verts, 0) + 1e-6
    head, blocks = [], []
    for l in lines:
        if l.startswith("g "):
            blocks.append([l])
        elif blocks:
            blocks[-1].append(l)
        else:
            head.append(l)

    def faces(block):
        return {int(t.split("/")[0]) - 1 for l in block if l.startswith("f ") for t in l.split()[1:4]}

    def number(block):
        m = re.match(r"g surface(\d+)", block[0])
        return int(m.group(1)) if m else -1

    drop = {i for i, b in enumerate(blocks) if number(b) in collision}
    for i, b in enumerate(blocks):
        idx = faces(b)
        others = [j for j, o in enumerate(blocks) if j != i and j not in drop and faces(o)]
        if i not in drop and others and idx and len(idx) < 64 and (np.ptp(verts[sorted(idx)], 0) >= 0.9 * full).all():
            drop.add(i)
    out = list(head)
    for i, b in enumerate(blocks):
        out.extend(l for l in b if not (i in drop and l.startswith("f ")))  # keep vertices (indices stay valid)
    return out


_FIXES = Path(__file__).resolve().parent / "model_fixes.json"
ASSEMBLIES = {k: v for k, v in (json.loads(_FIXES.read_text()).get("_assemblies", {}) if _FIXES.exists() else {}).items()
              if not k.startswith("_")}
PLACEHOLDER = 0.01  # metres: a model smaller than this across is an invisible marker (vg_hullgun: one 0.1 mm triangle)
VARIANTS = {k.lower(): v for k, v in (json.loads(_FIXES.read_text()).get("_variants", {}) if _FIXES.exists() else {}).items()}
_ALPHA = {}


def see_through(png):
    """Whether a texture has see-through parts (alpha below half on at least 2% of it) - written as map_d so
    viewers like Blender use its alpha."""
    if png not in _ALPHA:
        from PIL import Image
        import numpy as np
        try:
            im = Image.open(png)
            _ALPHA[png] = im.mode in ("RGBA", "LA") and float((np.asarray(im.getchannel("A")) < 128).mean()) >= 0.02
        except Exception:
            _ALPHA[png] = False
    return _ALPHA[png]


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
        collision = {k for k, x in enumerate(links_cache[archive].get(m["name"], [])) if x.get("how") == "collision"}
        obj = drop_hulls(obj, collision)
        V = np.array([[float(x) for x in l.split()[1:4]] for l in obj if l.startswith("v ")])
        if len(V) and np.linalg.norm(V.max(0) - V.min(0)) < PLACEHOLDER:
            continue
        # the model as linked, plus any hand-listed variants (model_fixes.json "_variants": a copy with some
        # textures swapped, e.g. the German multiplayer flag with and without its flag texture)
        versions = [("", texs, files)] + [
            (f"__{v}", [swap.get(t, t) for t in texs], [f if swap.get(t, t) == t else None for t, f in zip(texs, files)])
            for v, swap in VARIANTS.get(key, {}).items()]
        for suffix, vtexs, vfiles in versions:
            vout = out.with_name(out.stem + suffix + ".obj")
            vout.write_text("\n".join(f"mtllib {vout.stem}.mtl" if l.startswith("mtllib") else l for l in obj) + "\n")
            mtl = []
            for k in range(m["surfaces"]):
                mtl += [f"newmtl s{k}", "Kd 1 1 1"]
                t = vtexs[k] if k < len(vtexs) else None
                if t and t in index:
                    src = vfiles[k] if k < len(vfiles) and vfiles[k] else index[t]
                    fn = re.sub(r"[^A-Za-z0-9_.$-]", "_", t) + ("" if src == index[t] else
                         "__" + re.sub(r"[^A-Za-z0-9_]", "_", src.split("/textures_named/")[0])) + ".png"
                    if fn not in copied:
                        shutil.copyfile(root / src, dest / "textures" / fn)
                        copied.add(fn)
                    mtl.append(f"map_Kd ../textures/{fn}")
                    if see_through(dest / "textures" / fn):  # propeller discs, camo nets, foliage
                        mtl.append(f"map_d ../textures/{fn}")
            vout.with_suffix(".mtl").write_text("\n".join(mtl) + "\n")
            stem, ext = key.rsplit(".", 1)
            catalog.append({"name": f"{stem}{suffix}.{ext}", "category": cat,
                            "file": str(vout.relative_to(dest)).replace("\\", "/"),
                            "archive": str(archive.relative_to(root)).replace("\\", "/"), "textures": vtexs,
                            "skinned": m["skinned"], "surfaces": m["surfaces"], "verts": m["verts"], "tris": m["tris"],
                            **({"variant_of": key} if suffix else {})})
    # stands with their guns on (model_fixes.json "_assemblies"): each piece at its offset (catalog metres) - the
    # vehicle skeleton's distance from the stand's tag to the gun's, so the gun sits on its stand
    entries = {c["name"].lower(): c for c in catalog}
    for name, a in ASSEMBLIES.items():
        pieces = [(entries.get(k.lower()), off) for k, off in a["pieces"]]
        if not all(c for c, _ in pieces):
            print(f"  ! assembly {name}: missing {[k for (k, _), (c, _) in zip(a['pieces'], pieces) if not c]}", flush=True)
            continue
        cat = pieces[0][0]["category"]
        out = dest / cat / f"{name}.obj"
        lines, mtl, nv, nvt = [f"mtllib {name}.mtl"], [], 0, 0
        for k, (c, off) in enumerate(pieces):
            src = (dest / c["file"]).read_text().splitlines()
            v = [l for l in src if l.startswith("v ")]
            vt = [l for l in src if l.startswith("vt ")]
            for l in v:
                x, y, z = (float(t) for t in l.split()[1:4])
                lines.append(f"v {x + off[0]:.5f} {y + off[1]:.5f} {z + off[2]:.5f}")
            lines += vt
            for l in src:
                if l.startswith("usemtl"):
                    lines.append(f"usemtl p{k}_{l.split(None, 1)[1]}")
                elif l.startswith("g "):
                    lines.append(f"g p{k}_{l[2:]}")
                elif l.startswith("f "):
                    refs = []
                    for r in l.split()[1:]:
                        t = r.split("/")
                        refs.append(f"{int(t[0]) + nv}" + (f"/{int(t[1]) + nvt}" if len(t) > 1 and t[1] else ""))
                    lines.append("f " + " ".join(refs))
            for l in (dest / c["file"]).with_suffix(".mtl").read_text().splitlines():
                mtl.append(f"newmtl p{k}_{l.split(None, 1)[1]}" if l.startswith("newmtl") else l)
            nv, nvt = nv + len(v), nvt + len(vt)
        out.write_text("\n".join(lines) + "\n")
        out.with_suffix(".mtl").write_text("\n".join(mtl) + "\n")
        catalog.append({"name": f"{name}.xemesh", "category": cat, "file": str(out.relative_to(dest)).replace("\\", "/"),
                        "archive": pieces[0][0]["archive"], "textures": sum((c["textures"] for c, _ in pieces), []),
                        "skinned": False, "surfaces": sum(c["surfaces"] for c, _ in pieces),
                        "verts": sum(c["verts"] for c, _ in pieces), "tris": sum(c["tris"] for c, _ in pieces),
                        "assembly_of": [c["name"] for c, _ in pieces]})
    (dest / "catalog.json").write_text(json.dumps(catalog, indent=1))
    listed = {c["file"] for c in catalog}
    old_models = [p for c in {c["category"] for c in catalog} for p in (dest / c).glob("*.obj")
                  if f"{c}/{p.name}" not in listed]
    for p in old_models:  # a model no longer catalogued (an invisible placeholder, a renamed file)
        p.unlink()
        p.with_suffix(".mtl").unlink(missing_ok=True)
    if old_models:
        print(f"removed {len(old_models)} models no longer catalogued", flush=True)
    stale = [p for p in (dest / "textures").glob("*.png") if p.name not in copied]
    for p in stale:  # left from an earlier build (an old decode or link): nothing refers to it any more
        p.unlink()
    if stale:
        print(f"removed {len(stale)} textures no model uses any more", flush=True)
    counts = {}
    for c in catalog:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    print(f"{len(catalog)} unique models: {counts}; {len(copied)} textures")


if __name__ == "__main__":
    main()
