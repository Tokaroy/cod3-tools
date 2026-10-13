"""Give every exported Call of Duty 3 model its textures: writes <archive>/models/<model>.mtl.

  python -I cod3/link_textures.py <extracted root> [archive subpath ...]
Needs extract_models.py output (models.json with per-surface materials, pack_names.json) and
extract_textures.py's texture_index.json (texture name -> PNG). For each surface the colour texture is:
  1. the diffuse texture bound in its material record;
  2. else a bound normal/specular map's colour sibling (name_n / name_s -> name_c);
  3. else the colour texture (name ending _c) whose name shares the most words with the material and
     model names, preferring names listed in the model's own pack; ties go to the texture whose painted
     area the surface's UVs cover best;
  4. else the colour texture among those the model's pack binds at load time (its name table after
     "image"/"physical") whose painted area the surface's UVs fit best.
Collision hulls and tags (materials col_*, tag_*) get none.
Also writes links.json per archive: model -> [texture per surface] and how each was chosen.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repair_texture as RT  # noqa: E402

STOP = {"c", "ng", "p", "v", "w", "o", "lod", "lod1", "lod2", "lod3", "lod4", "d1", "xemesh", "xecell", "mil",
        "lvl", "env", "prp", "dec", "x", "of", "n", "s", "a", "us", "ge", "the", "ng2", "col", "tag",
        "sta", "frn", "msc", "mp", "ng1", "ng3", "pg",  # prop-name prefixes match nothing
        "world", "weapon"}  # on every weapon texture: says nothing about which weapon
# words naming one part of a model: a texture for that part only fits materials for that part
PART_WORDS = {"glass", "gls", "wheel", "wheels", "tire", "tires", "tread", "lens", "interior", "engine", "seat",
              "canvas", "reflect", "track", "tracks", "eye", "eyes", "teeth", "camo", "head", "helm", "gear"}
DIFFUSE = ("diffuse_texture", "diffuse_map_texture", "diffuse__w_alpha__texture", "diffuse_texture_texture")
RES = 64


NON_COLOUR = ("_n", "_s", "_a", "_spec", "_mask", "_t", "_tb", "_sw", "_h", "_ao", "_lm", "_refl", "_env",
              "_environment", "_hdr", "_n_ng", "_s_ng")
COLOUR = ("_c", "_diff", "_col", "_color", "_colour", "_c_ng", "_d", "_world")


def is_colour(n):
    """Colour (diffuse) maps: _c and the other suffixes the weapons use; also suffix-less material
    textures (cloth$..., wood$...) that aren't normal/specular maps."""
    n = n.lower()
    if n.endswith(NON_COLOUR) or n.startswith(("lm", "hud_", "__")):
        return False
    return n.endswith(COLOUR) or "$" in n


# the game's weapon codenames -> words used in their texture names
ALIASES = {"usrfl": {"m1garand", "garand"}, "usrfl_vm": {"m1garand"}, "gemp44": {"mp44"}, "gesmg": {"mp40"},
           "usshot": {"shotgun"}, "ussmg": {"tommy"}, "gelmg": {"mg34", "mg42"}, "rfl": {"k98", "kar98"},
           "pist": {"p38", "colt"}, "brsmg": {"sten"}, "usrokt": {"bazooka"}, "gerokt": {"panzerschrek"},
           "30cal": {"30cal", "browning"}, "usbar": {"bar"}, "brrfl": {"enfield"}, "usnade": {"nade"},
           "genade": {"genade"}, "usarms": {"usarms"}, "panzezriv": {"panzer", "panzeriv"},  # the game's own spelling
           "m1": {"m1garand"}}


def runtime_texture(bound, model, index):
    """The texture the game puts in at runtime for a binding that isn't a texture of its own:
    soldier pieces bind a placeholder (c_uspriv_ng_head_text) that the game fills with that head's or
    body's sheet (c_uspriv_head2_c, of_c_pnzgren_head2_c, c_brtinf1_head2_c), and some names carry an
    "of_" prefix only their texture lacks, or the other way round."""
    if not bound:
        return None
    if bound.startswith("of_") and bound[3:] in index:
        return bound[3:]
    if "of_" + bound in index:
        return "of_" + bound
    ph = re.fullmatch(r"(c_\w+?)_ng_(head|body)_text", bound)
    if not ph:
        return None
    kind = ph.group(2)
    mtype = model.lower().split("_ng", 1)[0]
    # this piece's own letter and number, when it is that kind of piece (c_uspriv_ng_lod2_head2,
    # c_pnzgren_ng_Lod2_head_4, c_wrmcht_ng_lod2_headb_1)
    num = re.search(kind + r"([a-z]?)_?(\d+)", model.lower().split("_ng", 1)[-1])
    tries = [(num.group(1), num.group(2))] if num else []
    tries += [("", num.group(2))] if num and num.group(1) else []
    for letter, n in tries + [("", "1")]:
        for t in dict.fromkeys((mtype, ph.group(1))):
            k = kind + letter
            names = [f"{t}_{k}{n}_c", f"of_{t}_{k}{n}_c", f"{t}1_{k}{n}_c", f"{t}_{k}{n}mp_c"]
            if n == "1":  # the first one is often unnumbered (of_c_wrmcht_head_c)
                names += [f"{t}_{k}_c", f"of_{t}_{k}_c"]
            for name in names:
                if name in index:
                    return name
    return None


def soldier_piece(model):
    """'head' or 'body' for a generic soldier's head or body piece (c_wrmcht2_ng_Lod2_head_2), else None."""
    m = re.search(r"_ng\w*?_(head|body)[a-z]?_?\d*(_lod\d)?\.xe", model.lower())
    return m.group(1) if m and model.lower().startswith("c_") else None


def words(name):
    ws = {w for w in re.split(r"[^a-z0-9]+", (name or "").lower()) if len(w) >= 2 and w not in STOP
          and not re.fullmatch(r"lod\d|\d+", w)}
    for w in list(ws):
        ws |= ALIASES.get(w, set())
        if len(w) >= 5 and w[-1].isdigit() and not w.rstrip("0123456789").endswith("_"):
            ws.add(w.rstrip("0123456789"))  # typewriter1 ~ typewriter2
    return ws


def overlap(a, b):
    """Words of a matched by words of b (prefixes and contained words count: 'sherman' ~ 'shermanm4a3',
    'body' ~ 'body2', 'mp44' ~ 'gemp44')."""
    return sum(1 for x in a if any(x == y or len(x) >= 3 and len(y) >= 3 and (x.startswith(y) or y.startswith(x))
                                   or len(y) >= 4 and y in x or len(x) >= 4 and x in y for y in b))


def obj_groups(path):
    """{group index: (uv array, triangles)} from an extract_models OBJ."""
    vt, groups, cur = [], {}, None
    for line in path.read_text().splitlines():
        if line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("g surface"):
            cur = int(line[9:])
            groups[cur] = []
        elif line.startswith("f ") and cur is not None:
            groups[cur].append([int(t.split("/")[1]) - 1 for t in line.split()[1:4]])
    vt = np.array(vt) if vt else np.zeros((0, 2))
    return vt, groups


def uv_mask(vt, faces):
    img = Image.new("L", (RES, RES), 0)
    dr = ImageDraw.Draw(img)
    for f in faces[:4000]:
        dr.polygon([((vt[i][0] % 1.0) * RES, (1 - vt[i][1] % 1.0) * RES) for i in f], fill=255)
    return np.asarray(img) > 0


_masks = {}


def tex_mask(path):
    if path not in _masks:
        try:
            im = Image.open(path).convert("RGBA")
            a = np.asarray(im.convert("RGB").resize((RES, RES), Image.BILINEAR)).astype(int)
            # painted and not decoding damage (stripes, pixel noise): a copy whose damaged band covers a
            # surface's UVs must lose to a clean copy
            bad = Image.fromarray(RT.damage_mask(np.asarray(im)) * 255).resize((RES, RES), Image.BILINEAR)
            _masks[path] = (a.sum(2) > 45) & (np.asarray(bad) < 128)
        except Exception:
            _masks[path] = None
    return _masks[path]


def texture_damage(root, paths):
    """Fraction of each PNG (relative path) that is decoding damage (repair_texture.damage_mask), cached in
    <root>/texture_damage.json by path and file size."""
    cache_file = root / "texture_damage.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    from concurrent.futures import ThreadPoolExecutor

    def one(rel):
        try:
            size = (root / rel).stat().st_size
            if rel in cache and cache[rel][0] == size:
                return rel, cache[rel]
            return rel, [size, float(RT.damage_mask(np.asarray(Image.open(root / rel).convert("RGBA"))).mean())]
        except Exception:
            return rel, [0, 0.0]
    with ThreadPoolExecutor(8) as ex:
        cache.update(dict(ex.map(one, paths)))
    cache_file.write_text(json.dumps(cache))
    return {k: v[1] for k, v in cache.items()}


def short(name):
    return (name or "").lower().split("\\")[-1]


def global_bindings(root: Path, index: dict):
    """(model, material) -> diffuse texture bound by any copy of that model in any archive: some packs
    leave the binding out and get it from another pack at load time."""
    out = {}
    for f in sorted(root.rglob("models/models.json")):
        for m in json.loads(f.read_text()):
            for mat in m.get("materials") or []:
                tx = {p: v.lower() for p, v in (mat.get("textures") or {}).items() if isinstance(v, str)}
                for p in DIFFUSE:
                    if tx.get(p) in index:
                        out.setdefault((short(m["name"]), (mat.get("name") or "").lower()), tx[p])
    return out


FIXES_FILE = Path(__file__).resolve().parent / "model_fixes.json"
# per-model fixes found by checking every model by eye: {model: {"surfaces": {"k" or "*": {"texture": name or
# null, "file": copy path relative to the models root}}, "note": why}}; under "_textures", texture name ->
# {"file": repaired copy, "note": why} for every model that uses it
FIXES = {k.lower(): v for k, v in json.loads(FIXES_FILE.read_text()).items()} if FIXES_FILE.exists() else {}
TEXTURE_FIXES = {k.lower(): v["file"] for k, v in FIXES.pop("_textures", {}).items()}


def best_copy(tex, um, root, index, copies):
    """The copy of `tex` (relative path) whose painted area covers this surface's UVs best. Some archives
    hold a copy decoded from a slightly wrong spot - big moved blocks, black where the model's pieces
    should be - so a surface that lands partly on black tries the other archives' copies."""
    def cover(rel):
        tm = tex_mask(root / rel)
        return -1.0 if tm is None else float((um & tm).sum() / um.sum())
    here = index[tex]
    c0 = cover(here)
    if c0 >= 0.97 or len(copies.get(tex, ())) < 2:
        return here
    alt = max(copies[tex][:16], key=cover)
    return alt if cover(alt) > c0 + 0.02 else here


def link_archive(archive: Path, root: Path, index: dict, colour: list, shared=None, copies=None):
    models = archive / "models"
    listing = json.loads((models / "models.json").read_text())
    pack_names = json.loads((models / "pack_names.json").read_text()) if (models / "pack_names.json").exists() else {}
    links = {}
    for m in listing:
        names_here = [n.lower() for n in pack_names.get(m["pack"], [])]
        own = set(names_here)
        start = names_here.index("physical") + 1 if "physical" in names_here else len(names_here)
        bound_region = [n for n in names_here[start:] if n in index and is_colour(n)]
        own_colour = [n for n in colour if n in own]
        mats = m.get("materials") or [{"name": None, "textures": {}}] * m["surfaces"]
        vt, groups = obj_groups(models / m["file"])
        chosen = []
        for k, mat in enumerate(mats):
            tex, how = None, None
            bound = {p: v.lower() for p, v in mat.get("textures", {}).items() if isinstance(v, str)}
            for p in DIFFUSE:
                if bound.get(p) in index:
                    tex, how = bound[p], "bound"
                elif p in bound and tex is None:
                    how = "bound-missing"  # named in the record but never decoded cleanly: don't guess
            if tex is None:
                for p in DIFFUSE:
                    r = runtime_texture(bound.get(p), m["name"], index)
                    if r:
                        tex, how = r, "runtime"
                        break
            piece = soldier_piece(m["name"])
            if tex is None and piece and not any(p in bound for p in DIFFUSE) \
                    and not any(v[-2:] in ("_n", "_s") for v in bound.values()):
                # a soldier's head or body binding nothing: the game gives it its sheet by number at runtime
                r = runtime_texture(m["name"].lower().split("_ng", 1)[0] + f"_ng_{piece}_text", m["name"], index)
                if r:
                    tex, how = r, "runtime"
            if tex is None and how in (None, "bound-missing") and shared:
                g = shared.get((short(m["name"]), (mat.get("name") or "").lower()))
                if g:
                    tex, how = g, "bound"
            if tex is None:
                for v in bound.values():
                    sib = next((s for s in (v[:-2] + "_c", "of_" + v[:-2] + "_c") if s in index), None)
                    if v[-2:] in ("_n", "_s") and sib:
                        tex, how = sib, "sibling"
                        break
            mname = (mat.get("name") or "").lower()
            part, _, owner = mname.partition(".")
            if re.match(r"(col|tag)(_|$)", part):
                tex, how = None, "collision"  # collision hulls and attach tags: not drawn
            if tex is None and how is None:
                w_part, w_owner = words(part), words(owner) | words(m["name"])
                best, ties = 0, []
                # textures named after this model win over ones that only share a part word ("body")
                pool = [n for n in colour if overlap(w_owner, words(n)) >= 1]
                for cand in (pool, colour) if pool else (colour,):
                    for n in cand:
                        wn = words(n)
                        op, oo, mine = overlap(w_part, wn), overlap(w_owner, wn), n in own
                        if not (op >= 1 or (oo >= 1 and mine) or oo >= 2):
                            continue
                        sc = 2 * op + 2 * oo + (0.5 if mine else 0)
                        sc -= 3 * sum(1 for x in wn & PART_WORDS if not overlap({x}, w_part))
                        if sc > best:
                            best, ties = sc, [n]
                        elif sc == best and n not in ties:
                            ties.append(n)
                    if ties:
                        break
                if ties:
                    tex, how = ties[0], "name"
                    if len(ties) > 1 and k in groups and len(vt):
                        um = uv_mask(vt, groups[k])

                        def fit(n):
                            tm = tex_mask(root / index[n])
                            return -1 if tm is None else (um & tm).sum() / max((um | tm).sum(), 1)
                        tex = max(ties[:40], key=fit)
                        how = "name+uv"
            if tex is not None and how != "bound" and k in groups and len(vt):
                # does the surface's UV layout land on painted parts of that sheet? If much of it falls
                # on empty (black) areas, another texture of the same model/character fits better
                um = uv_mask(vt, groups[k])
                if um.sum():
                    def cover(n):
                        tm = tex_mask(root / index[n])
                        return -1 if tm is None else (um & tm).sum() / um.sum()
                    c0 = cover(tex)
                    if c0 < 0.5:
                        w_owner = words(owner) | words(m["name"])
                        w_part = words(part)
                        family = [n for n in colour if overlap(w_owner, words(n)) >= 1
                                  and (not w_part or overlap(w_part, words(n)) >= 1)][:60]
                        better = max(family, key=cover, default=None)
                        if better and better != tex and cover(better) > c0 + 0.3:
                            tex, how = better, "cover"
            if tex is None and how is None and bound_region and k in groups and len(vt):
                # last resort: the colour textures this pack binds at load time (its name table after
                # "image"/"physical"), the one whose painted area fits this surface's UVs best
                um = uv_mask(vt, groups[k])
                fits = []
                for n in bound_region[:80]:
                    if re.match(r"(water|decal)\$", n) or "flag" in n or re.match(r"(of_)?c_", n):
                        continue  # water, decals, flags and uniforms only ever belong to their own models
                    tm = tex_mask(root / index[n])
                    # a sheet painted edge to edge "fits" any UV layout; only atlases with empty space
                    # between their islands can tell which model they belong to
                    if tm is not None and tm.mean() < 0.85:
                        fits.append(((um & tm).sum() / max((um | tm).sum(), 1), n))
                if fits:
                    f, n = max(fits)
                    if f > 0.5:
                        tex, how = n, "pack+uv"
            copy = index.get(tex) if tex else None
            if tex and copies and k in groups and len(vt):
                um = uv_mask(vt, groups[k])
                if um.sum():
                    copy = best_copy(tex, um, root, index, copies)
            if tex in TEXTURE_FIXES:  # a repaired copy (repair_texture.py) of a texture with no clean one
                copy = TEXTURE_FIXES[tex]
            chosen.append((tex, how, copy))
        # hand-checked fixes (cod3/model_fixes.json): per surface a texture name, a copy, or none
        fix = FIXES.get(short(m["name"]), {})
        for key, f in fix.get("surfaces", {}).items():
            ks = range(len(chosen)) if key == "*" else [int(key)]
            for k in ks:
                if k >= len(chosen):
                    continue
                tex, how, copy = chosen[k]
                if "texture" in f:
                    tex = f["texture"].lower() if f["texture"] else None
                    copy = index.get(tex) if tex else None
                if "file" in f:
                    copy = f["file"]
                elif tex in TEXTURE_FIXES:
                    copy = TEXTURE_FIXES[tex]
                chosen[k] = (tex, "fix", copy)
        lines = []
        for k, (tex, _, copy) in enumerate(chosen):
            lines += [f"newmtl s{k}", "Kd 1 1 1"]
            if tex:
                try:
                    rel = os.path.relpath(root / copy, models).replace("\\", "/")
                except ValueError:  # another drive
                    rel = str(root / copy).replace("\\", "/")
                lines.append(f"map_Kd {rel}")
        (models / (Path(m["file"]).stem + ".mtl")).write_text("\n".join(lines) + "\n")
        links[m["name"]] = [dict({"texture": t, "how": h}, **({"file": c} if c and c != index.get(t) else {}))
                            for t, h, c in chosen]
    (archive / "links.json").write_text(json.dumps(links, indent=1))
    found = sum(1 for v in links.values() for x in v if x["texture"])
    total = sum(len(v) for v in links.values())
    return found, total


if __name__ == "__main__":
    root = Path(sys.argv[1])
    index = json.loads((root / "texture_index.json").read_text())
    colour = sorted(n for n in index if is_colour(n))
    # A texture a material binds by name, whose every copy came out a little noisy (a bad patch on an
    # otherwise right sheet, score <= 1.25), still beats no texture - but only for that binding, never
    # for name guessing. Saved for texture_audit.py as texture_index_bound.json.
    scores = json.loads((root / "texture_scores.json").read_text()) if (root / "texture_scores.json").exists() else {}
    extra = {}
    for path, noise in sorted(((k, v) for k, v in scores.items() if v is not None), key=lambda kv: kv[1]):
        n = Path(path).stem.lower()
        if n not in index and n not in extra and noise <= 1.25:
            extra[n] = path
    (root / "texture_index_bound.json").write_text(json.dumps(extra, indent=0))
    index = {**index, **extra}
    targets = [root / a for a in sys.argv[2:]] or sorted(p.parent.parent for p in root.rglob("models/models.json"))
    # copies placed from the pack header (apkf.py) are exact: never treated as noise or decoding damage
    exact = set(json.loads((root / "texture_exact.json").read_text())) if (root / "texture_exact.json").exists() else set()
    copies = {}  # texture name -> every archive's usable copy (relative paths), least noisy first
    for path, noise in sorted(((k, v) for k, v in scores.items() if v is not None and (v <= 0.9 or k in exact)),
                              key=lambda kv: kv[1]):
        copies.setdefault(Path(path).stem.lower(), []).append(path)
    # a colour texture whose chosen copy is mostly decoding damage (a pack whose layout came out wrong):
    # another archive's clean copy instead, or none - no texture beats a sheet of noise
    damage = texture_damage(root, sorted(({index[n] for n in index if is_colour(n)} |
                                          {c for n, cs in copies.items() if is_colour(n) for c in cs[:8]}) - exact))
    dropped = 0
    for n in [n for n in index if is_colour(n) and index[n] not in exact and damage.get(index[n], 0) > 0.5]:
        clean = [c for c in copies.get(n, []) if damage.get(c, 1) < 0.1]
        if clean:
            index[n] = clean[0]
        else:
            del index[n]
            dropped += 1
    for n in copies:
        copies[n] = [c for c in copies[n] if c in exact or damage.get(c, 0) <= 0.5]
    colour = sorted(n for n in colour if n in index)
    print(f"{dropped} colour textures dropped: every copy is mostly decoding damage", flush=True)
    shared = global_bindings(root, index)
    F = T = 0
    for a in targets:
        f, t = link_archive(a, root, index, colour, shared, copies)
        F, T = F + f, T + t
        print(f"{a.relative_to(root)}: {f}/{t} surfaces textured", flush=True)
    print(f"Done: {F}/{T} surfaces textured", flush=True)
