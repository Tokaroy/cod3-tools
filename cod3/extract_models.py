"""Export every named Call of Duty 3 model (.xemesh and .xecell) as OBJ, read exactly from the game's records.

Runs over extract_all.py output (each archive folder has manifest.json and packs/*.bin):
  python -I cod3/extract_models.py <extracted root> [archive subpath ...] [--redo] [--reverse] [--shard=K/N]
Writes <archive>/models/<name>.obj, one group per surface, in Y-up metres, plus models.json listing
every model with its pack, surfaces, vertex and triangle counts and each surface's material (name and the
textures bound in its record), and pack_names.json (each pack's name table, for link_textures.py). Pack N of the file list is the N-th
pack in the .cod file; within a pack the mesh-bearing asset groups come in the same order as the file
list's .xemesh/.xecell entries sorted by (type, index).
cod3_xemesh.py documents the record layout. Multi-part rigid models (weapon viewmodels: frame, slide,
magazine...) have each part in its own bone's space; cod3_parts.py places them from the archive's
animation skeleton whose bones carry the part names ("placed" in models.json).
"""
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apkf  # noqa: E402
import cod3_parts as PT  # noqa: E402
import cod3_xemesh as X  # noqa: E402
from cod3_tuck import Tucker  # noqa: E402

INCH = 0.0254
MODELS_ROOT = None  # set from the command line: the disc-wide skeleton scan lives there
_TUCKER = []  # cod3_tuck.Tucker, made on first use (needs MODELS_ROOT and the soldier animations)
DEBUG_PARTS = __import__("os").environ.get("COD3_DEBUG_PARTS", "").lower()  # print part placement for models with this in their name


def pack_files(archive: Path, manifest: dict):
    """{pack number in the file list: path}. Pack numbers follow the packs' order in the .cod file, where
    compressed sections and GPU packs are interleaved (extract_all.pack_order)."""
    order = manifest.get("pack_order")
    if order is None:
        from cod3_kapf import Kapf
        from extract_all import pack_order
        order = pack_order(Kapf(Path(manifest["archive"])))
    return {i: archive / "packs" / f"{t}.bin" for i, t in enumerate(order) if (archive / "packs" / f"{t}.bin").exists()}


def tucker():
    """Gear tucked in on human characters (cod3_tuck.py); None if the soldier animations aren't extracted."""
    if not _TUCKER:
        anims = Path(MODELS_ROOT).parent / "anims" if MODELS_ROOT else None
        try:
            _TUCKER.append(Tucker(MODELS_ROOT, anims) if anims and (anims / "soldier" / "index.json").exists() else None)
        except SystemExit as e:
            print(f"    ! {e}", flush=True)
            _TUCKER.append(None)
    return _TUCKER[0]


def safe(name: str):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.split("\\")[-1])


def write_obj(path: Path, parts):
    """One group per surface, each with its own material slot s<k> (link_textures.py writes the .mtl)."""
    lines, base = [f"mtllib {path.stem}.mtl"], 0
    for k, m in enumerate(parts):
        p = np.c_[m["pos"][:, 0], m["pos"][:, 2], -m["pos"][:, 1]] * INCH  # COD Z-up inches -> Y-up metres
        uv = m["uv"] if m["uv"] is not None else np.zeros((len(p), 2))
        lines += [f"g surface{k}", f"usemtl s{k}"]
        lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in p]
        lines += [f"vt {u:.5f} {1 - v:.5f}" for u, v in uv]
        t = m["tris"] + base + 1
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in t]
        base += len(p)
    path.write_text("\n".join(lines) + "\n")


def header_bases(d, data0, groups):
    """{mesh object word offset: GPU block start}. From the pack header (apkf.py: exact) for every object
    whose block fits its surfaces' stored bounds there; searched (cod3_xemesh.object_bases) only for the
    rest. A searched block can land a vertex or two off and still fit the bounds - the triangles then join
    the wrong vertices (long slivers), as the scoped FG42 world model's third surface did."""
    objs = {w: list(g) for surfs in groups for w, g in itertools.groupby(surfs, key=lambda s: s["object"])}
    out = {}
    try:
        lay = apkf.layout(d)
        meshes = sorted((o for o in lay.objects if o.gpu is not None), key=lambda o: o.record)
        starts = [o.record for o in meshes]
        import bisect
        for w, g in objs.items():
            rec = data0 + 4 * w
            k = bisect.bisect_right(starts, rec) - 1
            if k >= 0 and rec < meshes[k].record + meshes[k].record_size and X.fits(d, meshes[k].gpu, g):
                out[w] = meshes[k].gpu
    except Exception:
        pass
    HEADER_STATS["header"] += len(out)
    if len(out) < len(objs):
        searched = X.object_bases(d, data0, groups)
        HEADER_STATS["searched"] += sum(1 for w in objs if w not in out and searched.get(w) is not None)
        for w in objs:
            if w in out:
                if searched.get(w) is not None and searched[w] != out[w]:
                    HEADER_STATS["differ"] += 1
            elif searched.get(w) is not None:
                out[w] = searched[w]
    return out


HEADER_STATS = {"header": 0, "searched": 0, "differ": 0}


# hand-checked part fixes (model_fixes.json "_parts"): {model: {part: {"rotate": [axis, degrees]} | {"hide": true}}}
_FIXES_FILE = Path(__file__).resolve().parent / "model_fixes.json"
PART_FIXES = {k.lower(): v for k, v in (json.loads(_FIXES_FILE.read_text()).get("_parts", {})
                                         if _FIXES_FILE.exists() else {}).items()}

PARKED_MARGIN = 6.0  # inches


def parked(parts):
    """Surfaces (indices) of parts an idle pose keeps away from the weapon: a weapon's idle animation moves
    what it isn't showing (a spent shell, a rocket not yet loaded) clear of the gun. A part whose box is
    more than PARKED_MARGIN from the main part's (the part with the most vertices) is left out."""
    import numpy as np
    by_part = {}
    for i, p in enumerate(parts):
        by_part.setdefault(p.get("part"), []).append(i)
    if len(by_part) < 2:
        return set()
    def box(idx):
        P = np.concatenate([parts[i]["pos"] for i in idx])
        return P.min(0), P.max(0)
    main = max(by_part, key=lambda k: sum(len(parts[i]["pos"]) for i in by_part[k]))
    lo, hi = box(by_part[main])
    drop = set()
    for k, idx in by_part.items():
        if k == main:
            continue
        a, b = box(idx)
        if ((a > hi + PARKED_MARGIN) | (b < lo - PARKED_MARGIN)).any():
            drop.update(idx)
    return drop


def export_archive(archive: Path):
    manifest = json.loads((archive / "manifest.json").read_text())
    files = manifest["files"]
    out = archive / "models"
    out.mkdir(exist_ok=True)
    packs = {}  # pack number -> (path, bytes, data0, groups, records end)
    for num, path in pack_files(archive, manifest).items():
        d = path.read_bytes()
        data0, groups = X.assets(d)
        packs[num] = (path, d, data0, groups, X.records_end(data0, groups) if groups else 0)
    listing, used, pack_names = [], set(), {}
    skeletons = PT.ArchiveSkeletons([v[1] for v in packs.values()], MODELS_ROOT)
    for num, (path, d, data0, groups, _) in packs.items():
        named = sorted([f for f in files if f["pack"] == num and f["name"].lower().endswith((".xemesh", ".xecell"))],
                       key=lambda f: (f["type"], f["index"]))
        if not groups:
            if named:
                print(f"    ! {path.name}: no mesh records for {len(named)} listed models", flush=True)
            continue
        bases = {w: (d, b) for w, b in header_bases(d, data0, groups).items()}
        lower_lods = X.lod_replacements(d, data0, groups)
        names = X.name_table(d, data0)
        pack_names[path.stem] = names
        if len(groups) != len(named):
            print(f"    ! {path.name}: {len(groups)} mesh groups, {len(named)} names listed", flush=True)
        for k, surfs in enumerate(groups):
            name = named[k]["name"] if k < len(named) else f"{path.stem}_asset{k}"
            parts, mats, src = [], [], set()
            part_of = {}  # object word -> its part name ("colt_mag" of "colt_mag.w_us_pist_vm")
            for w in {s["object"] for s in surfs}:
                i = (X.u32(d, data0 + 4 * w) & 0xFFFFFF) // 8
                part_of[w] = names[i].split(".")[0].lower() if i < len(names) and "." in names[i] else None
            placement = {} if any(s.get("palette") for s in surfs) else                 skeletons.placement(list(dict.fromkeys(v for v in part_of.values() if v)), name)
            for part in set(part_of.values()):  # "tag_turret01" (an alternate part) uses tag_turret's bone
                base = re.sub(r"\d+$", "", part or "")
                if part and part not in placement and base in placement:
                    placement[part] = placement[base]
            wheel_src = None  # (a wheel part in its bone's space, its index) to repeat at the other wheel bones
            chain_src = {}  # part -> (it in its bone's space, its index, chain bones): a weapon's belt of rounds
            for w, g in itertools.groupby(surfs, key=lambda s: s["object"]):
                g = list(g)
                if w in lower_lods:
                    continue  # a lower-detail copy of another object of this model
                if w not in bases:  # vertex data stored in another pack of the archive
                    for onum, (opath, od, _, _, oend) in packs.items():
                        b = X.gpu_base(od, g, oend) if onum != num else None
                        if b is not None:
                            bases[w] = (od, b)
                            src.add(opath.stem)
                            break
                if w not in bases:
                    continue
                vd, b = bases[w]
                for s in g:
                    if "vb" not in s or "ib" not in s:
                        continue
                    try:
                        with np.errstate(all="ignore"):
                            m = X.decode(vd, b, s)
                    except Exception:
                        continue
                    if m["pos"] is None or not len(m["tris"]) or not np.isfinite(m["pos"]).all():
                        continue
                    if s.get("palette") and m["bones"] is not None and tucker():
                        # a human character: its gear on the accessory bones where the game's standing pose
                        # puts it, not spread out as in the bind pose (cod3_tuck.py)
                        m["pos"] = tucker().tuck(m["pos"], s["palette"], m["bones"], m["weights"])
                    if part_of.get(w) in placement:  # into the model's space from its bone's
                        R, t = placement[part_of[w]]
                        if part_of[w].startswith("tag_wheel") and wheel_src is None:
                            wheel_src = (dict(m), mat_index_next := len(parts))
                        chain = [b for b in placement if re.fullmatch(re.escape(part_of[w]) + r"\d+", b)]
                        if len(chain) >= 3 and skeletons.posed_by_idle and part_of[w] not in chain_src:
                            chain_src[part_of[w]] = (dict(m), len(parts), chain)
                        c = m["pos"].mean(0)
                        # some models (truck and jeep wheels) already store a part where its bone is: only
                        # move parts whose geometry sits around their own origin, i.e. in bone space
                        moved = np.linalg.norm(c - t) >= 0.5 * np.linalg.norm(c)
                        if moved:
                            m["pos"] = m["pos"] @ R + t
                        if DEBUG_PARTS and DEBUG_PARTS in name.lower():
                            lo, hi = m["pos"].min(0), m["pos"].max(0)
                            print(f"    {name[:-7]:28} {part_of[w]:24} centre {np.round(c, 1)} bone {np.round(t, 1)} "
                                  f"rot-diag {np.round(np.diag(R), 2)} moved={moved} now {np.round(lo, 1)}..{np.round(hi, 1)}")
                    elif DEBUG_PARTS and DEBUG_PARTS in name.lower():
                        print(f"    {name[:-7]:28} {str(part_of.get(w)):24} (no bone) {np.round(m['pos'].min(0), 1)}..{np.round(m['pos'].max(0), 1)}")
                    m["part"] = part_of.get(w)
                    pfix = PART_FIXES.get(name.lower(), {}).get(part_of.get(w) or "", {})
                    if pfix.get("hide"):
                        continue
                    if "rotate" in pfix:  # [axis 0-2, degrees] about the part's own origin (its pivot)
                        ax, deg = pfix["rotate"]
                        a = np.radians(deg)
                        R = np.eye(3)
                        i, j = [k for k in range(3) if k != ax]
                        R[i, i] = R[j, j] = np.cos(a)
                        R[i, j], R[j, i] = -np.sin(a), np.sin(a)
                        m["pos"] = m["pos"] @ R.T
                    parts.append(m)
                    try:
                        mat = X.material(d, data0, s["material"], names)
                    except Exception:
                        mat = {"name": None, "params": {}}
                    mats.append({"name": mat["name"],
                                 "textures": {k2: v for k2, v in mat["params"].items() if k2 and "texture" in k2 and v}})
            if wheel_src is not None:  # the wheel model again at every wheel bone that has no part of its own
                src, k = wheel_src
                have = set(part_of.values())
                for bone, (R, t) in placement.items():
                    if bone.startswith("tag_wheel") and bone not in have:
                        parts.append(dict(src, pos=src["pos"] @ R + t))
                        mats.append(mats[k] if k < len(mats) else {"name": None, "textures": {}})
            have = set(part_of.values())
            for part, (src, k, chain) in chain_src.items():  # the part again at every bone of its chain
                for bone in chain:
                    if bone not in have:
                        R, t = placement[bone]
                        parts.append(dict(src, pos=src["pos"] @ R + t, part=part))
                        mats.append(mats[k] if k < len(mats) else {"name": None, "textures": {}})
            hidden = []
            if placement and skeletons.posed_by_idle:
                drop = parked(parts)
                hidden = sorted({parts[i].get("part") or "?" for i in drop})
                parts = [p for i, p in enumerate(parts) if i not in drop]
                mats = [m for i, m in enumerate(mats) if i not in drop]
            if not parts:
                print(f"    ! {name}: vertex data not found", flush=True)
                continue
            fn = safe(name).rsplit(".", 1)[0] + ".obj"
            if fn in used:
                fn = f"{fn[:-4]}__{path.stem}_{k}.obj"
            used.add(fn)
            write_obj(out / fn, parts)
            listing.append({"name": name, "file": fn, "pack": path.stem, "data": sorted(src) or [path.stem],
                            "surfaces": len(parts), "of_surfaces": len(surfs), "materials": mats,
                            "skinned": any(p["bones"] is not None for p in parts),
                            "placed": sorted(placement), **({"hidden": hidden} if hidden else {}),
                            "verts": int(sum(len(p["pos"]) for p in parts)),
                            "tris": int(sum(len(p["tris"]) for p in parts))})
    (out / "models.json").write_text(json.dumps(listing, indent=1))
    (out / "pack_names.json").write_text(json.dumps(pack_names))
    return listing


if __name__ == "__main__":
    redo = "--redo" in sys.argv
    shard = next((a for a in sys.argv[1:] if a.startswith("--shard=")), None)  # --shard=K/N: every N-th archive
    args = [a for a in sys.argv[1:] if a not in ("--redo", "--reverse") and not a.startswith("--shard=")]
    root = Path(args[0])
    MODELS_ROOT = root
    targets = [root / a for a in args[1:]] or sorted(p.parent for p in root.rglob("manifest.json"))
    if "--reverse" in sys.argv:
        targets.reverse()
    if shard:
        k, n = map(int, shard.split("=")[1].split("/"))
        targets = targets[k::n]
    t0, total = time.time(), 0
    for a in targets:
        if not redo and (a / "models" / "models.json").exists():
            continue  # already exported (pass --redo to export again)
        try:
            got = export_archive(a)
            total += len(got)
            print(f"{a.relative_to(root)}: {len(got)} models ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:
            print(f"{a.relative_to(root)}: FAILED {e!r}", flush=True)
    print(f"Done: {total} models; mesh blocks placed from the header {HEADER_STATS['header']}, searched {HEADER_STATS['searched']}", flush=True)
