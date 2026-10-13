"""Decode every Call of Duty 3 texture under its real name, from extract_all.py's unpacked packs.

  python -I cod3/extract_textures.py <extracted root> [archive subpath ...] [--redo] [--shard=K/N] [--no-index]
Writes <archive>/textures_named/<name>.png and textures.json (name, pack, size, format), and
<root>/texture_index.json mapping every texture name to its best (largest) PNG across the disc.

Texture records: two 0xFFFF0000 words, then the Xenos fetch constant; the record's name id sits 0x38
bytes before the fetch constant (cod3_xemesh.texture_names). Each texture's pixel data position comes
straight from the pack header (apkf.py: records and GPU data laid out in header order - exact for every
texture record on the disc). Only if a header can't be read does the old guesswork run: the allocation
list matched to tile seams (exact_layout), else a search near where the previous texture ended.
"""
import dataclasses
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apkf  # noqa: E402
import cod3_textures as T  # noqa: E402
import cod3_xemesh as X  # noqa: E402
import xenos_fetch as xenos  # noqa: E402


def image_score(d, at, fc):
    """Lower is more image-like: neighbour differences relative to contrast; None if it can't decode
    or is flat (padding)."""
    try:  # the tiled layout depends on width only, so the top rows decode on their own
        a = T.decode(d, at, dataclasses.replace(fc, height=min(fc.height, 64)))[:, :, :3].astype(np.int16)
    except Exception:
        return None
    if a.size == 0:
        return None
    sd = float(a.std())
    if sd < 3:
        return None
    return T.smooth(a) / (sd + 1.0)


def seam_ratio(a):
    """How much bigger the jumps across 32-pixel tile boundaries are than elsewhere. A correctly placed
    texture flows across them (~1-1.8); one decoded from a slightly wrong offset breaks there (> 3)."""
    a = a.astype(np.float32)
    if a.ndim == 3:
        a = a[:, :, :3].mean(2)
    best = 1.0
    for d in (np.abs(np.diff(a, axis=1)).mean(0), np.abs(np.diff(a, axis=0)).mean(1)):
        idx = np.arange(len(d))
        b = idx % 32 == 31
        if b.any() and (~b).any():
            best = max(best, float(d[b].mean() / (d[~b].mean() + 1e-3)))
    return best


def seam_score(d, at, fc):
    """seam_ratio of the top rows decoded at `at` (lower is better); None if it can't decode or is flat."""
    try:
        a = T.decode(d, at, dataclasses.replace(fc, height=min(fc.height, 128)))[:, :, :3]
    except Exception:
        return None
    if a.size == 0 or float(a.std()) < 3:
        return None
    return seam_ratio(a)


def fit_cost(d, at, fc):
    """How wrong a decode at `at` looks: its tile-seam ratio, or 9 for noise (no seams, but random)."""
    sc = image_score(d, at, fc)
    if sc is None or sc > 0.9:
        return 9.0
    return min(seam_score(d, at, fc) or 9.0, 9.0)


def block_seam(d, at, fc):
    """Extra cost for big-block jumps: rows every 128 pixels that change much more than the rows between
    them. A data start off by a large aligned step (e.g. 128 KB) moves whole 128-256 pixel bands
    of a texture; each band still looks clean, so the 32-pixel seam test can't see it. 0 when it flows."""
    if fc.height < 256:
        return 0.0
    try:
        a = T.decode(d, at, fc)[:, :, :3].astype(np.float32).mean(2)
    except Exception:
        return 0.0
    dy = np.abs(np.diff(a, axis=0)).mean(1)
    edge = np.arange(len(dy)) % 128 == 127
    return max(0.0, float(dy[edge].mean() / (dy[~edge].mean() + 1e-3)) - 1.3)


def locate(d, start, fc, window):
    best = None
    for at in range(max(0, start), min(start + window, len(d) - 0x1000), 0x1000):
        sc = image_score(d, at, fc)
        if sc is None or sc > 0.9:
            continue
        sm = seam_score(d, at, fc) or 9.0
        if best is None or (sm, sc) < best[0]:
            best = ((sm, sc), at)
            if sm < 1.3:
                break
    return best[1] if best else None


def alloc_chain(d, limit):
    """The pack header's GPU allocation list: 12-byte entries (record offset, CPU record size, GPU size),
    one per object in GPU-data order; textures have a 0x60-byte record, objects without GPU data size 0.
    Returns the longest run of such entries (it may start with a few unrelated words)."""
    best = []
    for start in range(0x18, max(0x18, limit - 12), 4):
        run, p = [], start
        while p + 12 <= limit:
            a, b, c = struct.unpack_from(">3I", d, p)
            if not (0 < b < 0x4000 and b % 4 == 0 and c < 0x1000000 and a < 0x400000):
                break
            run.append((a, b, c))
            p += 12
        if sum(1 for x in run if x[1] == 0x60) > sum(1 for x in best if x[1] == 0x60):
            best = run
    return best


def exact_layout(d, recs, data0):
    """Pixel-data position of every texture record, from the allocation list: GPU data starts at a 64 KB
    boundary, objects follow back to back, textures on 4 KB boundaries. The chain start and the data start
    are chosen so the textures' tile seams fit best (seam_ratio); any texture that still doesn't fit is
    searched for around its spot. None if most of them don't fit."""
    chain = alloc_chain(d, min(data0 or 0x4000, 0x8000))
    anchors = {}  # GPU block size -> known start(s) of mesh blocks of that size
    try:
        _, groups = X.assets(d)
        if groups:
            import itertools
            bases = X.object_bases(d, data0, groups)
            for surfs in groups:
                for w, g in itertools.groupby(surfs, key=lambda s: s["object"]):
                    if bases.get(w) is not None:
                        anchors.setdefault(X.block_size(list(g)), set()).add(bases[w])
    except Exception:
        pass
    first_tex = next((i for i, x in enumerate(chain) if x[1] == 0x60), None)
    if first_tex is None:
        return None
    cands = []
    for s0 in range(0, first_tex + 1):
        entries = chain[s0:]
        if sum(1 for x in entries if x[1] == 0x60) < len(recs):
            continue
        total = sum(x[2] for x in entries)
        starts = {g for g in range(0x10000, 0x100000, 0x10000)} | {(len(d) - total) & ~0xFFF}
        pos = 0
        for _, b, c in entries:  # anchor on mesh blocks, whose real positions are known
            if b == 0x60:
                pos = (pos + 0xFFF) & ~0xFFF
            elif c in anchors:
                starts |= {base - pos for base in anchors[c] if base - pos >= 0}
            pos += c
        starts = sorted(starts)
        for g in starts:
            if g + total > len(d) + 0x10000:
                continue
            pos, at, k = g, [], 0
            for _, b, c in entries:
                if b == 0x60:
                    pos = (pos + 0xFFF) & ~0xFFF  # GPU textures start on 4 KB boundaries
                    if k < len(recs):
                        at.append(pos)
                        k += 1
                pos += c
            seams = [fit_cost(d, q, fc) for (_, fc), q in zip(recs, at)]
            cands.append((sum(seams), at, seams))
    # the closest few candidates: also check the full height for big-block jumps (the top rows can't show them)
    cands.sort(key=lambda c: c[0])
    best = None
    for cost, at, seams in cands[:4]:
        if best is not None and cost > best[0] + 3:
            break
        big = sorted(range(len(at)), key=lambda i: -recs[i][1].width * recs[i][1].height)[:6]
        total_cost = cost + sum(block_seam(d, at[i], recs[i][1]) for i in big)
        if best is None or total_cost < best[0]:
            best = (total_cost, at, seams)
    if best is None or sum(1 for x in best[2] if x < 2.5) < max(1, len(recs) // 2):
        return None
    at = list(best[1])
    for i, ((_, fc), sm) in enumerate(zip(recs, best[2])):
        if sm >= 2.5:  # this one is still off: look around its computed spot
            q = locate(d, at[i] - 0x80000, fc, 0x100000)
            if q is not None and fit_cost(d, q, fc) < sm:
                at[i] = q
    return at


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.$-]", "_", name)


def header_textures(d):
    """(data0, [(fetch constant position, fetch constant)], [pixel data offset]) from the pack header
    (apkf.py) - exact; None if the header can't be read."""
    try:
        lay = apkf.layout(d)
    except Exception:
        return None
    recs, at = [], []
    for o in lay.by_tag("TEX"):
        p = o.record + 0x48
        if o.gpu is None or d[p - 8:p] != b"\xff\xff\x00\x00\xff\xff\x00\x00":
            continue  # no pixel data (render targets and the like)
        try:
            fc = xenos.parse_fetch_constant(d[p:p + 24])
        except Exception:
            continue
        if fc.format_code in (6, 18, 19, 20):
            recs.append((p, fc))
            at.append(o.gpu)
    return lay.data0, recs, at


def export_archive(archive: Path):
    out = archive / "textures_named"
    out.mkdir(exist_ok=True)
    listing = []
    for path in sorted((archive / "packs").glob("*.bin")):
        d = path.read_bytes()
        header = header_textures(d)
        if header is not None and header[1]:
            data0, recs, exact = header
            placed = "header"
        else:
            recs = T.records(d)
            if not recs:
                continue
            data0 = X.find_data0(d) or X.data0_from_textures(d, [p for p, _ in recs])
            exact = exact_layout(d, recs, data0)
            placed = "exact" if exact is not None else "searched"
        names = X.texture_names(d, data0, [p for p, _ in recs])
        sizes = T.data_sizes(d, len(recs))
        pos = (recs[-1][0] + 0xFFF) & ~0xFFF
        for i, (p, fc) in enumerate(recs):
            if exact is not None:
                at = exact[i]  # from the allocation list
            else:  # fallback: search near the previous texture's end for the most image-like decode
                first = i == 0 or pos is None
                at = locate(d, pos if pos is not None else (recs[-1][0] + 0xFFF) & ~0xFFF, fc,
                            0x400000 if first else 0x40000)
            if at is None:
                pos = None
                continue
            try:
                img = T.decode(d, at, fc)
            except Exception:  # runs past the end of the pack
                pos = None
                continue
            name = names[i] or f"{path.stem}_{i:02d}"
            fn = safe(name) + ".png"
            if (out / fn).exists() and any(l["file"] == fn for l in listing):
                fn = f"{safe(name)}__{path.stem}_{i}.png"
            Image.fromarray(img, "RGBA").save(out / fn)
            listing.append({"name": name, "file": fn, "pack": path.stem, "width": fc.width, "height": fc.height,
                            "format": fc.format_name, "placed": placed})
            pos = at + (sizes[i] if i < len(sizes) else T.mip_bytes(fc))
    (archive / "textures.json").write_text(json.dumps(listing, indent=1))
    return listing


_W = np.random.default_rng(3).integers(1, 2**63, 32 * 32 * 3, dtype=np.uint64)


def stripes(a):
    """Fraction of 8x8 blocks that are decoding damage: hard 1-pixel stripes (strong change along one
    axis, almost none along the other - the yellow/black bars) or colourful pixel noise (strong change
    along both axes in saturated colour), where a texture is decoded from slightly wrong data."""
    g = a[..., :3].astype(np.int16).mean(2)
    h, w = g.shape[0] // 8 * 8, g.shape[1] // 8 * 8
    if h == 0 or w == 0:
        return 0.0
    g = g[:h, :w]
    dx = np.pad(np.abs(np.diff(g, axis=1)), ((0, 0), (0, 1)))
    dy = np.pad(np.abs(np.diff(g, axis=0)), ((0, 1), (0, 0)))
    bx = dx.reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    by = dy.reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    hi, lo = np.maximum(bx, by), np.minimum(bx, by)
    rgb = a[:h, :w, :3].astype(np.int16)
    sat = (rgb.max(2) - rgb.min(2)).reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    return float(((hi > 40) & (hi > 4 * (lo + 1))).mean() + ((lo > 40) & (sat > 40)).mean())


def tile_hashes(a):
    """Hashes of the 32x32 tiles that have detail (flat tiles - black atlas background - match anything).
    Copies of one texture decoded from slightly different offsets share tiles at shifted positions."""
    h, w = a.shape[0] // 32 * 32, a.shape[1] // 32 * 32
    if h == 0 or w == 0:
        return np.zeros(0, np.uint64)
    t = (a[:h, :w, :3] // 8).reshape(h // 32, 32, w // 32, 32, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 32 * 32 * 3)
    t = t[t.std(1) > 0.5]
    return np.unique((t.astype(np.uint64) * _W).sum(1))


def texture_stats(path):
    """(noise score, 16x16 thumbnail, stripe fraction, tile hashes) for one decoded PNG. Noise score:
    neighbour differences over contrast on a 256x256 centre crop - real textures ~0.1-0.45, noise > 0.9;
    None when flat."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return None, None, 0.0, None
    thumb = np.asarray(im.resize((16, 16), Image.BILINEAR)).astype(np.float32)
    full = np.asarray(im)
    st, hashes = stripes(full), tile_hashes(full)
    a = full.astype(np.int16)
    h, w = a.shape[:2]
    a = a[h // 4:h // 4 + 256, w // 4:w // 4 + 256]
    sd = float(a.std())
    if sd < 3:
        return None, thumb, st, hashes
    sc = float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / (sd + 1)
    if seam_ratio(full) >= 2.5:
        sc = max(sc, 0.95)  # tile-scrambled: decoded from a slightly wrong offset - treat as unusable
    return sc, thumb, st, hashes


def build_index(root: Path, workers=6):
    """texture name -> the copy to trust. Archives repeat textures; correct copies are identical, while
    copies decoded from the wrong place differ. Ranking, per name:
      1. within a family of related copies - sharing >= 30% of their detailed 32x32 tiles at any position,
         i.e. one texture decoded from slightly different offsets - copies with clearly more stripes than
         the family's cleanest lose (several archives can share one bad decode and outvote the clean copy);
         unrelated copies are left alone (they are usually other textures, clean or not);
      2. more copies agreeing exactly (16x16 thumbnails);
      3. more related copies, then not clearly more damaged than the cleanest copy of the name, then the
         least noisy, then the largest.
    Plain noise (score > 0.9) is never used - except for copies placed from the pack header, which are exact:
    there a high score just means a fine-grained surface (asphalt, gravel, rust), not a bad decode. Also writes
    texture_scores.json, texture_stripes.json and texture_exact.json (copies placed from the header)."""
    from multiprocessing import Pool
    copies = []
    for lst in root.rglob("textures.json"):
        for t in json.loads(lst.read_text()):
            copies.append((t["name"].lower(), lst.parent / "textures_named" / t["file"], t["width"] * t["height"],
                           t.get("placed") == "header"))
    with Pool(workers) as pool:
        stats = pool.map(texture_stats, [c[1] for c in copies], chunksize=32)
    groups, table, striped, exact = {}, {}, {}, []
    for (key, path, area, header), (sc, thumb, st, hashes) in zip(copies, stats):
        rel = str(path.relative_to(root)).replace("\\", "/")
        table[rel] = sc
        striped[rel] = round(st, 4)
        if header:
            exact.append(rel)
        if thumb is None or (sc is not None and sc > 0.9 and not header):
            continue
        groups.setdefault(key, []).append((rel, sc, area, thumb, st, hashes))
    index = {}
    for key, g in groups.items():
        if len(g) == 1:
            index[key] = g[0][0]
            continue
        n = len(g)
        rel = np.zeros((n, n), bool)
        for i in range(n):
            for j in range(i + 1, n):
                hi, hj = g[i][5], g[j][5]
                if len(hi) and len(hj):
                    shared = len(np.intersect1d(hi, hj, assume_unique=True))
                    rel[i, j] = rel[j, i] = shared >= 0.3 * min(len(hi), len(hj))

        cleanest = min(c[4] for c in g)

        def rank(i):
            c = g[i]
            fam = [g[j][4] for j in range(n) if rel[i, j]] + [c[4]]
            dirty = c[4] > min(fam) + 0.015
            agree = sum(1 for o in g if o is not c and o[3].shape == c[3].shape
                        and float(np.abs(o[3] - c[3]).mean()) < 12)
            sc = c[1]
            quality = 0 if sc is not None and sc <= 0.55 else 1 if sc is not None else 2
            return (dirty, -agree, -int(rel[i].sum()), c[4] > cleanest + 0.015, quality, round(sc or 0, 2), -c[2])
        index[key] = g[min(range(n), key=rank)][0]
    (root / "texture_index.json").write_text(json.dumps(dict(sorted(index.items())), indent=0))
    (root / "texture_scores.json").write_text(json.dumps(table, indent=0))
    (root / "texture_stripes.json").write_text(json.dumps(striped, indent=0))
    (root / "texture_exact.json").write_text(json.dumps(sorted(exact), indent=0))
    return len(index)


if __name__ == "__main__":
    redo = "--redo" in sys.argv
    shard = next((a for a in sys.argv[1:] if a.startswith("--shard=")), None)  # --shard=K/N: every N-th archive
    args = [a for a in sys.argv[1:] if a not in ("--redo", "--no-index") and not a.startswith("--shard=")]
    root = Path(args[0])
    targets = [root / a for a in args[1:]] or sorted(p.parent for p in root.rglob("manifest.json"))
    if shard:
        k, n = map(int, shard.split("=")[1].split("/"))
        targets = targets[k::n]
    t0 = time.time()
    for a in targets:
        if not redo and (a / "textures.json").exists():
            continue
        try:
            got = export_archive(a)
            print(f"{a.relative_to(root)}: {len(got)} textures ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:
            print(f"{a.relative_to(root)}: FAILED {e!r}", flush=True)
    if "--no-index" in sys.argv:  # parallel shards: build the index once, after all of them
        print("Done (index not rebuilt)", flush=True)
    else:
        print(f"Done: {build_index(root)} unique texture names", flush=True)
