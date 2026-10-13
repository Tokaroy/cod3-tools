"""Read Call of Duty 3 .xemesh assets exactly, from the game's own records in an APKF pack.

Layout (big-endian; worked out from mp/forest/zn17 gpu3, the Sherman and GMC husks):
  * CPU data starts at `data0`; record links are 32-bit WORD offsets from it.
  * Objects carry ids 0xFC0000xx. After the surfaces, one dependency group per asset:
      u32 ntypes, u32 own word offset + 2, ntypes x (fourcc 'MAT'/'MESH'/'SKEL'..., u32 count,
      u32 word offset of its count x (u32 id, u32 word offset) pairs). MESH pairs are the mesh objects.
  * Mesh object: id, 0, nsurf, surface table (word off; nsurf x (0, surface word off)), ..., sphere.
  * Surface (R): sphere f32x4, min f32x4, max f32x4, f32 ?, u32 material (word off), ...,
      R+0x48 u32 vertex count, R+0x58 u32 index count, R+0x60 word offset of the stride word,
      skinned only: R+0x70 bone palette (u16 x 48), then u32 stride (R+0x70 rigid / R+0xD0 skinned),
      u32 (own word offset + 3), 0, then the vertex declaration: (u32 offset, u32 format, u32 usage<<16|index<<8) ... until 0x00FF0000,
      then a Xenos vertex fetch constant (dword0 = address|3, dword1 = size_in_dwords<<2 | endian),
      then after the next 0xFFFF0000 word (u32 index buffer offset, u32 size in bytes) - a triangle list.
    Vertex and index addresses are relative to a GPU block per mesh object, found by checking
    each candidate start against that object's surfaces' bounding boxes.
  Usages are Direct3D 9: 0 position, 1 blendweight, 2 blendindices, 3 normal, 5 texcoord, 10 color.
"""
import struct

import numpy as np

USAGE = {0: "position", 1: "blendweight", 2: "blendindices", 3: "normal", 5: "texcoord", 6: "tangent",
         7: "binormal", 10: "color"}
F_FLOAT3, F_FLOAT2 = 0x2A23B9, 0x2C23A5


def u32(d, o):
    return struct.unpack_from(">I", d, o)[0]


def find_data0(d):
    """data0 from any surface's stride word P: the word after it holds (P's own word offset) + 3."""
    p = d.find(struct.pack(">I", F_FLOAT3))
    while p >= 0:
        P = p - 0x10  # stride word
        if p % 2 == 0 and P >= 0 and u32(d, P) in range(12, 129) and u32(d, P + 0xC) == 0:
            w = u32(d, P + 4) - 3
            if 0 < 4 * w <= P:
                return P - 4 * w
        p = d.find(struct.pack(">I", F_FLOAT3), p + 2)
    return None


def surface(d, data0, w):
    R = data0 + 4 * w
    s = {"at": R, "min": struct.unpack_from(">3f", d, R + 0x10), "max": struct.unpack_from(">3f", d, R + 0x20),
         "material": u32(d, R + 0x34), "nverts": u32(d, R + 0x48), "nidx": u32(d, R + 0x58), "palette": None}
    me = u32(d, R + 0x60)
    P = next((q for q in range(R + 0x70, R + 0x200, 4) if u32(d, q + 4) == me + 3), None)
    if P is None:
        raise ValueError("no stride")
    if P > R + 0x70:  # skinned: bone palette (u16 list, zero padded) between header and stride
        pal = list(struct.unpack_from(f">{(P - R - 0x70) // 2}H", d, R + 0x70))
        while len(pal) > 1 and pal[-1] == 0:
            pal.pop()
        s["palette"] = pal
    s["stride"] = u32(d, P)
    decl, o = [], P + 0xC
    while u32(d, o) != 0x00FF0000 and len(decl) < 16:
        off, fmt, use = struct.unpack_from(">III", d, o)
        decl.append((off, fmt, (use >> 16) & 0xFF, (use >> 8) & 0xFF))
        o += 12
    s["decl"] = decl
    for f in range(o, o + 0x60, 4):
        a, b = struct.unpack_from(">II", d, f)
        if a & 3 == 3 and ((b >> 2) & 0xFFFFFF) * 4 == s["nverts"] * s["stride"] and s["nverts"]:
            s["vb"] = a & ~3
            for e in range(f + 8, f + 0x40, 4):
                if u32(d, e) == 0xFFFF0000:
                    s["ib"], s["ib_size"] = struct.unpack_from(">II", d, e + 4)
                    break
            break
    return s


def assets(d):
    """[(mesh asset ordinal, [surface dicts])] for every MESH asset in the pack, in index order."""
    data0 = find_data0(d)
    if data0 is None:
        return None, []
    out = []
    p = 0
    while True:
        p = d.find(b"HSEM", p)  # 'MESH' fourcc as stored
        if p < 0:
            break
        # walk back to the group start: u32 ntypes, u32 word offset of the type list (= g + 8),
        # then 12-byte type entries (fourcc, count, word offset of that type's (id, word offset) pairs)
        for k in range(1, 9):
            g = p - 12 * (k - 1) - 8
            if g >= 0 and u32(d, g) >= k and u32(d, g) < 16 and data0 + 4 * u32(d, g + 4) == g + 8:
                break
        else:
            p += 4
            continue
        types = [(d[g + 8 + 12 * i:g + 12 + 12 * i], u32(d, g + 12 + 12 * i), u32(d, g + 16 + 12 * i))
                 for i in range(u32(d, g))]
        objs, end = [], g + 8 + 12 * len(types)
        for t, c, at in types:
            pairs = data0 + 4 * at
            end = max(end, pairs + 8 * c)
            if t == b"HSEM":
                objs += [u32(d, pairs + 8 * i + 4) for i in range(c)]
        surfs = []
        for w in objs:
            M = data0 + 4 * w
            n, table = u32(d, M + 8), u32(d, M + 12)
            if n > 512:
                continue
            for j in range(n):
                sw = u32(d, data0 + 4 * table + 8 * j + 4)
                try:
                    surfs.append(surface(d, data0, sw) | {"object": w})
                except (struct.error, ValueError):
                    pass
        out.append(surfs)
        p = end
    return data0, out


def records_end(data0, mesh_assets):
    """A point past this pack's surface records; GPU blocks only start after it."""
    return max([s["at"] + 0x100 for surfs in mesh_assets for s in surfs] + [data0 or 0])


def object_bases(d, data0, mesh_assets):
    """{mesh object word offset: GPU block start}. Blocks follow each other in object order, so each
    block is first tried right after the previous one (or right before the next), then searched for
    beyond the CPU records."""
    import itertools
    objs = []
    for surfs in mesh_assets:
        for w, g in itertools.groupby(surfs, key=lambda s: s["object"]):
            objs.append((w, list(g)))
    cpu_end = records_end(data0, mesh_assets)
    bases, hint = {}, None
    for w, g in objs:
        b = gpu_base(d, g, cpu_end, hint)
        bases[w] = b
        hint = None if b is None else b + block_size(g)
    for k in range(len(objs) - 2, -1, -1):  # backward: block k ends where block k+1 starts
        w, g = objs[k]
        nxt = bases[objs[k + 1][0]]
        if nxt is not None and bases[w] != nxt - block_size(g) and fits(d, nxt - block_size(g), g):
            bases[w] = nxt - block_size(g)
    return bases


def block_size(surfs):
    """Bytes of the object's GPU block: its vertex and index buffers, back to back."""
    return max([s["vb"] + s["nverts"] * s["stride"] for s in surfs if "vb" in s]
               + [s["ib"] + s["ib_size"] for s in surfs if "ib" in s] + [0])


def pos_offset(s):
    return next((off for off, fmt, use, _ in s["decl"] if use == 0 and fmt == F_FLOAT3), None)


def _usable(surfs):
    return [s for s in surfs if "vb" in s and s["nverts"] > 2 and s["decl"] and pos_offset(s) is not None]


def fits(d, base, surfs):
    """Every vertex inside its surface's box (the boxes are stored in the records) and every index valid."""
    test = _usable(surfs)
    if not test or base < 0 or base + block_size(surfs) > len(d):
        return False
    for s in surfs:  # cheap reject first: the start of each index buffer
        if "ib" in s and s["nidx"] and s["ib_size"] == s["nidx"] * 2:
            k = min(s["nidx"], 64)
            if max(struct.unpack_from(f">{k}H", d, base + s["ib"])) >= s["nverts"]:
                return False
    seen, boxes = [], []
    with np.errstate(all="ignore"):
        for s in test:
            lo, hi = np.array(s["min"]) - 0.01, np.array(s["max"]) + 0.01
            o = base + s["vb"]
            raw = np.frombuffer(d[o:o + s["nverts"] * s["stride"]], np.uint8).reshape(s["nverts"], s["stride"])
            vs = raw[:, pos_offset(s):pos_offset(s) + 12].copy().view(">f4")  # every vertex, not a sample
            if not ((vs >= lo) & (vs <= hi)).all():
                return False
            seen.append(vs)
            boxes.append((lo, hi))
    # the boxes are the vertices' own bounds (per surface or per object), so together they must reach them
    allv = np.concatenate(seen)
    lo, hi = np.min([b[0] for b in boxes], 0), np.max([b[1] for b in boxes], 0)
    tol = 0.02 * (hi - lo) + 0.02
    if (np.abs(allv.min(0) - lo) > tol).any() or (np.abs(allv.max(0) - hi) > tol).any():
        return False
    for s in surfs:  # and every index must point at a vertex of its surface
        if "ib" in s and s["nidx"] and s["ib_size"] == s["nidx"] * 2:
            idx = np.frombuffer(d[base + s["ib"]:base + s["ib"] + s["nidx"] * 2], ">u2")
            if len(idx) < s["nidx"] or idx.max() >= s["nverts"]:
                return False
    return True


def gpu_base(d, surfs, min_start, hint=None):
    """Start of this object's GPU block: the hint (end of the previous block) when it fits, else a search
    for the first offset where sampled vertices of every surface lie in their boxes (2-byte aligned)."""
    if hint is not None:
        for h in dict.fromkeys([hint] + [(hint + a - 1) & ~(a - 1) for a in (0x10, 0x20, 0x80, 0x100, 0x1000)]):
            if fits(d, h, surfs):
                return h
    test = _usable(surfs)[:6]
    if not test:
        return None
    best = None
    for parity in (0, 2):  # bases are 4-byte aligned in most packs, 2-byte in some
        n = (len(d) - parity) // 4
        f = np.frombuffer(d[parity:parity + n * 4], ">f4")
        ok = np.ones(n, bool)
        with np.errstate(all="ignore"):
            for s in test:
                lo, hi = np.array(s["min"]) - 0.01, np.array(s["max"]) + 0.01
                for vi in (0, s["nverts"] - 1, s["nverts"] // 2):
                    a = (s["vb"] + vi * s["stride"] + pos_offset(s)) // 4
                    for c in range(3):
                        v = np.full(n, np.nan, np.float32)
                        if a + c < n:
                            v[:n - a - c] = f[a + c:]
                        ok &= (v >= lo[c]) & (v <= hi[c])
        for j in np.flatnonzero(ok):
            base = int(j) * 4 + parity
            if base >= min_start and (best is None or base < best) and fits(d, base, surfs):
                best = base
                break
    return best


def decode(d, base, s):
    """Positions (n,3), uv (n,2) or None, triangles (m,3), bone indices/weights or None."""
    n, st = s["nverts"], s["stride"]
    raw = np.frombuffer(d[base + s["vb"]:base + s["vb"] + n * st], np.uint8).reshape(n, st)
    out = {"pos": None, "uv": None, "bones": None, "weights": None}
    for off, fmt, use, idx in s["decl"]:
        if use == 0 and fmt == F_FLOAT3 and out["pos"] is None:
            out["pos"] = raw[:, off:off + 12].copy().view(">f4").astype(np.float64)
        elif use == 5 and idx == 0 and fmt == F_FLOAT2:
            out["uv"] = raw[:, off:off + 8].copy().view(">f4").astype(np.float64)
        elif use == 2:
            out["bones"] = raw[:, off:off + 4].copy()
        elif use == 1:
            out["weights"] = raw[:, off:off + 4].copy()
    isz = 4 if s["ib_size"] == s["nidx"] * 4 else 2
    idx = np.frombuffer(d[base + s["ib"]:base + s["ib"] + s["nidx"] * isz], ">u4" if isz == 4 else ">u2").astype(np.int64)
    tris = idx[:len(idx) // 3 * 3].reshape(-1, 3)
    out["tris"] = tris[(tris < n).all(1)]
    return out


def name_table(d, data0):
    """The pack's names: 32-byte entries (u32 hash, 28-byte NUL-padded name) ending shortly before data0.
    An object id 0xFC000000 | 8*k names entry k."""
    if data0 is None:
        return []

    def ok(o):
        n = d[o + 4:o + 32].split(b"\0")[0]
        return 0 < len(n) and all(32 <= c < 127 for c in n) and not any(d[o + 5 + len(n):o + 32])

    def empty(o):
        return not any(d[o:o + 32])

    best = []
    for end in range(data0, max(data0 - 0x100, 0), -4):  # end of the table (a few header words sit between)
        o, k, gap = end - 32, 0, 0
        while o >= 0 and (ok(o) or (empty(o) and gap == 0)):  # a lone empty entry can sit in the table
            gap = 1 if empty(o) else 0
            o -= 32
            k += 1
        if gap:  # don't start on an empty entry
            o += 32
            k -= 1
        if k > len(best):
            best = [d[q + 4:q + 32].split(b"\0")[0].decode("latin1") for q in range(o + 32, end, 32)]
    return best


def data0_from_textures(d, descriptor_positions):
    """data0 for packs without meshes: a texture record holds, 0x34 bytes before its fetch constant, the
    word offset of the spot 0x1C before the fetch constant."""
    votes = {}
    for p in descriptor_positions:
        if p >= 0x38 and u32(d, p - 0x38) >> 24 == 0xFC:
            c = p - 0x1C - 4 * u32(d, p - 0x34)
            if 0 <= c < p:
                votes[c] = votes.get(c, 0) + 1
    return max(votes, key=votes.get) if votes else None


def texture_names(d, data0, descriptor_positions):
    """Name of each texture whose fetch constant sits at the given position (its record holds the name
    id 0x38 bytes before the fetch constant)."""
    if data0 is None:
        data0 = data0_from_textures(d, descriptor_positions)
    names = name_table(d, data0)
    out = []
    for p in descriptor_positions:
        w = u32(d, p - 0x38) if p >= 0x38 else 0
        k = (w & 0xFFFFFF) // 8
        out.append(names[k] if w >> 24 == 0xFC and k < len(names) else None)
    return out


def material(d, data0, w, names=None):
    """Material at word offset w: its name and {parameter: value}. Texture parameters (type 11) give the
    bound texture object's name, or None when the game binds it at load time."""
    names = names if names is not None else name_table(d, data0)
    nm = lambda i: names[(i & 0xFFFFFF) // 8] if i >> 24 == 0xFC and (i & 0xFFFFFF) // 8 < len(names) else None
    M = data0 + 4 * w
    out = {"name": nm(u32(d, M)), "fx": nm(u32(d, M + 4)), "params": {}}
    o = M + 0x40
    while o + 0x1C <= len(d) and u32(d, o) >> 24 == 0xFC:
        nid, _, _, _, typ, val, size = struct.unpack_from(">7I", d, o)
        p = data0 + 4 * val
        if typ == 11:
            tw = u32(d, p) if p + 4 <= len(d) else 0xCDCDCDCD
            out["params"][nm(nid)] = None if tw == 0xCDCDCDCD else nm(u32(d, data0 + 4 * tw + 0x10))
        elif typ in (2, 3) and p + 4 <= len(d):
            out["params"][nm(nid)] = struct.unpack_from(">f" if typ == 3 else ">I", d, p)[0]
        o += 0x24
    return out


def skeletons(d, data0, names=None):
    """Every skeleton in the pack: {name: [bone dicts]}. A skeleton record is (name id, bone count,
    word offset of its bones = own offset + 4, pad), followed by 0x90-byte bones: two 4x4 float matrices
    (rows; the 4th row is the translation), then name id, parent index (-1 for the root), 2 pad words."""
    names = names if names is not None else name_table(d, data0)
    nm = lambda i: names[(i & 0xFFFFFF) // 8] if i >> 24 == 0xFC and (i & 0xFFFFFF) // 8 < len(names) else None
    out = {}
    end = min(len(d) - 0x10, data0 + 0x400000)
    for o in range(data0, end, 4):
        a = u32(d, o)
        if a >> 24 != 0xFC:
            continue
        n, ptr = u32(d, o + 4), u32(d, o + 8)
        if not (1 <= n <= 512) or ptr != (o - data0) // 4 + 4 or o + 0x10 + n * 0x90 > len(d):
            continue
        bones = []
        for i in range(n):
            b = o + 0x10 + i * 0x90
            parent = struct.unpack_from(">i", d, b + 0x84)[0]
            if u32(d, b + 0x80) >> 24 != 0xFC or not (-1 <= parent < n):
                bones = None
                break
            m1 = np.array(struct.unpack_from(">16f", d, b)).reshape(4, 4)
            m2 = np.array(struct.unpack_from(">16f", d, b + 0x40)).reshape(4, 4)
            bones.append({"name": nm(u32(d, b + 0x80)), "parent": parent, "m1": m1, "m2": m2})
        if bones:
            out[nm(a) or f"skeleton_{o:x}"] = bones
    return out


def lod_replacements(d, data0, mesh_assets):
    """Word offsets of mesh objects that are only lower-detail replacements of another object. A mesh
    object with u32 1 at +0x14 links (+0x18) to a LOD record whose first word is its replacement object
    (then the switch distance, e.g. 2000.0, and the object's tag name). Exporting those too draws every
    detail level on top of each other."""
    import itertools
    out = set()
    for surfs in mesh_assets:
        for w, _ in itertools.groupby(surfs, key=lambda s: s["object"]):
            M = data0 + 4 * w
            if M + 0x1C <= len(d) and u32(d, M + 0x14) == 1:
                rec = data0 + 4 * u32(d, M + 0x18)
                if rec + 4 <= len(d):
                    out.add(u32(d, rec))
    return out
