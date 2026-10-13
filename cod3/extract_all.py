"""Extract everything from the user's own Call of Duty 3 (Xbox 360) disc: every .cod archive's sections
and GPU packs, every texture (PNG), every mesh (OBJ + MTL, skinned and rigid), and a manifest.

Usage: python -I cod3/extract_all.py <COD3 game folder> <output folder>
Output per archive (e.g. sp/saint_lo/chars.cod -> <out>/sp/saint_lo/chars/):
  manifest.json            every file name in the archive with its size, pack and index
  packs/section<N>.bin     unpacked compressed sections;  packs/gpu<N>.bin  uncompressed GPU packs
  textures/<pack>_<n>_<w>x<h>_<fmt>.png
  meshes/<pack>_<offset>.obj (+ .mtl when a texture of the archive fits its UVs; see retexture.py)
Re-running skips archives that already have a manifest. Sounds: see extract_cod3_audio.py.
Meshes are converted to Y-up metres (x, z, -y) * 0.0254 so ordinary 3D viewers show them upright.
"""
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cod3_kapf import Kapf  # noqa: E402
from cod3_mesh import find_vertex_buffers, decode as decode_skinned  # noqa: E402
import apkf  # noqa: E402
import cod3_textures as T  # noqa: E402
import xenos_fetch as xenos  # noqa: E402
from retexture import retexture  # noqa: E402

RIGID = 52
INCH = 0.0254


def gpu_regions(k: Kapf):
    """Uncompressed APKF packs as bytes (see gpu_spans)."""
    return [k.d[a:b] for a, b in gpu_spans(k)]


def pack_order(k: Kapf):
    """Pack tags in file order - the asset list's pack number indexes this list. Sections and GPU packs
    are interleaved in the file, e.g. ["gpu0", "section0", "gpu1", ...]."""
    tagged = [(s, f"section{i}") for i, s in enumerate(k.section_starts())]
    tagged += [(a, f"gpu{i}") for i, (a, _) in enumerate(gpu_spans(k))]
    return [t for _, t in sorted(tagged)]


def gpu_spans(k: Kapf):
    """Uncompressed APKF packs: every "APKF" pack start outside the NCH-compressed chunks, each running
    to the next pack start / chunk / end of file. Some archives (all of mp/) have no NCH chunks at all."""
    d, cover, p = k.d, [], 0
    while True:
        p = d.find(b"NCH\0", p)
        if p < 0:
            break
        total = struct.unpack_from(">I", d, p + 24)[0]
        cover.append((p, (p + total + 0x7FFF) & ~0x7FFF))
        p += 4
    def covered(x):
        return any(a <= x < b for a, b in cover)
    starts, p = [], 0
    while True:
        p = d.find(b"APKF", p)
        if p < 0:
            break
        if p % 0x1000 == 0 and not covered(p):
            starts.append(p)
        p += 4
    bounds = sorted(set(starts + [a for a, _ in cover] + [len(d)]))
    return [(s, next(b for b in bounds if b > s)) for s in starts]


def textures(data: bytes, tag: str, out: Path):
    """Every texture of the pack, unnamed, at the position its header gives (apkf.py); extract_textures.py
    writes the same pixels under their real names."""
    try:
        lay = apkf.layout(data)
    except Exception:
        return []
    saved = []
    for i, o in enumerate(o for o in lay.by_tag("TEX") if o.gpu is not None):
        try:
            fc = xenos.parse_fetch_constant(data[o.record + 0x48:o.record + 0x60])
            img = T.decode(data, o.gpu, fc)
        except Exception:
            continue
        name = f"{tag}_{i:02d}_{fc.width}x{fc.height}_{fc.format_name}.png"
        Image.fromarray(img, "RGBA").save(out / name)
        saved.append(name)
    return saved


def rigid_meshes(raw: bytes, min_verts=60):
    """52-byte rigid vertices (props, weapons, scenery): position, normal, UV at byte 16; u16 tris after."""
    found, used = [], []
    for ph in range(0, RIGID, 4):
        cnt = (len(raw) - ph) // RIGID
        if cnt < min_verts:
            continue
        rec = np.frombuffer(raw[ph:ph + cnt * RIGID], np.uint8).reshape(cnt, RIGID)
        pos = rec[:, 0:12].copy().view(">f4")
        with np.errstate(all="ignore"):
            ok = np.isfinite(pos).all(1) & (np.abs(pos) < 4000).all(1) & (np.abs(pos) > 1e-3).any(1)
            uv = rec[:, 16:24].copy().view(">f4")
            ok &= np.isfinite(uv).all(1) & (np.abs(uv) < 64).all(1)
        d = np.diff(np.r_[0, ok.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            n = b - a
            start = ph + a * RIGID
            if n < min_verts or any(abs(start - u) < RIGID * n for u in used):
                continue
            idx = np.frombuffer(raw[ph + b * RIGID:ph + b * RIGID + min(600000, len(raw) - ph - b * RIGID) // 2 * 2], ">u2").astype(np.int64)
            bad = np.flatnonzero(idx >= n)
            m = (bad[0] if len(bad) else len(idx)) // 3 * 3
            tris = idx[:m].reshape(-1, 3)
            tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
            if len(tris) < min_verts // 2 or len(np.unique(tris)) < n * 0.5:
                continue
            used.append(start)
            found.append((start, pos[a:b].astype(np.float64), uv[a:b].astype(np.float64), tris))
    return found


def write_obj(path: Path, pos, uv, tris, texture=None):
    p = np.c_[pos[:, 0], pos[:, 2], -pos[:, 1]] * INCH  # COD Z-up inches -> Y-up metres
    lines = []
    if texture:
        mtl = path.with_suffix(".mtl")
        mtl.write_text(f"newmtl m\nKd 1 1 1\nmap_Kd ../textures/{texture}\n")
        lines += [f"mtllib {mtl.name}", "usemtl m"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in p]
    lines += [f"vt {u:.5f} {1 - v:.5f}" for u, v in uv]
    lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in tris]
    path.write_text("\n".join(lines) + "\n")


def extract_archive(cod: Path, out: Path):
    k = Kapf(cod)
    (out / "packs").mkdir(parents=True, exist_ok=True)
    (out / "textures").mkdir(exist_ok=True)
    (out / "meshes").mkdir(exist_ok=True)
    files = []
    for name, size, pack, idx in k.names():
        if name and name.isprintable() and "." in name:
            files.append({"name": name, "size": size, "pack": pack, "type": idx >> 16, "index": idx & 0xFFFF})
    datas = []
    for i, s in enumerate(k.section_starts()):
        try:
            data = k.section_data(s)
        except Exception as e:
            print(f"    ! section {i}: {e}", flush=True)
            continue
        (out / "packs" / f"section{i}.bin").write_bytes(data)
        datas.append((f"section{i}", data))
    for i, data in enumerate(gpu_regions(k)):
        (out / "packs" / f"gpu{i}.bin").write_bytes(data)
        datas.append((f"gpu{i}", data))
    n_tex = n_mesh = 0
    for tag, data in datas:
        tex = textures(data, tag, out / "textures")
        n_tex += len(tex)
        meshes = []
        try:
            for start, count in find_vertex_buffers(data):
                m = decode_skinned(data, start, count)
                if len(m["tris"]):
                    meshes.append((start, m["pos"].astype(np.float64), m["uv"].astype(np.float64), m["tris"]))
        except Exception as e:
            print(f"    ! skinned meshes in {tag}: {e}", flush=True)
        try:
            meshes += rigid_meshes(data)
        except Exception as e:
            print(f"    ! rigid meshes in {tag}: {e}", flush=True)
        for start, pos, uv, tris in meshes:
            keep = np.linalg.norm(pos, axis=1) > 0  # (unused phantom vertices are harmless)
            write_obj(out / "meshes" / f"{tag}_{start:08x}.obj", pos, uv, tris)
            n_mesh += 1
    retexture(out)  # pick each mesh's texture from the whole archive
    (out / "manifest.json").write_text(json.dumps({"archive": str(cod), "build": k.build, "files": files,
                                                   "pack_order": pack_order(k),
                                                   "textures": n_tex, "meshes": n_mesh}, indent=1))
    return len(files), n_tex, n_mesh


def main():
    game, out = Path(sys.argv[1]), Path(sys.argv[2])
    cods = sorted(game.rglob("*.cod"))
    t0 = time.time()
    totals = [0, 0, 0]
    for i, cod in enumerate(cods, 1):
        rel = cod.relative_to(game).with_suffix("")
        dest = out / rel
        if (dest / "manifest.json").exists():
            continue
        try:
            nf, nt, nm = extract_archive(cod, dest)
            totals = [a + b for a, b in zip(totals, (nf, nt, nm))]
            print(f"[{i}/{len(cods)}] {rel}: {nf} files listed, {nt} textures, {nm} meshes ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:
            print(f"[{i}/{len(cods)}] {rel}: FAILED {e}", flush=True)
    print(f"Done: {totals[1]} textures, {totals[2]} meshes from {len(cods)} archives in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
