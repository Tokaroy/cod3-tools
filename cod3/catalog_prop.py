"""Turn a COD3 catalog model (OBJ + MTL + PNG textures) into a textured rigid LGO1 prop for the Slip & Skid mod.

  python -I cod3/catalog_prop.py <catalog obj> <out .lgo> [--centre]
The catalog's OBJ space is COD's converted to Y-up metres (x, z, -y); Unity's is (-x, z, -y) (as the
soldier exporter), so x is mirrored and the triangle winding reversed. One bone ("root"), one rigid part
per material, textures kept (at most 512 px). --centre moves the model's box centre to the origin.
"""
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_TEX = 512


def read(obj: Path):
    V, VT, groups, cur = [], [], {}, None
    for line in obj.read_text().splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            VT.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("usemtl"):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("f "):
            groups.setdefault(cur, []).append([tuple(int(x) - 1 if x else -1 for x in (t.split("/") + [""])[:2])
                                               for t in line.split()[1:4]])
    maps, name = {}, None
    for line in obj.with_suffix(".mtl").read_text().splitlines():
        if line.startswith("newmtl"):
            name = line.split(None, 1)[1].strip()
        elif line.startswith("map_Kd") and name:
            maps[name] = (obj.parent / line.split(None, 1)[1].strip()).resolve()
    return np.array(V), np.array(VT) if VT else np.zeros((0, 2)), groups, maps


def normals(p, t):
    n = np.zeros_like(p)
    for a, b, c in t:
        f = np.cross(p[b] - p[a], p[c] - p[a])
        n[a] += f; n[b] += f; n[c] += f
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def main():
    obj, out = Path(sys.argv[1]), Path(sys.argv[2])
    V, VT, groups, maps = read(obj)
    if "--centre" in sys.argv:
        V = V - (V.min(0) + V.max(0)) / 2
    V = V * np.array([-1.0, 1.0, 1.0])  # catalog OBJ -> Unity: mirror x
    textures, tex_of, parts = [], {}, []
    for mat, faces in groups.items():
        path = maps.get(mat)
        if path is not None and path.exists() and path not in tex_of:
            im = Image.open(path).convert("RGBA")
            if max(im.size) > MAX_TEX:
                s = MAX_TEX / max(im.size)
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
            tex_of[path] = len(textures)
            textures.append(im)
        # one vertex per (position, uv) corner pair used by this material
        keys, remap, tris = [], {}, []
        for face in faces:
            tri = []
            for k in face:
                if k not in remap:
                    remap[k] = len(keys)
                    keys.append(k)
                tri.append(remap[k])
            tris.append(tri[::-1])  # mirrored: reverse winding
        p = np.array([V[v] for v, _ in keys])
        uv = np.array([VT[t] if t >= 0 else (0.0, 0.0) for _, t in keys])
        t = np.array(tris, np.int64)
        parts.append((tex_of.get(path, -1), p, normals(p, t), uv, t))
    with open(out, "wb") as f:
        f.write(b"LGO1")
        f.write(struct.pack("<i", 1))
        f.write(struct.pack("<i", 4)); f.write(b"root")
        f.write(struct.pack("<i3f4f3f", -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1))
        f.write(struct.pack("<i", len(textures)))
        for im in textures:
            f.write(struct.pack("<ii", *im.size))
            f.write(np.asarray(im, dtype=np.uint8).tobytes())  # top row first (the loader flips)
        f.write(struct.pack("<i", len(parts)))
        for tex, *_ in parts:
            f.write(struct.pack("<4fi", 1.0, 1.0, 1.0, 1.0, tex))
        f.write(struct.pack("<i", len(parts)))
        for k, (_, p, n, uv, t) in enumerate(parts):
            f.write(struct.pack("<iiii", 0, 0, k, len(p)))
            f.write(p.astype("<f4").tobytes()); f.write(n.astype("<f4").tobytes()); f.write(uv.astype("<f4").tobytes())
            idx = t.reshape(-1)
            f.write(struct.pack("<i", len(idx))); f.write(idx.astype("<i4").tobytes())
    print(f"{out.name}: {sum(len(t) for *_, t in parts)} tris, {len(parts)} parts, {len(textures)} textures")


if __name__ == "__main__":
    main()
