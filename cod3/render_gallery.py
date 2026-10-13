"""Contact sheets of extracted Call of Duty 3 models (OBJ + MTL from extract_all.py).

Each model is drawn textured (per-pixel: z-buffer, interpolated UVs, simple lighting) from a
three-quarter view. Largest models first; tiny fragments skipped.
Usage: python -I render_gallery.py <extracted root> <out prefix> [max models] [min triangles]
Writes <out prefix>_01.png, _02.png ... (24 models per sheet).
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TILE, COLS, ROWS = 320, 6, 4


def load_obj(path: Path):
    """Vertices, UVs, faces, per face a texture path (the .mtl's map_Kd for its usemtl) and per face the
    material colour (Kd; multiplies the texture, or is the colour when there's none)."""
    v, vt, f, ftex, fkd, mats, kds, cur = [], [], [], [], [], {}, {}, None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
            ftex.append(mats.get(cur))
            fkd.append(kds.get(cur, (1.0, 1.0, 1.0)))
        elif line.startswith("usemtl"):
            cur = line.split(None, 1)[1].strip()
        elif line.startswith("mtllib"):
            mtl = path.parent / line.split(None, 1)[1]
            if mtl.exists():
                name = None
                for ml in mtl.read_text().splitlines():
                    if ml.startswith("newmtl"):
                        name = ml.split(None, 1)[1].strip()
                    elif ml.startswith("Kd ") and name:
                        kds[name] = tuple(float(x) for x in ml.split()[1:4])
                    elif ml.startswith("map_Kd") and name:
                        t = Path(ml.split(None, 1)[1].strip())
                        mats[name] = t if t.is_absolute() else (path.parent / t).resolve()
    if cur is None and mats:  # single-material files (no usemtl per group)
        ftex = [next(iter(mats.values()))] * len(f)
    if cur is None and kds:
        fkd = [next(iter(kds.values()))] * len(f)
    return np.array(v), np.array(vt) if vt else None, np.array(f, dtype=int), ftex, fkd


def render(path: Path):
    """Per-pixel software render: z-buffer, UVs interpolated across each triangle, wrapped texture
    lookups, simple directional light (two-sided)."""
    v, vt, f, ftex, fkd = load_obj(path)
    if len(f) == 0:
        return None
    images = {}
    for t in set(ftex):
        if t is not None and t.exists():
            im = np.asarray(Image.open(t).convert("RGBA")).astype(np.float32)
            a = im[..., 3]
            # a cut-out texture: alpha is (nearly) on/off with real holes; others use alpha for gloss etc.
            if not ((a < 10).mean() > 0.05 and ((a >= 10) & (a < 245)).mean() < 0.1):
                im[..., 3] = 255
            images[t] = im
    a = np.radians(35)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    tilt = np.radians(-15)
    Tm = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    p = v @ (Tm @ R).T
    mn, mx = p.min(0), p.max(0)
    sc = (TILE - 40) / max(float((mx - mn)[:2].max()), 1e-6)
    cx, cy = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2
    sx = (p[:, 0] - cx) * sc + TILE / 2
    sy = TILE / 2 + 8 - (p[:, 1] - cy) * sc
    z = p[:, 2]
    zbuf = np.full((TILE, TILE), -np.inf, np.float32)
    out = np.empty((TILE, TILE, 3), np.float32)
    out[:] = (34, 37, 44)
    light = np.array([0.4, 0.7, 0.6]); light /= np.linalg.norm(light)
    have_uv = vt is not None and len(vt) == len(v)
    for i, (i0, i1, i2) in enumerate(f):
        x0, y0, x1, y1, x2, y2 = sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        xa, xb = int(max(0, np.floor(min(x0, x1, x2)))), int(min(TILE - 1, np.ceil(max(x0, x1, x2))))
        ya, yb = int(max(0, np.floor(min(y0, y1, y2)))), int(min(TILE - 1, np.ceil(max(y0, y1, y2))))
        if xa > xb or ya > yb:
            continue
        gy, gx = np.mgrid[ya:yb + 1, xa:xb + 1]
        px, py = gx + 0.5, gy + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        zz = w0 * z[i0] + w1 * z[i1] + w2 * z[i2]
        sub = zbuf[ya:yb + 1, xa:xb + 1]
        draw = inside & (zz > sub)
        if not draw.any():
            continue
        q = p[[i0, i1, i2]]
        n = np.cross(q[1] - q[0], q[2] - q[0]); nn = np.linalg.norm(n)
        shade = 0.45 + 0.55 * abs(float(n @ light) / nn) if nn > 0 else 0.6
        img = images.get(ftex[i])
        if img is not None and have_uv:
            u = (w0 * vt[i0, 0] + w1 * vt[i1, 0] + w2 * vt[i2, 0]) % 1.0
            vv = (w0 * vt[i0, 1] + w1 * vt[i1, 1] + w2 * vt[i2, 1]) % 1.0
            h, w = img.shape[:2]
            texel = img[((1 - vv) * (h - 1)).astype(int), (u * (w - 1)).astype(int)]
            draw = draw & (texel[..., 3] >= 128)  # cut-outs (propeller discs, spokes, foliage) are see-through
            if not draw.any():
                continue
            col = texel[..., :3] * np.array(fkd[i])
        elif fkd[i] != (1.0, 1.0, 1.0):  # plain coloured material
            col = np.broadcast_to(np.array(fkd[i]) * 235.0, w0.shape + (3,))
        else:
            col = np.broadcast_to(np.array([170.0, 170.0, 165.0]), w0.shape + (3,))
        sub[draw] = zz[draw]
        out[ya:yb + 1, xa:xb + 1][draw] = col[draw] * shade
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main():
    root, prefix = Path(sys.argv[1]), sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 96
    min_tris = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    objs = []
    for o in root.rglob("*.obj"):
        n = sum(1 for line in open(o) if line.startswith("f "))
        if n >= min_tris:
            objs.append((n, o))
    objs.sort(reverse=True)
    objs = objs[:limit]
    per = COLS * ROWS
    for s in range(0, len(objs), per):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * (TILE + 18)), (20, 22, 26))
        d = ImageDraw.Draw(sheet)
        for k, (n, o) in enumerate(objs[s:s + per]):
            try:
                img = render(o)
            except Exception:
                img = None
            x, y = (k % COLS) * TILE, (k // COLS) * (TILE + 18)
            if img:
                sheet.paste(img, (x, y + 18))
            label = f"{o.relative_to(root).parent.parent.as_posix()}/{o.stem} ({n} tris)"
            d.text((x + 4, y + 3), label[:52], fill=(255, 220, 90))
        out = f"{prefix}_{s // per + 1:02d}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
