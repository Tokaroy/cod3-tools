"""Second pass over extract_all.py output: give every OBJ the texture that really fits it.

extract_all.py only tried textures from the same pack, but characters and vehicles keep their
textures in neighbouring packs. Here every colour texture of the archive is a candidate, and the
fit is measured by overlap (IoU) between the model's UV layout (rasterised) and the texture's painted
area (anything not near-black background). Normal maps (grey, low saturation) and white specular
maps are skipped. Rewrites each <mesh>.mtl (or creates it) to the best match above a minimum fit.

Usage: python -I retexture.py <extracted root> [archive subpath ...]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RES = 96
MIN_IOU = 0.35


def obj_uv(path: Path):
    vt, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("vt "):
            vt.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            faces.append([int(t.split("/")[1]) - 1 for t in line.split()[1:4] if "/" in t])
    return np.array(vt), [f for f in faces if len(f) == 3]


def uv_mask(vt, faces):
    img = Image.new("L", (RES, RES), 0)
    d = ImageDraw.Draw(img)
    for f in faces:
        pts = [((vt[i][0] % 1.0) * RES, (1 - vt[i][1] % 1.0) * RES) for i in f]  # OBJ vt has v flipped
        d.polygon(pts, fill=255)
    return np.asarray(img) > 0


def texture_mask(path: Path):
    a = np.asarray(Image.open(path).convert("RGB").resize((RES, RES), Image.BILINEAR)).astype(int)
    sat = (a.max(2) - a.min(2)).mean()
    if a.mean() > 215 or sat < 7:
        return None  # spec / normal map
    return a.sum(2) > 45


def retexture(archive: Path):
    texdir, meshdir = archive / "textures", archive / "meshes"
    if not texdir.exists() or not meshdir.exists():
        return 0, 0
    masks = {}
    for t in texdir.glob("*.png"):
        m = texture_mask(t)
        if m is not None and m.mean() > 0.05:
            masks[t.name] = m
    done = fitted = 0
    for o in meshdir.glob("*.obj"):
        vt, faces = obj_uv(o)
        if len(vt) == 0 or not faces:
            continue
        um = uv_mask(vt, faces)
        if um.mean() < 0.002:
            continue
        best = None
        for name, tm in masks.items():
            inter = np.logical_and(um, tm).sum()
            union = np.logical_or(um, tm).sum()
            iou = inter / union if union else 0.0
            cover = inter / max(um.sum(), 1)
            score = iou * 0.6 + cover * 0.4
            if best is None or score > best[0]:
                best = (score, name, iou, cover)
        done += 1
        lines = o.read_text().splitlines()
        lines = [l for l in lines if not l.startswith(("mtllib", "usemtl"))]
        if best and best[3] > 0.85 and best[2] > MIN_IOU:
            o.with_suffix(".mtl").write_text(f"newmtl m\nKd 1 1 1\nmap_Kd ../textures/{best[1]}\n")
            lines = [f"mtllib {o.with_suffix('.mtl').name}", "usemtl m"] + lines
            fitted += 1
        else:
            o.with_suffix(".mtl").unlink(missing_ok=True)
        o.write_text("\n".join(lines) + "\n")
    return done, fitted


if __name__ == "__main__":
    root = Path(sys.argv[1])
    targets = [root / a for a in sys.argv[2:]] or [p.parent for p in root.rglob("manifest.json")]
    for a in targets:
        done, fitted = retexture(a)
        print(f"{a.relative_to(root)}: {fitted}/{done} models textured", flush=True)
