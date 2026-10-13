"""Stick-figure preview of a decoded COD3 animation (extract_anims.py output) on a model's bind pose.

  python -I preview_anim.py <anims folder>/<skeleton> <anim name> <model pack .bin> <out.png> [frames]
The model pack holds a character's mesh skeleton (skeleton_c_* objects, cod3_xemesh.skeletons); its m1
matrices are the bind transforms relative to each parent. Animated local rotations and positions replace
them where the animation has them. Quaternions are (x, y, z, w); COD space is Z-up, drawn from the side.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cod3_xemesh as X  # noqa: E402


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bind_local(pack):
    """name -> (local rotation 3x3, local offset) from the model skeleton's m1 matrices: m1 is each bone's
    bind transform relative to its parent, row-vector convention (rotation rows, translation in row 3) -
    the same frames the animation's quaternions use (the anim skeleton's defaults match m1 exactly)."""
    d = Path(pack).read_bytes()
    sk = next(iter(X.skeletons(d, X.find_data0(d)).values()))
    return {b["name"]: (b["m1"][:3, :3], b["m1"][3, :3]) for b in sk}


def pose(index, rot, pos, f, local):
    """World joint positions (row-vector chain: world = local @ parent_world)."""
    bones, parents = index["bones"], index["parents"]
    n = len(bones)
    P, W = [None] * n, [None] * n
    for i in range(n):
        R, t = local.get(bones[i], (np.eye(3), np.zeros(3)))
        if not np.isnan(rot[f, i, 0]):
            R = qmat(rot[f, i])
        if not np.isnan(pos[f, i, 0]):
            t = pos[f, i]
        p = parents[i]
        if p < 0 or W[p] is None:
            W[i], P[i] = R, t
        else:
            W[i] = R @ W[p]
            P[i] = t @ W[p] + P[p]
    return P


def draw(index, rot, pos, local, frames, out):
    tile = 220
    img = Image.new("RGB", (tile * len(frames), tile + 20), (25, 28, 34))
    dr = ImageDraw.Draw(img)
    for k, f in enumerate(frames):
        P = pose(index, rot, pos, f, local)
        pts = np.array([p for p in P if p is not None])
        ox = k * tile
        def sc(p):  # side view: x right, z up (inches); 3 px per inch
            return ox + tile / 2 + (p[0] - pts[:, 0].mean()) * 2.5, tile - 10 - p[2] * 2.5
        for i, p in enumerate(index["parents"]):
            if p >= 0 and P[i] is not None and P[p] is not None and index["bones"][i].startswith("bip01"):
                dr.line([sc(P[p]), sc(P[i])], fill=(240, 200, 90), width=2)
        dr.text((ox + 4, tile + 4), f"frame {f}", fill=(200, 200, 200))
    img.save(out)


if __name__ == "__main__":
    folder, name, pack, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    index = json.loads((folder / "index.json").read_text())
    z = np.load(folder / f"{name}.npz")
    n = z["rot"].shape[0]
    frames = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else \
        sorted({0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1})
    draw(index, z["rot"], z["pos"], bind_local(pack), frames, out)
    print(out)
