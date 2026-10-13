"""Repair decoding damage in a COD3 texture that has no clean copy anywhere on the disc.

  python -I cod3/repair_texture.py <models root> <texture copy, relative to the root> [...]
Some textures survive in one pack only, with part of their data overwritten (yellow/black 1-pixel bars,
colourful pixel noise). The damaged 8x8 blocks are found with the same test as extract_textures.stripes,
grown by one block, and filled from the clean pixels around them (OpenCV inpainting). Writes
<root>/repaired/<same relative path> and prints how much was repaired; point a model at it with a
"file" entry in cod3/model_fixes.json.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def damage_mask(a):
    """Per-pixel mask of the damaged 8x8 blocks (stripes or saturated noise)."""
    g = a[..., :3].astype(np.int16).mean(2)
    h, w = g.shape[0] // 8 * 8, g.shape[1] // 8 * 8
    g = g[:h, :w]
    dx = np.pad(np.abs(np.diff(g, axis=1)), ((0, 0), (0, 1)))
    dy = np.pad(np.abs(np.diff(g, axis=0)), ((0, 1), (0, 0)))
    bx = dx.reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    by = dy.reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    hi, lo = np.maximum(bx, by), np.minimum(bx, by)
    rgb = a[:h, :w, :3].astype(np.int16)
    sat = (rgb.max(2) - rgb.min(2)).reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    # decoding stripes alternate every pixel (x[i] ~ x[i+2]); real fine detail (vent slats) doesn't
    d2x = np.pad(np.abs(g[:, 2:] - g[:, :-2]), ((0, 0), (0, 2))).reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    d2y = np.pad(np.abs(g[2:, :] - g[:-2, :]), ((0, 2), (0, 0))).reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    alternating = np.where(bx >= by, bx > 2.5 * (d2x + 1), by > 2.5 * (d2y + 1))
    bad = ((hi > 40) & (hi > 4 * (lo + 1)) & alternating) | ((lo > 40) & (sat > 40)) | ((lo > 20) & (sat > 70))
    bad = cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # damage edges leak into neighbours
    # a clean-looking block boxed in by damage on both sides is damage too (noise patches have quiet spots)
    bad |= cv2.morphologyEx(bad.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[:h, :w] = np.kron(bad, np.ones((8, 8), np.uint8))
    return mask


def repair(src, dst):
    a = np.asarray(Image.open(src).convert("RGBA"))
    mask = damage_mask(a)
    rgb = cv2.inpaint(np.ascontiguousarray(a[..., :3]), mask, 6, cv2.INPAINT_TELEA)
    alpha = cv2.inpaint(np.ascontiguousarray(a[..., 3]), mask, 6, cv2.INPAINT_TELEA)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(dst)
    return mask.mean()


if __name__ == "__main__":
    root = Path(sys.argv[1])
    for rel in sys.argv[2:]:
        frac = repair(root / rel, root / "repaired" / rel)
        print(f"{rel}: {100 * frac:.1f}% repaired -> repaired/{rel}")
