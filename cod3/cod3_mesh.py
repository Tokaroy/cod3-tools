"""Find and decode Call of Duty 3 (Xbox 360) skinned character meshes in an unpacked .cod GPU region.

Vertex buffers are runs of 60-byte vertices (big-endian):
   0  float2 uv  (repeated 3 times: 0..23)
  24  u32 colour (0xffffffff)
  28  u32 packed normal, 32 u32 packed tangent
  36  float3 position (inches, Z up)
  48  u8[4] bone indices, 52 u8[4] weights (sum 255), 56 u32 packed binormal
followed directly by a u16 big-endian triangle list.

Usage: python -I cod3_mesh.py <gpu region .bin> <out dir>   (writes <offset>.npz per mesh + a contact sheet)
"""
import sys
from pathlib import Path

import numpy as np

STRIDE = 60


def find_vertex_buffers(raw: bytes, min_verts=100):
    """Runs of 60-byte records whose skin weights (bytes 52..55) sum to 255 and whose position is sane."""
    n = len(raw)
    b = np.frombuffer(raw, dtype=np.uint8)
    out = []
    for phase in range(0, STRIDE, 4):
        cnt = (n - phase) // STRIDE
        rec = b[phase:phase + cnt * STRIDE].reshape(cnt, STRIDE)
        wsum = rec[:, 52:56].astype(np.int32).sum(1)
        pos = rec[:, 36:48].copy().view(">f4")
        with np.errstate(all="ignore"):
            sane = np.isfinite(pos).all(1) & (np.abs(pos) < 200).all(1)
        good = (np.abs(wsum - 255) <= 1) & sane & (rec[:, 48:52].max(1) < 128)
        d = np.diff(np.r_[0, good.astype(np.int8), 0])
        for st, en in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if en - st >= min_verts:
                out.append((phase + st * STRIDE, int(en - st)))
    return sorted(out)


def decode(raw: bytes, start: int, count: int):
    """The weight test can miss the last vertex or two; take the count whose following triangle
    list is longest and most compact."""
    best = None
    for extra in range(0, 4):
        m = _decode(raw, start, count + extra)
        t = m["tris"]
        if len(t) == 0:
            continue
        edge = np.median(np.linalg.norm(m["pos"][t[:, 0]] - m["pos"][t[:, 1]], axis=1))
        score = len(t) / (1.0 + edge)
        if best is None or score > best[0]:
            best = (score, m)
    return best[1] if best else _decode(raw, start, count)


def _decode(raw: bytes, start: int, count: int):
    rec = np.frombuffer(raw[start:start + count * STRIDE], dtype=np.uint8).reshape(count, STRIDE)
    # The real vertex starts at byte 36 (position, bones, weights), so its UV, colour, normal and
    # tangent are the first 36 bytes of the next 60-byte block.
    nxt = np.frombuffer(raw[start + STRIDE:start + (count + 1) * STRIDE], dtype=np.uint8).reshape(count, STRIDE)
    uv = nxt[:, 0:8].copy().view(">f4").astype(np.float32)
    pos = rec[:, 36:48].copy().view(">f4").astype(np.float32)
    bones = rec[:, 48:52].astype(np.int32)
    weights = rec[:, 52:56].astype(np.float32) / 255.0
    # Triangle list straight after the vertices; stop at the first index out of range.
    ib_start = start + count * STRIDE
    idx = np.frombuffer(raw[ib_start:ib_start + 400000 // 2 * 2], dtype=">u2").astype(np.int32)
    bad = np.flatnonzero(idx >= count)
    n = (bad[0] if len(bad) else len(idx)) // 3 * 3
    tris = idx[:n].reshape(-1, 3)
    # Degenerate tail (padding zeros) is dropped.
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return dict(pos=pos, uv=uv, bones=bones, weights=weights, tris=tris[keep], packed=nxt[:, 28:36].copy())


def render(m, size=256):
    from PIL import Image, ImageDraw
    p = m["pos"]
    img = Image.new("RGB", (size * 2, size), (35, 38, 44))
    d = ImageDraw.Draw(img)
    mn, mx = p.min(0), p.max(0)
    sc = (size - 20) / max(float((mx - mn).max()), 1e-6)
    for view, ox in ((0, 0), (1, size)):
        order = np.argsort(p[m["tris"]].mean(1)[:, 1 - view] * (1 if view == 0 else -1))
        for t in m["tris"][order]:
            q = p[t]
            a, b = (q[:, 0], q[:, 2]) if view == 0 else (q[:, 1], q[:, 2])
            n = np.cross(q[1] - q[0], q[2] - q[0])
            shade = int(90 + 140 * abs(n[1 - view]) / (np.linalg.norm(n) + 1e-9))
            pts = [(ox + 10 + (x - mn[view]) * sc, size - 10 - (z - mn[2]) * sc) for x, z in zip(a, b)]
            d.polygon(pts, fill=(shade, shade, shade))
    return img


if __name__ == "__main__":
    raw = Path(sys.argv[1]).read_bytes()
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageDraw
    tiles = []
    for start, count in find_vertex_buffers(raw):
        m = decode(raw, start, count)
        np.savez(out / f"{start:08x}.npz", **m)
        img = render(m)
        ImageDraw.Draw(img).text((4, 4), f"{start:#x} v{count} t{len(m['tris'])}", fill=(255, 255, 0))
        tiles.append(img)
        print(f"{start:#x}: {count} verts, {len(m['tris'])} tris, bones {m['bones'].max() + 1}")
    if tiles:
        cols = 3
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (512 * cols, 256 * rows))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols) * 512, (i // cols) * 256))
        sheet.save(out / "sheet.png")
