"""Decode the textures in unpacked Call of Duty 3 .cod sections (see cod3_kapf.py).

Each section's TEX descriptors carry an Xbox 360 texture fetch constant (24 bytes, 28 bytes into a
96-byte record). The pixel data follow the section header back to back; the first texture starts at
the first 64 KB boundary after the header, each one padded to 4 KB. We find the start by trying
offsets and keeping the one whose decode is smoothest (real images have small neighbour differences).

Usage: python -I cod3_textures.py <section .bin files...> <out dir>
"""
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xenos_fast  # noqa: E402
import xenos_fetch as xenos  # noqa: E402


def mip_bytes(fc) -> int:
    """Bytes of the full mip chain, roughly as the Xbox lays it out (4 KB aligned)."""
    block = 8 if fc.format_code == 18 else 16
    texel = 1 if fc.format_code == 6 else 4  # A8R8G8B8 is 1x1 "blocks" of 4 bytes
    if fc.format_code == 6:
        block = 4
    total, w, h = 0, fc.width, fc.height
    for _ in range(max(1, fc.mipmaps)):
        bw, bh = max(1, (w + texel - 1) // texel), max(1, (h + texel - 1) // texel)
        total += max(4096, ((bw + 31) // 32 * 32) * ((bh + 31) // 32 * 32) * block)
        w, h = max(1, w // 2), max(1, h // 2)
    return (total + 0xFFF) & ~0xFFF


def descriptors(s: bytes):
    out = []
    for o in range(0, min(0x10000, len(s)) - 24, 4):  # descriptors sit after the name table, before the pixels
        try:
            fc = xenos.parse_fetch_constant(s[o:o + 24])
        except Exception:
            continue
        if fc.format_code in (18, 19, 20) and fc.tiled and 16 <= fc.width <= 2048 and 16 <= fc.height <= 2048 \
                and fc.width & (fc.width - 1) == 0 and fc.height & (fc.height - 1) == 0 and 1 <= fc.mipmaps <= 12:
            out.append((o, fc))
    return out


def records(s: bytes):
    """Every texture record's fetch constant (DXT1/3/5 and A8R8G8B8), found by the two 0xFFFF0000 words
    that precede it; in record order, which is also the order of the pixel data."""
    out, p = [], s.find(b"\xff\xff\x00\x00\xff\xff\x00\x00")
    while p >= 0 and p < 0x80000:
        o = p + 8
        try:
            fc = xenos.parse_fetch_constant(s[o:o + 24])
            if fc.format_code in (6, 18, 19, 20) and 1 <= fc.width <= 4096 and 1 <= fc.height <= 4096:
                out.append((o, fc))
        except Exception:
            pass
        p = s.find(b"\xff\xff\x00\x00\xff\xff\x00\x00", p + 4)
    return out


def data_sizes(s: bytes, count: int):
    """The section header lists each texture as (descriptor offset, 0x60, data size)."""
    w = np.frombuffer(s[:0x4000], dtype=">u4")
    sizes = []
    for i in range(len(w) - 2):
        if w[i + 1] == 0x60 and w[i + 2] % 0x1000 == 0 and 0x1000 <= w[i + 2] <= 0x800000 and w[i] < 0x10000:
            sizes.append(int(w[i + 2]))
    return sizes[:count] if len(sizes) >= count else []


def smooth(rgb):
    return np.abs(np.diff(rgb, axis=1)).mean() + np.abs(np.diff(rgb, axis=0)).mean()


def decode(s, o, fc):
    return xenos_fast.decode_rgba(s[o:o + mip_bytes(fc)], fc)


if __name__ == "__main__":
    *files, out = sys.argv[1:]
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for f in files:
        s = Path(f).read_bytes()
        ds = descriptors(s)
        if not ds:
            continue
        # Find where the first texture starts (after the descriptors).
        first = (ds[-1][0] + 0xFFF) & ~0xFFF
        best = None
        for start in range(first, first + 0x20000, 0x1000):
            sc = smooth(decode(s, start, ds[0][1])[:, :, :3].astype(int))
            if best is None or sc < best[0]:
                best = (sc, start)
        pos = best[1]
        sizes = data_sizes(s, len(ds))
        for i, (o, fc) in enumerate(ds):
            try:
                img = decode(s, pos, fc)
            except Exception as e:
                print(f"  ! {Path(f).stem} texture {i}: {e}")
                break
            name = f"{Path(f).stem}_{i:02d}_{fc.width}x{fc.height}_{fc.format_name}.png"
            Image.fromarray(img, "RGBA").save(out / name)
            print(f"{name} at {pos:#x}")
            pos += sizes[i] if i < len(sizes) else mip_bytes(fc)
