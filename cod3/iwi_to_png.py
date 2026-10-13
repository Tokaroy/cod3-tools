"""Convert Call of Duty 4 / World at War .iwi images (version 5/6; DXT1/3/5, ARGB) to PNG.

  python -I cod3/iwi_to_png.py <iwi folder> <png folder>
Header: "IWi", version, format (11 DXT1, 12 DXT3, 13 DXT5, 1 ARGB8, 2 RGB8), usage, width, height, depth
(u16 each), then four u32 mip end offsets; the largest mip is stored last (offsets[1]..offsets[0]).
"""
import io
import struct
import sys
from pathlib import Path

from PIL import Image

FOURCC = {11: b"DXT1", 12: b"DXT3", 13: b"DXT5"}


def iwi_image(data):
    if data[:3] != b"IWi":
        raise ValueError("not an IWi file")
    fmt = data[4]
    w, h, _ = struct.unpack_from("<HHH", data, 6)
    ends = struct.unpack_from("<4I", data, 12)
    start = ends[1] if ends[1] < ends[0] else 28
    pixels = data[start:ends[0]]
    if fmt in FOURCC:
        header = struct.pack("<4sI6I44s8I5I", b"DDS ", 124, 0x81007, h, w, len(pixels), 0, 1, b"\0" * 44,
                             32, 4, int.from_bytes(FOURCC[fmt], "little"), 0, 0, 0, 0, 0, 0x1000, 0, 0, 0, 0)
        return Image.open(io.BytesIO(header + pixels))
    if fmt == 1:
        return Image.frombuffer("RGBA", (w, h), pixels[:w * h * 4], "raw", "BGRA", 0, 1)
    if fmt == 2:
        return Image.frombuffer("RGB", (w, h), pixels[:w * h * 3], "raw", "BGR", 0, 1)
    raise ValueError(f"format {fmt}")


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    ok = bad = 0
    for f in sorted(src.glob("*.iwi")):
        try:
            iwi_image(f.read_bytes()).save(dst / (f.stem + ".png"))
            ok += 1
        except Exception as e:
            bad += 1
            print(f"{f.name}: {e}")
    print(f"{ok} converted, {bad} failed")
