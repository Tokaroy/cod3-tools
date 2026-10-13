"""Xbox 360 (Xenos) texture fetch constants and texture block layouts, for the COD3 texture tools.

A texture fetch constant is six 32-bit words (big-endian in the title's data). The fields used here, as
Xenia documents them (xe_gpu_texture_fetch_t, src/xenia/gpu/xenos.h), counting bits from the least
significant end of each word:
  word 0: bit 31 tiled, bits 22-30 pitch (in 32-texel units)
  word 1: bits 0-5 format, bits 6-7 endianness, bits 12-31 base address (4 KB units)
  word 2 (2D): bits 0-12 width - 1, bits 13-25 height - 1
  word 4: bits 6-9 the highest mip level (mip count - 1; checked against 800+ of the title's records)
"""
import struct
from dataclasses import asdict, dataclass

FORMATS = {6: "A8R8G8B8", 18: "DXT1", 19: "DXT3", 20: "DXT5"}


class FormatError(ValueError):
    """Malformed or unsupported texture data."""


@dataclass
class FetchConstant:
    width: int
    height: int
    format_code: int
    format_name: str
    mipmaps: int
    tiled: bool
    endian: int
    pitch: int = 0

    def manifest(self) -> dict:
        return asdict(self)


def bits(word: int, lo: int, count: int) -> int:
    return (word >> lo) & ((1 << count) - 1)


def parse_fetch_constant(data: bytes) -> FetchConstant:
    if len(data) < 24:
        raise FormatError("fetch constant needs 24 bytes")
    w = struct.unpack(">6I", data[:24])
    if bits(w[0], 0, 2) != 2:  # constant type 2 = texture
        raise FormatError("not a texture fetch constant")
    fmt = bits(w[1], 0, 6)
    return FetchConstant(width=bits(w[2], 0, 13) + 1, height=bits(w[2], 13, 13) + 1, format_code=fmt,
                         format_name=FORMATS.get(fmt, f"XENOS_{fmt}"), mipmaps=bits(w[4], 6, 4) + 1,
                         tiled=bool(bits(w[0], 31, 1)), endian=bits(w[1], 6, 2), pitch=bits(w[0], 22, 9) * 32)


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def block_layout(fetch: FetchConstant) -> tuple:
    """(block size in texels, bytes per block) of a supported format."""
    if fetch.format_code == 18:
        return 4, 8
    if fetch.format_code in (19, 20):
        return 4, 16
    if fetch.format_code == 6:
        return 1, 4
    raise FormatError(f"unsupported Xbox texture format {fetch.format_name}")
