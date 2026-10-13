"""Xbox 360 texture decoding with numpy.

Untiles an Xbox 360 texture's base mip, swaps endianness and decodes DXT1/DXT3/DXT5 or 8888.
"""
import numpy as np

from xenos_fetch import FormatError, align, block_layout as _layout


def _tiled_offsets(bw, bh, width, log_bpb):
    y, x = np.mgrid[0:bh, 0:bw].astype(np.int64)
    aligned = align(width, 32)
    macro = ((x >> 5) + (y >> 5) * (aligned >> 5)) << (log_bpb + 7)
    micro = ((x & 7) + ((y & 0xE) << 2)) << log_bpb
    off = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4)
    addr = (((off & ~0x1FF) << 3) + ((y & 16) << 7) + ((off & 0x1C0) << 2)
            + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (off & 0x3F))
    return addr >> log_bpb


def linearize(data: bytes, fc):
    bs, bpb = _layout(fc)
    bw, bh = max(1, (fc.width + bs - 1) // bs), max(1, (fc.height + bs - 1) // bs)
    src = np.frombuffer(data, np.uint8)
    if fc.tiled:
        idx = _tiled_offsets(bw, bh, align(fc.width, 128 if bs == 4 else 32) // bs, bpb.bit_length() - 1)
    else:
        pitch = max(bw, ((fc.pitch or fc.width) + bs - 1) // bs)
        y, x = np.mgrid[0:bh, 0:bw]
        idx = y * pitch + x
    idx = idx.reshape(-1)
    if (idx.max() + 1) * bpb > len(src):
        raise FormatError("texture payload truncated")
    blocks = src[(idx[:, None] * bpb + np.arange(bpb)).reshape(-1)].reshape(-1, bpb)
    if bs == 4:
        blocks = blocks.reshape(-1, bpb // 2, 2)[:, :, ::-1].reshape(-1, bpb)  # 16-bit swap
    else:
        blocks = blocks[:, ::-1]
    return blocks, bw, bh


def _565(v):
    r, g, b = (v >> 11) & 31, (v >> 5) & 63, v & 31
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.int32)


def _dxt(blocks, fmt):
    n = len(blocks)
    cb = blocks[:, 8:16] if fmt != 18 else blocks[:, 0:8]
    c0 = cb[:, 0].astype(np.int32) | (cb[:, 1].astype(np.int32) << 8)
    c1 = cb[:, 2].astype(np.int32) | (cb[:, 3].astype(np.int32) << 8)
    bits = cb[:, 4:8].copy().view("<u4").reshape(-1).astype(np.int64)
    a, b = _565(c0), _565(c1)
    four = (c0 > c1) | (fmt != 18)
    pal = np.zeros((n, 4, 4), np.int32)
    pal[:, 0, :3], pal[:, 1, :3] = a, b
    pal[:, 2, :3] = np.where(four[:, None], (2 * a + b) // 3, (a + b) // 2)
    pal[:, 3, :3] = np.where(four[:, None], (a + 2 * b) // 3, 0)
    pal[:, :3, 3] = 255
    pal[:, 3, 3] = np.where(four, 255, 0)
    sel = (bits[:, None] >> (np.arange(16) * 2)) & 3
    px = np.take_along_axis(pal, sel[:, :, None].repeat(4, 2), 1)  # n,16,4
    if fmt == 19:
        ab = blocks[:, :8].astype(np.int32)
        px[:, :, 3] = np.stack([(ab & 15) * 17, (ab >> 4) * 17], -1).reshape(n, 16)
    elif fmt == 20:
        a0, a1 = blocks[:, 0].astype(np.int32), blocks[:, 1].astype(np.int32)
        ap = np.zeros((n, 8), np.int32)
        ap[:, 0], ap[:, 1] = a0, a1
        big = a0 > a1
        for i in range(1, 7):
            ap[:, i + 1] = np.where(big, ((7 - i) * a0 + i * a1) // 7, 0)
        for i in range(1, 5):
            ap[:, i + 1] = np.where(big, ap[:, i + 1], ((5 - i) * a0 + i * a1) // 5)
        ap[:, 6] = np.where(big, ap[:, 6], 0)
        ap[:, 7] = np.where(big, ap[:, 7], 255)
        abits = np.zeros(n, np.int64)
        for k in range(6):
            abits |= blocks[:, 2 + k].astype(np.int64) << (8 * k)
        asel = (abits[:, None] >> (np.arange(16) * 3)) & 7
        px[:, :, 3] = np.take_along_axis(ap, asel, 1)
    return px


def decode_rgba(data: bytes, fc) -> np.ndarray:
    blocks, bw, bh = linearize(data, fc)
    if fc.format_code in (18, 19, 20):
        px = _dxt(blocks, fc.format_code).reshape(bh, bw, 4, 4, 4).transpose(0, 2, 1, 3, 4)
        img = px.reshape(bh * 4, bw * 4, 4)[:fc.height, :fc.width]
    elif fc.format_code == 6:
        img = blocks.reshape(bh, bw, 4)[:, :, [2, 1, 0, 3]]  # after reversal: B G R A order -> RGBA
        img = img[:fc.height, :fc.width]
    else:
        raise FormatError(f"unsupported Xbox texture format {fc.format_name}")
    return np.ascontiguousarray(img.astype(np.uint8))
