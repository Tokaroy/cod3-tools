"""Exact layout of a Call of Duty 3 APKF asset pack (a .cod section or uncompressed GPU pack), from its header.

The header (big-endian) describes every object in the pack, so where each record and each block of GPU
data sits follows exactly - no searching:
  0x00 "APKF", u32 version (0x04010000), u32 4, u32 -1, u32 2, u32 header size H, u32 0
  0x1C group table: 28 bytes per asset type, ended by a zero word:
       u32 type (a FourCC such as "TEX", "MESH", "MAT", "FX", "MLTI", or a hash), u32 ?, u32 ?,
       u32 offset of the type's entry list (from the word after this one), u32 count,
       u32 record alignment, u32 flags (high byte 1 = the type has GPU data; low 24 bits = its GPU alignment)
  entry lists: per object (name entry offset, record size) or, for types with GPU data,
       (name entry offset, record size, GPU data size)
  name entries: u32 hash, char[16] name
Records start at data0 = (H rounded down to 16) + 0x50, raised to the largest record alignment of any type
(some types need 0x80 or 0x800), one after another in group and entry order, each aligned to its type's
record alignment. The GPU data start at the first 64 KB boundary after the last record and hold every GPU
object, in the same order, each aligned to its type's GPU alignment (textures 4 KB).

Checked on a disc-wide sample (60 packs): every texture record and every mesh object's GPU block
(vertices inside the surface's stored bounds, indices valid) lands exactly where this layout says.
"""
import struct
from dataclasses import dataclass, field

DATA_ALIGN = 0x10000


def _u32(d, o):
    return struct.unpack_from(">I", d, o)[0]


def _align(v, a):
    return (v + a - 1) // a * a if a > 1 else v


@dataclass
class Group:
    tag: str
    count: int
    record_align: int
    flags: int
    entries: list = field(default_factory=list)  # (name entry offset, record size[, GPU size])

    @property
    def gpu_align(self):
        return self.flags & 0xFFFFFF if self.flags >> 24 == 1 else None


@dataclass
class Object:
    tag: str
    index: int          # within its group
    record: int         # file offset of its record
    record_size: int
    gpu: int = None     # file offset of its GPU data (None: no GPU data)
    gpu_size: int = 0


@dataclass
class Layout:
    data0: int
    records_end: int
    gpu_start: int
    gpu_end: int
    groups: list
    objects: list

    def by_tag(self, tag):
        return [o for o in self.objects if o.tag == tag]


def _tag(raw):
    t = raw[::-1].strip(b"\0")
    return t.decode("ascii") if t and t.isalpha() else raw.hex()


def groups(d):
    if d[:4] != b"APKF":
        raise ValueError("not an APKF pack")
    out, p, first = [], 0x1C, None
    while (first is None or p < first) and p + 28 <= len(d) and _u32(d, p) != 0:
        raw, _, _, rel, count, ralign, flags = struct.unpack_from(">4s6I", d, p)
        at = p + 0xC + rel
        if count > 0x10000 or at >= len(d):
            raise ValueError(f"bad group at {p:#x}")
        first = at if first is None else min(first, at)
        gpu = flags >> 24 == 1
        size = 12 if gpu else 8
        entries = [struct.unpack_from(">3I" if gpu else ">2I", d, at + size * i) for i in range(count)]
        out.append(Group(_tag(raw), count, ralign, flags, entries))
        p += 28
    return out


def layout(d) -> Layout:
    gs = groups(d)
    data0 = _align((_u32(d, 0x14) & ~0xF) + 0x50, max([0x10] + [g.record_align for g in gs]))
    cpu, objs = data0, []
    for g in gs:
        for i, e in enumerate(g.entries):
            cpu = _align(cpu, g.record_align)
            objs.append(Object(g.tag, i, cpu, e[1], None, e[2] if len(e) == 3 else 0))
            cpu += e[1]
    gpu = start = _align(cpu, DATA_ALIGN)
    for g in gs:
        if g.gpu_align is None:
            continue
        for o in objs:
            if o.tag == g.tag and o.gpu is None and o.gpu_size:
                gpu = _align(gpu, g.gpu_align or 1)
                o.gpu = gpu
                gpu += o.gpu_size
    if gpu > len(d) + DATA_ALIGN:
        raise ValueError(f"layout runs past the pack ({gpu:#x} > {len(d):#x})")
    return Layout(data0, cpu, start, gpu, gs, objs)


def texture_offsets(d, fetch_positions):
    """{fetch constant position: pixel data offset} for texture records (fetch constant at record + 0x48)."""
    lay = layout(d)
    by_record = {o.record: o.gpu for o in lay.objects if o.gpu is not None}
    return {p: by_record.get(p - 0x48) for p in fetch_positions}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    for f in sys.argv[1:]:
        d = Path(f).read_bytes()
        lay = layout(d)
        print(f"{f}: data0 {lay.data0:#x}, records end {lay.records_end:#x}, GPU data {lay.gpu_start:#x}..{lay.gpu_end:#x}"
              f" (pack {len(d):#x})")
        for g in lay.groups:
            print(f"  {g.tag:8} {g.count:4} objects, record align {g.record_align:#x}"
                  + (f", GPU align {g.gpu_align:#x}" if g.gpu_align is not None else ""))
