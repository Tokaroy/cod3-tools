"""Call of Duty 3 (Xbox 360) .cod archives: the KAPF container and its LZO1X-compressed "NCH" chunks.

Layout (worked out from the files; little-endian unless noted):
  0x00 "KAPF", f32 version, u32 header size (0x30), u32 section count, u32 entry table offset,
       u32 path table offset, u32 ?, char[32] build name ("CODAUTO30")
  sections (32 bytes each) follow the header: u32 ?, u32 ?, u32 data offset, u32 unpacked size, ...
  file entries (20 bytes each, from the entry table offset): u32 name offset, u32 path offset,
       u32 size, u32 offset in the unpacked section, u16 ?, u16 section/flags
  Section data is a run of "NCH\\0" chunks, each 0x20-byte header (big-endian):
       u32 packed size, u32 hash, u32 unpacked size, u32 unpacked offset, u32 hash, u32 size+0x20, u32 1
  followed by an LZO1X stream. Chunks start on 0x8000-byte boundaries.

Usage:
  python -I cod3_kapf.py list <file.cod>
  python -I cod3_kapf.py unpack <file.cod> <out dir>      (writes each section's unpacked data)
"""
import struct
import sys
from pathlib import Path


def lzo1x_decompress(src: bytes, out_len: int) -> bytes:
    dst = bytearray()
    ip = 0
    n = len(src)

    def copy_match(m_pos: int, count: int):
        if m_pos < 0:
            raise ValueError(f"LZO back-reference before start (pos {m_pos})")
        for k in range(count):
            dst.append(dst[m_pos + k])

    state = "start"
    t = 0
    while True:
        if state == "start":
            t = src[ip]
            if t > 17:
                ip += 1
                t -= 17
                if t < 4:
                    state = "match_next"
                    continue
                dst += src[ip:ip + t]; ip += t
                state = "first_literal_run"
                continue
            state = "loop"
            continue

        if state == "loop":
            t = src[ip]; ip += 1
            if t >= 16:
                state = "match"
                continue
            if t == 0:
                while src[ip] == 0:
                    t += 255; ip += 1
                t += 15 + src[ip]; ip += 1
            dst += src[ip:ip + t + 3]; ip += t + 3
            state = "first_literal_run"
            continue

        if state == "first_literal_run":
            t = src[ip]; ip += 1
            if t >= 16:
                state = "match"
                continue
            m_pos = len(dst) - (1 + 0x0800) - (t >> 2) - (src[ip] << 2); ip += 1
            copy_match(m_pos, 3)
            state = "match_done"
            continue

        if state == "match":
            if t >= 64:
                m_pos = len(dst) - 1 - ((t >> 2) & 7) - (src[ip] << 3); ip += 1
                copy_match(m_pos, (t >> 5) - 1 + 2)
            elif t >= 32:
                t &= 31
                if t == 0:
                    while src[ip] == 0:
                        t += 255; ip += 1
                    t += 31 + src[ip]; ip += 1
                m_pos = len(dst) - 1 - ((src[ip] | (src[ip + 1] << 8)) >> 2); ip += 2
                copy_match(m_pos, t + 2)
            elif t >= 16:
                m_pos = len(dst) - ((t & 8) << 11)
                t &= 7
                if t == 0:
                    while src[ip] == 0:
                        t += 255; ip += 1
                    t += 7 + src[ip]; ip += 1
                m_pos -= (src[ip] | (src[ip + 1] << 8)) >> 2; ip += 2
                if m_pos == len(dst):
                    break  # end of stream
                m_pos -= 0x4000
                copy_match(m_pos, t + 2)
            else:
                m_pos = len(dst) - 1 - (t >> 2) - (src[ip] << 2); ip += 1
                copy_match(m_pos, 2)
            state = "match_done"
            continue

        if state == "match_done":
            t = src[ip - 2] & 3
            if t == 0:
                state = "loop"
                continue
            state = "match_next"
            continue

        if state == "match_next":
            dst += src[ip:ip + t]; ip += t
            t = src[ip]; ip += 1
            state = "match"
            continue

        if ip >= n:
            break
    if len(dst) != out_len:
        raise ValueError(f"LZO produced {len(dst)} bytes, expected {out_len}")
    return bytes(dst)


class Kapf:
    def __init__(self, path):
        self.path = Path(path)
        self.d = self.path.read_bytes()
        d = self.d
        if d[:4] != b"KAPF":
            raise ValueError("not a KAPF archive")
        self.header_size, self.section_count, self.entry_off, self.path_off, self.unk = struct.unpack_from("<5I", d, 8)
        self.build = d[0x1C:0x3C].split(b"\0")[0].decode()

    def chunks(self, start: int):
        """Yield (packed bytes, unpacked size, unpacked offset) for the NCH chunks from start."""
        pos = start
        while pos + 0x20 <= len(self.d) and self.d[pos:pos + 4] == b"NCH\0":
            packed, _, size, offset, _, total, method = struct.unpack_from(">7I", self.d, pos + 4)
            yield self.d[pos + 0x20:pos + 0x20 + packed], size, offset
            pos += total
            pos = (pos + 0x7FFF) & ~0x7FFF

    def section_data(self, start: int) -> bytes:
        out = bytearray()
        for packed, size, offset in self.chunks(start):
            if offset == 0 and out:
                break  # the next section's first chunk
            if offset != len(out):
                raise ValueError(f"chunk at unpacked offset {offset:#x}, expected {len(out):#x}")
            out += lzo1x_decompress(packed, size)
        return bytes(out)

    def section_starts(self):
        """Data offsets of each section's first NCH chunk (scanned; header fields not fully known)."""
        starts, pos = [], 0
        while True:
            pos = self.d.find(b"NCH\0", pos)
            if pos < 0:
                break
            _, _, _, offset, _, total, _ = struct.unpack_from(">7I", self.d, pos + 4)
            if offset == 0:
                starts.append(pos)
            pos += 4
        return starts

    def names(self):
        out, pos = [], self.entry_off
        while pos + 20 <= self.path_off:
            name_off, path_off, a, b, c = struct.unpack_from("<5I", self.d, pos)
            if name_off == 0 and path_off == 0:
                break
            name = self.d[name_off:self.d.find(b"\0", name_off)].decode("latin-1")
            out.append((name, a, b, c))
            pos += 20
        return out


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    k = Kapf(path)
    if cmd == "list":
        print(f"{path}: build {k.build}, {k.section_count} section(s), sections at {[hex(s) for s in k.section_starts()]}")
        for name, a, b, c in k.names():
            print(f"  {name:60s} {a:#10x} {b:#10x} {c:#10x}")
    elif cmd == "unpack":
        out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(k.section_starts()):
            data = k.section_data(s)
            (out / f"{k.path.stem}.section{i}.bin").write_bytes(data)
            print(f"section {i}: {len(data):#x} bytes")
