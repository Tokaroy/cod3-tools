"""Find code in the unpacked COD3 executable (xex_unpack.py output, loaded at 0x82000000) that builds a
given address with lis/addi (or lis/ori), and disassemble around it.

  python -I ppc_xref.py <default.img> <hex address> [--dis N]
"""
import struct
import sys

import capstone

BASE = 0x82000000


def xrefs(img, target):
    hi, lo = (target + 0x8000) >> 16 & 0xFFFF, target & 0xFFFF
    lo_s = lo - 0x10000 if lo & 0x8000 else lo
    n = len(img) // 4
    words = struct.unpack(f">{n}I", img[:n * 4])
    out = []
    for i, w in enumerate(words):
        if w >> 26 == 15 and (w >> 16) & 31 == 0 and w & 0xFFFF == hi:  # lis rD, hi
            rd = (w >> 21) & 31
            for j in range(i + 1, min(i + 24, n)):
                v = words[j]
                op = v >> 26
                if op == 14 and (v >> 16) & 31 == rd and struct.unpack(">h", struct.pack(">H", v & 0xFFFF))[0] == lo_s:
                    out.append(BASE + j * 4)  # addi rX, rD, lo
                    break
                if op in (24,) and (v >> 21) & 31 == rd and v & 0xFFFF == lo:  # ori
                    out.append(BASE + j * 4)
                    break
    return out


def function_start(img, addr):
    """Walk back to the function prologue (mflr r12 = 0x7d8802a6) or a blr before it."""
    a = addr
    while a > BASE:
        w = struct.unpack_from(">I", img, a - BASE)[0]
        if w == 0x7D8802A6:
            return a
        if w == 0x4E800020 and a != addr:
            return a + 4
        a -= 4
    return BASE


def disasm(img, start, count):
    """One line per word; words capstone can't decode (VMX128 etc.) are named by vmx128.py or shown raw."""
    md = capstone.Cs(capstone.CS_ARCH_PPC, capstone.CS_MODE_32 | capstone.CS_MODE_BIG_ENDIAN)
    try:
        from vmx128 import decode_any
    except ImportError:
        decode_any = lambda w: None
    out = []
    for k in range(count):
        a = start + 4 * k
        code = img[a - BASE:a - BASE + 4]
        if len(code) < 4:
            break
        ins = next(md.disasm(code, a), None)
        if ins is not None:
            out.append(f"{a:08x}: {ins.mnemonic:8} {ins.op_str}")
        else:
            w = struct.unpack(">I", code)[0]
            out.append(f"{a:08x}: {decode_any(w) or '.word'} {w:08x}")
    return out


if __name__ == "__main__":
    img = open(sys.argv[1], "rb").read()
    target = int(sys.argv[2], 16)
    n = int(sys.argv[sys.argv.index("--dis") + 1]) if "--dis" in sys.argv else 0
    for x in xrefs(img, target):
        f = function_start(img, x)
        print(f"ref at {x:08x} in function {f:08x}")
        if n:
            print("\n".join(disasm(img, f, n)))
