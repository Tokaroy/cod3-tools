"""Xbox 360 VMX128 / AltiVec vector instructions in Python, for cod3_emu.py (Unicorn has no VMX128).

Decoding follows Xenia's opcode table (xenia-project/xenia, src/xenia/cpu/ppc/ppc_opcode_lookup_gen.cc,
ppc_decode_data.h, BSD licence); behaviour follows ppc_emit_altivec.cc and the AltiVec manual.
Vector registers: 128 x 16 bytes, element 0 = most significant (big-endian), floats are float32.
"""
import struct

import numpy as np


def bits(v, a, b):  # PowerPC bit numbering (0 = MSB), like Xenia's ExtractBits
    return (v >> (31 - b)) & ((1 << (b - a + 1)) - 1)


MEM_OPS = {0b00000000011: "lvsl128", 0b00001000011: "lvsr128", 0b00010000011: "lvewx128",
           0b00011000011: "lvx128", 0b00110000011: "stvewx128", 0b00111000011: "stvx128",
           0b01011000011: "lvxl128", 0b01111000011: "stvxl128", 0b10000000011: "lvlx128",
           0b10001000011: "lvrx128", 0b10100000011: "stvlx128", 0b10101000011: "stvrx128",
           0b11000000011: "lvlxl128", 0b11001000011: "lvrxl128", 0b11100000011: "stvlxl128",
           0b11101000011: "stvrxl128"}
VX_OPS = {0b00000001010: "vaddfp", 0b00001001010: "vsubfp", 0b00010000000: "vadduwm",
          0b00010001100: "vmrghw", 0b00110001100: "vmrglw", 0b01010001100: "vspltw",
          0b01110001100: "vspltisw", 0b01100001100: "vspltisb", 0b01101001100: "vspltish",
          0b10000000100: "vand", 0b10001000100: "vandc", 0b10010000100: "vor", 0b10011000100: "vxor",
          0b10100000100: "vnor", 0b10000001010: "vmaxfp", 0b10001001010: "vminfp",
          0b00110000100: "vslw", 0b01010000100: "vsrw", 0b01110000100: "vsraw",
          0b01101001010: "vcfsx", 0b01100001010: "vcfux", 0b01111001010: "vctsxs", 0b01110001010: "vctuxs",
          0b00100001010: "vrefp", 0b00101001010: "vrsqrtefp", 0b10010000000: "vsubuwm",
          0b00011001110: "vpkuwus", 0b01010001110: "vupklsb", 0b01001001110: "vupkhsh",
          0b01011001110: "vupklsh", 0b01000001110: "vupkhsb", 0b00001001110: "vpkuwum",
          0b01001001010: "vrfiz", 0b01011001010: "vrfim", 0b01000001010: "vrfin", 0b01010001010: "vrfip",
          0b00111000100: "vsl", 0b01011000100: "vsr", 0b10001001100: "vsro", 0b10000001100: "vslo",
          0b01010000000: "vadduws", 0b00110000010: "vmaxsw", 0b01110000010: "vminsw"}
VC_OPS = {0b0011000110: "vcmpeqfp", 0b0111000110: "vcmpgefp", 0b1011000110: "vcmpgtfp",
          0b0010000110: "vcmpequw", 0b1110000110: "vcmpgtsw", 0b1010000110: "vcmpgtuw"}
VA_OPS = {0b101010: "vsel", 0b101011: "vperm", 0b101100: "vsldoi", 0b101110: "vmaddfp",
          0b101111: "vnmsubfp"}
OP5 = {0b000001: "vaddfp128", 0b000101: "vsubfp128", 0b001001: "vmulfp128", 0b001101: "vmaddfp128",
       0b010001: "vmaddcfp128", 0b010101: "vnmsubfp128", 0b011001: "vmsum3fp128", 0b011101: "vmsum4fp128",
       0b100001: "vand128", 0b100101: "vandc128", 0b101001: "vnor128", 0b101101: "vor128",
       0b110001: "vxor128", 0b110101: "vsel128", 0b111001: "vslo128", 0b111101: "vsro128",
       0b100000: "vpkshss128", 0b100100: "vpkshus128", 0b101000: "vpkswss128", 0b101100: "vpkswus128",
       0b110000: "vpkuhum128", 0b110100: "vpkuhus128", 0b111000: "vpkuwum128", 0b111100: "vpkuwus128"}
OP6_A = {0b0100011: "vcfpsxws128", 0b0100111: "vcfpuxws128", 0b0101011: "vcsxwfp128",
         0b0101111: "vcuxwfp128", 0b0110011: "vrfim128", 0b0110111: "vrfin128", 0b0111011: "vrfip128",
         0b0111111: "vrfiz128", 0b1100011: "vrefp128", 0b1100111: "vrsqrtefp128",
         0b1101011: "vexptefp128", 0b1101111: "vlogefp128", 0b1110011: "vspltw128",
         0b1110111: "vspltisw128", 0b1111111: "vupkd3d128"}
OP6_C = {0b000000: "vcmpeqfp128", 0b001000: "vcmpgefp128", 0b010000: "vcmpgtfp128",
         0b011000: "vcmpbfp128", 0b100000: "vcmpequw128"}


def decode(w):
    """Instruction name for an opcode 4/5/6 word, or None."""
    op = w >> 26
    if op == 4:
        n = MEM_OPS.get((bits(w, 21, 27) << 4) | bits(w, 30, 31))
        if n:
            return n
        n = VX_OPS.get(bits(w, 21, 31))
        if n:
            return n
        n = VC_OPS.get(bits(w, 22, 31))
        if n:
            return n
        n = VA_OPS.get(bits(w, 26, 31))
        if n:
            return n
        if bits(w, 27, 27):
            return "vsldoi128"
    elif op == 5:
        if (bits(w, 22, 22) << 5 | bits(w, 27, 27)) == 0:
            return "vperm128"
        return OP5.get((bits(w, 22, 25) << 2) | bits(w, 27, 27))
    elif op == 6:
        if ((bits(w, 21, 22) << 5) | bits(w, 26, 27)) == 0b0100001:
            return "vpermwi128"
        n = {0b1100001: "vpkd3d128", 0b1110001: "vrlimi128"}.get((bits(w, 21, 23) << 4) | bits(w, 26, 27))
        if n:
            return n
        n = OP6_A.get(bits(w, 21, 27))
        if n:
            return n
        return OP6_C.get((bits(w, 22, 24) << 3) | bits(w, 27, 27))
    return None


# register fields (Xenia formats); x86 bitfields are LSB-first, so field offsets count from bit 0 = LSB
def vx128(w):  # VD, VA, VB
    vd = ((w >> 21) & 31) | (((w >> 2) & 3) << 5)
    va = ((w >> 16) & 31) | (((w >> 5) & 1) << 5) | (((w >> 10) & 1) << 6)
    vb = ((w >> 11) & 31) | ((w & 3) << 5)
    return vd, va, vb


def vx128_1(w):  # VD/VS, RA, RB
    return ((w >> 21) & 31) | (((w >> 2) & 3) << 5), (w >> 16) & 31, (w >> 11) & 31


def vx128_2_vc(w):
    return (w >> 6) & 7


def vx128_3(w):  # VD, VB, UIMM
    return ((w >> 21) & 31) | (((w >> 2) & 3) << 5), ((w >> 11) & 31) | ((w & 3) << 5), (w >> 16) & 31


def vx128_4(w):  # VD, VB, IMM, z
    return ((w >> 21) & 31) | (((w >> 2) & 3) << 5), ((w >> 11) & 31) | ((w & 3) << 5), (w >> 16) & 31, (w >> 6) & 3


def vx128_5_sh(w):
    return (w >> 6) & 15


def vx128_p(w):  # VD, VB, PERM
    return (((w >> 21) & 31) | (((w >> 2) & 3) << 5), ((w >> 11) & 31) | ((w & 3) << 5),
            ((w >> 16) & 31) | (((w >> 6) & 7) << 5))


def vx(w):  # standard AltiVec VD, VA, VB, VC
    return (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31, (w >> 6) & 31


class VMX:
    """128 vector registers and the instructions COD3's animation code uses."""

    def __init__(self):
        self.vr = np.zeros((128, 16), np.uint8)

    def f(self, i):
        return self.vr[i].view(">f4").astype(np.float32)

    def u(self, i):
        return self.vr[i].view(">u4").copy()

    def set_f(self, i, v):
        self.vr[i] = np.asarray(v, np.float32).astype(">f4").view(np.uint8)

    def set_u(self, i, v):
        self.vr[i] = np.asarray(v, np.uint32).astype(">u4").view(np.uint8)

    def execute(self, emu, w):
        name = decode(w)
        if name is None:
            raise RuntimeError(f"unknown vector instruction {w:08x}")
        with np.errstate(all="ignore"):
            getattr(self, "op_" + name)(emu, w)
        return name

    # memory (VX128_1): EA = (RA|0) + RB, 16-byte aligned
    def _ea(self, emu, w):
        _, ra, rb = vx128_1(w)
        return ((0 if ra == 0 else emu.reg(f"r{ra}")) + emu.reg(f"r{rb}")) & 0xFFFFFFF0

    def op_lvx128(self, emu, w):
        vd = vx128_1(w)[0]
        self.vr[vd] = np.frombuffer(emu.read(self._ea(emu, w), 16), np.uint8)

    def op_stvx128(self, emu, w):
        vs = vx128_1(w)[0]
        emu.uc.mem_write(self._ea(emu, w), self.vr[vs].tobytes())

    # standard AltiVec (VX / VA forms, registers 0-31)
    def op_vspltw(self, emu, w):
        vd, uimm, vb, _ = vx(w)
        self.set_u(vd, np.repeat(self.u(vb)[uimm & 3], 4))

    def op_vaddfp(self, emu, w):
        vd, va, vb, _ = vx(w)
        self.set_f(vd, self.f(va) + self.f(vb))

    def op_vsubfp(self, emu, w):
        vd, va, vb, _ = vx(w)
        self.set_f(vd, self.f(va) - self.f(vb))

    def op_vxor(self, emu, w):
        vd, va, vb, _ = vx(w)
        self.vr[vd] = self.vr[va] ^ self.vr[vb]

    def op_vor(self, emu, w):
        vd, va, vb, _ = vx(w)
        self.vr[vd] = self.vr[va] | self.vr[vb]

    def op_vand(self, emu, w):
        vd, va, vb, _ = vx(w)
        self.vr[vd] = self.vr[va] & self.vr[vb]

    def op_vrsqrtefp(self, emu, w):
        vd, _, vb, _ = vx(w)
        self.set_f(vd, np.float32(1) / np.sqrt(self.f(vb)))

    def op_vmaddfp(self, emu, w):  # D = A*C + B
        vd, va, vb, vc = vx(w)
        self.set_f(vd, self.f(va) * self.f(vc) + self.f(vb))

    def op_vperm(self, emu, w):  # D[i] = (A||B)[C[i] & 31]
        vd, va, vb, vc = vx(w)
        both = np.concatenate([self.vr[va], self.vr[vb]])
        self.vr[vd] = both[self.vr[vc] & 31]

    # VMX128 (VX128 form)
    def op_vmulfp128(self, emu, w):
        vd, va, vb = vx128(w)
        self.set_f(vd, self.f(va) * self.f(vb))

    def op_vaddfp128(self, emu, w):
        vd, va, vb = vx128(w)
        self.set_f(vd, self.f(va) + self.f(vb))

    def op_vmaddfp128(self, emu, w):  # D = A*B + D
        vd, va, vb = vx128(w)
        self.set_f(vd, self.f(va) * self.f(vb) + self.f(vd))

    def op_vmaddcfp128(self, emu, w):  # D = A*D + B
        vd, va, vb = vx128(w)
        self.set_f(vd, self.f(va) * self.f(vd) + self.f(vb))

    def op_vmsum3fp128(self, emu, w):
        vd, va, vb = vx128(w)
        s = np.float32((self.f(va)[:3] * self.f(vb)[:3]).sum(dtype=np.float32))
        self.set_f(vd, [s] * 4)

    def op_vmsum4fp128(self, emu, w):
        vd, va, vb = vx128(w)
        s = np.float32((self.f(va) * self.f(vb)).sum(dtype=np.float32))
        self.set_f(vd, [s] * 4)

    def op_vor128(self, emu, w):
        vd, va, vb = vx128(w)
        self.vr[vd] = self.vr[va] | self.vr[vb]

    def op_vxor128(self, emu, w):
        vd, va, vb = vx128(w)
        self.vr[vd] = self.vr[va] ^ self.vr[vb]

    def op_vsel128(self, emu, w):  # D = (A & ~D) | (B & D)
        vd, va, vb = vx128(w)
        m = self.vr[vd]
        self.vr[vd] = (self.vr[va] & ~m) | (self.vr[vb] & m)

    def op_vpermwi128(self, emu, w):
        vd, vb, p = vx128_p(w)
        src = self.u(vb)
        self.set_u(vd, [src[(p >> 6) & 3], src[(p >> 4) & 3], src[(p >> 2) & 3], src[p & 3]])

    def op_vspltw128(self, emu, w):
        vd, vb, uimm = vx128_3(w)
        self.set_u(vd, np.repeat(self.u(vb)[uimm & 3], 4))

    def op_vupkd3d128(self, emu, w):
        vd, vb, imm = vx128_3(w)
        kind = imm >> 2
        if kind != 4:  # only VPACK_NORMSHORT4 is used by the animation code
            raise RuntimeError(f"vupkd3d128 type {kind} not implemented")
        src = self.u(vb)
        shorts = [(src[2] >> 16) & 0xFFFF, src[2] & 0xFFFF, (src[3] >> 16) & 0xFFFF, src[3] & 0xFFFF]
        out = []
        for s in shorts:
            s = s - 0x10000 if s & 0x8000 else s
            v = (0x40400000 + s) & 0xFFFFFFFF
            out.append(0x7FC00000 if v == 0x403F8000 else v)  # -32768: quiet NaN, like the hardware
        self.set_u(vd, out)


# standard AltiVec loads/stores are primary opcode 31 (X form); extended opcode -> name
X31 = {103: "lvx", 359: "lvxl", 231: "stvx", 487: "stvxl", 71: "lvewx", 199: "stvewx", 6: "lvsl", 38: "lvsr"}


def decode_any(w):
    op = w >> 26
    if op in (4, 5, 6):
        return decode(w)
    if op == 31:
        return X31.get((w >> 1) & 0x3FF)
    return None


def _x31(self, emu, w):
    vd, ra, rb = (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
    return vd, ((0 if ra == 0 else emu.reg(f"r{ra}")) + emu.reg(f"r{rb}")) & 0xFFFFFFFF


def _op_lvx(self, emu, w):
    vd, ea = _x31(self, emu, w)
    self.vr[vd] = np.frombuffer(emu.read(ea & ~15, 16), np.uint8)


def _op_stvx(self, emu, w):
    vs, ea = _x31(self, emu, w)
    emu.uc.mem_write(ea & ~15, self.vr[vs].tobytes())


def _op_lvewx(self, emu, w):
    vd, ea = _x31(self, emu, w)
    ea &= ~3
    self.vr[vd][ea & 15:(ea & 15) + 4] = np.frombuffer(emu.read(ea, 4), np.uint8)


def _op_stvewx(self, emu, w):
    vs, ea = _x31(self, emu, w)
    ea &= ~3
    emu.uc.mem_write(ea, self.vr[vs][ea & 15:(ea & 15) + 4].tobytes())


def _op_lvsl(self, emu, w):
    vd, ea = _x31(self, emu, w)
    self.vr[vd] = np.arange(16, dtype=np.uint8) + (ea & 15)


def _op_lvsr(self, emu, w):
    vd, ea = _x31(self, emu, w)
    self.vr[vd] = np.arange(16, dtype=np.uint8) + (16 - (ea & 15))


VMX.op_lvx = VMX.op_lvxl = _op_lvx
VMX.op_stvx = VMX.op_stvxl = _op_stvx
VMX.op_lvewx, VMX.op_stvewx, VMX.op_lvsl, VMX.op_lvsr = _op_lvewx, _op_stvewx, _op_lvsl, _op_lvsr


def _execute(self, emu, w):
    name = decode_any(w)
    if name is None:
        raise RuntimeError(f"unknown vector instruction {w:08x}")
    with np.errstate(all="ignore"):
        getattr(self, "op_" + name)(emu, w)
    return name


VMX.execute = _execute


def scan(image, base):
    """{address: word} of every vector instruction in the code image."""
    n = len(image) // 4
    words = np.frombuffer(image[:n * 4], ">u4")
    op = words >> 26
    cand = np.flatnonzero((op == 4) | (op == 5) | (op == 6) | (op == 31))
    out = {}
    for i in cand:
        w = int(words[i])
        if decode_any(w):
            out[base + 4 * int(i)] = w
    return out


# unaligned load/store halves (Xbox 360 additions to AltiVec): lvlx loads from EA to the end of its 16-byte
# block into the LEFT of the register; lvrx loads from the block start up to EA into the RIGHT.
X31.update({519: "lvlx", 551: "lvrx", 775: "lvlxl", 807: "lvrxl", 647: "stvlx", 679: "stvrx",
            903: "stvlxl", 935: "stvrxl"})


def _lvlx_at(self, emu, vd, ea):
    eb = ea & 15
    v = np.zeros(16, np.uint8)
    v[:16 - eb] = np.frombuffer(emu.read(ea, 16 - eb), np.uint8)
    self.vr[vd] = v


def _lvrx_at(self, emu, vd, ea):
    eb = ea & 15
    v = np.zeros(16, np.uint8)
    if eb:
        v[16 - eb:] = np.frombuffer(emu.read(ea & ~15, eb), np.uint8)
    self.vr[vd] = v


def _stvlx_at(self, emu, vs, ea):
    eb = ea & 15
    emu.uc.mem_write(ea, self.vr[vs][:16 - eb].tobytes())


def _stvrx_at(self, emu, vs, ea):
    eb = ea & 15
    if eb:
        emu.uc.mem_write(ea & ~15, self.vr[vs][16 - eb:].tobytes())


def _x31_wrap(fn):
    def op(self, emu, w):
        vd, ea = _x31(self, emu, w)
        fn(self, emu, vd, ea)
    return op


def _x128_wrap(fn):
    def op(self, emu, w):
        vd = vx128_1(w)[0]
        _, ra, rb = vx128_1(w)
        ea = ((0 if ra == 0 else emu.reg(f"r{ra}")) + emu.reg(f"r{rb}")) & 0xFFFFFFFF
        fn(self, emu, vd, ea)
    return op


VMX.op_lvlx = VMX.op_lvlxl = _x31_wrap(_lvlx_at)
VMX.op_lvrx = VMX.op_lvrxl = _x31_wrap(_lvrx_at)
VMX.op_stvlx = VMX.op_stvlxl = _x31_wrap(_stvlx_at)
VMX.op_stvrx = VMX.op_stvrxl = _x31_wrap(_stvrx_at)
VMX.op_lvlx128 = VMX.op_lvlxl128 = _x128_wrap(_lvlx_at)
VMX.op_lvrx128 = VMX.op_lvrxl128 = _x128_wrap(_lvrx_at)
VMX.op_stvlx128 = VMX.op_stvlxl128 = _x128_wrap(_stvlx_at)
VMX.op_stvrx128 = VMX.op_stvrxl128 = _x128_wrap(_stvrx_at)
