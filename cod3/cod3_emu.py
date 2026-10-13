"""Run pieces of Call of Duty 3's own code (the user's default.xex, unpacked by xex_unpack.py) in the
Unicorn PowerPC emulator - used to decode animations exactly as the game does.

  emu = Emu("default.img")  # xex_unpack.py output
  addr = emu.load(data)            # copy bytes into emulated memory, returns their address
  emu.hook(0x821282a8, fn)         # replace a game function: fn(emu) sets r3 etc.
  emu.call(0x8216e3d8, r3=addr)    # call a function with arguments, returns r3
"""
import struct

from unicorn import Uc, UC_ARCH_PPC, UC_MODE_PPC32, UC_MODE_BIG_ENDIAN, UC_HOOK_CODE, UC_HOOK_INTR, UcError
from unicorn.ppc_const import UC_PPC_REG_PC, UC_PPC_REG_LR, UC_PPC_REG_MSR

BASE = 0x82000000
HEAP = 0x40000000
STACK = 0x70000000
RETURN_TRAP = 0x7FFF0000  # calls return here; emulation stops
REG = {f"r{i}": getattr(__import__("unicorn.ppc_const", fromlist=["x"]), f"UC_PPC_REG_{i}") for i in range(32)}
FREG = {f"f{i}": getattr(__import__("unicorn.ppc_const", fromlist=["x"]), f"UC_PPC_REG_FPR{i}") for i in range(32)}


class Emu:
    def __init__(self, image_path, heap_size=0x08000000):
        img = open(image_path, "rb").read()
        # The Xenon is a 64-bit PowerPC run with 32-bit addresses. Unicorn's 64-bit PowerPC mode doesn't
        # run at all, so this uses 32-bit mode and performs the 64-bit instructions the game's code uses
        # (mostly std/ld when saving registers) in Python when they trap (_on_trap).
        self.uc = Uc(UC_ARCH_PPC, UC_MODE_PPC32 | UC_MODE_BIG_ENDIAN)
        self.uc.reg_write(UC_PPC_REG_MSR, self.uc.reg_read(UC_PPC_REG_MSR) | 0x2000)  # MSR[FP]: enable the FPU
        size = (len(img) + 0xFFFF) & ~0xFFFF
        self.uc.mem_map(BASE, size)
        self.uc.mem_write(BASE, img)
        self.uc.mem_map(HEAP, heap_size)
        self.uc.mem_map(STACK - 0x100000, 0x200000)
        self.uc.mem_map(RETURN_TRAP, 0x1000)
        self.uc.mem_write(RETURN_TRAP, struct.pack(">I", 0x4E800020) * 0x400)  # blr (never reached)
        self.heap = HEAP
        self.hooks = {}
        self.trace = None
        self.uc.hook_add(UC_HOOK_CODE, self._on_code)
        self.uc.hook_add(UC_HOOK_INTR, self._on_trap)
        self.hi = {}  # register -> (low 32 bits, high 32 bits) left by the last 64-bit load/op (see _v64)
        from vmx128 import VMX, scan
        self.vmx = VMX()  # vector registers; VMX128/AltiVec instructions are performed in Python
        self.vector_at = scan(img, BASE)  # every vector instruction, run by _on_code instead of Unicorn

    # memory
    def alloc(self, n, align=16):
        self.heap = (self.heap + align - 1) & ~(align - 1)
        a = self.heap
        self.heap += n
        return a

    def load(self, data, align=16):
        a = self.alloc(len(data), align)
        self.uc.mem_write(a, bytes(data))
        return a

    def read(self, a, n):
        return bytes(self.uc.mem_read(a, n))

    def u32(self, a):
        return struct.unpack(">I", self.read(a, 4))[0]

    def w32(self, a, v):
        self.uc.mem_write(a, struct.pack(">I", v & 0xFFFFFFFF))

    def reg(self, name, value=None):
        r = REG.get(name) or FREG.get(name)
        if value is None:
            return self.uc.reg_read(r)
        self.uc.reg_write(r, value)

    # hooks: replace a function; fn(emu) runs instead and the call returns to LR
    def hook(self, addr, fn):
        self.hooks[addr] = fn

    def _on_code(self, uc, addr, size, _):
        if self.trace:
            self.trace(self, addr)
        w = self.vector_at.get(addr)
        if w is not None:
            self.vmx.execute(self, w)
            uc.reg_write(UC_PPC_REG_PC, addr + 4)
            return
        fn = self.hooks.get(addr)
        if fn is not None:
            fn(self)
            uc.reg_write(UC_PPC_REG_PC, uc.reg_read(UC_PPC_REG_LR))
        elif addr == RETURN_TRAP:
            uc.emu_stop()

    def _on_trap(self, uc, intno, _):
        """A program exception: perform the 64-bit instruction (at PC-4) in Python and carry on."""
        pc = uc.reg_read(UC_PPC_REG_PC) - 4
        w = self.u32(pc)
        op, rs, ra, d = w >> 26, (w >> 21) & 31, (w >> 16) & 31, w & 0xFFFF
        sd = struct.unpack(">h", struct.pack(">H", d & 0xFFFC))[0]
        base = 0 if ra == 0 else self.reg(f"r{ra}")
        if op in (4, 5, 6):  # VMX128 / AltiVec
            self.vmx.execute(self, w)
        elif op == 62 and w & 3 in (0, 1):  # std / stdu
            ea = (base + sd) & 0xFFFFFFFF
            self.uc.mem_write(ea, struct.pack(">q", self._v64(rs)))
            if w & 3 == 1:
                self.reg(f"r{ra}", ea)
        elif op == 58 and w & 3 in (0, 1, 2):  # ld / ldu / lwa
            ea = (base + sd) & 0xFFFFFFFF
            if w & 3 == 2:
                v = struct.unpack(">i", self.read(ea, 4))[0] & 0xFFFFFFFF
                hi = 0xFFFFFFFF if v & 0x80000000 else 0
            else:
                hi, v = struct.unpack(">II", self.read(ea, 8))
                if w & 3 == 1:
                    self.reg(f"r{ra}", ea)
            self.reg(f"r{rs}", v)
            self.hi[rs] = (v, hi)
        elif op == 31 and (w >> 1) & 0x3FF in (149, 181, 21, 53):  # stdx stdux ldx ldux
            xo = (w >> 1) & 0x3FF
            rb = (w >> 11) & 31
            ea = (base + self.reg(f"r{rb}")) & 0xFFFFFFFF
            if xo in (149, 181):
                self.uc.mem_write(ea, struct.pack(">q", self._v64(rs)))
            else:
                hi, v = struct.unpack(">II", self.read(ea, 8))
                self.reg(f"r{rs}", v)
                self.hi[rs] = (v, hi)
            if xo in (181, 53):
                self.reg(f"r{ra}", ea)
        elif not self._alu64(w):
            raise RuntimeError(f"unsupported instruction {w:08x} (opcode {op}) at {pc:08x}")
        uc.reg_write(UC_PPC_REG_PC, pc + 4)

    # 64-bit integer/FP instructions the compiler uses on 32-bit values. A register's 64-bit value is
    # taken as its 32-bit value sign-extended (how 32-bit code on the Xenon keeps them).
    def _v64(self, r):
        """A register's 64-bit value: the full value a 64-bit load/op left there if its low half is unchanged
        since (Unicorn only keeps 32 bits), otherwise the 32-bit value sign-extended."""
        v = self.reg(f"r{r}") & 0xFFFFFFFF
        known = self.hi.get(r)
        if known is not None and known[0] == v:
            full = (known[1] << 32) | v
            return full - (1 << 64) if full >> 63 else full
        return v - (1 << 32) if v & 0x80000000 else v

    def _set(self, r, v, rc=False):
        v &= (1 << 64) - 1
        self.reg(f"r{r}", v & 0xFFFFFFFF)
        self.hi[r] = (v & 0xFFFFFFFF, v >> 32)
        if rc:
            sv = v - (1 << 64) if v >> 63 else v
            self._cr0(sv)

    def _cr0(self, sv):
        from unicorn.ppc_const import UC_PPC_REG_CR
        cr = self.uc.reg_read(UC_PPC_REG_CR)
        f = 0b1000 if sv < 0 else 0b0100 if sv > 0 else 0b0010
        self.uc.reg_write(UC_PPC_REG_CR, (cr & 0x0FFFFFFF) | (f << 28))

    def _alu64(self, w):
        op, rs, ra, rb = w >> 26, (w >> 21) & 31, (w >> 16) & 31, (w >> 11) & 31
        M64 = (1 << 64) - 1

        def mask(mb, me):
            if mb <= me:
                return ((1 << (64 - mb)) - 1) & ~((1 << (63 - me)) - 1) & M64
            return (((1 << (64 - mb)) - 1) | ~((1 << (63 - me)) - 1)) & M64

        def rotl(v, n):
            v &= M64
            return ((v << n) | (v >> (64 - n))) & M64 if n else v

        if op == 30:  # rldicl / rldicr / rldic / rldimi (MD form) and rldcl/rldcr (MDS)
            sh = ((w >> 11) & 31) | (((w >> 1) & 1) << 5)
            mbe = ((w >> 6) & 31) | (((w >> 5) & 1) << 5)
            kind = (w >> 2) & 7
            v = self._v64(rs)
            if kind == 0:
                r = rotl(v, sh) & mask(mbe, 63)
            elif kind == 1:
                r = rotl(v, sh) & mask(0, mbe)
            elif kind == 2:
                r = rotl(v, sh) & mask(mbe, 63 - sh)
            elif kind == 3:
                m = mask(mbe, 63 - sh)
                r = (rotl(v, sh) & m) | (self._v64(ra) & M64 & ~m)
            elif kind == 4:  # rldcl / rldcr: shift from rB
                n = self.reg(f"r{rb}") & 63
                mb = ((w >> 6) & 31) | (((w >> 5) & 1) << 5)
                r = rotl(v, n) & (mask(mb, 63) if not (w >> 1) & 1 else mask(0, mb))
            else:
                return False
            self._set(ra, r, w & 1)
            return True
        if op == 31:
            xo = (w >> 1) & 0x3FF
            if xo == 986:  # extsw
                self._set(ra, self._v64(rs), w & 1)
            elif xo in (413, 414):  # sradi (XS form: sh from bits)
                sh = rb | (((w >> 1) & 1) << 5)
                self._set(ra, self._v64(rs) >> sh, w & 1)
            elif xo == 794:  # srad
                n = self.reg(f"r{rb}") & 127
                self._set(ra, self._v64(rs) >> min(n, 63), w & 1)
            elif xo == 27:  # sld
                n = self.reg(f"r{rb}") & 127
                self._set(ra, 0 if n > 63 else self._v64(rs) << n, w & 1)
            elif xo == 539:  # srd
                n = self.reg(f"r{rb}") & 127
                self._set(ra, 0 if n > 63 else (self._v64(rs) & M64) >> n, w & 1)
            elif xo == 58:  # cntlzd
                v = self._v64(rs) & M64
                self._set(ra, 64 - v.bit_length(), w & 1)
            elif xo in (233, 745):  # mulld
                self._set(rs, self._v64(ra) * self._v64(rb), w & 1)
            elif xo in (489, 1001):  # divd
                b = self._v64(rb)
                self._set(rs, 0 if b == 0 else int(self._v64(ra) / b), w & 1)
            elif xo in (457, 969):  # divdu
                b = self._v64(rb) & M64
                self._set(rs, 0 if b == 0 else (self._v64(ra) & M64) // b, w & 1)
            elif xo == 73:  # mulhd
                self._set(rs, (self._v64(ra) * self._v64(rb)) >> 64, w & 1)
            elif xo == 9:  # mulhdu
                self._set(rs, ((self._v64(ra) & M64) * (self._v64(rb) & M64)) >> 64, w & 1)
            else:
                return False
            return True
        if op == 63:  # fctid / fctidz / fcfid (FPR holds the 64-bit integer)
            xo = (w >> 1) & 0x3FF
            fd, fb = rs, rb
            import math
            bits = self.uc.reg_read(FREG[f"f{fb}"])
            val = struct.unpack(">d", struct.pack(">Q", bits & M64))[0]
            if xo in (814, 815):
                if math.isnan(val):
                    iv = -(1 << 63)
                else:
                    iv = int(val) if xo == 815 else int(round(val))
                    iv = max(-(1 << 63), min((1 << 63) - 1, iv))
                self.uc.reg_write(FREG[f"f{fd}"], iv & M64)
            elif xo == 846:
                iv = bits & M64
                iv = iv - (1 << 64) if iv >> 63 else iv
                self.uc.reg_write(FREG[f"f{fd}"], struct.unpack(">Q", struct.pack(">d", float(iv)))[0])
            else:
                return False
            return True
        return False

    def call(self, addr, max_instr=50_000_000, **regs):
        for k, v in regs.items():
            self.reg(k, v)
        self.reg("r1", STACK)
        self.uc.reg_write(UC_PPC_REG_LR, RETURN_TRAP)
        try:
            self.uc.emu_start(addr, RETURN_TRAP, count=max_instr)
        except UcError as e:
            pc = self.uc.reg_read(UC_PPC_REG_PC)
            raise RuntimeError(f"emulation fault at {pc:08x}: {e}") from None
        return self.reg("r3")
