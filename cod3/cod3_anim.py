"""Decode Call of Duty 3 animations by running the game's own NAL decoder (from the user's default.xex,
unpacked with xex_unpack.py) in the Unicorn emulator (cod3_emu.py).

What the game does, and what is run here:
  * A skeleton (.xeskel, e.g. "soldier") lists channels; each names its codec (NAL_Packed16Entropy-
    Quaternion for bone rotations, NAL_EntropyFloat3 for positions, NAL_EntropyTrajectoryPosition-
    Orientation for root motion, COD_Note for notetracks). The skeleton class's init (0x8216e3d8)
    looks each codec up in a registry (0x821282a8) and lets it lay out its data (codec slot 0x18).
  * An animation file (.xeanim) = header (version 0x10201, skeleton names) + linked anim objects
    (version 0x10301: name, duration, fps, frame count, block table, entropy-coded bitstreams).
    The anim class (vtable 0x82027d84) init (0x82170ce0) turns offsets into pointers.
  * Frames are decoded a block at a time by 0x821710a0 into a cache from a pool (0x82174560).
"""
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cod3_emu import Emu  # noqa: E402

ANIM_VTABLE = 0x82027D84
SKEL_INIT = 0x8216E3D8
ANIM_INIT = 0x82170CE0
DECODE_BLOCK = 0x821710A0
REGISTRY_LOOKUP = 0x821282A8
POOL = 0x82914EE0
# registry: codec name -> function table. Each registration node in the executable (0x825b31e0..) is
# [class info, next, name, link, ...]; the codec object is what the LINK points at (verified: every
# function table is then used exactly once). COD_Note's node is elsewhere (0x825f51c4).
CODECS = {
    "nal_rle8int1": 0x82027E20, "nal_entropyfloat1": 0x82027E90, "nal_signalcounter": 0x82027DB0,
    "nal_packed8float1": 0x82027F00, "nal_float3": 0x82027F70, "nal_entropyfloat3": 0x82027FE0,
    "nal_packed8entropyfloat3": 0x82028050, "nal_packed16entropyfloat3": 0x820280C0,
    "nal_float4": 0x82028130, "nal_entropyfloat4": 0x820281A0, "nal_packed8entropyfloat4": 0x82028210,
    "nal_packed16entropyfloat4": 0x82028280, "nal_packed16entropyiksp": 0x82028670,
    "nal_quaternion": 0x820282F0, "nal_entropyquaternion": 0x82028360,
    "nal_packed8entropyquaterni": 0x820283D0, "nal_packed16entropyquaterni": 0x82028440,
    "nal_positionorientation": 0x820284B0, "nal_entropypositionorienta": 0x82028520,
    "nal_trajectorypositionorien": 0x82028590, "nal_entropytrajectorypositi": 0x82028600,
    "cod_note": 0x8206CFB4,
}


class AnimDecoder:
    def __init__(self, image=None):
        # the memory image xex_unpack.py writes from the title's default.xex
        self.emu = Emu(image or os.environ.get("COD3_IMAGE", "default.img"))
        e = self.emu
        self.blr = e.load(struct.pack(">I", 0x4E800020))
        stub_vt = e.alloc(0x100)
        for i in range(64):
            e.w32(stub_vt + 4 * i, self.blr)
        self.objects = {}
        for name, vt in CODECS.items():
            obj = e.alloc(16)
            e.w32(obj, vt)
            self.objects[name] = obj
        self.stub = e.alloc(16)
        e.w32(self.stub, stub_vt)
        self.used = []
        e.hook(REGISTRY_LOOKUP, self._lookup)
        # the anim cache pool: [pool] = allocator object whose slot 1 allocates
        alloc_fn = e.alloc(16)
        e.uc.mem_write(alloc_fn, struct.pack(">I", 0x4E800020))
        # padded: the codecs store whole 16-byte vectors and may run a little past a block (the console pool pads too)
        e.hook(alloc_fn, lambda em: em.reg("r3", em.alloc(em.reg("r4") + 0x100, 128)))
        alloc_vt = e.alloc(0x40)
        for i in range(16):
            e.w32(alloc_vt + 4 * i, alloc_fn)
        alloc_obj = e.alloc(16)
        e.w32(alloc_obj, alloc_vt)
        for i in range(4):
            e.w32(POOL + 4 * i, 0)
        e.w32(POOL, alloc_obj)

    def _lookup(self, e):
        key = e.read(e.reg("r4"), 32)
        name = key[4:].split(b"\0")[0].decode("latin1").lower()
        obj = next((o for n, o in self.objects.items() if name.startswith(n[:len(name)]) or n.startswith(name)), None)
        self.used.append((name, obj is not None))
        node = e.alloc(0x40)
        e.w32(node + 0x20, obj if obj is not None else self.stub)
        e.reg("r3", node)

    def load_skeleton(self, data, offset):
        """data = the pack holding the .xeskel, offset = the skeleton object (4 bytes before 0x10302)."""
        self.skel = self.emu.load(data[offset:offset + 0x40000])
        self.emu.call(SKEL_INIT, r3=self.skel)
        return self.skel

    def load_anim(self, data, file_offset):
        """file_offset = the anim file header (version 0x10201). Returns the first anim object address."""
        e = self.emu
        size = 0x40000
        f = e.load(data[file_offset:file_offset + size])
        anim = f + struct.unpack_from(">I", data, file_offset + 0x34)[0]
        e.w32(anim, ANIM_VTABLE)
        e.w32(anim + 0x30, self.skel)
        e.call(ANIM_INIT, r3=anim)
        return anim

    def decode_block(self, anim, block, level=0):
        e = self.emu
        ctx = e.alloc(0x40)
        e.w32(ctx + 0x10, anim)
        e.call(DECODE_BLOCK, r3=ctx, r4=block, r5=level)
        return e.u32(e.u32(anim + 0x60) + 4 * block)  # the cache entry for this block


EVAL_FRAME = 0x821714A0  # (ctx, frame, &{_, level, pose}, &{_, _, defaults}, level): block decode + sample


def _ensure(self):
    if not hasattr(self, "_ctx"):
        e = self.emu
        n = e.u32(self.skel + 0x7C)  # pose elements (124 for the soldier)
        self.elements = n
        self._ctx = e.alloc(0x40)
        self._aux = e.alloc(0x40)
        # the sampler writes element i at byte offset table[i] of the pose (the game maps elements onto a
        # character's bones this way); here every element gets its own 32-byte slot (the trajectory
        # element writes two vectors: orientation and position)
        self._pose = e.alloc(n * 32 + 0x100)
        self._defaults = e.alloc(n * 32 + 0x100)
        table = e.alloc(n * 4)
        for i in range(n):
            e.w32(table + 4 * i, 32 * i)
            e.uc.mem_write(self._defaults + 32 * i, struct.pack(">8f", 0, 0, 0, 1, 0, 0, 0, 0))
        e.w32(self._aux + 0x10, table)
        self._a = e.alloc(0x20)
        self._b = e.alloc(0x20)


def evaluate(self, anim, frame, level=0):
    """The pose at a frame: n x 8 floats per element - a quaternion (rotation elements), xyz(w)
    (position elements), or orientation + position (the trajectory, element 0)."""
    _ensure(self)
    e = self.emu
    e.w32(self._ctx + 0x10, anim)
    e.w32(self._ctx + 0x0C, self.skel)
    e.w32(self._ctx + 0x28, self._aux)
    e.w32(self._ctx + 0x2C, 0)
    e.w32(self._a + 8, self._pose)
    e.w32(self._b + 8, self._defaults)
    e.call(EVAL_FRAME, r3=self._ctx, r4=frame, r5=self._a, r6=self._b, r7=level)
    raw = e.read(self._pose, self.elements * 32)
    return [struct.unpack_from(">8f", raw, 32 * i) for i in range(self.elements)]


def anim_info(self, anim):
    e = self.emu
    name = e.read(anim + 0x0C, 28).split(b"\0")[0].decode("latin1")
    return {"name": name, "duration": struct.unpack(">f", e.read(anim + 0x38, 4))[0],
            "fps": struct.unpack(">f", e.read(anim + 0x40, 4))[0], "frames": e.u32(anim + 0x44)}


AnimDecoder.evaluate = evaluate
AnimDecoder.anim_info = anim_info
