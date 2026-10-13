"""Export every Call of Duty 3 animation, decoded by the game's own code (cod3_anim.py) - bone by bone.

  python -I cod3/extract_anims.py <extracted root> <out folder> [anim archive ...]
Defaults to sp/anim and mp/anim. Writes <out>/<skeleton>/<anim>.npz with
  rot   (frames, bones, 4)  local rotation quaternions (x, y, z, w) per skeleton bone (NaN = not animated)
  pos   (frames, bones, 3)  local positions (NaN where the animation doesn't move that bone)
  root  (frames, 8)         trajectory: orientation quaternion + position (root motion)
  fps, duration
and <out>/<skeleton>/index.json (bone names and parents, every animation with frame count and length).
Animation files: an .xeanim is a file header (version 0x10201, skeleton names) followed by linked anim
objects; each is decoded block by block and sampled per frame at full detail (level 0).
"""
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cod3_anim as A  # noqa: E402

ANIM_FILE = struct.pack(">I", 0x00010201)
SKEL_VERSION = struct.pack(">I", 0x00010302)


def find_skeletons(data):
    """{name: object offset} of every skeleton object (version 0x10302) in a pack."""
    out = {}
    for m in re.finditer(re.escape(SKEL_VERSION), data):
        o = m.start() - 4
        if o >= 0 and o % 4 == 0:
            name = data[o + 0xC:o + 0x28].split(b"\0")[0]
            if name and all(32 <= c < 127 for c in name):
                out[name.decode()] = o
    return out


def anim_files(data):
    """Offsets of animation file headers (version 0x10201) whose first anim offset is sane."""
    out = []
    for m in re.finditer(re.escape(ANIM_FILE), data):
        o = m.start()
        if o % 4 == 0 and o + 0x40 <= len(data):
            first = struct.unpack_from(">I", data, o + 0x34)[0]
            nskel = struct.unpack_from(">I", data, o + 0xC)[0]
            if 0 < first < 0x10000 and 1 <= nskel <= 8:
                out.append(o)
    return out


def skeleton_info(dec):
    e = dec.emu
    sk = dec.skel
    nb = e.u32(sk + 0x70)
    bones, parents = [], []
    for i in range(nb):
        r = sk + 0x5F0 + 0x30 * i
        bones.append(e.read(r + 4, 28).split(b"\0")[0].decode("latin1"))
        v = struct.unpack(">h", e.read(r + 0x2C, 2))[0]  # parent index counted without "fakeroot"
        parents.append(v + 1 if v >= 0 else -1)
    elems = []
    base = e.u32(sk + 0x80)
    for i in range(e.u32(sk + 0x7C)):
        r = base + 0x28 * i
        elems.append((e.read(r + 4, 28).split(b"\0")[0].decode("latin1"), e.u32(r + 0x20)))
    return bones, parents, elems


def export_anim(dec, anim, bones, elems):
    info = dec.anim_info(anim)
    n = info["frames"]
    nb = len(bones)
    rot = np.full((n, nb, 4), np.nan, np.float32)
    pos = np.full((n, nb, 3), np.nan, np.float32)
    root = np.zeros((n, 8), np.float32)
    for f in range(n):
        pose = dec.evaluate(anim, f)
        for i, (kind, b) in enumerate(elems):
            v = pose[i]
            if kind == "orientation":
                rot[f, b] = v[:4]
            elif kind == "position":
                pos[f, b] = v[:3]
            elif kind == "trajectory":
                root[f] = v
    return info, rot, pos, root


def main():
    root_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    archives = sys.argv[3:] or ["sp/anim", "mp/anim"]
    glob = (root_dir / "sp/global/packs/section0.bin").read_bytes()
    skel_at = find_skeletons(glob)
    print("skeletons:", {k: hex(v) for k, v in skel_at.items()}, flush=True)
    t0 = time.time()
    for arch in archives:
        packs = {pack: pack.read_bytes() for pack in sorted((root_dir / arch / "packs").glob("*.bin"))}
        # skeletons: the shared ones in sp/global, else the archive's own (weapons carry theirs: 30cal, k98 ...)
        local = {name: (data, o) for data in packs.values() for name, o in find_skeletons(data).items()}
        for pack, data in packs.items():
            for fo in anim_files(data):
                skel_name = data[fo + 0x4C:fo + 0x68].split(b"\0")[0].decode("latin1")
                if skel_name in skel_at:
                    skel_data, skel_off = glob, skel_at[skel_name]
                elif skel_name in local:
                    skel_data, skel_off = local[skel_name]
                else:
                    print(f"  {arch}/{pack.name} @{fo:x}: skeleton {skel_name!r} not found", flush=True)
                    continue
                dec = A.AnimDecoder()  # fresh emulator per file keeps memory small
                dec.load_skeleton(skel_data, skel_off)
                bones, parents, elems = skeleton_info(dec)
                folder = out / skel_name
                folder.mkdir(parents=True, exist_ok=True)
                idx_path = folder / "index.json"
                index = json.loads(idx_path.read_text()) if idx_path.exists() else \
                    {"skeleton": skel_name, "bones": bones, "parents": parents, "anims": {}}
                try:
                    anim = dec.load_anim(data, fo)
                    info, rot, pos, rt = export_anim(dec, anim, bones, elems)
                except Exception as ex:
                    print(f"  {arch}/{pack.name} @{fo:x}: FAILED {ex}", flush=True)
                    continue
                name = re.sub(r"[^A-Za-z0-9_.-]", "_", info["name"])
                np.savez_compressed(folder / f"{name}.npz", rot=rot, pos=pos, root=rt,
                                    fps=info["fps"], duration=info["duration"])
                index["anims"][name] = {"frames": info["frames"], "fps": info["fps"],
                                        "duration": round(info["duration"], 4), "source": f"{arch}/{pack.name}"}
                idx_path.write_text(json.dumps(index, indent=1))
                print(f"{skel_name}/{name}: {info['frames']} frames ({time.time() - t0:.0f} s)", flush=True)
    print(f"Done ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
