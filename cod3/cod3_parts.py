"""Place the rigid parts of multi-part COD3 models (weapon viewmodels: frame, slide, magazine, hammer...).

A model like w_us_pist_VM.xemesh is several mesh objects named "<part>.<model>" (colt, colt_slide,
colt_mag, colt_hammer). Each part's vertices are in its own bone's space; the bone tree and its rest pose
come from an animation skeleton (.xeskel, version 0x10302) in the same archive whose bones carry the part
names. The rest pose is the skeleton's per-element defaults (16 bytes each: quaternion x,y,z,w for an
orientation element, x,y,z for a position element), read after the game's own skeleton init has run in
the emulator (cod3_anim.py). Row-vector chain, as for the soldier: W = R_local @ W_parent,
P = t_local @ W_parent + P_parent; a part vertex v goes to v @ W + P (COD space, inches).
"""
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SKEL_VERSION = struct.pack(">I", 0x00010302)


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bone_names(d, off, limit=0x4000, count=None):
    """Bone names of the skeleton object at `off`, straight from the file (records 0x30 apart, name at +4).
    Returns (record start relative to off, names) for the longest run - or, given the bone count, the
    first run at least that long (a short skeleton must not pick up the next skeleton's longer list) -
    or (None, [])."""
    def name_at(o):
        s = d[o + 4:o + 32].split(b"\0")[0]
        return s.decode() if s and all(32 < c < 127 for c in s) else None
    best = (None, [])
    for o in range(off + 0x60, min(off + limit, len(d) - 0x30), 4):
        if name_at(o) is None or (best[0] is not None and o < off + best[0] + 0x30 * len(best[1])):
            continue
        run, q = [], o
        while q + 0x30 <= len(d) and name_at(q):
            run.append(name_at(q))
            q += 0x30
        if count is not None and len(run) >= count:
            return (o - off, run)
        if len(run) > len(best[1]):
            best = (o - off, run)
    return best if count is None else (None, [])


def find_skeletons(d):
    out = {}
    p = d.find(SKEL_VERSION)
    while p >= 0:
        o = p - 4
        if o >= 0 and o % 4 == 0:
            name = d[o + 0xC:o + 0x28].split(b"\0")[0]
            if name and all(32 <= c < 127 for c in name):
                out[name.decode()] = o
        p = d.find(SKEL_VERSION, p + 1)
    return out


def rest_pose(d, off, frame=None):
    """{bone name: (3x3 world rotation, world position)} of the skeleton at `off`, via the game's init.
    `frame` = (local rotations (bones, 4), local positions (bones, 3)), e.g. a weapon's idle animation
    (extract_anims.py; NaN = not animated): where given, they replace the rest values - the pose the game
    shows the model in, with animated parts (a bazooka's rocket, a shotgun's shell) in place."""
    import cod3_anim as A
    dec = A.AnimDecoder()
    dec.load_skeleton(d, off)
    e, sk = dec.emu, dec.skel
    nb, ne = e.u32(sk + 0x70), e.u32(sk + 0x7C)
    blob = e.read(sk, 0x8000)
    start, names = bone_names(blob, 0, 0x8000 - 0x30, count=nb)
    if start is None or len(names) < nb:
        return {}
    names = names[:nb]
    # each record ends with (int16 parent index, int16 own index); both count bones without a leading
    # "tag_origin" fakeroot (own index -1), so resolve the parent through the own indices
    tails = [struct.unpack_from(">hh", blob, start + 0x30 * k + 0x2C) for k in range(nb)]
    position = {own: k for k, (_, own) in enumerate(tails) if own >= 0}
    parents = [position.get(par, -1) if par >= 0 else -1 for par, _ in tails]
    base, dp = e.u32(sk + 0x80), e.u32(sk + 0x94)
    rot, pos = [np.array([0, 0, 0, 1.0])] * nb, [np.zeros(3)] * nb
    raw = e.read(dp, ne * 16)
    for i in range(ne):
        kind = e.read(base + 0x28 * i + 4, 28).split(b"\0")[0]
        b = e.u32(base + 0x28 * i + 0x20)
        v = np.array(struct.unpack_from(">4f", raw, 16 * i), dtype=np.float64)
        if b >= nb or not np.isfinite(v[:3]).all():
            continue
        if kind == b"orientation" and abs(np.linalg.norm(v) - 1) < 0.01:
            rot[b] = v
        elif kind == b"position":
            pos[b] = v[:3]
    if frame is not None and len(frame[0]) == nb:
        for i in range(nb):
            if np.isfinite(frame[0][i]).all() and abs(np.linalg.norm(frame[0][i]) - 1) < 0.01:
                rot[i] = np.asarray(frame[0][i], dtype=np.float64)
            if np.isfinite(frame[1][i]).all():
                pos[i] = np.asarray(frame[1][i], dtype=np.float64)
    world = {}
    for i in range(nb):
        R, t = qmat(rot[i]), pos[i]
        p = parents[i]
        if 0 <= p < i and names[p] in world:
            PR, PP = world[names[p]]
            world[names[i]] = (R @ PR, t @ PR + PP)
        else:
            world[names[i]] = (R, t)
    return world


SCAN_FILE = "skeleton_scan.json"  # in the models root's parent: every skeleton on the disc (scan_skeletons)


def model_key(name):
    """A model or skeleton name reduced to the vehicle/weapon it is: v_sherman_firefly_d1.xemesh ->
    sherman_firefly, v_sherman_firefly_polish -> sherman_firefly_polish."""
    n = name.lower().rsplit(".", 1)[0]
    n = re.sub(r"^(v_|w_|p_husk_|p_)", "", n)
    return re.sub(r"(_d\d|_ai|_placed|_backseat|_vm|_world)+$", "", n)


def own_skeleton(skel, model):
    """Is this skeleton the model's own? The same vehicle, or a variant of it (firefly / firefly_polish)."""
    s, m = model_key(skel), model_key(model)
    return bool(s and m) and (s == m or s.startswith(m + "_") or m.startswith(s + "_"))


def scan_skeletons(models_root):
    """Every animation skeleton in every archive: {name: [[pack path relative to the root, offset, bones]]}."""
    out = {}
    for packs in sorted(Path(models_root).rglob("packs")):
        for p in sorted(packs.glob("*.bin")):
            d = p.read_bytes()
            offs = sorted(find_skeletons(d).items(), key=lambda kv: kv[1])
            for k, (name, off) in enumerate(offs):
                end = offs[k + 1][1] if k + 1 < len(offs) else len(d)
                _, names = bone_names(d, off, min(0x4000, end - off))
                if names:
                    out.setdefault(name, []).append([str(p.relative_to(models_root)).replace("\\", "/"), off, names])
    return out


class ArchiveSkeletons:
    """Skeletons of an archive's packs, by bone names, plus (given the disc-wide scan) every other archive's
    skeletons for models whose own skeleton lives elsewhere; rest poses are computed on first use."""

    def __init__(self, packs, models_root=None):
        self.entries = []  # (skeleton name, bone name set, pack bytes or pack path, offset)
        for d in packs:
            offs = sorted(find_skeletons(d).items(), key=lambda kv: kv[1])
            for k, (name, off) in enumerate(offs):
                # stop at the next skeleton: skeletons sit back to back and a scan could read its neighbour's bones
                end = offs[k + 1][1] if k + 1 < len(offs) else len(d)
                _, names = bone_names(d, off, min(0x4000, end - off))
                if names:
                    self.entries.append((name, set(n.lower() for n in names), d, off))
        self.others = []
        scan = Path(models_root) / SCAN_FILE if models_root else None
        if scan and scan.exists():
            for name, copies in json.loads(scan.read_text()).items():
                rel, off, names = copies[0]
                self.others.append((name, set(n.lower() for n in names), Path(models_root) / rel, off))
        self._poses = {}
        self._idle = {}
        self.posed_by_idle = False  # whether the last placement() used an idle animation frame
        # decoded weapon animations (extract_anims.py <root> <models root's parent>/anims_weapons <weapon archive>)
        self.anims = Path(models_root).parent / "anims_weapons" if models_root else None

    def idle_frame(self, skeleton):
        """First frame of the skeleton's idle animation (not the aim-down-sights one), if decoded."""
        if skeleton not in self._idle:
            folder = self.anims / skeleton if self.anims else None
            idle = sorted(p for p in folder.glob("*idle*.npz") if "ads" not in p.stem.lower())                 if folder is not None and folder.is_dir() else []
            z = np.load(idle[0]) if idle else None
            self._idle[skeleton] = (z["rot"][0], z["pos"][0]) if z is not None else None
        return self._idle[skeleton]

    def placement(self, part_names, model=""):
        """{part name: (R, t)} from the model's own skeleton (named after it, in this archive or any other),
        or else from the skeleton naming the most of its specific parts (weapon parts like colt_mag; at
        least half, two or more). Generic vehicle tags (tag_body, tag_wheel_front_left) never pick a
        skeleton on their own: every vehicle has them, and another vehicle's bones scatter the parts."""
        want = [n.lower() for n in part_names if n]
        if len(want) < 2:
            return {}
        own = [e for e in self.entries + self.others if model and own_skeleton(e[0], model)]
        if own:
            # the closest name first (firefly before firefly_polish), then the one naming the most parts
            name, bones, d, off = max(own, key=lambda e: (model_key(e[0]) == model_key(model),
                                                         sum(1 for n in want if n in e[1])))
        else:
            specific = [n for n in want if not n.startswith("tag_")]
            if len(specific) < 2 or not self.entries:
                return {}
            name, bones, d, off = max(self.entries, key=lambda e: sum(1 for n in specific if n in e[1]))
            hits = sum(1 for n in specific if n in bones)
            if hits < 2 or hits * 2 < len(specific):
                return {}
        self.posed_by_idle = self.idle_frame(name) is not None
        key = (str(d) if isinstance(d, Path) else id(d), off)
        if key not in self._poses:
            try:
                self._poses[key] = {k.lower(): v for k, v in
                                    rest_pose(d.read_bytes() if isinstance(d, Path) else d, off,
                                              self.idle_frame(name)).items()}
            except Exception:
                self._poses[key] = {}
        pose = self._poses[key]
        out = {n: pose[n] for n in want if n in pose}
        # wheel bones too: a vehicle draws its one wheel model at every tag_wheel_* bone
        out.update({n: v for n, v in pose.items() if n.startswith("tag_wheel") and n not in out})
        return out


if __name__ == "__main__":  # python -I cod3_parts.py <models root>: writes <root>/skeleton_scan.json
    root = Path(sys.argv[1])
    found = scan_skeletons(root)
    (root / SCAN_FILE).write_text(json.dumps(found))
    print(f"{len(found)} skeletons -> {root / SCAN_FILE}")
