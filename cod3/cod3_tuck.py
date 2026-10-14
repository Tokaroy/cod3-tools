"""Tuck a COD3 character's gear in: the accessory bones' places in a standing pose.

COD3's human rig (the same 92 bones in the same order for every soldier and hero) hangs pouches, canteens,
shovels, grenades, backpacks and so on from accessory bones ("bip01 laccside", "bip01 raccsidefront",
"bip01 chestacc", "bip01 backpack", ...). In the bind pose these sit spread out from the body - the left side
pouch 41 cm to the side, the chest pouches up to 47 cm forward - and every animation pulls them in against
the body. A bind-pose export therefore shows the gear floating where the arms hang. tuck() moves just the
accessory bones to where the game's standing idle puts them (each relative to its parent at bind), so the
T-posed model wears its gear the way the game shows it; every other bone stays at bind.

  from cod3_tuck import Tucker
  t = Tucker(models_root, anims_root)        # anims_root: extract_anims.py output (soldier/<clip>.npz)
  pos = t.tuck(pos, palette, bones, weights)  # COD space (inches), the mesh's own palette/bone bytes/weights
"""
import json
from pathlib import Path

import numpy as np

import cod3_xemesh as X

ARCHIVE = "sp/saint_lo/chars"
SKELETON = "skeleton_c_pnzgren_ng_body1"
CLIP, FRAME = "c_idle_stand_lookaround2", 0
RIG_BONES = 92


def _qmat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def is_accessory(name):
    return "acc" in name or "backpack" in name


class Tucker:
    def __init__(self, models_root, anims_root):
        models_root, anims_root = Path(models_root), Path(anims_root)
        bones = self._skeleton(models_root)
        self.names = [b["name"] for b in bones]
        parents = [b["parent"] for b in bones]
        world = []  # bind: column-convention rotation and position (inches), from the stored inverse bind
        for b in bones:
            inv = np.eye(4)
            inv[:3, :3], inv[3, :3] = b["m2"][:3, :3], b["m2"][3, :3]
            W = np.linalg.inv(inv)
            world.append((W[:3, :3].T, W[3, :3]))
        index = json.loads((anims_root / "soldier" / "index.json").read_text())
        z = np.load(anims_root / "soldier" / f"{CLIP}.npz")
        rot, pos = z["rot"][FRAME], z["pos"][FRAME]
        anim = index["bones"]
        # one 3x4 skin matrix per bone: identity, or bind -> tucked for an accessory bone
        self.skin = np.tile(np.eye(4)[:3], (len(bones), 1, 1))
        self.tucked = []
        for i, n in enumerate(self.names):
            if not is_accessory(n) or parents[i] < 0 or n not in anim:
                continue
            a = anim.index(n)
            if np.isnan(rot[a, 0]) and np.isnan(pos[a, 0]):
                continue
            PR, Pp = world[parents[i]]
            bR, bp = world[i]
            Rl = _qmat(rot[a]).T if not np.isnan(rot[a, 0]) else PR.T @ bR
            tl = pos[a] if not np.isnan(pos[a, 0]) else PR.T @ (bp - Pp)
            R, t = PR @ Rl, Pp + PR @ tl  # tucked world (parent at bind)
            A = R @ bR.T
            self.skin[i] = np.c_[A, t - A @ bp]
            self.tucked.append(n)

    @staticmethod
    def _skeleton(models_root):
        a = models_root / ARCHIVE
        man = json.loads((a / "manifest.json").read_text())
        for p in sorted((a / "packs").glob("*.bin")):
            d = p.read_bytes()
            data0 = X.find_data0(d)
            if data0 is None:
                continue
            sks = X.skeletons(d, data0)
            if SKELETON in sks:
                return sks[SKELETON]
        raise SystemExit(f"cod3_tuck: {SKELETON} not found in {a} ({len(man['files'])} files)")

    def tuck(self, pos, palette, bones, weights):
        """Skinned positions with the accessory bones tucked in (unchanged if none of them is used)."""
        if palette is None or bones is None or not len(palette) or max(palette) >= RIG_BONES:
            return pos
        pal = np.asarray(palette, np.int64)
        bi = pal[np.clip(bones.astype(np.int64), 0, len(pal) - 1)]
        w = weights.astype(np.float64) if weights is not None else np.tile([255.0, 0, 0, 0], (len(pos), 1))
        w /= np.maximum(w.sum(1, keepdims=True), 1e-6)
        S = np.einsum("nj,njab->nab", w, self.skin[bi])  # blended 3x4 per vertex
        if np.allclose(S, np.eye(4)[:3]):
            return pos
        return np.einsum("nab,nb->na", S[:, :, :3], pos) + S[:, :, 3]
