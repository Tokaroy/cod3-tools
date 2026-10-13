"""Check every decoded texture against the UVs of the models that use it, and re-decode the ones that fit
better a whole 64 KB step away.

  python -I cod3/uv_refit.py <models root> [archive subpath ...] [--shard=K/N] [--dry]
A pack's GPU data start is chosen by image statistics (extract_textures.exact_layout); when it lands one
64 KB step off, every texture of the pack is decoded from the wrong spot. The pieces still look like the
right kind of image (a uniform sheet shows other parts of the same uniform), so the noise tests can't see
it - but the models' UVs then fall on black between the pieces. For each texture some model in the
archive is linked to, this decodes it at its offset plus k * 64 KB (k = -4..4) and measures how much of
those surfaces' UV area lands on painted texels, less the sheet's painted share (chance), skipping decodes
that are noise. The pack's best shift is applied to its textures when at
least two of them clearly fit better there (single textures that "fit" elsewhere usually land on a
neighbouring texture, so they're left alone).
Rewrites textures_named/<name>.png and writes <archive>/uv_refit.json (texture -> shift, fit before/after).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cod3_textures as T  # noqa: E402
import cod3_xemesh as X  # noqa: E402
import extract_textures as E  # noqa: E402
import link_textures as L  # noqa: E402
import repair_texture as RT  # noqa: E402

RES = 128
STEP = 0x10000
SHIFTS = range(-4, 5)


def painted(img):
    return np.asarray(Image.fromarray(img[:, :, :3]).resize((RES, RES), Image.BILINEAR)).astype(int).sum(2) > 45


def uv_evidence(archive: Path):
    """texture name -> [UV masks of the surfaces linked to it] for this archive's models."""
    L.RES = RES
    models = archive / "models"
    if not (models / "models.json").exists() or not (archive / "links.json").exists():
        return {}
    links = json.loads((archive / "links.json").read_text())
    out = defaultdict(list)
    for m in json.loads((models / "models.json").read_text()):
        chosen = links.get(m["name"], [])
        if not any(c.get("texture") for c in chosen):
            continue
        vt, groups = L.obj_groups(models / m["file"])
        for k, c in enumerate(chosen):
            if c.get("texture") and k in groups and len(vt) and len(out[c["texture"]]) < 6:
                um = L.uv_mask(vt, groups[k])
                if um.sum() >= 20:
                    out[c["texture"]].append(um)
    return out


def fit(d, at, fc, masks):
    try:
        a = T.decode(d, at, fc)
    except Exception:
        return -1.0, None
    if a.size == 0 or float(a[:, :, :3].std()) < 3:
        return -1.0, None
    p = painted(a)
    if RT.damage_mask(a).mean() > 0.05:
        return -1.0, a  # decoding noise or stripes: not the texture
    # how much better than chance the UVs land on paint: a sheet painted edge to edge "covers" any UVs,
    # so a wrong decode of a full sheet (a wall, noise) must not beat the right decode of an atlas
    cover = float(np.mean([(um & p).sum() / um.sum() for um in masks]))
    return cover - float(p.mean()), a


def refit_archive(archive: Path, dry=False):
    evidence = uv_evidence(archive)
    report = {}
    if not evidence:
        return report
    listing = json.loads((archive / "textures.json").read_text()) if (archive / "textures.json").exists() else []
    files = {(t["pack"], t["name"].lower()): t["file"] for t in listing}
    for path in sorted((archive / "packs").glob("*.bin")):
        d = path.read_bytes()
        recs = T.records(d)
        if not recs:
            continue
        data0 = X.find_data0(d) or X.data0_from_textures(d, [p for p, _ in recs])
        names = X.texture_names(d, data0, [p for p, _ in recs])
        exact = E.exact_layout(d, recs, data0)
        if exact is None:
            continue
        todo = [(i, str(n).lower()) for i, n in enumerate(names) if n and str(n).lower() in evidence
                and (path.stem, str(n).lower()) in files]
        if not todo:
            continue
        table = {}  # (index, shift) -> fit
        for i, n in todo:
            for s in SHIFTS:
                table[i, s] = fit(d, exact[i] + s * STEP, recs[i][1], evidence[n])[0]
        # one shift for the whole pack (its data start was misjudged), backed by at least two textures:
        # a single texture "fitting better" elsewhere usually lands on a neighbouring texture instead
        pack_best = max(SHIFTS, key=lambda s: sum(table[i, s] for i, _ in todo))
        better = sum(1 for i, _ in todo if table[i, pack_best] > table[i, 0] + 0.03)
        gain = sum(table[i, pack_best] - table[i, 0] for i, _ in todo) / len(todo)
        if pack_best == 0 or better < 2 or gain <= 0.03:
            continue
        for i, n in todo:
            s = pack_best
            if table[i, s] <= table[i, 0] + 0.02:
                continue
            f, img = fit(d, exact[i] + s * STEP, recs[i][1], evidence[n])
            report[n] = {"pack": path.stem, "shift": s, "before": round(table[i, 0], 3), "after": round(f, 3)}
            if not dry and img is not None:
                Image.fromarray(img, "RGBA").save(archive / "textures_named" / files[(path.stem, n)])
    if not dry:
        (archive / "uv_refit.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    dry = "--dry" in sys.argv
    shard = next((a for a in sys.argv[1:] if a.startswith("--shard=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = Path(args[0])
    targets = [root / a for a in args[1:]] or sorted(p.parent for p in root.rglob("textures.json"))
    if shard:
        k, n = map(int, shard.split("=")[1].split("/"))
        targets = targets[k::n]
    total = 0
    for a in targets:
        try:
            rep = refit_archive(a, dry)
        except Exception as e:
            print(f"{a.relative_to(root)}: FAILED {e!r}", flush=True)
            continue
        total += len(rep)
        for n, r in rep.items():
            print(f"  {a.relative_to(root)} {n}: shift {r['shift']:+d} x 64 KB, fit {r['before']} -> {r['after']}", flush=True)
        print(f"{a.relative_to(root)}: {len(rep)} textures re-placed", flush=True)
    print(f"Done: {total} textures re-placed", flush=True)
