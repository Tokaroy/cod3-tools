"""Re-place character uniform variant textures (camo, dirty, island versions) by matching their plain sibling.

  python -I cod3/fix_variant_textures.py <models root> [archive ...]
A character's variants (c_dixon_body4_c, c_huxley_body3_c, ...) share the plain uniform's UV layout. When a
pack's data start is ambiguous, a variant can be decoded from a slightly wrong spot: big bands of the atlas
move, every band still looks clean, and the model shows black patches where its UVs miss the moved pieces.
For each variant, the pixel data is searched +-1 MB around its spot (4 KB steps) for the decode whose
layout best matches the plain texture (c_<who>_body_c) from the texture index (grey 64x64 correlation;
near-ties go to the smallest move). It is replaced only if the
match is clearly better than the exported PNG (0.6-0.99 and >= 0.3 above it; ~1.0 means it found the
plain texture's own pixels); the PNG in textures_named is rewritten
and listed in <root>/variant_fixes.json. Run link_textures/catalog_models afterwards.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cod3_textures as T  # noqa: E402
import cod3_xemesh as X  # noqa: E402
import extract_textures as E  # noqa: E402

VARIANT = re.compile(r"^((?:of_)?c_[a-z0-9]+(?:_[a-z])?_(?:body|head))(\d*)_c$")


def small(rgb):
    g = np.asarray(Image.fromarray(rgb[..., :3]).convert("L").resize((64, 64), Image.BILINEAR)).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def siblings(name, index):
    m = VARIANT.match(name)
    if not m:
        return []
    stem, base = m.group(1), m.group(1) + "_c"
    if m.group(2) == "" or base not in index:
        return []  # the plain texture itself, or no plain one to compare with (other variants can differ)
    return [base]


def fix_archive(archive: Path, root: Path, index: dict, fixes: list):
    listing_path = archive / "textures.json"
    if not listing_path.exists():
        return
    listing = json.loads(listing_path.read_text())
    wanted = {t["name"].lower(): t for t in listing if siblings(t["name"].lower(), index)}
    if not wanted:
        return
    for path in sorted((archive / "packs").glob("*.bin")):
        d = path.read_bytes()
        recs = T.records(d)
        if not recs:
            continue
        data0 = X.find_data0(d) or X.data0_from_textures(d, [p for p, _ in recs])
        names = [(n or "").lower() for n in X.texture_names(d, data0, [p for p, _ in recs])]
        if not any(n in wanted for n in names):
            continue
        exact = E.exact_layout(d, recs, data0)
        if exact is None:
            continue
        for i, n in enumerate(names):
            entry = wanted.get(n)
            if entry is None or entry["pack"] != path.stem:
                continue
            fc = recs[i][1]
            refs = [small(np.asarray(Image.open(root / index[s]).convert("RGB"))) for s in siblings(n, index)
                    if index[s].split("/textures_named/")[0] != str(archive.relative_to(root)).replace("\\", "/")]
            if not refs:
                continue

            def match(at):
                try:
                    s = small(T.decode(d, at, fc))
                except Exception:
                    return -1.0
                return max(float((s * r).mean()) for r in refs)
            png = archive / "textures_named" / entry["file"]  # what the last export wrote
            now = max(float((small(np.asarray(Image.open(png).convert("RGB"))) * r).mean()) for r in refs)                 if png.exists() else -1.0
            scores = [(match(exact[i] + k * 0x1000), k) for k in range(-256, 257) if exact[i] + k * 0x1000 >= 0]
            top = max(scores, default=(-1.0, 0))[0]
            best = min((c for c in scores if c[0] >= top - 0.02), key=lambda c: abs(c[1]), default=(-1.0, 0))
            # ~1.0 is the plain texture's own pixels (a variant differs by its dirt or camo): not a fix
            if 0.6 <= best[0] < 0.99 and best[0] >= now + 0.3:
                at = exact[i] + best[1] * 0x1000
                Image.fromarray(T.decode(d, at, fc), "RGBA").save(archive / "textures_named" / entry["file"])
                fixes.append({"archive": str(archive.relative_to(root)).replace("\\", "/"), "texture": n,
                              "shift": best[1] * 0x1000, "match_before": round(now, 3), "match_after": round(best[0], 3)})
                print(f"  {n}: moved {best[1] * 0x1000:+#x} (match {now:.2f} -> {best[0]:.2f})", flush=True)


def main():
    root = Path(sys.argv[1])
    index = json.loads((root / "texture_index.json").read_text())
    targets = [root / a for a in sys.argv[2:]] or sorted(p.parent for p in root.rglob("textures.json"))
    fixes = []
    for a in targets:
        before = len(fixes)
        fix_archive(a, root, index, fixes)
        if len(fixes) > before:
            print(f"{a.relative_to(root)}: {len(fixes) - before} fixed", flush=True)
    (root / "variant_fixes.json").write_text(json.dumps(fixes, indent=1))
    print(f"Done: {len(fixes)} variant textures re-placed", flush=True)


if __name__ == "__main__":
    main()
