"""Audit every catalog model's texture links and rank the suspicious ones.

  python -I cod3/texture_audit.py <models root> <catalog folder> [--top N]
For each drawn surface (collision hulls skipped) it scores:
  name  - does the texture name share words with the model / material names (link_textures.words)?
  cover - fraction of the surface's UV area that lands on painted (non-black) texels (128x128)
  clean - the texture copy's tile-seam ratio (extract_textures.seam_ratio; > 2.5 = partly scrambled)
and how the link was made (bound/sibling are taken from the game's own records). Writes
<catalog>/texture_audit.json and prints the worst models; a model's score is its worst big surface.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import link_textures as L  # noqa: E402
from extract_textures import seam_ratio  # noqa: E402

RES = 128
_tex = {}


def tex_info(path):
    if path not in _tex:
        try:
            im = Image.open(path).convert("RGB")
            a = np.asarray(im)
            small = np.asarray(im.resize((RES, RES), Image.BILINEAR)).astype(int)
            _tex[path] = (small.sum(2) > 45, seam_ratio(a))
        except Exception:
            _tex[path] = (None, 9.0)
    return _tex[path]


def uv_mask(vt, faces):
    img = Image.new("L", (RES, RES), 0)
    dr = ImageDraw.Draw(img)
    for f in faces[:6000]:
        dr.polygon([((vt[i][0] % 1.0) * RES, (1 - vt[i][1] % 1.0) * RES) for i in f], fill=255)
    return np.asarray(img) > 0


def audit(root, catalog):
    index = json.loads((root / "texture_index.json").read_text())
    if (root / "texture_index_bound.json").exists():  # bound-only copies (link_textures.py)
        index.update(json.loads((root / "texture_index_bound.json").read_text()))
    cat = json.loads((catalog / "catalog.json").read_text())
    results = []
    links_cache = {}
    models_cache = {}
    for c in cat:
        a = c["archive"]
        if a not in links_cache:
            links_cache[a] = json.loads((root / a / "links.json").read_text())
            models_cache[a] = {m["name"].lower().split("\\")[-1]: m for m in
                               json.loads((root / a / "models" / "models.json").read_text())}
        m = models_cache[a].get(c["name"])
        link = next((v for k, v in links_cache[a].items() if k.lower().split("\\")[-1] == c["name"]), None)
        if m is None or link is None:
            continue
        vt, groups = L.obj_groups(root / a / "models" / m["file"])
        surfs = []
        for k, (mat, ln) in enumerate(zip(m.get("materials") or [], link)):
            if ln.get("how") in ("collision",) or k not in groups:
                continue
            tris = len(groups[k])
            tex = ln.get("texture")
            if tex is None:
                surfs.append({"k": k, "tris": tris, "material": mat.get("name"), "texture": None,
                              "how": ln.get("how"), "bad": 1.0, "why": "untextured"})
                continue
            mask, seam = tex_info(root / index[tex]) if tex in index else (None, 9.0)
            um = uv_mask(vt, groups[k]) if len(vt) else None
            cover = float((um & mask).sum() / max(um.sum(), 1)) if (um is not None and mask is not None) else 0.0
            mname = (mat.get("name") or "").lower()
            part, _, owner = mname.partition(".")
            wt = L.words(tex)
            name_ok = L.overlap(L.words(owner) | L.words(c["name"]) | L.words(part), wt) > 0
            trusted = ln.get("how") in ("bound", "sibling")
            why = []
            bad = 0.0
            if not trusted and not name_ok:
                bad += 0.5
                why.append("name mismatch")
            if cover < 0.6:
                bad += (0.6 - cover)
                why.append(f"UVs on {cover:.0%} painted")
            if seam > 2.5:
                bad += 0.3
                why.append(f"texture partly scrambled ({seam:.1f})")
            if ln.get("how") in ("pack+uv", "cover"):
                bad += 0.1
                why.append(ln["how"])
            surfs.append({"k": k, "tris": tris, "material": mat.get("name"), "texture": tex, "how": ln.get("how"),
                          "cover": round(cover, 2), "seam": round(seam, 2), "bad": round(bad, 2), "why": why})
        big = [s for s in surfs if s["tris"] >= 0.05 * max(1, c["tris"])] or surfs
        score = max((s["bad"] for s in big), default=0.0)
        results.append({"name": c["name"], "category": c["category"], "file": c["file"], "tris": c["tris"],
                        "score": round(score, 2), "surfaces": surfs})
    results.sort(key=lambda r: (-r["score"], -r["tris"]))
    (catalog / "texture_audit.json").write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    root, catalog = Path(sys.argv[1]), Path(sys.argv[2])
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 40
    res = audit(root, catalog)
    flagged = [r for r in res if r["score"] >= 0.3]
    by_cat = {}
    for r in flagged:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"{len(flagged)} of {len(res)} models flagged: {by_cat}")
    for r in res[:top]:
        worst = max(r["surfaces"], key=lambda s: s["bad"], default=None)
        print(f"  {r['score']:.2f} {r['category']:10} {r['name']:42} {worst['texture'] if worst else ''} "
              f"{worst['why'] if worst else ''}")
