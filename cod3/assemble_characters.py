"""Assemble COD3's generic soldiers from their pieces, the way the game builds them at runtime.

  python -I cod3/assemble_characters.py <catalog folder>
Soldier types (c_pnzgren, c_wrmcht, c_uspriv, ...) come as numbered bodies, heads (with the hands), helmets
or hats and gear sets, all modelled in the same space on the same skeleton - so a soldier is just its
pieces in one model. For each type (most detailed LOD only), every body is combined with every head; the
helmet and gear sets are cycled across those combinations so each one appears. Writes
<catalog>/characters_assembled/<type>_b<body>_h<head>.obj/.mtl (textures from ../textures) and
<catalog>/characters_assembled/index.json (soldier -> its pieces).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PARTS = {"body": "body", "head": "head", "helm": "helm", "helmet": "helm", "hat": "helm", "cap": "helm",
         "beret": "helm", "gear": "gear"}


def parse(name):
    """(soldier type, piece kind, variant, lod) from names like c_pnzgren_ng_Lod2_body_1,
    c_uspriv_ng_lod2_head1, c_polinf_ng_head1_lod2, c_wrmcht_ng_lod2_headb_1 - or None."""
    stem = name.lower().rsplit(".", 1)[0]
    if "_ng" not in stem:
        return None
    kind_type, rest = stem.split("_ng", 1)
    tokens = [t for t in rest.strip("_").split("_") if t]
    lod = next((int(t[3:]) for t in tokens if re.fullmatch(r"lod\d", t)), 2)
    tokens = [t for t in tokens if not re.fullmatch(r"lod\d", t)]
    for i, t in enumerate(tokens):
        m = re.fullmatch(r"(body|head|helmet|helm|hat|cap|beret|gear)([a-z]*)(\d*)", t)
        if m:
            variant = (m.group(2) + m.group(3) + "".join(tokens[i + 1:])) or "1"
            return kind_type, PARTS[m.group(1)], variant, lod
    return None


def read_obj(path):
    v, vt, faces, mtl_lib = [], [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append(line)
        elif line.startswith("vt "):
            vt.append(line)
        elif line.startswith(("f ", "g ", "usemtl")):
            faces.append(line)
        elif line.startswith("mtllib"):
            mtl_lib = line.split(None, 1)[1]
    mtl = (path.parent / mtl_lib).read_text() if mtl_lib and (path.parent / mtl_lib).exists() else ""
    return v, vt, faces, mtl


def combine(pieces, out_obj):
    lines, mtl_lines, nv, nvt = [f"mtllib {out_obj.stem}.mtl"], [], 0, 0
    for k, piece in enumerate(pieces):
        v, vt, faces, mtl = read_obj(piece)
        lines += v + vt
        for line in faces:
            if line.startswith("usemtl"):
                line = f"usemtl p{k}_{line.split(None, 1)[1]}"
            elif line.startswith("g "):
                line = f"g p{k}_{line[2:]}"
            elif line.startswith("f "):
                refs = []
                for r in line.split()[1:]:
                    a = r.split("/")
                    refs.append(f"{int(a[0]) + nv}" + (f"/{int(a[1]) + nvt}" if len(a) > 1 and a[1] else ""))
                line = "f " + " ".join(refs)
            lines.append(line)
        for line in mtl.splitlines():
            mtl_lines.append(f"newmtl p{k}_{line.split(None, 1)[1]}" if line.startswith("newmtl") else line)
        nv, nvt = nv + len(v), nvt + len(vt)
    out_obj.write_text("\n".join(lines) + "\n")
    out_obj.with_suffix(".mtl").write_text("\n".join(mtl_lines) + "\n")


def main():
    catalog = Path(sys.argv[1])
    out = catalog / "characters_assembled"
    out.mkdir(exist_ok=True)
    types = defaultdict(lambda: defaultdict(dict))  # type -> kind -> variant -> (lod, obj)
    for c in json.loads((catalog / "catalog.json").read_text()):
        if c["category"] != "characters":
            continue
        p = parse(c["name"])
        if p is None:
            continue
        kind_type, kind, variant, lod = p
        have = types[kind_type][kind].get(variant)
        if have is None or lod < have[0]:  # most detailed LOD wins
            types[kind_type][kind][variant] = (lod, catalog / c["file"])
    index = {}
    for kind_type, kinds in sorted(types.items()):
        bodies, heads = sorted(kinds.get("body", {})), sorted(kinds.get("head", {}))
        helms, gears = sorted(kinds.get("helm", {})), sorted(kinds.get("gear", {}))
        if not bodies or not heads:
            continue
        n = 0
        for b in bodies:
            for h in heads:
                pieces = [kinds["body"][b][1], kinds["head"][h][1]]
                names = [f"body {b}", f"head {h}"]
                if helms:
                    hm = helms[n % len(helms)]
                    pieces.append(kinds["helm"][hm][1])
                    names.append(f"helm {hm}")
                if gears:
                    g = gears[n % len(gears)]
                    pieces.append(kinds["gear"][g][1])
                    names.append(f"gear {g}")
                name = f"{kind_type}_b{b}_h{h}"
                combine(pieces, out / f"{name}.obj")
                index[name] = names
                n += 1
        print(f"{kind_type}: {len(bodies)} bodies, {len(heads)} heads, {len(helms)} helmets, {len(gears)} gear -> {n}",
              flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"Done: {len(index)} assembled soldiers")


if __name__ == "__main__":
    main()
