"""Check extracted COD3 models against what the game drew (Xenia + RenderDoc, dumped by capture_dump.py).

  python -I cod3/capture_check.py <models root> <report.json> <dump file> [dump file ...] [--archives=mp,sp/saint_lo]
For every captured draw:
  1. topology: the surface of ours whose triangle list is the same guest index list, triangle for triangle;
  2. shape: the one 4x4 transform taking our vertices (model space, as exported) to the clip-space corners
     the game drew (least squares; clip space is linear in model space before the divide). The residual,
     relative to the drawn spread, is ~0 when our geometry is the game's (rigid draws; skinned surfaces
     are posed by bones and only get the topology check);
  3. placement (models built from bone-placed parts only - vehicles, viewmodels; other models' surfaces
     are materials or detail levels of one object): every part of one copy shares the model's transform. A surface
     that needs a different one is misplaced; the report gives how far our copy is from where the game
     drew it, in metres, measured in the model's own space.
Our surfaces come from <root>/<archive>/models/*.obj (one "g surfaceK" group each).
"""
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_dump(path):
    d = Path(path).read_bytes()
    o, draws = 0, []
    while o + 12 <= len(d):
        eid, n, _ = struct.unpack_from("<3I", d, o)
        o += 12
        g = np.frombuffer(d, "<u4", n, o).astype(np.int64)
        o += 4 * n
        clip = np.frombuffer(d, "<f4", 4 * n, o).reshape(n, 4).astype(np.float64)
        o += 16 * n
        draws.append((eid, g, clip))
    return draws


def read_obj(path):
    """[(surface index, positions (m x 3), triangles (t x 3) relative to the surface)]."""
    verts, faces, out, cur = [], [], [], None
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append(line.split()[1:4])
        elif line.startswith("g surface"):
            cur = [int(line[9:]), len(verts), []]
            out.append(cur)
        elif line.startswith("f ") and cur is not None:
            cur[2].append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    V = np.array(verts, dtype=np.float64) if verts else np.zeros((0, 3))
    res = []
    for i, (k, start, f) in enumerate(out):
        end = out[i + 1][1] if i + 1 < len(out) else len(V)
        if f:
            t = np.array(f, dtype=np.int64) - start
            res.append((k, V[start:end], t))
    return res


MULTIPART = set()  # models built from parts placed by bones (vehicles, viewmodels)


def library(root, archives):
    """(triangle count, first triangle) -> [(archive, model file, surface, positions, triangles)]."""
    lib = defaultdict(list)
    for a in archives:
        for mj in sorted((root / a).rglob("models/models.json")):
            arch = mj.parent.parent
            for m in json.loads(mj.read_text()):
                try:
                    surfs = read_obj(mj.parent / m["file"])
                except Exception:
                    continue
                for k, V, t in surfs:
                    lib[(len(t), tuple(t[0]))].append((str(arch.relative_to(root)).replace("\\", "/"),
                                                       m["file"], k, V, t, m.get("skinned", False)))
                    MULTIPART.add(m["file"]) if m.get("placed") else None
    return lib


def fit(V, clip):
    A = np.c_[V, np.ones(len(V))]
    M, *_ = np.linalg.lstsq(A, clip, rcond=None)
    pred = A @ M
    spread = np.sqrt(((clip - clip.mean(0)) ** 2).sum(1).mean()) + 1e-9
    return M, float(np.sqrt(((pred - clip) ** 2).sum(1).mean()) / spread)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    root, report, dumps = Path(args[0]), Path(args[1]), args[2:]
    archives = opt.get("archives", "mp").split(",")
    lib = library(root, archives)
    print(f"library: {sum(len(v) for v in lib.values())} surfaces", flush=True)
    results = []
    for dump in dumps:
        per_model = defaultdict(list)  # (archive, file) -> [(surface, V, g, clip, M, resid)]
        draws = read_dump(dump)
        matched = 0
        for eid, g, clip in draws:
            T = g.reshape(-1, 3)
            cands = [c for c in lib.get((len(T), tuple(T[0])), []) if np.array_equal(c[4], T)]
            if not cands:
                continue
            matched += 1
            # several models can share a simple triangle list (boxes): the one whose shape fits wins
            best = None
            for arch, file, k, V, _, skinned in cands[:200]:
                if np.unique(g).max() >= len(V):
                    continue
                M, resid = fit(V[g], clip)
                if best is None or resid < best[0]:
                    best = (resid, arch, file, k, V, M, skinned)
            if best is None:
                c = cands[0]
                results.append({"dump": Path(dump).stem, "event": int(eid), "model": c[1], "archive": c[0],
                                "surface": c[2], "status": "vertex count differs"})
                continue
            resid, arch, file, k, V, M, skinned = best
            per_model[(arch, file)].append((k, V, g, clip, M, resid, skinned, int(eid)))
            status = "skinned (topology only)" if skinned else ("shape ok" if resid < 0.002 else
                     "shape close (vertex animation?)" if resid < 0.03 else "shape differs")
            results.append({"dump": Path(dump).stem, "event": int(eid), "model": file, "archive": arch, "surface": k,
                            "candidates": len(cands), "skinned": bool(skinned), "shape_residual": round(resid, 5),
                            "status": status})
        # placement: each rigid surface against the model's best-fitting (largest) surface's transform
        for (arch, file), surfs in per_model.items():
            if file not in MULTIPART:
                continue  # one object (its surfaces are materials or detail levels of it): nothing to place
            rigid = [s for s in surfs if not s[6] and s[5] < 0.002]
            if len({s[0] for s in rigid}) < 2:
                continue
            for k, V, g, clip, M, resid, _, eid in rigid:
                # the model is drawn once per copy in the level: compare with each draw of its other
                # surfaces and keep the nearest - that one belongs to the same copy
                best = None
                for k2, V2, g2, clip2, M2, _, _, _ in rigid:
                    if k2 == k:
                        continue
                    # where the game drew this surface, in model space through the other surface's transform
                    drawn, *_ = np.linalg.lstsq(M2[:3].T, (clip - M2[3]).T, rcond=None)
                    off = float(np.sqrt(((drawn.T - V[g]) ** 2).sum(1).mean()))
                    if best is None or off < best[0]:
                        best = (off, k2)
                if best is None:
                    continue
                for r in results:
                    if r["dump"] == Path(dump).stem and r["event"] == eid:
                        r["placement_m"], r["placement_vs_surface"] = round(best[0], 4), best[1]
                        if best[0] > 0.02:
                            r["status"] = "misplaced"
        print(f"{Path(dump).stem}: {len(draws)} draws, {matched} matched to our surfaces", flush=True)
    report.write_text(json.dumps(results, indent=1))
    by = defaultdict(int)
    for r in results:
        by[r["status"]] += 1
    print(dict(by))
    for r in results:
        if r["status"] in ("shape differs", "misplaced", "vertex count differs"):
            extra = f" residual {r.get('shape_residual')}" + (f" off {r['placement_m']} m" if "placement_m" in r else "")
            print(f"  {r['status']:22} {r['archive']}/{r['model']} surface {r['surface']}{extra}")
