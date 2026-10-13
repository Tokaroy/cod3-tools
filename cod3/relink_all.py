"""Relink every model's textures, rebuild the catalog and audit it, one after the other.

  python -I cod3/relink_all.py [models root] [catalog folder] [--models] [--index] [--variants]
--models first re-exports every model (extract_models.py --redo, 4 shards in parallel).
--index first rebuilds texture_index.json (extract_textures.py; archives already exported are skipped).
--variants re-places character uniform variant textures first (fix_variant_textures.py).
--refit re-decodes textures whose pack data start was a 64 KB step off, judged by the models' UVs
  (uv_refit.py, 6 shards in parallel; needs links.json from an earlier link).
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if a not in ("--models", "--index", "--variants", "--refit")]
if len(args) < 2:
    sys.exit(__doc__)
root, catalog = args[0], args[1]
if "--models" in sys.argv:
    print("== extract_models.py (4 shards)", flush=True)
    jobs = [subprocess.Popen([sys.executable, "-I", str(here / "extract_models.py"), root, "--redo", f"--shard={k}/4"])
            for k in range(4)]
    if any(j.wait() for j in jobs):
        sys.exit("extract_models failed")
if "--refit" in sys.argv:
    print("== uv_refit.py (6 shards)", flush=True)
    jobs = [subprocess.Popen([sys.executable, "-I", str(here / "uv_refit.py"), root, f"--shard={k}/6"]) for k in range(6)]
    if any(j.wait() for j in jobs):
        sys.exit("uv_refit failed")
steps = [["extract_textures.py", root]] if "--index" in sys.argv else []
steps += [["fix_variant_textures.py", root]] if "--variants" in sys.argv else []
for step in steps + [["link_textures.py", root], ["catalog_models.py", root, catalog],
             ["texture_audit.py", root, catalog, "--top", "45"]]:
    print(f"== {step[0]}", flush=True)
    subprocess.run([sys.executable, "-I", str(here / step[0]), *step[1:]], check=True)
print("ALL DONE", flush=True)
