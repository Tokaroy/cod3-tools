"""Rebuild texture_index.json (and texture_scores/stripes/exact.json) from the textures already exported.

  python -I cod3/rebuild_index.py <models root>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_textures  # noqa: E402

if __name__ == "__main__":  # the index build uses a process pool, whose workers re-import this file
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    print(f"{extract_textures.build_index(root)} unique texture names", flush=True)
