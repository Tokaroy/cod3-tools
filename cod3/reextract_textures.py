"""Re-decode every texture on the disc from the pack headers (extract_textures.py --redo, 4 shards in
parallel), then rebuild texture_index.json once.

  python -I cod3/reextract_textures.py <models root>
"""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    jobs = [subprocess.Popen([sys.executable, "-I", str(here / "extract_textures.py"), root, "--redo", "--no-index",
                              f"--shard={k}/4"]) for k in range(4)]
    if any(j.wait() for j in jobs):
        sys.exit("extract_textures failed")
    sys.path.insert(0, str(here))
    import extract_textures
    print(f"ALL DONE: {extract_textures.build_index(Path(root))} unique texture names", flush=True)


if __name__ == "__main__":  # the index build uses a process pool, whose workers re-import this file
    main()
