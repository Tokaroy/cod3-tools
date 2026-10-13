"""How many of the models each archive lists (.xemesh/.xecell) extract_models.py exported, and which are missing.

  python -I cod3/model_coverage.py <extracted root> [--list]
"""
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1])
    listed_all, got_all, missing = set(), set(), {}
    archives = 0
    for listing in sorted(root.rglob("models/models.json")):
        archive = listing.parent.parent
        files = json.loads((archive / "manifest.json").read_text())["files"]
        listed = {f["name"].lower().split("\\")[-1] for f in files if f["name"].lower().endswith((".xemesh", ".xecell"))}
        got = {m["name"].lower().split("\\")[-1] for m in json.loads(listing.read_text())}
        archives += 1
        listed_all |= listed
        got_all |= got
        miss = sorted(listed - got)
        if miss:
            missing[str(archive.relative_to(root))] = miss
    print(f"{archives} archives: {len(got_all & listed_all)} of {len(listed_all)} unique listed models exported")
    never = sorted(listed_all - got_all)
    print(f"{len(never)} never exported anywhere")
    if "--list" in sys.argv:
        for a, m in missing.items():
            print(f"  {a}: {len(m)} missing: {', '.join(m[:8])}{' ...' if len(m) > 8 else ''}")


if __name__ == "__main__":
    main()
