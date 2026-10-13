"""Convert every Call of Duty 3 (Xbox 360) sound bank to named .wav files.

The .wbk files are Treyarch "NSLB" banks (XMA), which vgmstream reads directly, with names.
Usage: python -I cod3/extract_cod3_audio.py <COD3 game folder> <output folder>
Writes <out>/<bank path>/<nnnn>_<name>.wav. Skips banks already converted. Set VGMSTREAM to the
vgmstream-cli.exe path (default %USERPROFILE%/Downloads/vgmstream/vgmstream-cli.exe).
Paths given to vgmstream must be Windows-style (G:/...), not MSYS (/g/...).
"""
import os
import subprocess
import sys
from pathlib import Path

VGM = Path(os.environ.get("VGMSTREAM", Path.home() / "Downloads/vgmstream/vgmstream-cli.exe"))

game, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
total = 0
for bank in sorted(game.rglob("*.wbk")):
    dest = out / bank.relative_to(game).with_suffix("")
    if dest.exists() and any(dest.glob("*.wav")):
        n = len(list(dest.glob("*.wav")))
    else:
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(VGM), "-S", "0", "-o", str(dest / "?04s_?n.wav"), str(bank)], capture_output=True)
        n = len(list(dest.glob("*.wav")))
    total += n
    print(f"{bank.relative_to(game)}: {n} sounds", flush=True)
print(f"Done: {total} .wav files under {out}", flush=True)
