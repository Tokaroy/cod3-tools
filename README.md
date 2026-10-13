# cod3-tools

Tools and format notes for extracting **Call of Duty 3 (Xbox 360)** assets (models, textures, sounds,
skeletons) from your own copy of the game, plus tools that check the results against what the game
actually draws.

**No game files are included or distributed here.** You need your own disc, extracted to a folder
(the files `default.xex`, `codmp_xenonf.xex`, `sp/`, `mp/` ...).

## Status (honest)

| What | State |
|---|---|
| Archives (`.cod`) | Unpacked fully (KAPF container, LZO1X chunks, APKF asset packs). |
| Meshes (`.xemesh`, `.xecell`) | Read from the game's own surface records. Verified **exact** against the running game for every model captured so far (47 models in Mayenne multiplayer, including world geometry and a multi-part weapon viewmodel). Not every model has been captured yet. |
| Multi-part models | Placed from animation skeletons (`.xeskel` rest pose, computed by running the title's own skeleton code in an emulator). Several vehicles have **no skeleton anywhere on the disc**, so their parts can't be placed yet (WC51, Harley, Kübelwagen, M5 Stuart, Tiger I, Panther, Kangaroo, Sherman M4A3). |
| Textures (`.xetex`) | Xenos-tiled DXT1/DXT3/DXT5/A8R8G8B8, byte-swapped. Decoding is fine; **finding where each texture's pixels start** is the weak point (see below). |
| Texture → material links | Mostly from the material records; soldier heads/bodies are filled in at runtime from placeholders, which the linker resolves the same way. |
| Sounds (`.wbk`, Treyarch NSLB) | Extracted with names through vgmstream. |
| Animations (`.xeanim`) | **Not decoded yet.** |
| SpeedTree vegetation (`.spt`) | **Not extracted yet** (trees/grass are drawn from these, not from `.xemesh`). |

### The texture problem, in detail

Each asset pack lists its GPU allocations in order, and the GPU data starts on a 64 KB boundary. The
start is chosen by how image-like the decodes look. When it is one 64 KB step off, every texture of that
pack decodes as **the right kind of image with its pieces in the wrong places** (a uniform sheet showing
other parts of the uniform), which image statistics cannot detect. `uv_refit.py` catches this by checking
where the models' UVs land; a few packs still give several textures the same position. If you know how
the pack header really describes texture placement, please open an issue.

## Requirements

Python 3.11+, `pip install -r requirements.txt`. For sounds: [vgmstream](https://vgmstream.org)
(`vgmstream-cli`). For capture checks: [Xenia Canary](https://github.com/xenia-canary/xenia-canary) and
[RenderDoc](https://renderdoc.org).

## Pipeline

```
python -I cod3/extract_all.py <disc folder> <work>/models        # unpack every archive's packs
python -I cod3/extract_models.py <work>/models                   # every .xemesh/.xecell -> OBJ, by name
python -I cod3/extract_textures.py <work>/models                 # every texture -> PNG, by name
python -I cod3/link_textures.py <work>/models                    # material -> colour texture (.mtl)
python -I cod3/catalog_models.py <work>/models <work>/catalog    # unique models by category, self-contained
python -I cod3/relink_all.py <work>/models <work>/catalog [--models] [--refit]   # the above in one go
```

Long steps take `--shard=K/N` to run in parallel. Per-model corrections found by eye go in
`cod3/model_fixes.json` (applied by `link_textures.py`).

Other tools:

- `assemble_characters.py` builds whole generic soldiers from body/head/helmet/gear pieces.
- `render_review.py`, `render_models.py`, `render_gallery.py` render review sheets (software renderer).
- `uv_refit.py` re-decodes packs whose texture data start was a 64 KB step off (judged by model UVs).
- `repair_texture.py` in-paints decoding damage in textures that have no clean copy anywhere.
- `extract_cod3_audio.py` extracts the `.wbk` sound banks with names.
- `xex_unpack.py` decrypts/unpacks `default.xex` to its memory image (needs `XEX_RETAIL_KEY`, the Xbox 360
  retail key, which is not included). `cod3_emu.py` / `cod3_anim.py` run the title's own code on it
  (Unicorn) to get skeleton rest poses.

## Checking against the game

1. Run the game in Xenia under RenderDoc and press F12 where the models you care about are on screen:
   `powershell -File cod3/xenia_capture.ps1 -Disc <disc> -Xenia <xenia_canary.exe> -RenderDoc <renderdoccmd.exe>`
2. Dump each capture's draws (inside RenderDoc's Python):
   `set CAPTURE_IN=<capture.rdc>` / `set CAPTURE_OUT=<file.draws>` then `qrenderdoc.exe --python cod3/capture_dump.py`
3. Compare: `python -I cod3/capture_check.py <work>/models report.json <file.draws> ... --archives=mp`

Each draw is matched to an extracted surface by its exact triangle list (the title's own index buffer);
then one 4x4 transform is fitted from our vertices to the clip-space positions the game drew. A residual
near zero means the geometry is exactly the game's; parts of multi-part models must share their model's
transform.

## Formats (short notes; the module docstrings have the details)

- **`.cod`** – "KAPF" archive (little-endian header and file list) of "NCH" chunks (big-endian headers,
  LZO1X payload) plus uncompressed GPU packs. Each section/pack is an "APKF" asset pack. (`cod3_kapf.py`)
- **Meshes** – record links are 32-bit word offsets from the pack's data start; `0xFC0000xx` ids are
  name-table indices (id / 8). One dependency group per asset (`MAT`, `MESH`, `SKEL` ...). Each mesh object
  has its own GPU block. Vertex declarations are Direct3D 9 style; positions are big-endian float3 in
  inches, Z up. Skinned vertices use a per-surface bone palette. (`cod3_xemesh.py`)
- **Textures** – Xenos fetch constants (24 bytes) after two `0xFFFF0000` words; tiled 32x32-block layout.
  (`cod3_textures.py`, `xenos_fetch.py`, `xenos_fast.py`)
- **Skeletons** – mesh skeletons: 0x90-byte bone records (inverse bind matrices). Animation skeletons
  (`.xeskel`, version 0x10302): 0x30-byte bone records, name at +4, (int16 parent, int16 own index) at
  +0x2C, both counted without the leading `tag_origin`. (`cod3_parts.py`)
- **Sounds** – Treyarch NSLB banks; vgmstream reads them with names.

## Credits

- [Xenia](https://github.com/xenia-canary/xenia-canary) (BSD) documents the Xenos texture fetch constant
  layout used in `xenos_fetch.py`.
- [CoD3Recomp](https://github.com/FrankHUN88453/CoD3Recomp) documents the game's asset banks and animation
  system (useful for the `.xeanim` work).
- [RenderDoc](https://renderdoc.org), [vgmstream](https://vgmstream.org).

## License

MIT (see `LICENSE`). Call of Duty 3 and its assets belong to Activision; this repository contains only
original code and notes.
