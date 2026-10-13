# cod3-tools

Tools and format notes for extracting **Call of Duty 3 (Xbox 360)** assets (models, textures, sounds,
skeletons) from your own copy of the game, plus tools that check the results against what the game
actually draws.

> **⚠️ Work in progress: not everything here is correct yet.**
> These tools are unfinished and the output still has known errors. A handful of models get the wrong
> texture or none, several vehicles can't be assembled, and vegetation isn't extracted at all. Only part of
> the geometry has been checked against the running game. Don't treat any
> output as verified unless the "Status" table below says so. **Help from others is very welcome:** see
> [Help wanted](#help-wanted).

**No game files are included or distributed here.** You need your own disc, extracted to a folder
(the files `default.xex`, `codmp_xenonf.xex`, `sp/`, `mp/` ...).

## Status (honest)

| What | State |
|---|---|
| Archives (`.cod`) | Unpacked fully (KAPF container, LZO1X chunks, APKF asset packs). |
| Asset pack layout | **Exact, from the pack header** (`apkf.py`): where every record and every block of GPU data sits. Checked on all 34,869 texture records on the disc; every one of the 30,386 mesh blocks is placed from it too (each checked against its surfaces' stored bounds). |
| Meshes (`.xemesh`, `.xecell`) | Read from the game's own surface records, at the GPU position the pack header gives (an earlier search could land a vertex or two off and scramble the triangles). Verified **exact** against the running game for every model captured so far (47 models in Mayenne multiplayer, including world geometry and a multi-part weapon viewmodel). Not every model has been captured yet. |
| Multi-part models | Placed from animation skeletons (`.xeskel` rest pose, computed by running the title's own skeleton code in an emulator). Weapons are posed with the first frame of their own idle animation when it has been decoded, so animated parts sit where the game holds them (a bazooka's rocket in the tube); parts the idle pose parks away from the gun (a spent shell) are left out and listed as `hidden` in `models.json`. Several vehicles have **no skeleton anywhere on the disc**, so their parts can't be placed yet (WC51, Harley, Kübelwagen, M5 Stuart, Tiger I, Panther, Kangaroo, Sherman M4A3). |
| Textures (`.xetex`) | Xenos-tiled DXT1/DXT3/DXT5/A8R8G8B8, byte-swapped, each at the exact position the pack header gives. Compared with a community HD set made from correctly decoded originals, the character and weapon textures match at 0.95 on average (most at 0.99-1.00). |
| Texture → material links | Mostly from the material records; soldier heads/bodies are filled in at runtime from placeholders, which the linker resolves the same way. 99.4% of surfaces get a texture. **A few models still get the wrong texture or none** (e.g. the C-47, a Sherman wreck, a Flakvierling variant). |
| Sounds (`.wbk`, Treyarch NSLB) | Extracted with names through vgmstream. |
| Animations (`.xeanim`) | Decoded by running the game's own decoder code in an emulator (`cod3_anim.py`, `extract_anims.py`; needs the unpacked `default.xex`). The compressed format itself isn't written down yet. |
| SpeedTree vegetation (`.spt`) | **Not extracted yet** (trees/grass are drawn from these, not from `.xemesh`). |

### How texture placement was solved

Texture records don't store where their pixels are (the address in the fetch constant is filled in at load
time). Earlier versions guessed each pack's GPU data start from how image-like the decodes looked; when the
guess was one 64 KB step off, every texture of that pack came out as the right kind of image with its pieces
in the wrong places, and image statistics can't see that.

The pack header describes everything needed (`apkf.py`): a table of asset types (record count, record
alignment, GPU alignment) and per-object entries (record size, GPU data size). Records start at
`(header size & ~0xF) + 0x50`, raised to the largest record alignment, and follow each other in table
order; the GPU data start at the next 64 KB boundary and hold every object's GPU data in the same order.
Laid out that way, every texture record on the disc (and every mesh block checked so far) lands exactly where it really is.

## Help wanted

This is a one-person project and a lot of it is still guesswork checked by eye. If you know Call of Duty 3,
Treyarch's Xbox 360 formats or the Xenos GPU, any of these would help a lot:

- **Wrong or missing textures on a few models:** the material-to-texture link for models like the C-47
  or the damaged Flakvierling.
- **Vehicle skeletons:** where the WC51, Harley, Kübelwagen, M5 Stuart, Tiger I, Panther, Kangaroo and
  Sherman M4A3 get their part positions, if not from a `.xeskel`.
- **`.xeanim` animations:** a written description of the compression (they decode today only by running the
  game's own code).
- **SpeedTree `.spt`:** the version COD3 uses, and how its trees and grass are built.
- **Checking results:** captures (Xenia + RenderDoc) of levels and vehicles not covered yet, or reports of
  models and textures that look wrong next to the game.

Please open an issue or a pull request, even for partial findings or a "this looks off" report.

## Requirements

Python 3.11+, `pip install -r requirements.txt`. For sounds: [vgmstream](https://vgmstream.org)
(`vgmstream-cli`). For capture checks: [Xenia Canary](https://github.com/xenia-canary/xenia-canary) and
[RenderDoc](https://renderdoc.org).

## Pipeline

```
python -I cod3/extract_all.py <disc folder> <work>/models        # unpack every archive's packs
python -I cod3/extract_anims.py <work>/models <work>/anims_weapons <weapon archive> ...   # optional: weapon idle poses
python -I cod3/extract_models.py <work>/models                   # every .xemesh/.xecell -> OBJ, by name
python -I cod3/extract_textures.py <work>/models                 # every texture -> PNG, by name
python -I cod3/link_textures.py <work>/models                    # material -> colour texture (.mtl)
python -I cod3/catalog_models.py <work>/models <work>/catalog    # unique models by category, self-contained
python -I cod3/relink_all.py <work>/models <work>/catalog [--models] [--index]   # link, catalog, audit in one go
```

Long steps take `--shard=K/N` to run in parallel. Per-model corrections found by eye go in
`cod3/model_fixes.json` (applied by `link_textures.py`).

Other tools:

- `assemble_characters.py` builds whole generic soldiers from body/head/helmet/gear pieces.
- `render_review.py`, `render_models.py`, `render_gallery.py` render review sheets (software renderer).
- `apkf.py` reads an asset pack's header and gives the exact position of every record and GPU block.
- `reextract_textures.py` re-decodes every texture (4 processes) and rebuilds the index once;
  `rebuild_index.py` rebuilds only the index.
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
- **Asset packs** – header with a type table (FourCC or hash, count, record alignment, GPU flag and
  alignment) and per-object (name entry, record size[, GPU size]) lists; records and GPU data in table
  order. (`apkf.py`)
- **Textures** – Xenos fetch constants (24 bytes) at texture record + 0x48, after two `0xFFFF0000` words;
  tiled 32x32-block layout. (`cod3_textures.py`, `xenos_fetch.py`, `xenos_fast.py`)
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
