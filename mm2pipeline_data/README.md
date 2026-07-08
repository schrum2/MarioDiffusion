# mm2pipeline_data

Every format conversion in the project lives in this package. Run any stage
through the single entry point (add `--help` for its options):

```
python -m mm2pipeline_data <command> [options]
```

Getting real levels into a dataset:

1. `extract` — pull levels from the HuggingFace dump into `.bcd` files
2. `toost` — decode each `.bcd` to level JSON and a PNG
3. `json-to-ascii` — flatten the JSON to ASCII grids
4. `dataset` — window the ASCII into a tile-id dataset, then split it

...then train the diffusion model on that dataset.

Getting a generated level back out:

5. `ascii-to-json` — rebuild level JSON from the model's ASCII
6. `swe` — write a `.swe` you can play in SMM: World Engine

(To preview a generated level in toost instead, use `json_to_bcd.py` at the repo
root to turn its JSON into a `.bcd`.)

Each command maps to a module you can also run directly, e.g.
`python -m mm2pipeline_data.extract` or `python -m mm2pipeline_data.dataset build`:

| Command         | Converts                          | Module               |
|-----------------|-----------------------------------|----------------------|
| `extract`       | HuggingFace dataset → `.bcd`      | `mm2pipeline_data.extract`|
| `toost`         | `.bcd` → level JSON + PNG         | `mm2pipeline_data.toost`  |
| `json-to-ascii` | level JSON → ASCII grid           | `mm2pipeline_data.ascii`  |
| `dataset`       | ASCII → tile-id dataset (+ split) | `mm2pipeline_data.dataset`|
| `ascii-to-json` | generated ASCII → level JSON      | `mm2pipeline_data.ascii`  |
| `swe`           | level JSON → playable `.swe`      | `mm2pipeline_data.swe`    |

Two more modules hold shared pieces rather than a stage: `mm2pipeline_data.tiles`
(object metadata / glyph vocabulary, from `mm2_tileset_we.json`) and
`mm2pipeline_data.bcd` (the binary `.bcd` codec).

## Requirements

`pip install -r requirements.txt` from the repo root covers everything. Stage
specifics:

- `extract` needs `datasets` and `pycryptodome`.
- `toost` needs `toost_stuff/bin/toost.exe` (bundled; or build it from
  [TheGreatRambler/toost](https://github.com/TheGreatRambler/toost)).
- `dataset build --with_images` needs `Pillow`.
- Everything else is standard library.

## Walkthrough: real levels → training dataset

This is what `bat\extract_levels_to_ascii.bat` + the `prepare-mario-maker`
/ `run_full_pipeline` scripts drive; the manual equivalent:

```bat
REM 1. Pull levels from the HuggingFace dump (writes .bcd files plus a
REM    level_metadata.json index of each level's server-side difficulty/tags).
python -m mm2pipeline_data extract --output_folder out\bcd --limit 100 --skip_3dworld --skip_items --skip_subworld_items

REM    Filters: --name / --name_count, --ids, --tag / --all-tags / --exclude-tag,
REM    --difficulty, --likes N (minimum), --dislikes N (maximum).
python -m mm2pipeline_data extract --output_folder out\bcd --tag Speedrun --difficulty Easy Normal --likes 1000 --limit 50

REM 2. Decode each .bcd to level JSON and a rendered PNG. The metadata index is
REM    folded into the JSONs automatically.
python -m mm2pipeline_data toost --input out\bcd -o out\json --images-output out\images

REM 3. Flatten the JSONs to ASCII grids. Also writes ascii\metadata.json
REM    (per-level name/difficulty/gamestyle/theme/tags, keyed by file stem).
python -m mm2pipeline_data json-to-ascii --input out\json --output_folder out\ascii

REM 4. Window the ASCII into fixed-size tile-id scenes. metadata.json is picked
REM    up automatically from the input folder (or pass --metadata explicitly).
python -m mm2pipeline_data dataset build --input out\ascii --output_folder dataset.json --tileset mm2_tileset_we.json --sliding_window --stride 20

REM    Useful options: --strip_goal (train without flagpoles), --window_h/--window_w,
REM    --min_tiles_pct (drop mostly-air windows; rejects go to *_dropped.json),
REM    --with_images (crop the matching PNG slice per sample),
REM    --convert_to_extended / --convert_to_vglc (alternate tile vocabularies).

REM 5. Split into train/validate/test (writes dataset-train.json etc.).
python -m mm2pipeline_data dataset split --input dataset.json --seed 0
```

## Walkthrough: generated ASCII → playable level

```bat
REM 1. Rebuild level JSON from the model's ASCII output (a .txt file or folder).
REM    Multi-tile objects are re-coalesced, scattered semisolid platforms are
REM    repaired into clean boxes, and a level with no goal gets a reachable
REM    end runway synthesized on its right edge.
python -m mm2pipeline_data ascii-to-json --input samples\ --output_folder out\gen_json --gamestyle smw --theme overworld

REM 2. Convert to a .swe save for Super Mario Maker: World Engine. A matching
REM    *_subworld.json next to the overworld file is included automatically.
python -m mm2pipeline_data swe --input out\gen_json -o out\swe --user <your SMMWE username>
```

Drop the `.swe` into SMMWE's save folder (the `--user` name must match the
logged-in SMMWE account; by default it is auto-detected from the SMMWE
settings file). To preview a generated level in toost instead, use
`python json_to_bcd.py <level.json> --toost-compat` from the repo root.

## More examples, per command

### extract

```bat
REM A quick 100-level test sample.
python -m mm2pipeline_data extract -o out\bcd --limit 100

REM Just a few specific levels by data_id.
python -m mm2pipeline_data extract -o out\bcd --ids 3000004 3000007

REM Levels whose name contains "mario" (first 25 matches).
python -m mm2pipeline_data extract -o out\bcd --name mario --name_count 25

REM Speedruns, Easy/Normal, 1000+ likes, at most 300 boos, no Art.
python -m mm2pipeline_data extract -o out\bcd --tag Speedrun --difficulty Easy Normal --likes 1000 --dislikes 300 --exclude-tag Art --limit 50

REM Everything, dropping levels that won't survive later stages (~100 GB stream).
python -m mm2pipeline_data extract -o out\bcd --skip_3dworld --skip_items --skip_subworld_items
```

### toost

```bat
REM JSON + PNG next to the .bcd folder (default out\bcd\json and out\bcd\images).
python -m mm2pipeline_data toost --input out\bcd

REM Explicit output folders.
python -m mm2pipeline_data toost --input out\bcd -o out\json --images-output out\images

REM Render without the grid overlay, and keep even 1-object subworlds.
python -m mm2pipeline_data toost --input out\bcd -o out\json --remove-grid --min-objects 0
```

### json-to-ascii

```bat
REM JSON folder -> ASCII grids, with metadata.json written into out\ascii.
python -m mm2pipeline_data json-to-ascii --input out\json --output_folder out\ascii

REM Write the metadata sidecar somewhere else.
python -m mm2pipeline_data json-to-ascii --input out\json --output_folder out\ascii --metadata_output out\meta.json
```

### dataset build

```bat
REM One best (busiest) 20x20 window per level.
python -m mm2pipeline_data dataset build --input out\ascii --output_folder dataset.json --tileset mm2_tileset_we.json

REM Every 20-wide window across each level (no overlap).
python -m mm2pipeline_data dataset build --input out\ascii --output_folder dataset.json --tileset mm2_tileset_we.json --sliding_window --stride 20

REM Train without flagpoles, and crop the matching PNG slice per sample.
python -m mm2pipeline_data dataset build --input out\ascii --output_folder dataset.json --tileset mm2_tileset_we.json --sliding_window --strip_goal --with_images

REM Extended tile vocabulary instead of the base one.
python -m mm2pipeline_data dataset build --input out\ascii --output_folder dataset.json --tileset extended_tiles.json --convert_to_extended --sliding_window --stride 20
```

### dataset split

```bat
REM Default 80 / 10 / 10.
python -m mm2pipeline_data dataset split --input dataset.json --seed 0

REM Custom ratios.
python -m mm2pipeline_data dataset split --input dataset.json --train_pct 0.7 --val_pct 0.15 --test_pct 0.15 --seed 42
```

### ascii-to-json

```bat
REM A single generated level.
python -m mm2pipeline_data ascii-to-json --input sample.txt --output_folder out\gen_json

REM A whole folder, as SMW / overworld.
python -m mm2pipeline_data ascii-to-json --input samples\ --output_folder out\gen_json --gamestyle smw --theme overworld

REM NSMBU castle, with a longer timer.
python -m mm2pipeline_data ascii-to-json --input samples\ --output_folder out\gen_json --gamestyle nsmbu --theme castle --timer 500
```

### swe

```bat
REM One level (subworld picked up automatically if it sits alongside).
python -m mm2pipeline_data swe --input out\gen_json\3000048_overworld.json

REM A whole folder to a .swe output folder.
python -m mm2pipeline_data swe --input out\gen_json -o out\swe

REM Set the author and displayed level name explicitly.
python -m mm2pipeline_data swe --input level_overworld.json -o mylevel.swe --user Patrick --name "Test Level"
```

## A few conventions

The tileset `mm2_tileset_we.json` is the shared glyph vocabulary — every
drawable object has to fold onto one of its glyphs (`mm2pipeline_data.tiles` checks
this on import).

Files stay matched by stem: a level is `<stem>_overworld.json` (and maybe
`<stem>_subworld.json`), its ASCII grid keeps the same stem, and the metadata
sidecars are keyed by it.

A level's difficulty and tags aren't in the `.bcd`, so they ride along on the
side: `extract` writes them to `level_metadata.json`, `toost` folds them into
the JSONs, `json-to-ascii` copies them into `metadata.json`, and `dataset build`
attaches them to each sample.

## History

These stages replaced the standalone scripts `extract_mm2_bcd.py`,
`toost_stuff/batch_convert.py`, `mm2_json_to_ascii.py`,
`build_dataset_with_ascii.py`, `split_mario_maker_data.py`,
`mm2_ascii_to_json.py` and `json_to_swe.py` (removed July 2026; see git
history for the originals).
