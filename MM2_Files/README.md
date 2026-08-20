# MM2_Files

The captioning, rendering, and evaluation layer that sits on top of the
`mm2pipeline_data` pipeline. Once that package has turned real levels into a
tile-id dataset, the scripts here caption it, draw it, and score whatever a model
generates from it:

- `MarioMaker_create_ascii_captions.py` — deterministic captions (ground summary + tile counts)
- `MarioMaker_llm_captions.py` — LLM captions via Ollama, a hosted API, or a local vision model
- `render_mm2.py` — draws real MM2 art for a scene (imported, no command line)
- `evaluate_mm2_metrics.py` — metrics for generated levels

This is a package, so run everything from the repo root. The two captioners can be
run as plain scripts; they add the repo root to `sys.path` themselves.

## Requirements

Everything is covered by `pip install -r requirements.txt` at the repo root.
Extras only some paths need:

- `render_mm2` needs `Pillow` and the bundled `toost_stuff/` assets (spritesheet
  + gamestyle tilesheets). Without them it falls back to flat colour tiles.
- `MarioMaker_llm_captions --backend ollama` needs a running [Ollama](https://ollama.com)
  server; the `smolvlm` backend needs `torch` + `transformers`; the `claude` /
  `openai` / `gemini` backends need an API key file.
- `evaluate_mm2_metrics` uses the GPU for edit-distance by default (`--cpu` to
  skip).

## Deterministic captions

Writes a caption per scene from the tileset tags — a ground/floor summary, a count
of every tile type, and a "blob of X" note when one piles up. `mm2pipeline_data
dataset build --captions` already runs this for you; run it directly to re-caption
an existing dataset.

```bat
python MM2_Files\MarioMaker_create_ascii_captions.py --dataset dataset.json --tileset Game_MM2\mm2_tileset_we.json --output dataset_captioned.json
```

By default the caption lands in the `caption` field. Use `--caption-mode keyed` to
store it as a list under `--caption-key` instead, so a scene can carry captions
from several sources at once:

```bat
python MM2_Files\MarioMaker_create_ascii_captions.py --dataset dataset.json --tileset Game_MM2\mm2_tileset_we.json --output dataset_captioned.json --caption-mode keyed --caption-key deterministic_captions
```

## LLM captions

Generates richer, human-style captions with a language model. The prompt hands the
model a symbol dictionary, pre-computed metadata (terrain heights, floor/ceiling,
object counts), and the level grid, then asks for captions that vary in length and
register. Runs resume: rerun against the same `--output` and already-captioned
scenes are skipped, with progress saved every 10 captions.

`--game` picks the tileset; `--backend` picks where the model runs. Local Ollama is
the default and needs no key:

```bat
python MM2_Files\MarioMaker_llm_captions.py --game MM2 --dataset dataset.json --output dataset_captioned.json --backend ollama --model qwen2.5:14b --num-captions 5
```

A missing Ollama model is pulled automatically the first time. To use a hosted
backend, point `--api-key-file` at a `.txt` file with the key on its first line:

```bat
python MM2_Files\MarioMaker_llm_captions.py --game MM2 --dataset dataset.json --output dataset_captioned.json --backend claude --api-key-file key.txt
```

Vision-capable models (Claude, GPT-4o, Gemini, or an Ollama vision model) get each
scene's rendered PNG sent alongside the grid automatically. Build the dataset with
`mm2pipeline_data dataset build --with_images` first so every sample has an `image`,
and pass `--images-dir` if the PNGs live somewhere other than next to the dataset:

```bat
python MM2_Files\MarioMaker_llm_captions.py --game MM2 --dataset dataset.json --output dataset_captioned.json --backend claude --api-key-file key.txt --images-dir out\images
```

Useful knobs: `--caption-mode keyed` (accumulate captions from several models under
per-model keys), `--grid-format tokens` (feed `T<NN>` tokens instead of ASCII —
easier for the model to count), `--num-ctx` / `--max-num-ctx` (Ollama context
window; grown per-scene so a wide grid can't starve generation). The full rendered
prompt for the last scene is dumped to `MM2_Prompt.txt` at the end of each run
(`--no-prompt-log` to print it to the console instead).

## Rendering

`render_mm2.py` turns a scene of tile ids back into real Mario Maker 2 artwork,
using the spritesheet and gamestyle tilesheets under `toost_stuff/`. Multi-tile
objects are reconstructed from their glyph blocks (a 2×2 patch of Thwomp becomes one
Thwomp, a run of `#` gets grass tops and inner corners, pipes get rims and bodies),
so scenes read the way toost would draw them without needing a toost checkout.

There's no command line — the ASCII data browser, the interactive generator, and
the sample-rendering code import it. `mm2_tiles(gamestyle)` returns one image per
tile id, and `_render_mm2_samples(...)` renders whole scenes to PNGs.

## Evaluation

Scores generated levels: broken-structure counts, average min edit distance
(diversity, and distance from a real dataset), caption adherence when entries carry
prompts, and same-source diversity. Two modes — walk a model directory or score one
`all_levels.json`:

```bat
REM Walk a model dir; writes an evaluation_metrics.json next to each all_levels.json.
python -m MM2_Files.evaluate_mm2_metrics --model_path MODEL_DIR --game MM2

REM Score a single file, with a real dataset for AMED_real.
python -m MM2_Files.evaluate_mm2_metrics --json MODEL-unconditional-samples-short\all_levels.json --real_json datasets\MM2_LevelsAndCaptions-regular.json
```

`--cpu` skips the GPU, `--override` recomputes over an existing metrics file, and
`--skip_adherence` drops caption scoring (use it when the prompts are LLM captions
rather than deterministic ones).

## Other files

- `toost_stuff/` — the bundled `toost.exe`, spritesheet, per-gamestyle tilesheets,
  and font used for decoding and rendering.
- `replacements.md` — notes on which MM2 objects get dropped or swapped when
  extracting and exporting to `.swe` (e.g. Fish Bone → Cheep Cheep).
- `MM2_Prompt.txt` — the last prompt log written by the LLM captioner.
- `dataset_captioned.json` — a sample captioned dataset.
