The Conversion Process
Files Added to MarioDiffusion
Pipeline:

init.py — makes the pipeline a proper importable package.
main.py — lets you run the whole pipeline from the command line.
bcd.py — decrypts and unpacks Mario Maker's raw .bcd level format.
extract.py — pulls the raw level dumps down from HuggingFace.
tiles.py — the tile/tileset matching logic (the good version of the old level.py).
ascii.py — turns decoded levels into the ASCII grids everything else trains on.
dataset.py — ties it all together into a dataset build command.
swe.py — exports playable .swe files (brought over because you asked for it).
paths.py — central place for all the file paths the pipeline needs.
README.md — docs for how to actually run the thing.
Game_MM2:

render_mm2.py — draws MM2 levels back out as images.
MarioMaker_create_ascii_captions.py — generates the deterministic text captions for levels.
MarioMaker_llm_captions.py — the LLM-based captioning alternative.
evaluate_mm2_metrics.py — scores generated MM2 levels.
MM2_Prompt.txt — the prompt text for the LLM captioner.
replacements.md — the tile-name mapping reference.
dataset_captioned.json — a sample captioned dataset to test against.
init.py — makes Game_MM2 importable as a module.
toost_stuff/bin/toost.exe — the binary that actually decodes .bcd into JSON/PNG.
toost_stuff/img/spritesheet.png — the one spritesheet the renderer needs.
toost_stuff/img/tile/22349-0.png — the single ground tile our SMW/theme-0 render path uses.
Misc:

captions/MM2_caption_generator.py — builds MM2 captions alongside the existing MM/LR ones.
captions/MM2_caption_match.py — checks whether a level matches its caption.
astar/MM2State.py — MM2's version of the A* pathfinding state, next to the other games'.
util/mm2_metrics.py — helper functions for the MM2 metrics.
Game_MM2/mm2_tileset_we.json — the actual MM2 tileset definition (68 tiles).
Files Edited already in MarioDiffusion
util/common_settings.py — added the MM2_* constants so every other file knows MM2's size and tiles.
requirements.txt — added pycryptodome, which the .bcd decryption needs.
.gitignore — so the .bcd/.swe junk stays out but the few real MM2 assets get tracked.
mm2pipeline_data/paths.py — repointed the paths at MarioDiffusion's folders instead of MarioMakerPCG's.
mm2pipeline_data/toost.py — fixed the paths and made it actually print toost's errors when it fails.
mm2pipeline_data/dataset.py — ripped out the extended-tileset stuff and added the caption/tokenizer options.
train_diffusion.py — added MM2 as another --game option to train on.
level_dataset.py — taught it how to load/visualize MM2 levels.
evaluate_caption_adherence.py — added the MM2 case so it can score MM2 too.
evaluate_tile_distribution.py — same deal, added MM2 to the tile-stats.
text_to_level_diffusion.py — added MM2 to the text-to-level generation path.
run_diffusion.py — added MM2 as a runnable game.
split_data.py — added MM2 handling for splitting datasets.
interactive_tile_level_generator.py — added MM2 to the interactive editor GUI.
ascii_data_browser.py — added MM2 so you can browse its levels.
evolve_interactive_conditional_diffusion.py — added MM2 to the conditional evolution tool.
evolve_interactive_unconditional_diffusion.py — added MM2 to the unconditional one too.
astar/astar_traversability_check.py — added MM2 so it can check if MM2 levels are beatable.
astar/astar_path_visualization.py — added MM2 to the path-drawing tool.
captions/MM2_caption_match.py — renamed the leftover bare "MM" references to "MM2" so they don't collide with Mega Man.