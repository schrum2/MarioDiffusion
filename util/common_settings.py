
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

# Jacob: I anticipate conflicts between this file and the original common_settings.py.
#        So, this needs to be looked over carefully.

# Mario Maker uses the canonical MM2 tileset (mm2_tileset_we.json). The MM data is
# encoded as sorted(tileset['tiles']) + the appended '_' padding tile, giving 68
# tile ids (0-67): 67 real tiles plus '_'. The trained block2vec embeddings and the
# scene data both use this 68-id range, so the tile count must match the tileset.
# Regenerate datasets and block_embeddings.pt after changing the tileset. Using a
# smaller tileset (e.g. the 17-tile extended_tiles.json) makes the model emit tile
# ids outside the tileset and assign_caption raises KeyError: <id> on lookup.

# Jacob: These were 20x20 in MarioMakerPCG but I set them back to 16x16. Will this cause problems?
MARIO_HEIGHT = 16
MARIO_WIDTH = 16
MARIO_TILESET = 'datasets/smb.json'

MARIO_TILE_PIXEL_DIM = 16
MARIO_TILE_COUNT = 13

# Jacob: Make sure other files use these:
#        (already used in evaluate_caption_adherence.py)

# Kept as aliases of the canonical MM2 tileset/count above so older "MM_EXTENDED"
# callers stay in sync.
MM_EXTENDED_TILE_COUNT = 68
MM_EXTENDED_TILESET = 'datasets/mm2_tileset_we.json'

# Mario Maker 2 (the canonical training tileset; see memory/canonical-mm-tileset).
# Tiles are rendered from img/spritesheet.png using the per-object {x,y,w,h}
# rectangles in toost's LevelData.hpp ObjectLocation map (see mm2_tiles()).
MM2_TILESET = 'datasets/mm2_tileset_we.json'
MM2_TILE_PIXEL_DIM = 16
# Jacob: Is 20x20 correct? I thought we switched to 32x32. I guess this is overridden? However, 
#        I'm using these in evaluate_caption_adherence.py now
MM2_WIDTH = 20
MM2_HEIGHT = 20
# Default game style used to pick sprites (data is generated SMW by default in
# mm2pipeline_data.ascii). One of: SMB1, SMB3, SMW, NSMBU, SM3DW.
MM2_GAMESTYLE = 'SMW'
# MM2 sky-blue backdrop (toost's canvas background, #5C94FC) — composited behind
# each (often transparent) sprite so the tile grid reads correctly.
MM2_SKY_COLOR = (0x5C, 0x94, 0xFC)


LR_HEIGHT = 32
LR_WIDTH = 32

LR_TILE_PIXEL_DIM = 8
LR_TILE_COUNT = 8

LR_TILESET = 'datasets/Loderunner.json'

MEGAMAN_HEIGHT = 14
MEGAMAN_WIDTH = 16

MM_TILE_PIXEL_DIM = 16
MM_SIMPLE_TILE_COUNT = 13
MM_FULL_TILE_COUNT = 41

MM_FULL_TILESET = 'datasets/MM.json'
MM_SIMPLE_TILESET = 'datasets/MM-simple-tileset.json'

# Mega Man Maker (MMLV) shares Mega Man's scene shape and pixel dim, but its tileset is the
# full VGLC set plus the conveyor-belt tiles ('>' / 'E'), so it has 2 extra tile types.
MMLV_TILE_COUNT = 43
MMLV_TILESET = 'datasets/MMLV.json'