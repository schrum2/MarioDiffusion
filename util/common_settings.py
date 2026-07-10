
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

# Mario 1

MARIO_HEIGHT = 16 
MARIO_WIDTH = 16
MARIO_TILESET = 'datasets/smb.json'

MARIO_TILE_PIXEL_DIM = 16
MARIO_TILE_COUNT = 13

# Mario Maker 2

# Mario Maker uses the canonical MM2 tileset (mm2_tileset_we.json). The MM2 data is
# encoded as sorted(tileset['tiles']) + the appended '_' padding tile, giving 68
# tile ids (0-67): 67 real tiles plus '_'. The trained block2vec embeddings and the
# scene data both use this 68-id range, so the tile count must match the tileset.
# Tiles are rendered from img/spritesheet.png using the per-object {x,y,w,h}
# rectangles in toost's LevelData.hpp ObjectLocation map (see mm2_tiles()).

MM2_TILE_COUNT = 68
MM2_TILESET = 'datasets/mm2_tileset_we.json'
MM2_TILE_PIXEL_DIM = 16
MM2_WIDTH = 20 # Patrick: 20x20 is just what I've been using. Changing this will not result in any crashes that I'm aware of.
MM2_HEIGHT = 20
MM2_GAMESTYLE = 'SMW' # Default game style used to pick sprites (data is generated SMW by default in mm2pipeline_data.ascii). One of: SMB1, SMB3, SMW, NSMBU, SM3DW.
MM2_SKY_COLOR = (0x5C, 0x94, 0xFC) #MM2 sky-blue backdrop 


# Loderunner

LR_HEIGHT = 32
LR_WIDTH = 32

LR_TILE_PIXEL_DIM = 8
LR_TILE_COUNT = 8

LR_TILESET = 'datasets/Loderunner.json'

# Mega Man

MEGAMAN_HEIGHT = 14
MEGAMAN_WIDTH = 16

MM_TILE_PIXEL_DIM = 16
MM_SIMPLE_TILE_COUNT = 13
MM_FULL_TILE_COUNT = 41

MM_FULL_TILESET = 'datasets/MM.json'
MM_SIMPLE_TILESET = 'datasets/MM-simple-tileset.json'

# Mega Man Maker (MMLV) shares Mega Man's scene shape and pixel dim, but its tileset is the
# full VGLC set plus the conveyor-belt tiles ('>' / 'E'), so it has 2 extra tile types.
MMLV_TILE_COUNT = 49
MMLV_TILESET = 'datasets/MMLV.json'