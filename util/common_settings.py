
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5

MARIO_HEIGHT = 16
MARIO_WIDTH = 16

MARIO_TILE_PIXEL_DIM = 16
MARIO_TILE_COUNT = 13

MARIO_TILESET = 'datasets/smb.json'

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