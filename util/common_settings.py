import os

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

# The single master metadata sidecar for downloaded Mega Man Maker levels: a global file in
# the repo, keyed by MMLV level id, holding each level's name/author/downloads/likes/dislikes.
# megaman/Bulk_Download.py writes/updates it as levels are downloaded; create_megaman_json_data.py
# reads it to attach that metadata to every generated sample. Resolved as an absolute path off
# the repo root so both entry points hit the same file regardless of their working directory.
MEGAMAN_METADATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'datasets', 'megaman_level_metadata.json')

# Shared game metadata and helpers used across the app
GAME_DISPLAY_NAMES = [
    "Mario",
    "Lode Runner",
    "Mega Man (Simple)",
    "Mega Man (Full)",
    "Mega Man (Maker)",
    "Mario Maker 2",
]
GAME_CLI_CHOICES = ["Mario", "LR", "MM-Simple", "MM-Full", "MMLV", "MM2"]
GAME_ALIASES = {
    "Mario": "Mario",
    "LR": "Lode Runner",
    "MM-Simple": "Mega Man (Simple)",
    "MM-Full": "Mega Man (Full)",
    "MMLV": "Mega Man (Maker)",
    "MM2": "Mario Maker 2",
}
MEGAMAN_GAMES = {"Mega Man (Simple)", "Mega Man (Full)", "Mega Man (Maker)"}


def normalize_game_name(game):
    if game is None:
        return "Mario Maker 2"
    if not isinstance(game, str):
        return str(game)
    normalized = game.strip()
    if normalized in GAME_ALIASES:
        return GAME_ALIASES[normalized]
    return normalized


def get_game_config(game=None):
    game_name = normalize_game_name(game)
    if game_name == "Mario":
        return {
            "name": "Mario",
            "cli_name": "Mario",
            "tileset": MARIO_TILESET,
            "tile_count": MARIO_TILE_COUNT,
            "width": MARIO_WIDTH,
            "height": MARIO_HEIGHT,
            "render_name": "Mario",
            "is_mario": True,
            "is_lode_runner": False,
            "is_mario_maker_2": False,
            "is_megaman": False,
            "supports_per_image_play": True,
            "is_composed_playable": True, 
        }
    if game_name == "Lode Runner":
        return {
            "name": "Lode Runner",
            "cli_name": "LR",
            "tileset": LR_TILESET,
            "tile_count": LR_TILE_COUNT,
            "width": LR_WIDTH,
            "height": LR_HEIGHT,
            "render_name": "LR",
            "is_mario": False,
            "is_lode_runner": True,
            "is_mario_maker_2": False,
            "is_megaman": False,
            "supports_per_image_play": True,
            "is_composed_playable": False, 
        }
    if game_name == "Mega Man (Simple)":
        return {
            "name": "Mega Man (Simple)",
            "cli_name": "MM-Simple",
            "tileset": MM_SIMPLE_TILESET,
            "tile_count": MM_SIMPLE_TILE_COUNT,
            "width": MEGAMAN_WIDTH,
            "height": MEGAMAN_HEIGHT,
            "render_name": "MM-Simple",
            "is_mario": False,
            "is_lode_runner": False,
            "is_mario_maker_2": False,
            "is_megaman": True,
            "supports_per_image_play": False,
            "is_composed_playable": False, 
        }
    if game_name == "Mega Man (Full)":
        return {
            "name": "Mega Man (Full)",
            "cli_name": "MM-Full",
            "tileset": MM_FULL_TILESET,
            "tile_count": MM_FULL_TILE_COUNT,
            "width": MEGAMAN_WIDTH,
            "height": MEGAMAN_HEIGHT,
            "render_name": "MM-Full",
            "is_mario": False,
            "is_lode_runner": False,
            "is_mario_maker_2": False,
            "is_megaman": True,
            "supports_per_image_play": False,
            "is_composed_playable": False, 
        }
    if game_name == "Mega Man (Maker)":
        return {
            "name": "Mega Man (Maker)",
            "cli_name": "MMLV",
            "tileset": MMLV_TILESET,
            "tile_count": MMLV_TILE_COUNT,
            "width": MEGAMAN_WIDTH,
            "height": MEGAMAN_HEIGHT,
            "render_name": "MMLV",
            "is_mario": False,
            "is_lode_runner": False,
            "is_mario_maker_2": False,
            "is_megaman": True,
            "supports_per_image_play": False,
            "is_composed_playable": False, 
        }
    if game_name == "Mario Maker 2":
        return {
            "name": "Mario Maker 2",
            "cli_name": "MM2",
            "tileset": MM2_TILESET,
            "tile_count": MM2_TILE_COUNT,
            "width": MM2_WIDTH,
            "height": MM2_HEIGHT,
            "render_name": "MM2",
            "is_mario": False,
            "is_lode_runner": False,
            "is_mario_maker_2": True,
            "is_megaman": False,
            "supports_per_image_play": False,
            "is_composed_playable": True, 
        }
    raise ValueError(f"Unsupported game selected: {game_name}")